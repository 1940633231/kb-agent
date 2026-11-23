"""Unified prompt/context budget across prompt, history, evidence and output."""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional

from agent.evidence_contract import (
    EvidenceObligation,
    assess_obligations,
)


class _CharCounter:
    @staticmethod
    def count(text: str) -> int:
        return len(text or "")


def _count(token_counter: Any, text: str) -> int:
    return int(token_counter.count(text or ""))


def _messages_tokens(token_counter: Any, messages: List[Dict[str, str]]) -> int:
    return sum(
        _count(token_counter, str(message.get("content") or ""))
        for message in messages or []
    )


def _history_turns(history: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    turns: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    for message in history or []:
        role = message.get("role")
        if role == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
        if role == "assistant":
            turns.append(current)
            current = []
    if current:
        turns.append(current)
    return turns


def trim_history(
        history: List[Dict[str, str]],
        token_counter: Any,
        max_tokens: int
        ) -> Dict[str, Any]:
    """Keep recent complete turns within max_tokens; drop oldest first."""
    turns = _history_turns(history)
    kept_reversed: List[List[Dict[str, str]]] = []
    used = 0
    skipped = 0
    for turn in reversed(turns):
        turn_tokens = _messages_tokens(token_counter, turn)
        if used + turn_tokens <= max_tokens:
            kept_reversed.append(turn)
            used += turn_tokens
        else:
            skipped += len(turn)
    kept = [message for turn in reversed(kept_reversed) for message in turn]
    return {
        "history": kept,
        "history_tokens": used,
        "skipped_history_turns": skipped,
    }


def _hit_facets(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Evidence chunk 的公司/年份/文档标签，用于给上下文加显式边界。"""
    metadata = hit.get("metadata") or {}
    document_id = str(
        hit.get("document_id") or metadata.get("document_id") or "")
    year = metadata.get("year")
    if year is None and "-" in document_id:
        year = document_id.rsplit("-", 1)[1]
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    return {
        "document_id": document_id,
        "company": metadata.get("company"),
        "year": year,
    }


def _chunk_label(index: int, chunk: Dict[str, Any]) -> str:
    """单条证据的头部标签：把公司/年份/文档写在数值旁边，避免跨文档混配。"""
    facets = _hit_facets(chunk)
    return "[{}]｜公司：{}｜年份：{}｜文档：{}".format(
        index,
        facets.get("company") or "未知",
        facets.get("year") or "未知",
        facets.get("document_id") or "未知",
    )


def _is_truncated_copy(current: Dict[str, Any], full: Dict[str, Any]) -> bool:
    """压缩器会把内容按 token 截断；被截断的副本要用完整证据替换。"""
    cur = str(current.get("content") or "")
    whole = str(full.get("content") or "")
    if not cur or not whole or cur == whole:
        return False
    return whole.startswith(cur) and len(cur) < len(whole)


def _obligation_checklist(
        obligations: Iterable[EvidenceObligation | Dict[str, Any]],
        results: List[Dict[str, Any]],
        chunk_numbers: Dict[str, int],
        ) -> str:
    """把证据义务渲染成"必答清单"，逐项指明对应证据编号或缺失。"""
    parsed = [
        item if isinstance(item, EvidenceObligation)
        else EvidenceObligation.from_dict(item)
        for item in obligations or []
    ]
    if not parsed:
        return ""
    by_id = {str(item.get("obligation_id")): item for item in results or []}
    lines = ["【必答清单】共{}项，逐项作答，缺一不可：".format(len(parsed))]
    for obligation in parsed:
        slot = "｜".join(part for part in (
            obligation.company,
            str(obligation.year) if obligation.year else None,
            obligation.metric,
        ) if part) or "未命名义务"
        result = by_id.get(obligation.obligation_id) or {}
        status = result.get("status")
        if status == "covered":
            numbers = [
                chunk_numbers[str(chunk_id)]
                for chunk_id in result.get("matched_chunks") or []
                if str(chunk_id) in chunk_numbers
            ]
            ref = "证据{}".format(
                "".join("[{}]".format(number) for number in numbers[:2])
            ) if numbers else "证据"
        elif status == "ambiguous":
            ref = "证据存在多口径，须列出各口径数值并标注口径"
        else:
            ref = "证据缺失（该项须明确说明无法回答，不得推测）"
        lines.append("- {}：{} → {}".format(
            obligation.obligation_id, slot, ref))
    return "\n".join(lines)


def _critical_evidence(
        hits: List[Dict[str, Any]],
        obligations: Iterable[EvidenceObligation | Dict[str, Any]],
        token_counter: Any
        ) -> Dict[str, Any]:
    parsed = [
        item if isinstance(item, EvidenceObligation)
        else EvidenceObligation.from_dict(item)
        for item in obligations or []
    ]
    if not parsed:
        return {"chunks": [], "tokens": 0, "missing": []}
    result = assess_obligations(hits, parsed)
    by_chunk = {
        str(item.get("chunk_id")): item for item in hits
        if item.get("chunk_id")
    }
    selected: List[Dict[str, Any]] = []
    seen = set()
    missing = []
    for item in result["results"]:
        if item["status"] not in ("covered", "ambiguous"):
            missing.append(item["obligation_id"])
            continue
        picked = 0
        for chunk_id in (item.get("matched_chunks") or [])[:2]:
            key = str(chunk_id)
            if key in seen or key not in by_chunk:
                continue
            seen.add(key)
            selected.append(by_chunk[key])
            picked += 1
            if picked >= 2:
                break
    tokens = sum(_count(token_counter, item.get("content") or "")
                 for item in selected)
    return {"chunks": selected, "tokens": tokens, "missing": missing}


def _preserve_critical_context(
        chunks: List[Dict[str, Any]],
        critical: List[Dict[str, Any]],
        available_tokens: int,
        token_counter: Any,
        per_chunk_overhead: int = 0,
        ) -> tuple[List[Dict[str, Any]], int, int]:
    def cost(item: Dict[str, Any]) -> int:
        return _count(token_counter, item.get("content") or "") + per_chunk_overhead

    final = list(chunks or [])
    critical_by_id = {
        str(item.get("chunk_id")): item for item in critical or []
        if item.get("chunk_id")
    }
    # 压缩器会把 chunk 截到 token 上限，可能正好把金额/年份截断（曾出现
    # "47,074,853,106." 这样的半截数字）。关键证据必须用完整原文替换，
    # 否则模型只能拒答或引用错值。
    for index, item in enumerate(final):
        key = str(item.get("chunk_id") or "")
        full = critical_by_id.get(key)
        if full is not None and _is_truncated_copy(item, full):
            final[index] = full
    existing = {
        str(item.get("chunk_id")) for item in final
        if item.get("chunk_id")
    }
    critical_ids = set(critical_by_id)
    content_used = sum(
        _count(token_counter, item.get("content") or "") for item in final)
    used = sum(cost(item) for item in final)
    removed = 0
    while used > available_tokens and final:
        removable = next(
            (index for index in range(len(final) - 1, -1, -1)
             if str(final[index].get("chunk_id")) not in critical_ids),
            None,
        )
        if removable is None:
            break
        removed_chunk = final.pop(removable)
        content_used -= _count(
            token_counter, removed_chunk.get("content") or "")
        used -= cost(removed_chunk)
        removed += 1
    for candidate in critical or []:
        chunk_id = str(candidate.get("chunk_id"))
        if chunk_id in existing:
            continue
        needed = cost(candidate)
        while used + needed > available_tokens and final:
            removable = next(
                (index for index in range(len(final) - 1, -1, -1)
                 if str(final[index].get("chunk_id")) not in critical_ids),
                None,
            )
            if removable is None:
                break
            removed_chunk = final.pop(removable)
            content_used -= _count(
                token_counter, removed_chunk.get("content") or "")
            used -= cost(removed_chunk)
            removed += 1
        if used + needed <= available_tokens:
            final.append(candidate)
            existing.add(chunk_id)
            content_used += _count(
                token_counter, candidate.get("content") or "")
            used += needed
    return final, content_used, removed


def build_context_with_budget(
        pipeline: Any,
        question: str,
        hits: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        obligations: Optional[
            Iterable[EvidenceObligation | Dict[str, Any]]] = None,
        max_context_tokens: Optional[int] = None,
        output_reserved_tokens: Optional[int] = None,
        ) -> Dict[str, Any]:
    """Build retrieval context while reserving prompt, history and output."""
    manager = pipeline.context_manager
    token_counter = getattr(manager, "token_counter", _CharCounter())
    max_tokens = int(
        max_context_tokens or getattr(manager, "max_context_tokens", 4096)
    )
    output_tokens = int(
        output_reserved_tokens
        if output_reserved_tokens is not None
        else 1024
    )
    base_messages = pipeline.prompt_builder.build(
        question, "", history=[])
    base_tokens = _messages_tokens(token_counter, base_messages)
    question_tokens = _count(token_counter, question)
    fixed_tokens = max(0, base_tokens - question_tokens)
    system_tokens = _count(
        token_counter,
        next((message.get("content") or "" for message in base_messages
              if message.get("role") == "system"), ""),
    )
    parsed_obligations = [
        item if isinstance(item, EvidenceObligation)
        else EvidenceObligation.from_dict(item)
        for item in obligations or []
    ]
    critical = _critical_evidence(hits, parsed_obligations, token_counter)
    # 只为"必答清单"预留固定开销；每条证据的标签开销在装填时按实际 chunk
    # 逐条计入，避免按候选总数粗估而把真正有用的证据挤出预算。
    overhead_reserve = (
        (24 + 46 * len(parsed_obligations)) if parsed_obligations else 0
    )
    remaining_for_history = max(
        0,
        max_tokens - fixed_tokens - overhead_reserve - question_tokens
        - output_tokens - critical["tokens"],
    )
    history_result = trim_history(
        history or [], token_counter, remaining_for_history)

    reserved = (
        fixed_tokens + overhead_reserve
        + history_result["history_tokens"] + output_tokens
    )
    request_manager = copy.copy(manager)
    request_manager.max_context_tokens = max_tokens
    request_manager.reserved_tokens = min(
        reserved, max(0, max_tokens - question_tokens))
    context_result = request_manager.build(question, hits)
    context_stats = dict(context_result.get("stats") or {})
    context_stats["query_tokens"] = question_tokens

    available_evidence = max(
        0, max_tokens - reserved - question_tokens)
    label_overhead = _count(token_counter, _chunk_label(1, {
        "company": "未知", "year": 2025, "document_id": "000000-2025"}))
    final_chunks, evidence_tokens, removed = _preserve_critical_context(
        context_result.get("chunks") or [],
        critical["chunks"],
        available_evidence,
        token_counter,
        per_chunk_overhead=label_overhead,
    )
    chunk_numbers = {
        str(chunk.get("chunk_id")): index
        for index, chunk in enumerate(final_chunks, 1)
        if chunk.get("chunk_id")
    }
    checklist_results = (
        assess_obligations(
            final_chunks, parsed_obligations, question)["results"]
        if parsed_obligations else []
    )
    checklist = _obligation_checklist(
        parsed_obligations, checklist_results, chunk_numbers)
    labels = [
        _chunk_label(index, chunk)
        for index, chunk in enumerate(final_chunks, 1)
    ]
    format_tokens = (
        _count(token_counter, checklist)
        + sum(_count(token_counter, label) for label in labels)
    )
    body = "\n\n".join(
        "{}\n{}".format(label, chunk.get("content") or "")
        for label, chunk in zip(labels, final_chunks)
    )
    context_text = (
        "{}\n\n【证据】\n{}".format(checklist, body) if checklist else body
    )
    fixed_reported = fixed_tokens + format_tokens
    total_estimate = (
        fixed_reported + question_tokens
        + history_result["history_tokens"] + evidence_tokens + output_tokens
    )
    overflow = total_estimate > max_tokens
    missing_in_context = [
        item["obligation_id"] for item in checklist_results
        if item.get("status") == "missing"
    ] if checklist_results else critical["missing"]
    budget_stats = {
        "max_context_tokens": max_tokens,
        "system_tokens": system_tokens,
        "fixed_prompt_tokens": fixed_reported,
        "checklist_tokens": _count(token_counter, checklist),
        "obligation_count": len(parsed_obligations),
        "query_tokens": question_tokens,
        "history_tokens": history_result["history_tokens"],
        "evidence_tokens": evidence_tokens,
        "reserved_output_tokens": output_tokens,
        "skipped_history_turns": history_result["skipped_history_turns"],
        "skipped_evidence_chunks": (
            int(context_stats.get("skipped_count") or 0) + removed),
        "required_evidence_tokens": critical["tokens"],
        "missing_obligations_in_context": missing_in_context,
        "budget_overflow": overflow,
        "prompt_token_estimate": total_estimate,
        "context_final_count": len(final_chunks),
        "context_manager": context_stats,
    }
    return {
        "context": context_text,
        "chunks": final_chunks,
        "history": history_result["history"],
        "budget": budget_stats,
    }
