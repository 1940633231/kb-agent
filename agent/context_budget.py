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
        for chunk_id in item.get("matched_chunks") or []:
            key = str(chunk_id)
            if key in seen or key not in by_chunk:
                continue
            seen.add(key)
            selected.append(by_chunk[key])
            break
    tokens = sum(_count(token_counter, item.get("content") or "")
                 for item in selected)
    return {"chunks": selected, "tokens": tokens, "missing": missing}


def _preserve_critical_context(
        chunks: List[Dict[str, Any]],
        critical: List[Dict[str, Any]],
        available_tokens: int,
        token_counter: Any,
        ) -> tuple[List[Dict[str, Any]], int, int]:
    final = list(chunks or [])
    existing = {
        str(item.get("chunk_id")) for item in final
        if item.get("chunk_id")
    }
    critical_ids = {
        str(item.get("chunk_id")) for item in critical or []
        if item.get("chunk_id")
    }
    used = sum(_count(token_counter, item.get("content") or "") for item in final)
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
        used -= _count(token_counter, removed_chunk.get("content") or "")
        removed += 1
    for candidate in critical or []:
        chunk_id = str(candidate.get("chunk_id"))
        if chunk_id in existing:
            continue
        needed = _count(token_counter, candidate.get("content") or "")
        while used + needed > available_tokens and final:
            removable = next(
                (index for index in range(len(final) - 1, -1, -1)
                 if str(final[index].get("chunk_id")) not in critical_ids),
                None,
            )
            if removable is None:
                break
            removed_chunk = final.pop(removable)
            used -= _count(
                token_counter, removed_chunk.get("content") or "")
            removed += 1
        if used + needed <= available_tokens:
            final.append(candidate)
            existing.add(chunk_id)
            used += needed
    return final, used, removed


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
    critical = _critical_evidence(hits, obligations or [], token_counter)
    remaining_for_history = max(
        0,
        max_tokens - fixed_tokens - question_tokens
        - output_tokens - critical["tokens"],
    )
    history_result = trim_history(
        history or [], token_counter, remaining_for_history)

    reserved = (
        fixed_tokens + history_result["history_tokens"] + output_tokens
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
    final_chunks, evidence_tokens, removed = _preserve_critical_context(
        context_result.get("chunks") or [],
        critical["chunks"],
        available_evidence,
        token_counter,
    )
    context_text = "\n\n".join(
        "[{}] {}".format(index, chunk.get("content") or "")
        for index, chunk in enumerate(final_chunks, 1)
    )
    total_estimate = reserved + question_tokens + evidence_tokens
    overflow = total_estimate > max_tokens
    budget_stats = {
        "max_context_tokens": max_tokens,
        "system_tokens": system_tokens,
        "fixed_prompt_tokens": fixed_tokens,
        "query_tokens": question_tokens,
        "history_tokens": history_result["history_tokens"],
        "evidence_tokens": evidence_tokens,
        "reserved_output_tokens": output_tokens,
        "skipped_history_turns": history_result["skipped_history_turns"],
        "skipped_evidence_chunks": (
            int(context_stats.get("skipped_count") or 0) + removed),
        "required_evidence_tokens": critical["tokens"],
        "missing_obligations_in_context": critical["missing"],
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
