"""生成侧修复：把"引不到出处的数字"改掉，而不是继续加重试。

诊断依据（实测，第二轮）：LLM 复核判定 `ar-fact-001/004` 的证据"够"是对的——
fact-001 的片段里有"14,769,360.50 万元（即 147,693,605,000 元）"，
fact-004 的四个季度归母净利润可以加总出 82,320,067,101.68 元。
错误发生在**生成侧**：拿了母公司利润表的数、把亿元/万元换错 10 倍、多值题漏答一半。
所以修复要在生成之后、定稿之前做一次，而不是让检索层再重试。

修复指令刻意收紧到三件事：替换为上下文里逐字出现的数值、标注所在表格/章节、
找不到就删掉并写明无法确认。不允许"再想想"，否则等于又给一次自由发挥的机会。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

REPAIR_SYSTEM_PROMPT = (
    "你是答案校订员。下面初稿里的部分数字无法在给定片段中逐字定位，或存在口径歧义"
    "（同一指标在多张表里出现）。请只做三件事：\n"
    "1. 把这些数字替换为片段中**逐字出现**的数值，并在其后用括号标注它所在表格或章节"
    "（例如“（主要会计数据）”“（合并利润表）”“（母公司利润表）”）；\n"
    "2. 同一指标在片段里有多个候选值时，只保留与问题口径一致的那个，并说明为什么不选另一个；\n"
    "   如果数值是由片段里若干数字**相加或相减**得到的，可以保留，但必须写明算式与每个加数；\n"
    "3. 片段里确实没有的，就删掉该数字并写明“无法确认”，不要用其它数字凑。\n"
    "保持原有的引用编号格式（如 [1][2]），不要新增片段里没有的内容。"
)


def build_repair_messages(question: str, evidence_text: str, draft: str,
                          ungrounded: List[str],
                          ambiguity: Optional[Dict[str, Any]] = None
                          ) -> List[Dict[str, str]]:
    ambiguous_line = ""
    if ambiguity and ambiguity.get("ambiguous"):
        ambiguous_line = (
            "\n口径歧义：指标“{}”在片段中出现多个候选值：{}\n"
        ).format(ambiguity.get("metric", ""), "、".join(ambiguity.get("values") or []))
    user_prompt = (
        "问题：{question}\n"
        "{ambiguous}"
        "\n可引用的片段：\n{evidence}\n"
        "\n初稿：\n{draft}\n"
        "\n无法逐字定位的数字：{ungrounded}\n"
        "\n请输出校订后的完整答案。"
    ).format(question=question, ambiguous=ambiguous_line, evidence=evidence_text,
             draft=draft, ungrounded="、".join(ungrounded) or "（见口径歧义）")
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def repair_answer(generator: Any, question: str, evidence_text: str, draft: str,
                  ungrounded: List[str],
                  ambiguity: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """调一次 LLM 做校订；异常返回 None，由调用方保留原稿。"""
    try:
        revised = generator.generate(
            build_repair_messages(question, evidence_text, draft, ungrounded, ambiguity))
    except Exception:  # noqa: BLE001 - 修复失败不应破坏已有答案
        return None
    revised = (revised or "").strip()
    return revised or None


CLAIM_REPAIR_SYSTEM_PROMPT = (
    "你是 Claim 定向校订器。只修正给定的硬违规 Claim，不要重写其它句子。"
    "每个修复必须是原答案中的原文字片段替换。"
    "不得引入证据片段中不存在的数字，不得改变与该 Claim 无关的引用或结论。"
    "只输出 JSON：{\"repairs\":[{\"claim_id\":\"claim-1\","
    "\"replacement\":\"替换后的原文字片段\"}]}。"
)


def build_claim_repair_messages(
        question: str,
        evidence_text: str,
        draft: str,
        claims: List[Dict[str, Any]],
        missing_obligations: Optional[List[Dict[str, Any]]] = None
        ) -> List[Dict[str, str]]:
    claim_block = "\n".join(
        "- {} [{}] {} | value={} | company={} | year={} | citations={}".format(
            item.get("claim_id"),
            item.get("claim_type"),
            item.get("text_span"),
            item.get("value"),
            item.get("company") or "",
            item.get("year") or "",
            item.get("citation_ids") or [],
        )
        for item in claims or []
    )
    obligation_block = "\n".join(
        "- {} {} {} {}".format(
            item.get("obligation_id"),
            item.get("company") or "",
            item.get("year") or "",
            item.get("metric") or "",
        ).strip()
        for item in missing_obligations or []
    ) or "无"
    user_prompt = (
        "问题：{question}\n\n"
        "需要修复的硬违规 Claim：\n{claims}\n\n"
        "缺失证据义务：\n{obligations}\n\n"
        "可引用证据：\n{evidence}\n\n"
        "原答案：\n{draft}"
    ).format(question=question, claims=claim_block,
             obligations=obligation_block, evidence=evidence_text, draft=draft)
    return [
        {"role": "system", "content": CLAIM_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _parse_claim_repairs(answer: Optional[str]) -> Dict[str, str]:
    if not answer:
        return {}
    match = re.search(r"\{.*\}", answer, re.S)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    repairs = payload.get("repairs") or []
    if not isinstance(repairs, list):
        return {}
    parsed: Dict[str, str] = {}
    for item in repairs:
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("claim_id") or "").strip()
        replacement = str(item.get("replacement") or "")
        if claim_id and replacement:
            parsed[claim_id] = replacement
    return parsed


def apply_claim_repairs(
        draft: str,
        claims: List[Dict[str, Any]],
        replacements: Dict[str, str]
        ) -> tuple[str, List[str]]:
    """Apply only the spans named by repaired claims, from end to start."""
    by_id = {
        str(item.get("claim_id")): item
        for item in claims or []
        if item.get("claim_id") in replacements
    }
    spans: Dict[tuple[int, int], tuple[str, str]] = {}
    for claim_id, item in by_id.items():
        start = int(item.get("start") or 0)
        end = int(item.get("end") or 0)
        if end <= start or end > len(draft):
            continue
        spans[(start, end)] = (claim_id, replacements[claim_id])
    repaired = draft
    applied: List[str] = []
    for (start, end), (claim_id, replacement) in sorted(
            spans.items(), key=lambda item: item[0][0], reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
        applied.append(claim_id)
    return repaired, sorted(applied)


def repair_claims(
        generator: Any,
        question: str,
        evidence_text: str,
        draft: str,
        claims: List[Dict[str, Any]],
        missing_obligations: Optional[List[Dict[str, Any]]] = None
        ) -> Dict[str, Any]:
    """Run one claim-targeted repair and apply only returned span replacements."""
    try:
        raw = generator.generate(build_claim_repair_messages(
            question, evidence_text, draft, claims, missing_obligations))
    except Exception:  # noqa: BLE001 - preserve the draft on external failure
        return {"answer": None, "applied_claim_ids": [], "raw": None}
    replacements = _parse_claim_repairs(raw)
    if not replacements:
        return {"answer": None, "applied_claim_ids": [], "raw": raw}
    allowed = {str(item.get("claim_id")) for item in claims or []}
    replacements = {
        claim_id: text for claim_id, text in replacements.items()
        if claim_id in allowed
    }
    if not replacements:
        return {"answer": None, "applied_claim_ids": [], "raw": raw}
    answer, applied = apply_claim_repairs(draft, claims, replacements)
    return {
        "answer": answer if applied else None,
        "applied_claim_ids": applied,
        "raw": raw,
    }
