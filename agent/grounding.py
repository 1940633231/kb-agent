"""答案落地校验：答案里的数字必须能在"被引用且含问题指标词"的片段里找到。

为什么需要它（实测）：`ar-fact-004` 问的是"归属于上市公司股东的净利润"，
模型引用的却是同一份年报里的"净利润 85,305,341,965.73"；`ar-table-004` 问
"研发投入合计"，模型答的是同一文档同年的"研发项目明细"。
这类错误**既不是编造、也不是单纯的年份/公司混入**——模型引用的是真实片段、
真实数字，只是选错了指标名或选错了表。年份/公司过滤救不了它们。

判定口径：
    对答案里每个候选数字，找到它出现在哪些引用片段中；只有当**至少一个**
    这样的片段包含问题的**最长指标词**（如"归属于上市公司股东"而不是"净利润"）时，
    才算"落地"。用最长词而不是"全部词"或"任意词"：
        - 要求"全部词"会把"研发投入合计占营业收入比例"这类多词问题误判（实测偏严 0.591）；
        - 要求"任意词"会让"净利润"这种短词放行错答案（`ar-fact-004` 就会漏判）。
    放宽到最长词后，离线复核从 0.591 升到 0.636，且仍然抓住 `ar-fact-004`。
    同时接受数量级等价写法（元 / 千元 / 万元 / 百万元 / 亿元互转），
    避免把模型自己的正确单位换算误判成未落地。

    **它衡量的是"引用是否支撑答案里的数字"，不是"答案是否正确"**：
    实测标出的题里，一部分是答案错了（`ar-fact-004`、`ar-table-004`、`ar-crossco-002`），
    另一部分是答案对但引用片段没覆盖到该数字（`ar-table-001`、`ar-table-003`、
    `ar-cross-002`、`ar-calc-001`）——后者是引用可信度问题，不是正确性问题。
    因此本模块只做**标记**、不参与路由，也不能直接当错误率使用。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from agent.grading import extract_facets, normalize

# 至少 4 位数字、可带小数，且不是 4 位年份——太小的数字（如 0.95、7.21）误匹配风险高
_NUMBER = re.compile(r"\d[\d,]{2,}\.?\d*")
MONEY_SCALES = (1.0, 1e3, 1e4, 1e6, 1e8)


def extract_numbers(text: str) -> List[str]:
    """抽候选数字：过滤年份与过短数字，去重保序。"""
    numbers: List[str] = []
    for raw in _NUMBER.findall(text or ""):
        plain = raw.replace(",", "").rstrip(".")
        digits = plain.replace(".", "")
        if len(digits) < 4 or (len(digits) == 4 and 1900 <= int(digits) <= 2100):
            continue
        if raw not in numbers:
            numbers.append(raw)
    return numbers


def _equivalents(number: str) -> List[str]:
    """数字本身 + 各金额数量级的等价写法（含 2 位小数版本）。"""
    try:
        value = float(number.replace(",", ""))
    except ValueError:
        return [number]
    variants = {number, number.replace(",", "")}
    for scale in MONEY_SCALES:
        scaled = value / scale
        variants.add("{:.2f}".format(scaled))
        variants.add("{:,.2f}".format(scaled))
        variants.add("{:.0f}".format(scaled))
        variants.add("{:,}".format(round(scaled)))
    return [v for v in variants if v and v != "0" and set(v) != {"0"}]


def verify_grounding(answer: Optional[str], citations: List[Dict],
                     chunks: List[Dict], question: str) -> Dict[str, Any]:
    """返回 {checked, grounded, ungrounded, verified}。"""
    numbers = extract_numbers(answer or "")
    if not numbers:
        return {"checked": 0, "grounded": [], "ungrounded": [], "verified": True}

    metric_tokens = [token for token in extract_facets(question)["keywords"]]
    anchor_tokens: List[str] = []
    if metric_tokens:
        longest = max(len(token) for token in metric_tokens)
        anchor_tokens = [token for token in metric_tokens if len(token) == longest]
    by_chunk = {chunk.get("chunk_id"): chunk for chunk in chunks or []}
    cited_texts: List[str] = []
    for citation in citations or []:
        chunk = by_chunk.get(citation.get("chunk_id"))
        if chunk is None:
            continue
        candidate = normalize(chunk.get("content") or "")
        if not anchor_tokens or any(normalize(t) in candidate for t in anchor_tokens):
            cited_texts.append(candidate)

    grounded: List[str] = []
    ungrounded: List[str] = []
    for number in numbers:
        variants = [normalize(v) for v in _equivalents(number)]
        hit = any(any(v and v in text for v in variants) for text in cited_texts)
        (grounded if hit else ungrounded).append(number)

    return {"checked": len(numbers), "grounded": grounded,
            "ungrounded": ungrounded, "verified": not ungrounded,
            "metric_tokens": metric_tokens, "anchor_tokens": anchor_tokens,
            "citation_chunks": len(cited_texts)}
