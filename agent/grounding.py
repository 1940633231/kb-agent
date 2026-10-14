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

# 报表/章节标识：同一指标在年报里常有多个口径（合并 vs 母公司），
# 实测错例 ar-fact-001 就是拿了"母公司利润表"的营业收入答合并口径的问题。
STATEMENT_MARKERS = ("合并", "母公司", "主要会计数据", "利润表", "资产负债表",
                     "现金流量表", "所有者权益变动表", "分季度")


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


def number_equivalents(number: str) -> List[str]:
    """数字的等价写法（含金额数量级换算），供引用覆盖等离线分析复用。"""
    return _equivalents(number)


def verify_grounding(answer: Optional[str], citations: List[Dict],
                     chunks: List[Dict], question: str,
                     accept_derived: bool = True) -> Dict[str, Any]:
    """返回 {checked, grounded, derived, ungrounded, verified}。

    accept_derived：把"由片段里若干数字加/减得到的值"也算通过。
    实测依据：`ar-fact-004` 的正确答案是四个季度归母净利润之**和**
    （82,320,067,101.68），`ar-calc-002` 是两年营业收入之**差**；
    如果一律要求逐字出现，这两类合法聚合会被误判成未落地。
    """
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
    raw_texts: List[str] = []
    for citation in citations or []:
        chunk = by_chunk.get(citation.get("chunk_id"))
        if chunk is None:
            continue
        candidate = normalize(chunk.get("content") or "")
        if not anchor_tokens or any(normalize(t) in candidate for t in anchor_tokens):
            cited_texts.append(candidate)
            # 抽数字必须用**原文**：归一化会去掉逗号与空格，把相邻数字粘成一个巨数
            raw_texts.append(chunk.get("content") or "")

    grounded: List[str] = []
    derived: List[str] = []
    ungrounded: List[str] = []
    numbers_by_text: List[float] = []
    for text in raw_texts:
        for token in extract_numbers(text):
            try:
                numbers_by_text.append(float(token.replace(",", "")))
            except ValueError:
                continue
    for number in numbers:
        variants = [normalize(v) for v in _equivalents(number)]
        hit = any(any(v and v in text for v in variants) for text in cited_texts)
        if hit:
            grounded.append(number)
            continue
        if accept_derived and _is_derived(number, numbers_by_text):
            derived.append(number)
            continue
        ungrounded.append(number)

    return {"checked": len(numbers), "grounded": grounded, "derived": derived,
            "ungrounded": ungrounded, "verified": not ungrounded,
            "metric_tokens": metric_tokens, "anchor_tokens": anchor_tokens,
            "citation_chunks": len(cited_texts)}


def _is_derived(number: str, pool: List[float], tolerance: float = 0.01) -> bool:
    """number 是否等于 pool 中若干数字的和或两数之差（相对误差 ≤1%）。"""
    try:
        target = float(number.replace(",", ""))
    except ValueError:
        return False
    if not pool or target == 0:
        return False

    def close(value: float) -> bool:
        return abs(value - target) <= abs(target) * tolerance

    if close(sum(pool)):
        return True
    if len(pool) >= 2:
        for left in pool:
            for right in pool:
                if left is right:
                    continue
                if close(left + right) or close(abs(left - right)):
                    return True
    return False


def detect_ambiguous_metric(question: str, chunks: List[Dict],
                            max_values: int = 4) -> Dict[str, Any]:
    """**同一份文档**里同一指标是否出现多个候选数值（口径/表格不同）。

    为什么要单独检测：`ar-fact-001` 问合并口径的营业收入，证据里同时有
    "主要会计数据"的 147,693,604,994.14 与"母公司利润表"的 81,415,570,788.59，
    两者都能通过"数字能否在片段里找到"的落地校验——落地校验管不了口径混淆。

    只看**单份文档内**的冲突：跨文档的多个数值是合法的（多公司/多年份题，
    每份文档各自一个值），不能据此触发修复。

    返回 {ambiguous, metric, values, document_id}。
    """
    tokens = extract_facets(question)["keywords"]
    if not tokens:
        return {"ambiguous": False, "metric": "", "values": []}
    longest = max(len(token) for token in tokens)
    anchors = [token for token in tokens if len(token) == longest]

    per_document: Dict[str, List[str]] = {}
    for chunk in chunks or []:
        content = chunk.get("content") or ""
        flat = normalize(content)
        if not any(normalize(anchor) in flat for anchor in anchors):
            continue
        document_id = chunk.get("document_id") or \
            (chunk.get("metadata") or {}).get("document_id") or "?"
        bucket = per_document.setdefault(document_id, [])
        for number in extract_numbers(content):
            if number not in bucket:
                bucket.append(number)

    for document_id, values in per_document.items():
        if len(values) >= 2:
            return {"ambiguous": True, "metric": anchors[0],
                    "values": values[:max_values], "document_id": document_id}
    return {"ambiguous": False, "metric": anchors[0], "values": []}


def has_statement_label(answer: str) -> bool:
    """答案里是否标注了口径/表格（合并、母公司、主要会计数据…）。"""
    text = answer or ""
    return any(marker in text for marker in STATEMENT_MARKERS)
