"""检索质量判断：低成本规则粗筛为主，分数只作参考（v2 §4.1）。

为什么不能只看分数（第零阶段实测）：
    应拒答题的 top-1 相似度是 0.609/0.615/0.725，可答题是 0.664~0.802，
    区间重叠：没有任何阈值能同时拦住该拒答的、又放过该回答的。
    而且 L0 用余弦、L1/L2 用 RRF/重排分数，刻度完全不同，阈值不可跨层复用。

因此判定以"问题要素是否被证据覆盖"为主：
    1. 年份覆盖：问题里出现的每个年份，证据里必须有该年份年报的片段
       （同公司不同年份语义高度相似，这是 L0 最典型的失败模式）；
    2. 公司覆盖：语料里的公司名要出现在证据文本里；
    3. 关键词覆盖：指标词（研发投入/职工薪酬/非利息净收入…）要出现在证据里；
    4. 分数只记录不判定，用于后续调参和回放。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_REGISTRY = PROJECT_ROOT / "data" / "docs" / "annual_reports" / "metadata.json"
ALIASES_REGISTRY = PROJECT_ROOT / "data" / "docs" / "annual_reports" / "company_aliases.json"

# 问题里的疑问/口语化成分 + 关系/比较类短语，都不参与关键词覆盖判定。
# 实测教训：漏掉"较上年末上升了"这类比较短语时，它会被当成"指标关键词"，
# 于是评分永远不通过、白烧两轮重试，最后把能答的题烧成拒答
# （ar-cross-003 就是这么错的）。
QUESTION_NOISE = (
    "是多少元", "是多少千元", "是多少亿元", "是多少百万元", "多少个百分点",
    "分别是多少", "保留两位小数", "单位不同", "分别是", "是多少", "有多少",
    "多少", "请问", "约为", "大约", "注意", "是否", "为什么", "哪些", "什么",
    "如何理解", "如何", "理解", "含义", "定义", "概念",
    "同年", "列出", "两位小数",
    # 关系/比较类短语
    "较上年末", "较上年底", "上年末", "上年同期", "上期", "同期", "同比", "环比",
    "百分点", "个百分点", "增减", "变动", "变化", "上升", "下降", "增加", "减少",
    # 单位词与泛化填充词：它们不是指标名，留在关键词里只会让覆盖判定假失败
    "分别", "比例", "明细", "披露", "实现", "情况", "数据",
    "分产品口径", "百万元", "千元", "万元", "亿元", "元",
    "两家", "一家", "三家",
    "两年", "三年", "多年", "本年", "当年", "全年",
    "合并利润表", "母公司利润表", "主要会计数据",
)

# 清洗后若一个 token 的字符全部来自这里，说明它是关系词残留（如"上升了"），不是指标名
RELATIONAL_CHARS = set(
    "较上年末同期同比环增减升降变动情况幅度分别为保留位小数单位注意与和及或是在于了"
)

KNOWN_METRICS = (
    "主要会计数据中的归母净利润",
    "主要会计数据中归母净利润",
    "合并利润表中的净利润",
    "合并利润表中净利润",
    "归属于上市公司股东的扣除非经常性损益后的净利润",
    "归属于上市公司股东的扣除非经常性损益的净利润",
    "扣除非经常性损益后的归母净利润",
    "扣除非经常性损益后的净利润",
    "归属于上市公司股东的净利润",
    "归属于母公司股东的净利润",
    "归属于本公司股东的净利润",
    "归属于母公司股东的营运利润",
    "归属于母公司股东营运利润",
    "经营活动产生的现金流量净额",
    "加权平均净资产收益率",
    "研发投入合计",
    "基本每股收益",
    "归母净利润",
    "扣非归母净利润",
    "归母营运利润",
    "营运利润",
    "经营现金流净额",
    "营业总收入",
    "营业收入",
    "研发费用",
    "销售费用",
    "净利润",
    "总资产",
    "资产总计",
)


def _load_companies() -> List[str]:
    """语料公司名来自抓取登记表，不在代码里硬编码。"""
    try:
        registry = json.loads(REPORTS_REGISTRY.read_text(encoding="utf-8"))
        names = sorted({entry["company"] for entry in registry["documents"]},
                       key=len, reverse=True)
        if names:
            return names
    except Exception:  # noqa: BLE001 - 语料缺失时退化为无实体匹配
        pass
    return []


COMPANIES = _load_companies()


def _load_company_aliases(companies: List[str]) -> Dict[str, List[str]]:
    try:
        payload = json.loads(ALIASES_REGISTRY.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        payload = {}
    return {
        company: [str(item) for item in payload.get(company, []) if item]
        for company in companies
    }


COMPANY_ALIASES = _load_company_aliases(COMPANIES)

# 纯寒暄/无信息量的输入：没有可检索的要素，直接走"不检索"分支
CHITCHAT = ("你好", "您好", "谢谢", "感谢", "在吗", "再见", "嗨", "hello", "hi", "hey")


def normalize(text: str) -> str:
    """去掉空白与千分位分隔符，便于跨换行/跨列的片段匹配。"""
    return re.sub(r"[\s,，\u3000]", "", text or "")


# 同一指标在不同年报里的写法差异（"研发费用" vs "研发投入"、
# "经营现金流净额" vs "经营活动产生的现金流量净额"）。判分、义务匹配、
# 答案审计共用这一份，避免三处各写一套而漏掉同一批别名。
METRIC_TERM_ALIASES = {
    "营业收入": ("营业总收入", "营业收入合计", "营业总收入合计", "总营业收入"),
    "营业总收入": ("营业收入", "营业总收入合计", "总营业收入"),
    "归母净利润": (
        "归属于上市公司股东的净利润",
        "归属于母公司股东的净利润",
        "归属于母公司所有者的净利润",
        "归属于本公司股东的净利润",
    ),
    "扣非归母净利润": (
        "扣除非经常性损益后的归母净利润",
        "扣除非经常性损益后的净利润",
        "归属于上市公司股东的扣除非经常性损益的净利润",
        "归属于上市公司股东的扣除非经常性损益后的净利润",
    ),
    "经营现金流净额": (
        "经营活动产生的现金流量净额",
        "经营活动现金流量净额",
        "经营活动现金流净额",
    ),
    "总资产": ("资产总计", "资产总额"),
    "研发费用": ("研发投入", "研发投入合计", "研发费用合计"),
    "研发投入": ("研发费用", "研发投入合计"),
}


def metric_mentioned(text: str, keyword: str) -> bool:
    """指标关键词是否在证据里出现，包含同义写法。"""
    normalized = normalize(text or "")
    if not keyword:
        return True
    if normalize(keyword) in normalized:
        return True
    return any(
        normalize(alias) in normalized
        for alias in METRIC_TERM_ALIASES.get(keyword, ())
    )


def company_terms(company: str) -> List[str]:
    terms = [company] + list(COMPANY_ALIASES.get(company, []))
    return sorted({item for item in terms if item}, key=len, reverse=True)


def company_position(question: str, company: str) -> int:
    positions = [
        question.find(term) for term in company_terms(company)
        if question.find(term) >= 0
    ]
    return min(positions) if positions else len(question) + 1


def company_mentioned(text: str, company: str) -> bool:
    normalized = normalize(text or "")
    return any(normalize(term) in normalized
               for term in company_terms(company))


def _extract_known_metrics(question: str) -> tuple[List[str], str]:
    occupied: List[tuple[int, int]] = []
    matches: List[tuple[int, str]] = []
    for phrase in sorted(KNOWN_METRICS, key=len, reverse=True):
        for match in re.finditer(re.escape(phrase), question):
            start, end = match.span()
            if any(start < used_end and end > used_start
                   for used_start, used_end in occupied):
                continue
            occupied.append((start, end))
            matches.append((start, phrase))
    matches.sort()
    remainder = question
    for start, phrase in sorted(matches, reverse=True):
        remainder = (
            remainder[:start] + " " + remainder[start + len(phrase):])
    return [phrase for _, phrase in matches], remainder


def extract_facets(question: str) -> Dict[str, List[str]]:
    """从问题里抽出可判定的要素：年份、公司、指标关键词。"""
    annual_question = re.sub(
        r"20\d{2}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?",
        " ",
        question,
    )
    years = sorted(set(re.findall(r"(20\d{2})", annual_question)))
    matches = []
    for name in COMPANIES:
        for term in company_terms(name):
            position = question.find(term)
            if position >= 0:
                matches.append((position, -len(term), name, term))
    companies = []
    matched_terms: Dict[str, List[str]] = {}
    for _, _, name, _ in sorted(matches):
        if name not in companies:
            companies.append(name)
    for name in companies:
        terms = [
            term for term in company_terms(name)
            if term in question
        ]
        matched_terms[name] = (
            [name] if name in terms else terms
        )

    metric_phrases, remainder = _extract_known_metrics(question)
    # 长短语优先替换，避免"是多少元"被"多少"先吃掉一半
    for noise in sorted(QUESTION_NOISE, key=len, reverse=True):
        remainder = remainder.replace(noise, " ")
    for name in companies:
        for term in matched_terms.get(name, [name]):
            remainder = remainder.replace(term, " ")
    remainder = re.sub(r"20\d{2}年?", " ", remainder)
    # 注意：不要把"在/于/对"当分隔符——它们会切开"归属于上市公司股东的净利润"
    remainder = re.sub(
        r"[0-9%．.、，,；;：:（）()《》\"'？?！!\[\]\s的与和及或是占较中]", " ",
        remainder)

    keywords: List[str] = []
    for token in re.split(r"\s+", remainder):
        token = token.strip()
        if len(token) < 2 or token in keywords:
            continue
        if all(char in RELATIONAL_CHARS for char in token):
            continue
        keywords.append(token)
    for metric in metric_phrases:
        if metric not in keywords:
            keywords.append(metric)
    return {"years": years, "companies": companies, "keywords": keywords}


def needs_knowledge_base(question: str) -> bool:
    """是否需要走知识库检索：有年份/公司，或有非寒暄的指标关键词。"""
    facets = extract_facets(question)
    if facets["years"] or facets["companies"]:
        return True
    return any(keyword not in CHITCHAT for keyword in facets["keywords"])


def _evidence_text(hits: Iterable[Dict]) -> str:
    return normalize(" ".join((hit.get("content") or "") for hit in hits))


def _evidence_documents(hits: Iterable[Dict]) -> Dict[str, Dict]:
    documents: Dict[str, Dict] = {}
    for hit in hits:
        metadata = hit.get("metadata") or {}
        document_id = hit.get("document_id") or metadata.get("document_id")
        if document_id:
            documents.setdefault(document_id, metadata)
    return documents


def grade_evidence(question: str, hits: List[Dict],
                   min_coverage: float = 0.6,
                   obligations: Optional[List[Dict[str, Any]]] = None,
                   route_coverage: Optional[List[Dict[str, Any]]] = None) -> Dict:
    """对当前证据打分并给出能否作答的判定。

    min_coverage: 指标关键词覆盖率下限，默认 0.6，容忍同义表述差异。
    """
    from agent.evidence_contract import (
        assess_obligations,
        build_obligations,
        scope_supported_by_corpus,
    )

    facets = extract_facets(question)
    text = _evidence_text(hits)
    documents = _evidence_documents(hits)
    if obligations is None:
        obligations = [item.as_dict() for item in build_obligations(question, facets)]
    obligation_result = assess_obligations(hits, obligations, question)
    unscoped_general = not facets["companies"] and not facets["years"]
    scope_supported = (
        True if unscoped_general
        else scope_supported_by_corpus(question, facets)
    )

    years_in_evidence = set()
    for document_id in documents:
        match = re.search(r"(20\d{2})", document_id)
        if match:
            years_in_evidence.add(match.group(1))

    missing_years = [year for year in facets["years"]
                     if year not in years_in_evidence]
    missing_companies = [
        name for name in facets["companies"]
        if not company_mentioned(text, name)
    ]
    keyword_hits = {kw: metric_mentioned(text, kw) for kw in facets["keywords"]}
    coverage = (sum(keyword_hits.values()) / len(keyword_hits)) if keyword_hits else 1.0

    reasons: List[str] = []
    if not hits:
        reasons.append("没有检索到任何候选片段")
    if missing_years:
        reasons.append("证据缺少问题涉及年份的年报片段: {}".format(",".join(missing_years)))
    if missing_companies:
        reasons.append("证据文本未出现公司: {}".format(",".join(missing_companies)))
    if keyword_hits and coverage < min_coverage:
        missing = [kw for kw, ok in keyword_hits.items() if not ok]
        reasons.append("指标关键词未覆盖: {}".format(",".join(missing)))
    if obligation_result["missing"]:
        reasons.append("缺少证据义务: {}".format(
            ",".join(obligation_result["missing"])))
    if obligation_result["ambiguous"]:
        reasons.append("证据义务存在多口径歧义: {}".format(
            ",".join(obligation_result["ambiguous"])))
    if not scope_supported:
        reasons.append("语料中没有问题点名的公司/年份年报")

    if unscoped_general:
        passed = (
            bool(hits)
            and not reasons
            and not obligation_result["refusal_expected"]
            and coverage >= min_coverage
        )
    else:
        passed = (scope_supported and not reasons
                  and not obligation_result["refusal_expected"]
                  and obligation_result["passed"])

    return {
        "passed": passed,
        "reason": "；".join(reasons) if reasons else (
            "证据已覆盖问题要素" if passed else "证据义务未完整覆盖"),
        "coverage": round(coverage, 3),
        "obligation_coverage": obligation_result["coverage"],
        "obligations": obligation_result["obligations"],
        "obligation_results": obligation_result["results"],
        "covered_obligations": obligation_result["covered"],
        "missing_obligations": [
            item for item in obligation_result["results"]
            if item["obligation_id"] in obligation_result["missing"]
        ],
        "ambiguous_obligations": [
            item for item in obligation_result["results"]
            if item["obligation_id"] in obligation_result["ambiguous"]
        ],
        "refusal_expected": (
            obligation_result["refusal_expected"]
            or (not scope_supported and not unscoped_general)
        ),
        "unscoped_general": unscoped_general,
        "route_coverage": route_coverage or [],
        "keyword_hits": keyword_hits,
        "missing_years": missing_years,
        "missing_companies": missing_companies,
        "candidate_count": len(hits),
        "document_count": len(documents),
        "documents": sorted(documents),
        "top1_score": float(hits[0].get("score", 0.0)) if hits else None,
        "facets": facets,
    }
