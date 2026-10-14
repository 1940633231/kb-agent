"""本轮优化的回归测试：指标关键词抽取、facet 证据裁剪、答案落地校验。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evidence import scope_evidence  # noqa: E402
from agent.grading import extract_facets  # noqa: E402
from agent.grounding import extract_numbers, verify_grounding  # noqa: E402


def hit(document_id: str, content: str = "内容") -> Dict[str, Any]:
    return {"chunk_id": "{}#1".format(document_id), "document_id": document_id,
            "content": content, "score": 0.5, "metadata": {"document_id": document_id}}


# ---- 指标关键词抽取：关系/比较短语不能当指标名 ----

def test_relational_phrase_is_not_a_metric_keyword():
    """回归：漏掉"较上年末上升了"会让评分永远不通过，白烧两轮重试还答成拒答。"""
    facets = extract_facets(
        "招商银行2023年房地产业不良贷款率是多少，较上年末上升了多少个百分点？")
    assert facets["keywords"] == ["房地产业不良贷款率"]


def test_ratio_phrase_splits_into_metrics():
    facets = extract_facets(
        "贵州茅台2025年研发投入合计占营业收入的比例约为多少（%，保留两位小数）？")
    assert facets["keywords"] == ["研发投入合计", "营业收入"]


def test_metric_phrase_is_not_split_by_prepositions():
    """不要把"在/于/对"当分隔符——否则会切开"归属于上市公司股东的净利润"。"""
    keywords = extract_facets("贵州茅台2025年归属于上市公司股东的净利润是多少元？")["keywords"]
    assert "净利润" in keywords
    assert any("归属" in token for token in keywords)
    assert all(token not in ("归属", "上市公司股东") for token in keywords)


def test_unit_words_are_not_metric_keywords():
    facets = extract_facets("宁德时代2025年营业收入较2024年增加约多少亿元？")
    assert facets["keywords"] == ["营业收入"]


# ---- facet 证据裁剪 ----

def test_scope_keeps_only_named_company_and_year():
    hits = [hit("600519-2023"), hit("600519-2024"), hit("600036-2023")]
    kept, stats = scope_evidence("贵州茅台2023年的营业收入是多少元？", hits)
    assert [h["document_id"] for h in kept] == ["600519-2023"]
    assert stats["scoped"] is True and stats["dropped"] == 2


def test_scope_keeps_all_named_facets_for_cross_company_question():
    hits = [hit("600519-2023"), hit("600036-2023"), hit("600519-2024")]
    kept, stats = scope_evidence(
        "贵州茅台与招商银行2023年的营业收入分别是多少？", hits)
    assert sorted(h["document_id"] for h in kept) == ["600036-2023", "600519-2023"]
    assert stats["companies"] == ["600036", "600519"]


def test_scope_with_unknown_company_filters_by_year_only():
    """问题点名的公司不在语料里（应拒答题）：只有年份可用，就只按年份裁。"""
    hits = [hit("600519-2023"), hit("300750-2024")]
    kept, stats = scope_evidence("比亚迪2024年的营业收入是多少？", hits)
    assert [h["document_id"] for h in kept] == ["300750-2024"]
    assert stats["scoped"] is True


def test_scope_falls_back_when_nothing_matches():
    """裁剪后为空必须回退：证据为空会直接变成必然拒答。"""
    hits = [hit("600519-2023")]
    kept, stats = scope_evidence("比亚迪2026年的营业收入是多少？", hits)
    assert len(kept) == len(hits)
    assert stats["scoped"] is False


def test_scope_without_named_facets_is_noop():
    hits = [hit("600519-2023")]
    kept, stats = scope_evidence("这些公司表现如何？", hits)
    assert kept == hits and stats["scoped"] is False


# ---- 答案落地校验 ----

def test_extract_numbers_skips_years_and_short_numbers():
    numbers = extract_numbers("2023年营业收入147,693,604,994.14元，同比增长0.95%")
    assert numbers == ["147,693,604,994.14"]


def test_grounding_flags_number_from_wrong_metric_name():
    """实测错例：问"归属于上市公司股东的净利润"，答的是同一份年报里的"净利润"。"""
    chunk = {"chunk_id": "c1", "document_id": "600519-2025",
             "content": "四、净利润（净亏损以“－”号填列） 85,305,341,965.73 76,370,300"}
    citation = {"number": 2, "chunk_id": "c1", "document_id": "600519-2025"}
    outcome = verify_grounding("净利润为 85,305,341,965.73 元 [2]", [citation], [chunk],
                              "贵州茅台2025年归属于上市公司股东的净利润是多少元？")
    assert outcome["verified"] is False
    assert outcome["ungrounded"] == ["85,305,341,965.73"]


def test_grounding_accepts_correct_number_and_unit_conversion():
    chunk = {"chunk_id": "c1", "document_id": "600519-2023",
             "content": "主要会计数据 营业收入 147,693,604,994.14 归属于上市公司股东的净利润"}
    citation = {"number": 1, "chunk_id": "c1", "document_id": "600519-2023"}
    verbatim = verify_grounding("营业收入 147,693,604,994.14 元 [1]", [citation],
                               [chunk], "贵州茅台2023年的营业收入是多少元？")
    assert verbatim["verified"] is True

    # 模型自己换成"万元"也应当算落地（数量级等价）
    converted = verify_grounding("营业收入约 14,769,360.50 万元 [1]", [citation],
                                 [chunk], "贵州茅台2023年的营业收入是多少元？")
    assert converted["verified"] is True


def test_grounding_ignores_citation_chunk_without_metric_keyword():
    """引用的片段里没有问题的指标词时，即使数字碰巧出现也不算落地。"""
    chunk = {"chunk_id": "c9", "document_id": "600519-2025",
             "content": "经营活动产生的现金流量净额 85,305,341,965.73"}
    citation = {"number": 1, "chunk_id": "c9", "document_id": "600519-2025"}
    outcome = verify_grounding("净利润 85,305,341,965.73", [citation], [chunk],
                              "贵州茅台2025年的净利润是多少元？")
    assert outcome["verified"] is False


def test_grounding_uses_longest_metric_token_as_anchor():
    """多指标词的问题（研发投入合计 + 营业收入）只要求最长词出现，避免误判。"""
    chunk = {"chunk_id": "c1", "document_id": "600519-2025",
             "content": "研发投入情况表 研发投入合计 803,132,232.31"}
    citation = {"number": 1, "chunk_id": "c1", "document_id": "600519-2025"}
    outcome = verify_grounding(
        "研发投入合计 803,132,232.31 元 [1]", [citation], [chunk],
        "贵州茅台2025年研发投入合计占营业收入的比例约为多少？")
    assert outcome["verified"] is True
    assert outcome["anchor_tokens"] == ["研发投入合计"]
