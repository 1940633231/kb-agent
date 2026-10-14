"""本轮优化的回归测试：指标关键词抽取、facet 证据裁剪、答案落地校验。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.evidence import scope_evidence  # noqa: E402
from agent.grading import extract_facets  # noqa: E402
from agent.grounding import extract_numbers, verify_grounding  # noqa: E402
from agent.llm_grade import LLMGrader  # noqa: E402
from fakes import FakeGenerator, FakeService, maotai_hit  # noqa: E402


def graph_api():
    """延迟导入主图：它依赖 production-rag（app 包），没有就跳过相关用例。"""
    pytest.importorskip("app.core.config", reason="需要 production-rag（app 包）")
    from agent.graph import ask, build_graph, default_runtime
    return ask, build_graph, default_runtime


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


# ---- LLM 语义评分：抓"规则假通过" ----

def test_llm_grader_parses_judgment():
    grader = LLMGrader(FakeGenerator('{"judgment": "insufficient", "reason": "缺数值", "missing": "营业收入"}'))
    outcome = grader.grade("贵州茅台2023年营业收入是多少元？", [maotai_hit(2023)])
    assert outcome["judgment"] == "insufficient"
    assert outcome["source"] == "llm"
    assert outcome["missing"] == "营业收入"


def test_llm_grader_falls_back_on_error_or_garbage():
    class Boom:
        def generate(self, messages):
            raise RuntimeError("api down")

    grader = LLMGrader(Boom())
    outcome = grader.grade("问题", [maotai_hit(2023)])
    assert outcome["source"] == "fallback"
    assert outcome["judgment"] == "answerable", "外部依赖异常时沿用规则结论"

    grader = LLMGrader(FakeGenerator("我不知道怎么判"))
    assert grader.grade("问题", [maotai_hit(2023)])["source"] == "fallback"


class FakeGrader:
    def __init__(self, judgment: str) -> None:
        self.judgment = judgment
        self.calls = 0

    def grade(self, question, hits):
        self.calls += 1
        return {"judgment": self.judgment, "reason": "fake", "missing": "",
                "source": "llm"}


def build_fake_graph(resolver=None, **runtime_kwargs):
    _, build_graph, default_runtime = graph_api()
    resolver = resolver or (lambda query: [maotai_hit(2023)])
    service = FakeService(resolver, generator=FakeGenerator())
    runtime = default_runtime(service=service, use_llm_rewrite=False,
                              **runtime_kwargs)
    return build_graph(runtime)


def test_llm_grader_flips_rule_pass_into_retry():
    """规则说"证据已覆盖"，LLM 复核说不足 → 应当触发重试而不是直接作答。"""
    ask, _, _ = graph_api()
    grader = FakeGrader("insufficient")
    graph = build_fake_graph(llm_grader=grader, generate_answers=False)
    result = ask(graph, "贵州茅台2023年的营业收入是多少元？", thread_id="t-llmgrade")

    assert grader.calls >= 1
    assert result["retry_count"] == 2, "被复核拦下后应走满重试"
    score = result["retrieval_score"]
    assert score["passed"] is False and score["rule_passed"] is True
    assert score["llm_review"]["judgment"] == "insufficient"


def test_llm_grader_answerable_keeps_single_pass():
    ask, _, _ = graph_api()
    grader = FakeGrader("answerable")
    graph = build_fake_graph(llm_grader=grader, generate_answers=False)
    result = ask(graph, "贵州茅台2023年的营业收入是多少元？", thread_id="t-llmok")
    assert result["retry_count"] == 0
    assert result["retrieval_score"]["passed"] is True


def test_numeric_discipline_reaches_the_prompt():
    """生成时必须带上数字纪律（数字逐字引用 / 换算先给原值 / 多公司逐条标注）。"""
    ask, build_graph, default_runtime = graph_api()
    captured = {}

    class RecordingGenerator(FakeGenerator):
        def generate(self, messages):
            captured["system"] = messages[0]["content"]
            return super().generate(messages)

    service = FakeService(lambda query: [maotai_hit(2023)],
                          generator=RecordingGenerator())
    runtime = default_runtime(service=service, use_llm_rewrite=False)
    graph = build_graph(runtime)
    ask(graph, "贵州茅台2023年的营业收入是多少元？", thread_id="t-prompt")

    system = captured["system"]
    assert "数字纪律" in system
    assert "逐字一致" in system
    assert "公司+年份=数值" in system


def test_default_runtime_routes_known_fields_out_of_extra():
    """回归：字段名与 AgentRuntime 同名的注入参数不能落进 extra（踩过两次）。"""
    _, _, default_runtime = graph_api()
    service = FakeService(lambda query: [maotai_hit(2023)])
    grader = FakeGrader("answerable")
    runtime = default_runtime(service=service, use_llm_rewrite=False,
                              llm_grader=grader, widened_top_k=7,
                              interrupt_on_low_confidence=True)
    assert runtime.llm_grader is grader
    assert runtime.widened_top_k == 7
    assert runtime.extra == {"interrupt_on_low_confidence": True}
