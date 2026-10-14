"""主图与检索子图的测试（用假服务替身，不加载模型、不调 LLM）。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# kb-agent 的检索链路同进程 import production-rag（app 包）。CI 里没有它时，
# 这个模块整体跳过，而不是让收集阶段直接报 import error。
pytest.importorskip("app.core.config", reason="需要 production-rag（app 包）")

from agent import graph as agent_graph  # noqa: E402
from agent.grading import extract_facets, grade_evidence  # noqa: E402
from agent.nodes import (OUT_OF_SCOPE, PENDING_REVIEW,  # noqa: E402
                         REFUSAL_NO_EVIDENCE)
from agent.review import pending_review, resume_review  # noqa: E402
from agent.retrieval_subgraph import build_retrieval_subgraph  # noqa: E402
from agent.rewriter import RuleRewriter  # noqa: E402
from agent.runtime import AgentRuntime  # noqa: E402
from rag_adapter.rag import accumulate_evidence  # noqa: E402
from rag_adapter.rag import enrich_citation  # noqa: E402

COMPANY_BY_CODE = {"600519": "贵州茅台", "300750": "宁德时代", "600036": "招商银行"}


def make_hit(document_id: str, content: str, score: float = 0.7) -> Dict:
    code, year = document_id.split("-")
    return {
        "chunk_id": "{}#1".format(document_id),
        "document_id": document_id,
        "score": score,
        "content": content,
        "start_offset": 0,
        "metadata": {"company": COMPANY_BY_CODE.get(code, code),
                     "year": int(year),
                     "document_title": "{}年年度报告".format(year)},
    }


def maotai_year_hit(year: int, score: float = 0.7) -> Dict:
    content = ("贵州茅台酒股份有限公司 {} 年年度报告 主要会计数据 "
               "营业收入 147,693,604,994.14 归属于上市公司股东的净利润").format(year)
    return make_hit("600519-{}".format(year), content, score)


class FakeRetriever:
    def __init__(self, resolver):
        self.resolver = resolver
        self.queries: List[str] = []

    def search(self, query, top_k=5, document_ids=None):
        self.queries.append(query)
        return self.resolver(query)[:top_k]


class FakeService:
    """只实现节点用到的最小接口（mode=vector 走 retriever.search 这条路径）。"""

    def __init__(self, resolver, mode: str = "vector", use_rerank: bool = False):
        self.mode = mode
        self.use_rerank = use_rerank
        self.strategy = "recursive"
        self.generator = None
        self._pipeline = FakePipeline(resolver)


class FakePipeline:
    """对齐 pipeline 的多路检索语义：_search_all 合并各路候选。"""

    def __init__(self, resolver):
        self.retriever = FakeRetriever(resolver)
        self.generator = None
        self.reranker = None
        self.top_k = 5
        self.rerank_candidate_pool = 50

    def _search_all(self, query, queries, document_ids=None):
        collected: List[Dict] = []
        for text in queries:
            collected.extend(self.retriever.search(text, top_k=self.top_k))
        return collected

    def _filter_active_versions(self, candidates):
        return candidates


def build_graph(resolver, **runtime_kwargs):
    service = FakeService(resolver)
    runtime = AgentRuntime(service=service, generate_answers=False,
                           rewriter=RuleRewriter(), **runtime_kwargs)
    return agent_graph.build_graph(runtime), service


class DictStore:
    """审核幂等表的最小替身（够 resume_review 的 get/put 用）。"""

    def __init__(self):
        self.data = {}

    def put(self, namespace, key, value):
        self.data[(tuple(namespace), key)] = dict(value)

    def get(self, namespace, key):
        item = self.data.get((tuple(namespace), key))
        if item is None:
            return None
        return type("Item", (), {"key": key, "value": item})()


QUESTION = "贵州茅台2023年的营业收入是多少元？"


def test_graph_compiles_and_passes_when_evidence_matches():
    graph, service = build_graph(lambda q: [maotai_year_hit(2023)])
    result = agent_graph.ask(graph, QUESTION, thread_id="t-ok")

    assert result["retrieval_score"]["passed"] is True
    assert result["retry_count"] == 0
    assert result["status"] == "retrieval_only"      # 测试里不生成答案
    assert len(service._pipeline.retriever.queries) == 1


def test_retry_rewrites_query_and_recovers():
    """第一轮只召回同公司其他年份（L0 实测的典型失败），改写后命中 2023 年报。"""

    def resolver(query: str) -> List[Dict]:
        if "2023年" in query and "年度报告" in query:
            return [maotai_year_hit(2023)]
        return [maotai_year_hit(2025)]

    graph, service = build_graph(resolver)
    result = agent_graph.ask(graph, QUESTION, thread_id="t-retry")

    assert result["retry_count"] == 1
    assert result["retrieval_score"]["passed"] is True
    assert len(result["rewrite_history"]) == 1
    assert result["rewrite_history"][0]["queries"]
    assert len(service._pipeline.retriever.queries) == 2


def test_gives_up_after_two_retries_and_flags_review():
    graph, service = build_graph(lambda q: [maotai_year_hit(2025)])
    result = agent_graph.ask(graph, QUESTION, thread_id="t-giveup")

    assert result["retry_count"] == 2, "重试上限为 2（v2 §4.1）"
    assert result["retrieval_score"]["passed"] is False
    assert result["needs_human_review"] is True
    assert result["next_action"] == "generate", "超限后基于现有证据作答（带保留）"
    decisions = [entry for entry in result["trace"] if entry["node"] == "decide_next_step"]
    assert decisions[-1]["action"] == "generate"
    assert len(service._pipeline.retriever.queries) == 3, "首次 + 2 次重试"
    budgets = [entry["budget"] for entry in result["trace"]
               if entry["node"] == "retrieve_knowledge"]
    assert budgets == [5, 15, 15], "重试轮放宽检索预算（方案允许的重试动作之一）"


def test_refuses_when_no_evidence_at_all():
    graph, _ = build_graph(lambda q: [])
    result = agent_graph.ask(graph, QUESTION, thread_id="t-empty")

    assert result["retry_count"] == 2
    assert result["answer"] == REFUSAL_NO_EVIDENCE
    assert result["status"] == "refused_no_evidence"
    assert result["citations"] == []


def test_human_review_interrupt_and_resume_approve():
    """低置信度 + 开启审核：图在 human_review 处暂停，Command 恢复后收敛结果。"""
    graph, _ = build_graph(lambda q: [maotai_year_hit(2025)],
                           extra={"interrupt_on_low_confidence": True})
    result = agent_graph.ask(graph, QUESTION, thread_id="t-review")

    payload = pending_review(graph, "t-review")
    assert payload is not None, "应挂起等待人工审核"
    assert payload["question"] == QUESTION
    assert payload["reason"]
    assert result["needs_human_review"] is True

    outcome = resume_review(graph, "t-review", payload["review_id"],
                            decision="approve", reviewer="alice")
    assert outcome["status"] == "resumed"
    assert outcome["final_status"] == "review_approved"
    assert pending_review(graph, "t-review") is None, "恢复后不应再有挂起项"


def test_human_review_reject_and_idempotency():
    store = DictStore()
    graph, _ = build_graph(lambda q: [maotai_year_hit(2025)],
                           extra={"interrupt_on_low_confidence": True})
    agent_graph.ask(graph, QUESTION, thread_id="t-reject")
    payload = pending_review(graph, "t-reject")
    assert payload

    first = resume_review(graph, "t-reject", payload["review_id"],
                          decision="reject", reviewer="bob", note="证据不足",
                          store=store)
    assert first["final_status"] == "review_rejected"

    # 重复提交同一 review_id：必须有幂等记录、返回同一结果，且不再推进图状态
    again = resume_review(graph, "t-reject", payload["review_id"],
                          decision="approve", reviewer="bob", store=store)
    assert again["status"] == "already_processed"
    assert again["final_status"] == "review_rejected"


def test_human_review_rejects_stale_review_id():
    graph, _ = build_graph(lambda q: [maotai_year_hit(2025)],
                           extra={"interrupt_on_low_confidence": True})
    agent_graph.ask(graph, QUESTION, thread_id="t-stale")
    outcome = resume_review(graph, "t-stale", "deadbeef", decision="approve")
    assert outcome["status"] == "review_id_mismatch"


def test_short_question_skips_retrieval():
    graph, service = build_graph(lambda q: [maotai_year_hit(2023)])
    result = agent_graph.ask(graph, "你好", thread_id="t-short")

    assert result["needs_retrieval"] is False
    assert result["status"] == "out_of_scope"
    assert result["answer"] == OUT_OF_SCOPE
    assert service._pipeline.retriever.queries == []


def test_trace_accumulates_across_nodes():
    """Reducer 生效：trace 累积而不是被最后一个节点覆盖。"""
    graph, _ = build_graph(lambda q: [maotai_year_hit(2023)])
    result = agent_graph.ask(graph, QUESTION, thread_id="t-trace")
    nodes = [entry["node"] for entry in result["trace"]]

    assert nodes[0] == "understand_query"
    assert "retrieve_knowledge" in nodes
    assert "grade_retrieval" in nodes
    assert "decide_next_step" in nodes
    assert nodes[-1] == "finalize_response"


def test_retrieval_subgraph_merges_and_dedupes():
    def resolver(query: str) -> List[Dict]:
        if "2023" in query:
            return [maotai_year_hit(2023)]
        return [maotai_year_hit(2024), maotai_year_hit(2024, score=0.9)]

    runtime = AgentRuntime(service=FakeService(resolver), top_k=5)
    subgraph = build_retrieval_subgraph(runtime)
    result = subgraph.invoke({"queries": ["q2023", "q2024"], "top_k": 5})

    ids = [hit["document_id"] for hit in result["candidates"]]
    assert sorted(ids) == ["600519-2023", "600519-2024"], "同 chunk 应去重"
    assert result["candidates"][0]["score"] == 0.9, "去重保留最高分"
    assert result["stats"]["queries"] == ["q2023", "q2024"]


def test_evidence_accumulates_across_retries():
    """自纠错只做加法：重试后第一轮证据仍在（L3 第一版负向结果的根因）。"""
    def resolver(query: str) -> List[Dict]:
        if "2025年" in query and "年度报告" in query:
            return [maotai_year_hit(2025, score=0.6)]
        return [maotai_year_hit(2023, score=0.9)]

    graph, _ = build_graph(resolver)
    result = agent_graph.ask(graph, "贵州茅台2023年与2025年的营业收入分别是多少元？",
                             thread_id="t-accumulate")

    document_ids = {hit["document_id"] for hit in result["retrieved_chunks"]}
    assert result["retry_count"] == 1
    assert document_ids == {"600519-2023", "600519-2025"}, "新证据追加而非覆盖"
    assert result["retrieved_chunks"][0]["score"] == 0.9, "先到的证据保持在前"


def test_facet_scope_drops_unnamed_year_from_evidence():
    """问题只点名 2023 时，混进来的 2025 片段要被裁掉（跨年污染实测 11/17）。"""
    graph, _ = build_graph(lambda q: [maotai_year_hit(2023, score=0.9),
                                      maotai_year_hit(2025, score=0.8)])
    result = agent_graph.ask(graph, QUESTION, thread_id="t-scope")

    document_ids = {hit["document_id"] for hit in result["retrieved_chunks"]}
    assert document_ids == {"600519-2023"}
    stats = result["retrieval_stats"]["scope"]
    assert stats["scoped"] is True and stats["dropped"] == 1
    assert any("scoped_out" in entry and entry["scoped_out"] == 1
               for entry in result["trace"] if entry["node"] == "retrieve_knowledge")


def test_accumulate_evidence_dedupes_and_caps():
    previous = [make_hit("600519-2023", "x", score=0.5)]
    incoming = [make_hit("600519-2023", "x", score=0.9),
                make_hit("600519-2024", "y", score=0.8)]
    merged = accumulate_evidence(previous, incoming, limit=10)
    assert [hit["document_id"] for hit in merged] == ["600519-2023", "600519-2024"]
    assert len(accumulate_evidence(previous, incoming, limit=1)) == 1


def test_grading_flags_year_mismatch_and_missing_keyword():
    hits = [maotai_year_hit(2025)]
    score = grade_evidence(QUESTION, hits)
    assert score["passed"] is False
    assert score["missing_years"] == ["2023"]

    ok = grade_evidence(QUESTION, [maotai_year_hit(2023)])
    assert ok["passed"] is True

    empty = grade_evidence(QUESTION, [])
    assert empty["passed"] is False
    assert empty["candidate_count"] == 0


def test_rule_rewriter_splits_multi_year_question():
    facets = extract_facets("招商银行2023年、2024年、2025年归属于本行股东的净利润分别是多少百万元？")
    assert facets["years"] == ["2023", "2024", "2025"]

    queries = RuleRewriter().rewrite(
        "招商银行2023年、2024年、2025年归属于本行股东的净利润分别是多少百万元？",
        "证据缺少问题涉及年份的年报片段: 2024,2025", [])
    assert len(queries) == 3
    assert all("年度报告" in query for query in queries)


def test_citation_enrichment_adds_title_and_page():
    """引用要能回到原文档与页码（M6）：上游只给 file_name=unknown，适配层补齐。"""
    offsets = {"600519-2024": {"title": "贵州茅台2024年年度报告", "company": "贵州茅台",
                               "year": 2024, "pages": [[1, 0, 100], [2, 102, 300]]}}
    citation = {"number": 3, "chunk_id": "c1", "document_id": "600519-2024",
                "file_name": "unknown", "start_offset": 150, "end_offset": 200}
    enriched = enrich_citation(citation, offsets)

    assert enriched["document_title"] == "贵州茅台2024年年度报告"
    assert enriched["file_name"] == "贵州茅台2024年年度报告"
    assert enriched["page"] == 2
    assert enriched["company"] == "贵州茅台"
