"""P0 multi-route provenance and completeness tests."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import graph as agent_graph  # noqa: E402
from agent.evidence_contract import (  # noqa: E402
    build_obligations,
    preserve_obligation_evidence,
)
from agent.retrieval_subgraph import build_retrieval_subgraph, merge_candidates  # noqa: E402
from agent.rewriter import RuleRewriter  # noqa: E402
from agent.runtime import AgentRuntime  # noqa: E402
from fakes import FakeService, make_hit  # noqa: E402
from rag_adapter.rag import (  # noqa: E402
    cap_evidence_preserving,
    retrieve_multi_with_routes,
)


def hit(company: str, ts_code: str, year: int, metric: str = "营业收入"):
    return make_hit(
        "{}-{}".format(ts_code, year),
        "{} {}年年度报告 {} 100,000,000.00".format(company, year, metric),
    )


def test_retrieve_multi_preserves_query_to_hit_mapping():
    service = FakeService(lambda query: [
        hit("贵州茅台", "600519", 2023 if "2023" in query else 2024)
    ])

    hits, routes = retrieve_multi_with_routes(
        service,
        ["贵州茅台2023年营业收入", "贵州茅台2024年营业收入"],
        top_k=5,
    )

    assert len(routes) == 2
    assert routes[0]["query_id"] == "route-1"
    assert routes[1]["query_id"] == "route-2"
    assert routes[0]["candidate_count"] == 1
    assert routes[1]["hits"][0]["document_id"] == "600519-2024"
    assert len(hits) == 2
    assert {tuple(item["retrieval_routes"]) for item in hits} == {
        ("route-1",), ("route-2",)
    }


def test_merge_candidates_merges_route_provenance():
    first = hit("贵州茅台", "600519", 2023)
    second = dict(first, score=0.9, retrieval_routes=["route-2"])
    first["score"] = 0.8
    first["retrieval_routes"] = ["route-1"]

    merged = merge_candidates([first, second])

    assert len(merged) == 1
    assert merged[0]["retrieval_routes"] == ["route-1", "route-2"]


def test_subgraph_reports_per_query_stats_and_obligation_coverage():
    service = FakeService(lambda query: [
        hit("贵州茅台", "600519", 2023 if "2023" in query else 2024)
    ])
    runtime = SimpleNamespace(service=service, top_k=5)
    graph = build_retrieval_subgraph(runtime)
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    obligations = [item.as_dict() for item in build_obligations(question)]

    result = graph.invoke({
        "queries": ["贵州茅台2023年营业收入", "贵州茅台2024年营业收入"],
        "obligations": obligations,
        "route_prefix": "r0-",
        "top_k": 5,
        "anchor": question,
    })

    stats = result["stats"]
    assert stats["per_query_returned"] == {"r0-route-1": 1, "r0-route-2": 1}
    assert stats["per_query_documents"]["r0-route-1"] == ["600519-2023"]
    assert stats["per_query_documents"]["r0-route-2"] == ["600519-2024"]
    assert result["route_coverage"][0]["covered"] == ["obligation-1"]
    assert result["route_coverage"][1]["covered"] == ["obligation-2"]


def test_route_coverage_uses_final_selected_evidence():
    question = "贵州茅台2023年的营业收入是多少元？"
    obligations = [item.as_dict() for item in build_obligations(question)]
    ambiguous = make_hit(
        "600519-2023",
        "贵州茅台2023年年度报告 合并营业收入 100 母公司营业收入 200",
        score=0.9,
    )
    clean = make_hit(
        "600519-2023",
        "贵州茅台2023年年度报告 营业收入 300",
        score=0.99,
    )
    service = FakeService(lambda query: [ambiguous, clean])
    runtime = SimpleNamespace(service=service, top_k=1)
    graph = build_retrieval_subgraph(runtime)

    result = graph.invoke({
        "queries": [question],
        "routes": [{
            "query_id": "route-1",
            "query": question,
            "obligation_ids": ["obligation-1"],
            "document_ids": ["600519-2023"],
        }],
        "obligations": obligations,
        "routing_enabled": True,
        "top_k": 1,
        "anchor": question,
    })

    assert result["route_coverage"][0]["covered"] == ["obligation-1"]
    assert result["route_coverage"][0]["ambiguous"] == []


def test_merged_candidates_do_not_hide_missing_year():
    service = FakeService(lambda query: [
        hit("贵州茅台", "600519", 2023),
        hit("贵州茅台", "600519", 2023, "净利润"),
    ])
    runtime = SimpleNamespace(service=service, top_k=5)
    graph = build_retrieval_subgraph(runtime)
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    obligations = [item.as_dict() for item in build_obligations(question)]

    result = graph.invoke({
        "queries": ["贵州茅台2023年营业收入", "贵州茅台2024年营业收入"],
        "obligations": obligations,
        "route_prefix": "r0-",
        "top_k": 5,
        "anchor": question,
    })

    all_covered = {
        obligation_id
        for route in result["route_coverage"]
        for obligation_id in route["covered"]
    }
    assert all_covered == {"obligation-1"}
    assert result["route_coverage"][1]["missing"] == ["obligation-2"]


def test_obligation_evidence_is_preserved_when_global_top_k_drops_it():
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    obligations = build_obligations(question)
    kept_hit = hit("贵州茅台", "600519", 2023)
    missing_hit = hit("贵州茅台", "600519", 2024)
    route_hits = [
        {"query_id": "r0-route-1", "query": "2023", "hits": [kept_hit]},
        {"query_id": "r0-route-2", "query": "2024", "hits": [missing_hit]},
    ]

    merged, added = preserve_obligation_evidence(
        [kept_hit], route_hits, obligations)

    assert [item["document_id"] for item in merged] == [
        "600519-2023", "600519-2024"
    ]
    assert added == ["obligation-2"]


def test_obligation_fill_survives_evidence_cap():
    ordinary = [
        make_hit("600519-2023", "ordinary-{}".format(index))
        for index in range(5)
    ]
    critical = make_hit("600519-2024", "销售费用 100,000")
    critical["obligation_fill"] = "obligation-2"

    kept = cap_evidence_preserving(
        ordinary + [critical], limit=3,
        protected_keys={critical["chunk_id"]},
    )

    assert critical in kept
    assert len(kept) == 3


def test_preserve_replaces_same_chunk_with_stale_metadata():
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    obligations = build_obligations(question)
    kept_hit = hit("贵州茅台", "600519", 2023)
    stale = hit("贵州茅台", "600519", 2024)
    stale["metadata"] = dict(stale["metadata"], year=2025)
    route_hit = hit("贵州茅台", "600519", 2024)
    route_hit["chunk_id"] = stale["chunk_id"]

    merged, added = preserve_obligation_evidence(
        [kept_hit, stale],
        [{"query_id": "r0-route-2", "query": "2024", "hits": [route_hit]}],
        obligations,
    )

    assert added == ["obligation-2"]
    assert merged[1]["metadata"]["year"] == 2024


def test_retry_targets_only_the_missing_obligation():
    calls = {"2024": 0}

    def resolver(query: str):
        if "2024" in query:
            calls["2024"] += 1
            # First attempt violates the route contract. FakeRetriever drops
            # it; the retry for route-2 then returns the in-scope document.
            if calls["2024"] == 1:
                return [hit("贵州茅台", "600519", 2023)]
            return [hit("贵州茅台", "600519", 2024)]
        return [hit("贵州茅台", "600519", 2023)]

    service = FakeService(resolver)
    runtime = AgentRuntime(
        service=service, generate_answers=False, rewriter=RuleRewriter(),
        max_retries=2,
    )
    graph = agent_graph.build_graph(runtime)
    result = agent_graph.ask(
        graph,
        "贵州茅台2023年与2024年的营业收入分别是多少元？",
        thread_id="p0-targeted-retry",
    )

    assert result["retry_count"] == 1
    assert result["retrieval_score"]["passed"] is True
    assert result["rewrite_history"][0]["obligations"] == ["obligation-2"]
    assert "2024年" in result["rewrite_history"][0]["queries"][0]


def test_unknown_company_refuses_without_wasting_retries():
    service = FakeService(lambda query: [])
    runtime = AgentRuntime(
        service=service, generate_answers=False, rewriter=RuleRewriter(),
        max_retries=2,
    )
    graph = agent_graph.build_graph(runtime)

    result = agent_graph.ask(
        graph, "华为2024年的研发投入是多少？", thread_id="p0-refusal")

    assert result["retrieval_score"]["refusal_expected"] is True
    assert result["retry_count"] == 0
    assert result["status"] == "refused_no_evidence"


def test_known_company_with_unsupported_year_refuses_without_retries():
    service = FakeService(lambda query: [hit("贵州茅台", "600519", 2025)])
    runtime = AgentRuntime(
        service=service, generate_answers=False, rewriter=RuleRewriter(),
        max_retries=2,
    )
    graph = agent_graph.build_graph(runtime)

    result = agent_graph.ask(
        graph, "贵州茅台2026年的营业收入是多少元？",
        thread_id="p0-future-year-refusal",
    )

    assert result["retrieval_score"]["refusal_expected"] is True
    assert result["retry_count"] == 0
    assert result["status"] == "refused_no_evidence"


def test_ambiguous_obligation_is_not_silently_accepted():
    content = (
        "贵州茅台2023年年度报告 合并营业收入 100,000,000.00 "
        "母公司营业收入 200,000,000.00"
    )
    service = FakeService(lambda query: [make_hit("600519-2023", content)])
    runtime = AgentRuntime(
        service=service, generate_answers=False, rewriter=RuleRewriter(),
        max_retries=0,
    )
    graph = agent_graph.build_graph(runtime)

    result = agent_graph.ask(
        graph, "贵州茅台2023年的营业收入是多少元？",
        thread_id="p0-ambiguous")

    assert result["retrieval_score"]["passed"] is False
    assert result["retrieval_score"]["ambiguous_obligations"]
    assert result["needs_human_review"] is True
    assert result["status"] == "ambiguous_evidence"
