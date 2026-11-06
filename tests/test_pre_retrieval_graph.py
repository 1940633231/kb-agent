"""Graph-level regressions for pre-retrieval routing."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import graph as agent_graph  # noqa: E402
from agent.rewriter import RuleRewriter  # noqa: E402
from agent.runtime import AgentRuntime  # noqa: E402
from fakes import FakeService, make_hit  # noqa: E402


def hit(company: str, code: str, year: int):
    return make_hit(
        "{}-{}".format(code, year),
        "{} {}年年度报告 营业收入 100,000,000.00".format(company, year),
    )


def graph_for(resolver, **runtime_kwargs):
    service = FakeService(resolver)
    runtime = AgentRuntime(
        service=service,
        generate_answers=False,
        rewriter=RuleRewriter(),
        **runtime_kwargs,
    )
    return agent_graph.build_graph(runtime), service


def test_multi_company_question_searches_each_company_route():
    question = "贵州茅台与招商银行2023年的营业收入分别是多少？"
    graph, service = graph_for(lambda query: [
        hit("贵州茅台", "600519", 2023)
        if "贵州茅台" in query
        else hit("招商银行", "600036", 2023)
    ])

    result = agent_graph.ask(graph, question, thread_id="route-companies")

    assert result["retrieval_score"]["passed"] is True
    assert len(result["routes"]) == 2
    assert len(service._pipeline.retriever.queries) == 2
    assert service._pipeline.retriever.document_filters == [
        {"600519-2023"}, {"600036-2023"}
    ]
    assert {item["document_id"] for item in result["retrieved_chunks"]} == {
        "600519-2023", "600036-2023"
    }


def test_multi_company_multi_year_plans_complete_matrix():
    question = (
        "贵州茅台与招商银行2023年、2024年的营业收入分别是多少？"
    )

    def resolver(query):
        code = "600519" if "贵州茅台" in query else "600036"
        company = "贵州茅台" if code == "600519" else "招商银行"
        year = 2024 if "2024" in query else 2023
        return [hit(company, code, year)]

    graph, service = graph_for(resolver)
    result = agent_graph.ask(graph, question, thread_id="route-matrix")

    planned = result["route_history"][0]["routes"]
    assert len(planned) == 4
    assert {
        (item["company"], item["year"], item["document_ids"][0])
        for item in planned
    } == {
        ("贵州茅台", 2023, "600519-2023"),
        ("贵州茅台", 2024, "600519-2024"),
        ("招商银行", 2023, "600036-2023"),
        ("招商银行", 2024, "600036-2024"),
    }
    assert len(service._pipeline.retriever.queries) == 4
    assert result["retrieval_score"]["passed"] is True


def test_retry_reuses_only_the_missing_route_and_document_scope():
    attempts = {"2024": 0}

    def resolver(query):
        if "2024" in query:
            attempts["2024"] += 1
            if attempts["2024"] == 1:
                return [hit("贵州茅台", "600519", 2023)]
            return [hit("贵州茅台", "600519", 2024)]
        return [hit("贵州茅台", "600519", 2023)]

    graph, service = graph_for(resolver)
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    result = agent_graph.ask(graph, question, thread_id="route-retry")

    assert result["retry_count"] == 1
    initial = result["route_history"][0]
    retry = result["route_history"][1]
    assert len(initial["routes"]) == 2
    assert retry["missing_obligations"] == ["obligation-2"]
    assert len(retry["routes"]) == 1
    assert retry["routes"][0]["query_id"] == "route-2"
    assert retry["routes"][0]["document_ids"] == ["600519-2024"]
    assert service._pipeline.retriever.document_filters[-1] == {
        "600519-2024"
    }


def test_unknown_entity_does_not_search_or_retry():
    graph, service = graph_for(lambda query: [])
    result = agent_graph.ask(
        graph, "华为2024年的研发投入是多少？", thread_id="route-refusal")

    assert result["retrieval_score"]["refusal_expected"] is True
    assert result["retry_count"] == 0
    assert service._pipeline.retriever.queries == []
    assert result["routes"] == []


def test_routing_switch_restores_legacy_single_query_behavior():
    graph, service = graph_for(
        lambda query: [hit("贵州茅台", "600519", 2025)],
        enable_pre_retrieval_routing=False,
        max_retries=0,
    )
    result = agent_graph.ask(
        graph,
        "贵州茅台2023年的营业收入是多少元？",
        thread_id="route-disabled",
    )

    assert result["routes"] == []
    assert result["route_history"] == []
    assert result["retrieval_stats"]["routing_enabled"] is False
    assert service._pipeline.retriever.queries == [
        "贵州茅台2023年的营业收入是多少元？"
    ]
