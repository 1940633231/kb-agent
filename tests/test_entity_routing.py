"""Pre-retrieval entity routing: aggregate obligations into document routes."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evidence_contract import (  # noqa: E402
    bind_routes_to_obligations,
    build_initial_routes,
    build_obligations,
    group_obligations_by_document,
    route_for_missing_obligations,
)


def test_single_company_single_year_builds_one_document_route():
    question = "贵州茅台2023年的营业收入是多少元？"
    obligations = build_obligations(question)

    routes = build_initial_routes(question, obligations)

    assert len(routes) == 1
    assert routes[0].query == question
    assert routes[0].document_ids == ("600519-2023",)
    assert routes[0].obligation_ids == ("obligation-1",)
    assert routes[0].unrouted is False


def test_single_company_multiple_years_builds_one_route_per_year():
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    routes = build_initial_routes(question, build_obligations(question))

    assert [(route.company, route.year) for route in routes] == [
        ("贵州茅台", 2023),
        ("贵州茅台", 2024),
    ]
    assert [route.document_ids for route in routes] == [
        ("600519-2023",),
        ("600519-2024",),
    ]


def test_multiple_companies_build_independent_routes_in_same_year():
    question = "贵州茅台与招商银行2023年的营业收入分别是多少？"
    routes = build_initial_routes(question, build_obligations(question))

    assert [(route.company, route.year) for route in routes] == [
        ("贵州茅台", 2023),
        ("招商银行", 2023),
    ]
    assert [route.document_ids for route in routes] == [
        ("600519-2023",),
        ("600036-2023",),
    ]


def test_multiple_companies_and_years_build_complete_matrix():
    question = (
        "贵州茅台与招商银行2023年、2024年的营业收入分别是多少？"
    )
    routes = build_initial_routes(question, build_obligations(question))

    assert len(routes) == 4
    assert {
        (route.company, route.year, route.document_ids[0])
        for route in routes
    } == {
        ("贵州茅台", 2023, "600519-2023"),
        ("贵州茅台", 2024, "600519-2024"),
        ("招商银行", 2023, "600036-2023"),
        ("招商银行", 2024, "600036-2024"),
    }


def test_multiple_metrics_in_one_company_year_share_a_document_route():
    question = "贵州茅台2023年营业收入、净利润分别是多少？"
    obligations = build_obligations(question)

    routes = build_initial_routes(question, obligations)

    assert len(routes) == 1
    assert len(routes[0].obligation_ids) == len(obligations)
    assert routes[0].document_ids == ("600519-2023",)
    assert routes[0].metrics == ("营业收入", "净利润")


def test_refusal_obligation_does_not_generate_a_route():
    question = "华为2024年的研发投入是多少？"
    obligations = build_obligations(question)

    assert obligations[0].expected_kind == "refusal"
    assert build_initial_routes(question, obligations) == []


def test_routes_are_bound_back_to_obligations_and_documents():
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    obligations = build_obligations(question)
    routes = build_initial_routes(question, obligations)

    bound = bind_routes_to_obligations(obligations, routes)
    grouped = group_obligations_by_document(bound)

    assert [item["route_ids"] for item in bound] == [
        ["route-1"], ["route-2"]
    ]
    assert [item.obligation_id for item in grouped["600519-2023"]] == [
        "obligation-1"
    ]
    assert [item.obligation_id for item in grouped["600519-2024"]] == [
        "obligation-2"
    ]


def test_missing_obligation_retry_keeps_route_id_and_document_scope():
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    obligations = build_obligations(question)
    routes = build_initial_routes(question, obligations)

    retry = route_for_missing_obligations(
        question,
        [obligations[1].as_dict()],
        [route.as_dict() for route in routes],
        rewritten_queries=["贵州茅台 2024年 营业收入 主要会计数据"],
    )

    assert len(retry) == 1
    assert retry[0].query_id == "route-2"
    assert retry[0].document_ids == ("600519-2024",)
    assert retry[0].obligation_ids == ("obligation-2",)
    assert "2024年" in retry[0].query
    assert "主要会计数据" in retry[0].query
