"""P0 evidence contract tests: one slot per required company/year/metric."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.evidence_contract import (  # noqa: E402
    assess_obligations,
    assess_route_coverage,
    build_obligations,
    scope_supported_by_corpus,
)
from fakes import make_hit  # noqa: E402


def hit(company: str, ts_code: str, year: int, metric: str,
        value: str = "100,000,000.00", extra: str = ""):
    content = "{}{}年年度报告 {} {} {}".format(
        company, year, metric, value, extra
    )
    return make_hit("{}-{}".format(ts_code, year), content)


def test_single_company_single_year_single_value():
    question = "贵州茅台2023年的营业收入是多少元？"
    obligations = build_obligations(question)

    result = assess_obligations(
        [hit("贵州茅台", "600519", 2023, "营业收入")], obligations, question)

    assert len(obligations) == 1
    assert obligations[0].expected_kind == "single_value"
    assert result["passed"] is True
    assert result["coverage"] == 1.0


def test_single_company_multiple_years_requires_each_year():
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    obligations = build_obligations(question)
    hits = [
        hit("贵州茅台", "600519", 2023, "营业收入"),
        hit("贵州茅台", "600519", 2024, "营业收入"),
    ]

    result = assess_obligations(hits, obligations, question)

    assert len(obligations) == 2
    assert obligations[0].expected_kind == "multi_value"
    assert result["passed"] is True
    assert result["covered"] == ["obligation-1", "obligation-2"]


def test_multiple_companies_are_compared_per_company():
    question = "贵州茅台与招商银行2023年的营业收入分别是多少？"
    obligations = build_obligations(question)
    hits = [
        hit("贵州茅台", "600519", 2023, "营业收入"),
        hit("招商银行", "600036", 2023, "营业收入"),
    ]

    result = assess_obligations(hits, obligations, question)

    assert len(obligations) == 2
    assert obligations[0].expected_kind == "comparison"
    assert result["passed"] is True


def test_multi_company_multi_year_builds_cross_product():
    question = (
        "贵州茅台与招商银行2023年、2024年的营业收入分别是多少？"
    )
    obligations = build_obligations(question)

    assert len(obligations) == 4
    assert {
        (item.company, item.year) for item in obligations
    } == {
        ("贵州茅台", 2023), ("贵州茅台", 2024),
        ("招商银行", 2023), ("招商银行", 2024),
    }


def test_merged_quantity_is_not_enough_when_one_obligation_is_missing():
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    obligations = build_obligations(question)
    hits = [
        hit("贵州茅台", "600519", 2023, "营业收入", "100"),
        hit("贵州茅台", "600519", 2023, "营业收入", "101"),
        hit("贵州茅台", "600519", 2023, "营业收入", "102"),
    ]

    result = assess_obligations(hits, obligations, question)

    assert result["passed"] is False
    assert result["missing"] == ["obligation-2"]


def test_calculation_requires_all_inputs():
    question = "宁德时代2025年营业收入较2024年增加约多少亿元？"
    obligations = build_obligations(question)
    hits = [hit("宁德时代", "300750", 2025, "营业收入")]

    result = assess_obligations(hits, obligations, question)

    assert len(obligations) == 2
    assert all(item.expected_kind == "calculation_input"
               for item in obligations)
    assert result["passed"] is False
    assert len(result["missing"]) == 1


def test_same_document_multiple_values_is_ambiguous():
    question = "贵州茅台2023年的营业收入是多少元？"
    obligations = build_obligations(question)
    content = (
        "贵州茅台2023年年度报告 合并营业收入 100,000,000.00 "
        "母公司营业收入 200,000,000.00"
    )

    result = assess_obligations(
        [make_hit("600519-2023", content)], obligations, question)

    assert result["passed"] is False
    assert result["ambiguous"] == ["obligation-1"]


def test_current_and_prior_year_columns_are_not_misread_as_caliber_conflict():
    question = "贵州茅台2023年的营业收入是多少元？"
    obligations = build_obligations(question)
    content = (
        "贵州茅台2023年年度报告 主要会计数据 营业收入 "
        "147,693,604,994.14 140,000,000,000.00"
    )

    result = assess_obligations(
        [make_hit("600519-2023", content)], obligations, question)

    assert result["passed"] is True
    assert result["ambiguous"] == []


def test_unknown_company_produces_refusal_obligation():
    question = "华为2024年的研发投入是多少？"
    obligations = build_obligations(question)
    result = assess_obligations([], obligations, question)

    assert len(obligations) == 1
    assert obligations[0].expected_kind == "refusal"
    assert result["refusal_expected"] is True
    assert result["passed"] is False


def test_general_question_without_company_or_year_has_no_refusal_obligation():
    question = "经营活动现金流如何理解？"
    obligations = build_obligations(question)
    result = assess_obligations([], obligations, question)

    assert obligations == []
    assert result["refusal_expected"] is False
    assert result["passed"] is False


def test_known_company_with_unsupported_year_is_outside_corpus_scope():
    question = "贵州茅台2026年的营业收入是多少元？"
    obligations = build_obligations(question)

    assert obligations
    assert scope_supported_by_corpus(question) is False


def test_cross_company_metrics_follow_each_company_clause():
    question = (
        "宁德时代2024年的研发费用与贵州茅台2024年的研发投入合计分别是多少？"
    )
    obligations = build_obligations(question)

    assert {(item.company, item.metric) for item in obligations} == {
        ("宁德时代", "研发费用"),
        ("贵州茅台", "研发投入合计"),
    }


def test_parenthetical_qualifiers_do_not_create_fake_obligations():
    question = "贵州茅台2025年茅台酒（分产品口径）的营业收入是多少元？"
    obligations = build_obligations(question)

    assert len(obligations) == 1
    assert "茅台酒" in obligations[0].metric
    assert "营业收入" in obligations[0].metric


def test_comparison_filler_does_not_become_a_metric():
    question = "贵州茅台与招商银行2023年的营业收入分别是多少？"
    obligations = build_obligations(question)

    assert {item.metric for item in obligations} == {"营业收入"}


def test_route_coverage_keeps_empty_and_covered_routes_distinct():
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    obligations = build_obligations(question)
    coverage = assess_route_coverage([
        {"query_id": "r0-route-1", "query": "2023", "hits": [
            hit("贵州茅台", "600519", 2023, "营业收入")
        ]},
        {"query_id": "r0-route-2", "query": "2024", "hits": [
            hit("贵州茅台", "600519", 2024, "营业收入")
        ]},
        {"query_id": "r0-route-3", "query": "empty", "hits": []},
    ], obligations)

    assert coverage[0]["covered"] == ["obligation-1"]
    assert coverage[1]["covered"] == ["obligation-2"]
    assert coverage[2]["covered"] == []
    assert coverage[2]["missing"] == ["obligation-1", "obligation-2"]


def test_thirty_fake_pass_cases_detect_the_missing_obligation():
    """The pool looks large enough, but exactly one required slot is absent."""
    companies = [
        ("贵州茅台", "600519"),
        ("宁德时代", "300750"),
        ("招商银行", "600036"),
    ]
    detected = 0
    cases = 0
    for company, ts_code in companies:
        for offset in range(5):
            year = 2020 + offset
            question = "{} {}年与{}年的营业收入分别是多少元？".format(
                company, year, year + 1)
            obligations = build_obligations(question)
            result = assess_obligations(
                [hit(company, ts_code, year, "营业收入")],
                obligations,
                question,
            )
            cases += 1
            detected += int(
                result["passed"] is False
                and len(result["missing"]) == 1
            )

    pairs = [
        (companies[0], companies[1]),
        (companies[0], companies[2]),
        (companies[1], companies[2]),
    ]
    for offset in range(5):
        year = 2020 + offset
        for left, right in pairs:
            question = "{}与{} {}年的营业收入分别是多少元？".format(
                left[0], right[0], year)
            obligations = build_obligations(question)
            result = assess_obligations(
                [hit(left[0], left[1], year, "营业收入")],
                obligations,
                question,
            )
            cases += 1
            detected += int(
                result["passed"] is False
                and len(result["missing"]) == 1
            )

    assert cases == 30
    assert detected / cases >= 0.9
