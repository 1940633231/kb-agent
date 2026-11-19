"""Company alias normalization tests."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evidence_contract import build_initial_routes, build_obligations  # noqa: E402
from agent.grading import extract_facets  # noqa: E402


def test_supported_company_aliases_are_canonicalized():
    cases = {
        "茅台2024年的销售费用是多少元？": "贵州茅台",
        "美的2024年的研发费用是多少千元？": "美的集团",
        "顺丰2024年的研发费用是多少千元？": "顺丰控股",
        "海康2024年的经营现金流净额是多少元？": "海康威视",
        "平安2024年的归母净利润是多少百万元？": "中国平安",
        "招行2024年的总资产是多少百万元？": "招商银行",
    }

    for question, canonical in cases.items():
        facets = extract_facets(question)
        assert facets["companies"] == [canonical]
        assert not any(alias in facets["keywords"]
                       for alias in ("茅台", "美的", "顺丰", "海康", "平安", "招行"))


def test_alias_questions_build_scoped_routes():
    question = "茅台2024年的销售费用是多少元？"
    routes = build_initial_routes(question, build_obligations(question))

    assert len(routes) == 1
    assert routes[0].company == "贵州茅台"
    assert routes[0].document_ids == ("600519-2024",)
