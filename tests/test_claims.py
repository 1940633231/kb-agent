"""P1 Claim extraction tests."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.claims import extract_claims  # noqa: E402


def test_numeric_claim_keeps_value_unit_company_and_year():
    claims = extract_claims(
        "贵州茅台2023年营业收入为 147,693,604,994.14 元 [1]。",
        "贵州茅台2023年的营业收入是多少元？",
    )

    numeric = [item for item in claims if item.claim_type == "numeric"]
    assert len(numeric) == 1
    assert numeric[0].company == "贵州茅台"
    assert numeric[0].year == 2023
    assert numeric[0].unit == "元"
    assert numeric[0].value == 147693604994.14
    assert numeric[0].citation_ids == [1]


def test_multi_company_sentence_binds_each_number_to_nearest_company():
    answer = (
        "贵州茅台营业收入 147,693,604,994.14 元 [1]，"
        "招商银行营业收入 339,123,000,000.00 元 [2]。"
    )
    claims = extract_claims(answer)
    numeric = [item for item in claims if item.claim_type == "numeric"]

    assert [item.company for item in numeric] == ["贵州茅台", "招商银行"]


def test_claim_spans_can_be_repaired_without_touching_other_sentences():
    claims = extract_claims(
        "贵州茅台营业收入为 999,888,777,666.55 元 [1]。\n"
        "公司经营稳定 [2]。"
    )

    assert claims[0].start == 0
    assert claims[0].end < len(
        "贵州茅台营业收入为 999,888,777,666.55 元 [1]。\n"
        "公司经营稳定 [2]。")
    assert claims[0].start < claims[0].end


def test_citation_claims_are_extracted_separately():
    claims = extract_claims("结论如下 [1][2]。")
    citations = [item for item in claims if item.claim_type == "citation"]

    assert [item.citation_ids for item in citations] == [[1], [2]]
