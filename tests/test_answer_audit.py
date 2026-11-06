"""P1 claim-level hard gate and soft review tests."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.answer_audit import audit_answer  # noqa: E402
from agent.evidence_contract import build_obligations  # noqa: E402
from fakes import make_hit  # noqa: E402


QUESTION = "贵州茅台2023年的营业收入是多少元？"


def chunk(content: str, document_id: str = "600519-2023"):
    return make_hit(document_id, content)


def citation(item, number: int = 1):
    return {
        "number": number,
        "chunk_id": item["chunk_id"],
        "document_id": item["document_id"],
        "page": 12,
    }


def test_correct_number_company_year_and_citation_pass():
    item = chunk("贵州茅台2023年年度报告 营业收入 147,693,604,994.14")
    audit = audit_answer(
        "贵州茅台2023年营业收入为 147,693,604,994.14 元 [1]。",
        [citation(item)],
        [item],
        QUESTION,
        build_obligations(QUESTION),
    )

    assert audit["hard_gate_passed"] is True
    assert audit["hard_violations"] == []


def test_number_missing_from_cited_chunk_is_hard_violation():
    item = chunk("贵州茅台2023年年度报告 营业收入 147,693,604,994.14")
    audit = audit_answer(
        "贵州茅台2023年营业收入为 999,888,777,666.55 元 [1]。",
        [citation(item)],
        [item],
        QUESTION,
    )

    assert any(item["code"] == "unsupported_number"
               for item in audit["hard_violations"])
    assert audit["hard_gate_passed"] is False


def test_wrong_company_or_year_is_hard_violation():
    item = chunk("招商银行2024年年度报告 营业收入 339,123,000,000.00",
                 "600036-2024")
    audit = audit_answer(
        "贵州茅台2023年营业收入为 339,123,000,000.00 元 [1]。",
        [citation(item)],
        [item],
        QUESTION,
    )

    assert any(item["code"] == "entity_year_mismatch"
               for item in audit["hard_violations"])


def test_wrong_metric_name_is_hard_violation():
    item = chunk("贵州茅台2023年年度报告 净利润 85,305,341,965.73")
    audit = audit_answer(
        "贵州茅台2023年营业收入为 85,305,341,965.73 元 [1]。",
        [citation(item)],
        [item],
        QUESTION,
    )

    assert any(item["code"] == "metric_mismatch"
               for item in audit["hard_violations"])


def test_unit_conversion_is_accepted():
    item = chunk("贵州茅台2023年年度报告 营业收入 147,693,604,994.14")
    audit = audit_answer(
        "贵州茅台2023年营业收入约 14,769,360.50 万元 [1]。",
        [citation(item)],
        [item],
        QUESTION,
    )

    assert audit["hard_gate_passed"] is True


def test_multi_value_missing_obligation_is_hard_violation():
    question = "贵州茅台2023年与2024年的营业收入分别是多少元？"
    item_2023 = chunk("贵州茅台2023年年度报告 营业收入 147,693,604,994.14")
    item_2024 = chunk("贵州茅台2024年年度报告 营业收入 170,899,152,276.34",
                      "600519-2024")
    audit = audit_answer(
        "贵州茅台2023年与2024年营业收入分别是 "
        "147,693,604,994.14 元 [1] 和 170,899,152,276.34 元 [2]。",
        [citation(item_2023, 1), citation(item_2024, 2)],
        [item_2023],
        question,
        build_obligations(question),
    )

    assert any(item["code"] == "missing_obligation"
               for item in audit["hard_violations"])


def test_same_document_caliber_conflict_is_soft_not_auto_repaired():
    merged = chunk("贵州茅台2023年年度报告 主要会计数据 营业收入 147,693,604,994.14")
    parent = chunk("贵州茅台2023年年度报告 母公司利润表 营业收入 81,415,570,788.59")
    audit = audit_answer(
        "贵州茅台2023年营业收入为 81,415,570,788.59 元 [1]。",
        [citation(parent)],
        [merged, parent],
        QUESTION,
    )

    assert audit["hard_gate_passed"] is True
    assert any(item["code"] == "ambiguous_caliber"
               for item in audit["soft_issues"])
    assert audit["repairable_claim_ids"] == []


def test_correct_citation_with_page_metadata_is_preserved():
    item = chunk("贵州茅台2023年年度报告 营业收入 147,693,604,994.14")
    cited = citation(item)
    audit = audit_answer(
        "贵州茅台2023年营业收入为 147,693,604,994.14 元 [1]。",
        [cited], [item], QUESTION,
    )

    assert audit["hard_gate_passed"] is True
    assert cited["page"] == 12


def test_uncited_numeric_claim_is_soft_when_number_may_be_correct():
    item = chunk("贵州茅台2023年年度报告 营业收入 147,693,604,994.14")
    audit = audit_answer(
        "贵州茅台2023年营业收入为 147,693,604,994.14 元。",
        [], [item], QUESTION,
    )

    assert audit["hard_gate_passed"] is True
    assert any(item["code"] == "uncited_numeric_claim"
               for item in audit["soft_issues"])
