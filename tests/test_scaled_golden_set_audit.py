"""Tests for automated scaled-golden-set source auditing."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evals.audit_scaled_golden_set import audit_record  # noqa: E402


def record() -> dict:
    return {
        "id": "case-1",
        "question": "公司甲2024年的营业收入是多少元？",
        "expected_behavior": "answer",
        "expected_docs": [{
            "company": "公司甲", "year": 2024, "ts_code": "000001",
        }],
        "expected_keywords": ["营业收入"],
        "expected_values": [{
            "label": "营业收入",
            "value": 1000000.0,
            "unit": "元",
            "display": "1,000,000元",
        }],
        "evidence_literals": ["1,000,000"],
        "source_pages": [{
            "doc": "000001-2024",
            "page": 2,
            "chunk_id": "c1",
        }],
        "required_evidence": [{"obligation_id": "obligation-1"}],
    }


def test_audit_passes_grounded_record():
    result = audit_record(
        record(),
        {"000001-2024": {"company": "公司甲", "year": 2024}},
        {"c1": {
            "chunk_id": "c1",
            "content": "公司甲2024年年度报告 营业收入 1,000,000 元",
            "start_offset": 120,
        }},
        {"000001-2024": {"pages": [[1, 0, 100], [2, 100, 300]]}},
    )

    assert result["bucket"] == "normal"
    assert result["risk_score"] == 0


def test_audit_fails_missing_literal_and_page_mismatch():
    result = audit_record(
        record(),
        {"000001-2024": {"company": "公司甲", "year": 2024}},
        {"c1": {
            "chunk_id": "c1",
            "content": "公司甲2024年年度报告 营业收入 999,999 元",
            "start_offset": 120,
        }},
        {"000001-2024": {"pages": [[1, 0, 200], [3, 200, 300]]}},
    )

    codes = {item["code"] for item in result["risks"]}
    assert result["bucket"] == "fail"
    assert "literal_not_in_source" in codes
    assert "page_mismatch" in codes
