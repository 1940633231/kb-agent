"""P1 offline contamination set: hard recall and clean-answer false positives."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.answer_audit import audit_answer  # noqa: E402
from agent.evidence_contract import build_obligations  # noqa: E402

CASES_PATH = PROJECT_ROOT / "evals" / "claim_audit_cases.jsonl"


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_claim_audit_contamination_set_has_50_clean_and_50_polluted():
    cases = load_cases()
    by_kind: dict[str, int] = {}
    for item in cases:
        by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1

    assert by_kind == {"correct": 50, "polluted": 50}


def test_hard_violation_recall_and_clean_answer_false_positive_budget():
    correct = []
    polluted = []
    for item in load_cases():
        audit = audit_answer(
            item["answer"],
            item["citations"],
            item["chunks"],
            item["question"],
            build_obligations(item["question"]),
        )
        if item["kind"] == "correct":
            correct.append(audit["hard_gate_passed"])
        else:
            polluted.append(bool(audit["hard_violations"]))

    hard_recall = sum(polluted) / len(polluted)
    false_positive_rate = 1.0 - (sum(correct) / len(correct))
    assert hard_recall >= 0.95
    assert false_positive_rate <= 0.05


def test_targeted_repair_changes_only_hard_claim_span():
    from agent.repair import apply_claim_repairs
    from agent.claims import extract_claims

    answer = (
        "贵州茅台2023年营业收入为 999,888,777,666.55 元 [1]。\n"
        "公司经营稳定 [2]。"
    )
    claims = [item.as_dict() for item in extract_claims(
        answer, "贵州茅台2023年的营业收入是多少元？")]
    repaired, applied = apply_claim_repairs(
        answer,
        claims,
        {
            "claim-1": (
                "贵州茅台2023年营业收入为 147,693,604,994.14 元 "
                "（主要会计数据）[1]。"
            )
        },
    )

    assert applied == ["claim-1"]
    assert "147,693,604,994.14" in repaired
    assert "公司经营稳定 [2]。" in repaired
