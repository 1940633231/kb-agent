"""Run the P1 claim-audit contamination set and write a reproducible report."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.answer_audit import audit_answer  # noqa: E402
from agent.evidence_contract import build_obligations  # noqa: E402
from agent.repair import apply_claim_repairs  # noqa: E402
from agent.claims import extract_claims  # noqa: E402

CASES_PATH = PROJECT_ROOT / "evals" / "claim_audit_cases.jsonl"
REPORT_PATH = PROJECT_ROOT / "evals" / "reports" / "claim_audit.json"
METADATA_PATH = (
    PROJECT_ROOT / "data" / "docs" / "annual_reports" / "metadata.json"
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    correct: list[bool] = []
    polluted: list[bool] = []
    rows = []
    for item in cases:
        audit = audit_answer(
            item["answer"],
            item["citations"],
            item["chunks"],
            item["question"],
            build_obligations(item["question"]),
        )
        hard = bool(audit["hard_violations"])
        if item["kind"] == "correct":
            correct.append(not hard)
        else:
            polluted.append(hard)
        rows.append({
            "case_id": item["case_id"],
            "kind": item["kind"],
            "expected_hard": item.get("expected_hard") or [],
            "hard_violations": audit["hard_violations"],
            "soft_issues": audit["soft_issues"],
        })

    repair_draft = (
        "贵州茅台2023年营业收入为 999,888,777,666.55 元 [1]。\n"
        "公司经营稳定。"
    )
    repair_claims = [
        item.as_dict() for item in extract_claims(
            repair_draft, "贵州茅台2023年的营业收入是多少元？")
    ]
    repaired, applied = apply_claim_repairs(
        repair_draft,
        repair_claims,
        {
            "claim-1": (
                "贵州茅台2023年营业收入为 147,693,604,994.14 元 "
                "（主要会计数据）[1]。"
            )
        },
    )
    repair_question = "贵州茅台2023年的营业收入是多少元？"
    repair_chunk = {
        "chunk_id": "600519-2023_chunk_1",
        "document_id": "600519-2023",
        "content": "贵州茅台2023年年度报告 主要会计数据 营业收入 147,693,604,994.14",
        "metadata": {
            "company": "贵州茅台",
            "year": 2023,
            "document_id": "600519-2023",
        },
    }
    repair_citations = [{
        "number": 1,
        "chunk_id": repair_chunk["chunk_id"],
        "document_id": repair_chunk["document_id"],
        "page": 11,
    }]
    repair_obligations = build_obligations(repair_question)
    repair_before = audit_answer(
        repair_draft, repair_citations, [repair_chunk],
        repair_question, repair_obligations)
    repair_after = audit_answer(
        repaired, repair_citations, [repair_chunk],
        repair_question, repair_obligations)

    true_positive = sum(polluted)
    false_positive = len(correct) - sum(correct)
    predicted_positive = true_positive + false_positive
    metrics = {
        "hard_violation_recall": (
            round(sum(polluted) / len(polluted), 4) if polluted else None
        ),
        "hard_gate_precision": (
            round(true_positive / predicted_positive, 4)
            if predicted_positive else None
        ),
        "clean_answer_false_positive_rate": (
            round(1.0 - sum(correct) / len(correct), 4) if correct else None
        ),
        "correct_cases": len(correct),
        "polluted_cases": len(polluted),
        "targeted_repair_applied_claims": applied,
        "targeted_repair_hard_before": len(
            repair_before["hard_violations"]),
        "targeted_repair_hard_after": len(repair_after["hard_violations"]),
        "targeted_repair_changed_unrelated_sentence": (
            "公司经营稳定。" not in repaired
        ),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "rule-based-claim-audit",
        "dataset": {
            "path": str(CASES_PATH),
            "sha256": sha256_of(CASES_PATH),
            "version": "p1-claim-audit-v1",
            "size": len(cases),
        },
        "corpus_metadata_sha256": sha256_of(METADATA_PATH),
        "metrics": metrics,
        "targeted_repair": {
            "before": repair_before,
            "after": repair_after,
            "answer_before": repair_draft,
            "answer_after": repaired,
        },
        "cases": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("report -> {}".format(REPORT_PATH))
    return 0 if (
        metrics["hard_violation_recall"] >= 0.95
        and metrics["clean_answer_false_positive_rate"] <= 0.05
        and metrics["targeted_repair_hard_after"] == 0
        and not metrics["targeted_repair_changed_unrelated_sentence"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
