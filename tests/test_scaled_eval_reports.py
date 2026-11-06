"""P3 scaled report contracts: repeat count, metric coverage and gate values."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "evals" / "reports"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def test_scaled_retrieval_and_l3_report_expected_metrics():
    for name in ("scaled_retrieval.json", "scaled_l3.json"):
        report = load(name)
        metrics = report["metrics"]
        assert report["dataset"]["evaluated"] == 120
        assert metrics["answerable_questions"] == 105
        assert "evidence_obligation_coverage" in metrics
        assert "missing_obligation_rate" in metrics
        assert "route_coverage" in metrics or "route_obligation_coverage" in metrics


def test_scaled_generation_has_three_runs_with_mean_and_stddev():
    report = load("scaled_generation.json")
    assert report["model"]
    assert report["dataset"]["manifest"]["version"] == "v1"
    assert report["config"]["repeats"] == 3
    assert len(report["runs"]) == 3
    assert report["dataset"]["evaluated"] == 120
    for metric in (
        "answer_value_rate",
        "answer_correctness",
        "claim_support_rate",
        "citation_full_coverage",
        "refusal_correctness",
        "context_overflow_rate",
    ):
        assert report["metrics"][metric]["runs"] == 3
        assert "mean" in report["metrics"][metric]
        assert "stddev" in report["metrics"][metric]
    assert report["metrics"]["context_overflow_rate"]["mean"] == 0.0
    assert report["hard_gate"]["precision"] >= 0.95
    assert report["hard_gate"]["recall"] >= 0.95


def test_scaled_generation_records_failures_per_repeat():
    report = load("scaled_generation.json")
    assert set(report["failures"]) == {"1", "2", "3"}
    assert all(isinstance(items, list)
               for items in report["failures"].values())


def test_claim_and_budget_reports_meet_p1_p2_gates():
    claim = load("claim_audit.json")
    budget = load("context_budget.json")
    assert claim["metrics"]["hard_violation_recall"] >= 0.95
    assert claim["metrics"]["clean_answer_false_positive_rate"] <= 0.05
    assert claim["model"]
    assert budget["metrics"]["cases"] == 105
    assert budget["dataset_version"]
    assert budget["metrics"]["budget_overflow_rate"] == 0.0


def test_reports_are_marked_stale_when_dataset_sha_changed():
    manifest_sha = json.loads(
        (PROJECT_ROOT / "evals" / "golden_set_scaled_120.manifest.json")
        .read_text(encoding="utf-8")
    )["sha256"]
    for name in (
        "scaled_retrieval.json",
        "scaled_l3.json",
        "scaled_generation.json",
    ):
        report = load(name)
        report_sha = (
            report.get("dataset", {}).get("manifest", {}).get("sha256")
        )
        if report_sha and report_sha != manifest_sha:
            assert report["status"] == "stale_dataset_manifest_mismatch"
