"""Routing-ablation runner configuration and summary regressions."""
from __future__ import annotations

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "evals" / "reports"
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import run_routing_ablation as ablation  # noqa: E402


def report(layer: str, metrics: dict, rows: list[dict]) -> dict:
    return {
        "layer": layer,
        "label": layer,
        "temporary": False,
        "formal_report": True,
        "dataset": {
            "path": "dataset.jsonl",
            "evaluated": 120,
            "manifest": {"sha256": "same-sha"},
        },
        "config": ablation.layer_config(layer, 5, 50),
        "metrics": metrics,
        "questions": rows,
        "elapsed_seconds": 1.0,
    }


def test_layer_configs_match_plan():
    l2 = ablation.layer_config("L2", 5, 50)
    l3a = ablation.layer_config("L3a", 5, 50)
    l3b = ablation.layer_config("L3b", 5, 50)

    assert l2["enable_pre_retrieval_routing"] is False
    assert l2["max_retries"] == 0
    assert l3a["enable_pre_retrieval_routing"] is True
    assert l3a["max_retries"] == 0
    assert l3b["enable_pre_retrieval_routing"] is True
    assert l3b["max_retries"] == 2
    for config in (l2, l3a, l3b):
        assert config["strategy"] == "recursive"
        assert config["mode"] == "hybrid"
        assert config["use_rerank"] is True
        assert config["top_k"] == 5
        assert config["route_candidate_limit"] == 50
        assert config["generate"] is False
        assert config["rule_rewrite"] is True
        assert config["llm_grader"] is False


def test_debug_paths_do_not_overwrite_formal_reports():
    formal = ablation.report_paths(False)
    debug = ablation.report_paths(True)

    assert formal["L2"].name == "routing_ablation_l2.json"
    assert formal["L3a"].name == "routing_ablation_l3a.json"
    assert formal["L3b"].name == "routing_ablation_l3b.json"
    assert formal["summary"].name == "routing_ablation_summary.json"
    assert all(path.name.endswith("_debug.json")
               for path in debug.values())


def test_ablation_reuses_one_service_for_all_layers(monkeypatch):
    service = object()
    seen_services = []

    def fake_evaluate(questions, args, service=None):
        seen_services.append(service)
        return {
            "metrics": {"enable_routing": args.no_pre_retrieval_routing},
            "rows": [{"id": item["id"], "seconds": 1.0}
                     for item in questions],
        }

    monkeypatch.setattr(ablation.agent_eval, "evaluate", fake_evaluate)
    manifest = {"counts": {"total": 120}}
    questions = [{"id": "q-{}".format(index)} for index in range(120)]

    reports = ablation.run_ablation(
        questions, service, manifest, top_k=5, route_candidate_limit=50)

    assert set(reports) == {"L2", "L3a", "L3b"}
    assert seen_services == [service, service, service]


def test_resume_accepts_only_contract_matching_reports(tmp_path):
    path = tmp_path / "routing_ablation_l2.json"
    payload = report(
        "L2",
        {"doc_recall@5": 0.91},
        [{"id": "q-{}".format(index)} for index in range(120)],
    )
    payload["dataset"]["manifest"]["sha256"] = "manifest-sha"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reused = ablation.reusable_report(
        path, "L2", {"sha256": "manifest-sha"}, 5, 50, False, 120)

    assert reused == payload
    assert ablation.reusable_report(
        path, "L2", {"sha256": "other-sha"}, 5, 50, False, 120
    ) is None
    assert ablation.reusable_report(
        path, "L2", {"sha256": "manifest-sha"}, 3, 50, False, 120
    ) is None


def test_official_layer_matrix_rebuilds_current_layers_only(tmp_path):
    rows = [{"id": "q-{}".format(index), "question_type": "fact"}
            for index in range(120)]
    reports = {
        layer: report(layer, {"doc_recall@5": 0.95}, rows)
        for layer in ablation.LAYERS
    }
    path = tmp_path / "layer_matrix.json"
    path.write_text(
        json.dumps({"layers": {"L0": {}, "L1": {}, "L3": {}},
                    "golden_set": {"total": 50}}),
        encoding="utf-8",
    )

    ablation.update_official_layer_matrix(reports, path=path)

    matrix = json.loads(path.read_text(encoding="utf-8"))
    assert set(matrix["layers"]) == {"L2", "L3a", "L3b"}
    assert matrix["golden_set"]["total"] == 120
    assert matrix["golden_set"]["by_type"] == {"fact": 120}
    assert matrix["dataset"]["manifest"]["sha256"] == "same-sha"


def test_summary_contains_deltas_latency_breakdown_and_gates():
    rows = {
        "L2": [
            {"question_type": "fact", "expected_behavior": "answer",
             "wanted_docs": ["600519-2023"], "doc_recall": 1.0,
             "route_coverage_rate": 0.9, "seconds": 10.0},
            {"question_type": "refuse", "expected_behavior": "refuse",
             "wanted_docs": [], "doc_recall": None,
             "route_coverage_rate": None, "seconds": 5.0},
        ],
        "L3a": [
            {"question_type": "fact", "expected_behavior": "answer",
             "wanted_docs": ["600519-2023"], "doc_recall": 1.0,
             "route_coverage_rate": 1.0, "seconds": 11.0},
            {"question_type": "refuse", "expected_behavior": "refuse",
             "wanted_docs": [], "doc_recall": None,
             "route_coverage_rate": None, "seconds": 5.0},
        ],
        "L3b": [
            {"question_type": "fact", "expected_behavior": "answer",
             "wanted_docs": ["600519-2023"], "doc_recall": 1.0,
             "route_coverage_rate": 1.0, "seconds": 12.0},
            {"question_type": "refuse", "expected_behavior": "refuse",
             "wanted_docs": [], "doc_recall": None,
             "route_coverage_rate": None, "seconds": 5.0},
        ],
    }
    metrics = {
        "L2": {"doc_recall@5": 0.90,
               "cross_document_contamination_rate": 0.01},
        "L3a": {"doc_recall@5": 0.96,
                "cross_document_contamination_rate": 0.0},
        "L3b": {"doc_recall@5": 0.97,
                "route_obligation_coverage": 0.97,
                "missing_obligation_rate": 0.02,
                "retry_rate": 0.08,
                "cross_document_contamination_rate": 0.0},
    }
    reports = {
        layer: report(layer, metrics[layer], rows[layer])
        for layer in ablation.LAYERS
    }

    summary = ablation.build_summary(reports)

    assert summary["deltas"]["L3a-L2"]["doc_recall@5"] == 0.06
    assert summary["deltas"]["L3b-L3a"]["doc_recall@5"] == 0.01
    assert summary["latency_breakdown"]["L2"][
        "single_company_single_year_mean_seconds"] == 10.0
    assert summary["latency_breakdown"]["L3a"][
        "single_company_single_year_mean_seconds"] == 11.0
    assert summary["acceptance"]["checks"][
        "l3a_single_company_latency_ratio"]["ratio"] == 1.1
    assert summary["acceptance"]["checks"][
        "l3a_single_company_latency_ratio"]["passed"] is True
    contamination = summary["acceptance"]["checks"][
        "l3a_l3b_cross_document_contamination_zero"]
    assert contamination["values"]["L2"] == 0.01
    assert contamination["passed"] is True
    assert summary["acceptance"]["all_passed"] is True


def test_formal_routing_ablation_reports_meet_acceptance_gates():
    reports = {
        layer: json.loads(
            (REPORTS / "routing_ablation_{}.json".format(layer.lower()))
            .read_text(encoding="utf-8")
        )
        for layer in ablation.LAYERS
    }
    summary = json.loads(
        (REPORTS / "routing_ablation_summary.json")
        .read_text(encoding="utf-8")
    )
    matrix = json.loads(
        (REPORTS / "layer_matrix.json").read_text(encoding="utf-8")
    )

    manifest_shas = {
        report["dataset"]["manifest"]["sha256"]
        for report in reports.values()
    }
    current_manifest = json.loads(
        (PROJECT_ROOT / "evals" / "golden_set_scaled_120.manifest.json")
        .read_text(encoding="utf-8")
    )
    assert len(manifest_shas) == 1
    assert manifest_shas == {current_manifest["sha256"]}
    for layer, report in reports.items():
        assert report["layer"] == layer
        assert report["temporary"] is False
        assert report["formal_report"] is True
        assert report["dataset"]["evaluated"] == 120
        assert len(report["questions"]) == 120
    assert reports["L2"]["metrics"]["doc_recall@5"] == 0.9762
    assert reports["L3a"]["metrics"]["doc_recall@5"] >= 0.95
    assert reports["L3b"]["metrics"]["route_obligation_coverage"] >= 0.95
    assert reports["L3b"]["metrics"]["missing_obligation_rate"] <= 0.05
    assert reports["L3b"]["metrics"]["retry_rate"] <= 0.0917
    assert reports["L3a"]["metrics"][
        "cross_document_contamination_rate"] == 0
    assert reports["L3b"]["metrics"][
        "cross_document_contamination_rate"] == 0
    assert summary["acceptance"]["all_passed"] is True
    assert set(matrix["layers"]) == {"L2", "L3a", "L3b"}
    assert matrix["golden_set"]["total"] == 120
    assert matrix["dataset"]["manifest"]["sha256"] in manifest_shas
