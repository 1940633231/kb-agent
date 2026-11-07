"""Run the 120-question pre-retrieval routing ablation.

L2, L3a, and L3b share one hybrid/rerank service so model loading happens once.
Formal runs write the four official reports; ``--limit`` runs use ``_debug``
filenames and never touch the formal reports or layer matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import run_agent_eval as agent_eval  # noqa: E402
from rag_adapter import rag  # noqa: E402
from rag_adapter.settings import load_env, use_project_root_as_cwd  # noqa: E402

DATASET = PROJECT_ROOT / "evals" / "golden_set_scaled_120.jsonl"
MANIFEST = PROJECT_ROOT / "evals" / "golden_set_scaled_120.manifest.json"
CORPUS_METADATA = (
    PROJECT_ROOT / "data" / "docs" / "annual_reports" / "metadata.json"
)
REPORTS = PROJECT_ROOT / "evals" / "reports"
FORMAL_QUESTION_COUNT = 120

LAYERS = {
    "L2": {
        "label": "hybrid + rerank，无预路由",
        "routing": False,
        "max_retries": 0,
    },
    "L3a": {
        "label": "L2 + 检索前公司/年份路由",
        "routing": True,
        "max_retries": 0,
    },
    "L3b": {
        "label": "L3a + 缺失义务定向重试",
        "routing": True,
        "max_retries": 2,
    },
}


def load_questions(limit: int = 0) -> List[Dict[str, Any]]:
    questions = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return questions[:limit] if limit else questions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_dataset() -> Dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual_dataset_sha = _sha256(DATASET)
    actual_corpus_sha = _sha256(CORPUS_METADATA)
    if actual_dataset_sha != manifest.get("sha256"):
        raise SystemExit(
            "dataset SHA mismatch: {} != {}".format(
                actual_dataset_sha, manifest.get("sha256")))
    if actual_corpus_sha != manifest.get("corpus_metadata_sha256"):
        raise SystemExit(
            "corpus metadata SHA mismatch: {} != {}".format(
                actual_corpus_sha, manifest.get("corpus_metadata_sha256")))
    return manifest


def dataset_meta(questions: Sequence[Dict[str, Any]],
                 manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(DATASET),
        "manifest_path": str(MANIFEST),
        "manifest": manifest,
        "evaluated": len(questions),
        "available": int(manifest["counts"]["total"]),
    }


def layer_config(layer: str, top_k: int,
                 route_candidate_limit: int) -> Dict[str, Any]:
    spec = LAYERS[layer]
    return {
        "strategy": "recursive",
        "mode": "hybrid",
        "use_rerank": True,
        "top_k": top_k,
        "route_candidate_limit": route_candidate_limit,
        "max_retries": spec["max_retries"],
        "generate": False,
        "rule_rewrite": True,
        "llm_grader": False,
        "enable_pre_retrieval_routing": spec["routing"],
        "strict_route_filter": True,
    }


def evaluate_layer(layer: str, questions: Sequence[Dict[str, Any]],
                   service: Any, manifest: Dict[str, Any], top_k: int,
                   route_candidate_limit: int, temporary: bool) -> Dict[str, Any]:
    config = layer_config(layer, top_k, route_candidate_limit)
    args = SimpleNamespace(
        strategy=config["strategy"],
        top_k=top_k,
        max_retries=config["max_retries"],
        generate=False,
        rule_rewrite=True,
        no_scope=False,
        no_pre_retrieval_routing=not config[
            "enable_pre_retrieval_routing"],
        no_strict_route_filter=False,
        route_candidate_limit=route_candidate_limit,
        no_llm_grade=True,
    )
    started = time.time()
    outcome = agent_eval.evaluate(questions, args, service=service)
    return {
        "layer": layer,
        "label": LAYERS[layer]["label"],
        "temporary": temporary,
        "formal_report": not temporary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 2),
        "dataset": dataset_meta(questions, manifest),
        "config": config,
        "metrics": outcome["metrics"],
        "questions": outcome["rows"],
    }


def run_ablation(questions: Sequence[Dict[str, Any]], service: Any,
                 manifest: Dict[str, Any], top_k: int = 5,
                 route_candidate_limit: int = 50,
                 on_layer_complete: Optional[Any] = None
                 ) -> Dict[str, Dict[str, Any]]:
    temporary = len(questions) != FORMAL_QUESTION_COUNT
    reports: Dict[str, Dict[str, Any]] = {}
    for layer in LAYERS:
        print("\n== {} {} ==".format(layer, LAYERS[layer]["label"]),
              flush=True)
        reports[layer] = evaluate_layer(
            layer,
            questions,
            service,
            manifest,
            top_k=top_k,
            route_candidate_limit=route_candidate_limit,
            temporary=temporary,
        )
        print(json.dumps(reports[layer]["metrics"], ensure_ascii=False,
                         indent=2), flush=True)
        if on_layer_complete is not None:
            on_layer_complete(layer, reports[layer])
    return reports


def report_paths(temporary: bool) -> Dict[str, Path]:
    suffix = "_debug" if temporary else ""
    paths = {
        layer: REPORTS / "routing_ablation_{}{}.json".format(
            layer.lower(), suffix)
        for layer in LAYERS
    }
    paths["summary"] = REPORTS / (
        "routing_ablation_summary{}.json".format(suffix))
    return paths


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.mean(clean), 6) if clean else None


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return round(ordered[lower], 6)
    weight = position - lower
    return round(
        ordered[lower] * (1.0 - weight) + ordered[upper] * weight,
        6,
    )


def latency_breakdown(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    seconds = [
        float(row["seconds"]) for row in rows
        if isinstance(row.get("seconds"), (int, float))
    ]
    single = [
        float(row["seconds"]) for row in rows
        if row.get("expected_behavior") == "answer"
        and row.get("question_type") in {"fact", "table"}
        and len(row.get("wanted_docs") or []) == 1
        and isinstance(row.get("seconds"), (int, float))
    ]
    by_type: Dict[str, Any] = {}
    for question_type in sorted({
            str(row.get("question_type")) for row in rows
            if row.get("question_type")}):
        typed = [
            row for row in rows
            if row.get("question_type") == question_type
        ]
        by_type[question_type] = {
            "questions": len(typed),
            "mean_seconds": _mean(row.get("seconds") for row in typed),
            "doc_recall": _mean(
                row.get("doc_recall") for row in typed),
            "route_obligation_coverage": _mean(
                row.get("route_coverage_rate") for row in typed),
        }
    return {
        "p50_seconds": _percentile(seconds, 0.50),
        "p95_seconds": _percentile(seconds, 0.95),
        "mean_seconds_per_query": _mean(seconds),
        "single_company_single_year_mean_seconds": _mean(single),
        "fact_mean_seconds": by_type.get("fact", {}).get("mean_seconds"),
        "table_mean_seconds": by_type.get("table", {}).get("mean_seconds"),
        "cross_document_mean_seconds": by_type.get(
            "cross_document", {}).get("mean_seconds"),
        "refusal_mean_seconds": by_type.get("refuse", {}).get(
            "mean_seconds"),
        "by_question_type": by_type,
    }


def metric_deltas(upper: Dict[str, Any],
                  lower: Dict[str, Any]) -> Dict[str, Optional[float]]:
    deltas: Dict[str, Optional[float]] = {}
    for key in sorted(set(upper) | set(lower)):
        upper_value = upper.get(key)
        lower_value = lower.get(key)
        if (isinstance(upper_value, (int, float))
                and not isinstance(upper_value, bool)
                and isinstance(lower_value, (int, float))
                and not isinstance(lower_value, bool)):
            deltas[key] = round(float(upper_value) - float(lower_value), 6)
    return deltas


def _gate_metric(name: str, metric: Optional[float], op: str,
                 threshold: float) -> Dict[str, Any]:
    passed = False
    if metric is not None:
        if op == ">=":
            passed = metric >= threshold
        elif op == "<=":
            passed = metric <= threshold
        elif op == "==":
            passed = metric == threshold
    return {
        "metric": name,
        "value": metric,
        "operator": op,
        "threshold": threshold,
        "passed": passed,
    }


def acceptance_checks(reports: Dict[str, Dict[str, Any]], top_k: int,
                      temporary: bool) -> Dict[str, Any]:
    metrics = {layer: report["metrics"] for layer, report in reports.items()}
    breakdowns = {
        layer: latency_breakdown(report["questions"])
        for layer, report in reports.items()
    }
    recall_key = "doc_recall@{}".format(top_k)
    l2_seconds = breakdowns["L2"]["single_company_single_year_mean_seconds"]
    l3a_seconds = breakdowns["L3a"][
        "single_company_single_year_mean_seconds"]
    latency_limit = (
        round(l3a_seconds / l2_seconds, 6)
        if l2_seconds else None
    )
    manifest_shas = {
        report.get("dataset", {}).get("manifest", {}).get("sha256")
        for report in reports.values()
    }
    question_counts = {
        report.get("dataset", {}).get("evaluated")
        for report in reports.values()
    }
    contamination_values = {
        layer: metrics[layer].get("cross_document_contamination_rate")
        for layer in LAYERS
    }
    routed_contamination = {
        layer: contamination_values[layer] for layer in ("L3a", "L3b")
    }
    checks = {
        "l3a_l3b_cross_document_contamination_zero": {
            "values": contamination_values,
            "routed_values": routed_contamination,
            "passed": all(
                value == 0 for value in routed_contamination.values()),
        },
        "l3a_doc_recall_at_k": _gate_metric(
            recall_key, metrics["L3a"].get(recall_key), ">=", 0.95),
        "l3b_route_obligation_coverage": _gate_metric(
            "route_obligation_coverage",
            metrics["L3b"].get("route_obligation_coverage"), ">=", 0.95),
        "l3b_missing_obligation_rate": _gate_metric(
            "missing_obligation_rate",
            metrics["L3b"].get("missing_obligation_rate"), "<=", 0.05),
        "l3b_retry_rate": _gate_metric(
            "retry_rate", metrics["L3b"].get("retry_rate"), "<=", 0.0917),
        "l3a_single_company_latency_ratio": {
            "metric": "L3a / L2 single-company single-year mean seconds",
            "l2_seconds": l2_seconds,
            "l3a_seconds": l3a_seconds,
            "ratio": latency_limit,
            "operator": "<=",
            "threshold": 1.10,
            "passed": latency_limit is not None and latency_limit <= 1.10,
        },
        "manifest_sha_consistent": {
            "values": sorted(sha for sha in manifest_shas if sha),
            "passed": len(manifest_shas) == 1 and None not in manifest_shas,
        },
        "all_reports_have_120_questions": {
            "values": sorted(
                count for count in question_counts if count is not None),
            "passed": (
                not temporary
                and question_counts == {FORMAL_QUESTION_COUNT}
            ),
        },
    }
    return {
        "temporary": temporary,
        "formal_report": not temporary,
        "all_passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def build_summary(reports: Dict[str, Dict[str, Any]], top_k: int = 5
                  ) -> Dict[str, Any]:
    first = reports["L2"]
    temporary = bool(first.get("temporary"))
    metrics = {
        layer: reports[layer]["metrics"] for layer in LAYERS
    }
    return {
        "temporary": temporary,
        "formal_report": not temporary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": first["dataset"],
        "configs": {
            layer: reports[layer]["config"] for layer in LAYERS
        },
        "elapsed_seconds": {
            layer: reports[layer].get("elapsed_seconds") for layer in LAYERS
        },
        "metrics": metrics,
        "deltas": {
            "L3a-L2": metric_deltas(metrics["L3a"], metrics["L2"]),
            "L3b-L3a": metric_deltas(metrics["L3b"], metrics["L3a"]),
        },
        "latency_breakdown": {
            layer: latency_breakdown(reports[layer]["questions"])
            for layer in LAYERS
        },
        "acceptance": acceptance_checks(reports, top_k, temporary),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def reusable_report(path: Path, layer: str, manifest: Dict[str, Any],
                    top_k: int, route_candidate_limit: int,
                    temporary: bool,
                    question_count: int) -> Optional[Dict[str, Any]]:
    """Load an existing report only when it matches the current contract."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("temporary") != temporary
        or payload.get("layer") != layer
        or payload.get("config") != layer_config(
            layer, top_k, route_candidate_limit)
        or payload.get("dataset", {}).get("manifest", {}).get("sha256")
        != manifest.get("sha256")
        or len(payload.get("questions") or []) != question_count
        or not (payload.get("metrics"))
    ):
        return None
    return payload


def update_official_layer_matrix(
        reports: Dict[str, Dict[str, Any]],
        path: Optional[Path] = None) -> Path:
    """Rebuild the matrix from the current 120-question ablation only."""
    path = path or REPORTS / "layer_matrix.json"
    first = reports["L2"]
    question_types: Dict[str, int] = {}
    for row in first["questions"]:
        question_type = str(row.get("question_type"))
        question_types[question_type] = question_types.get(question_type, 0) + 1
    matrix = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": first["config"]["strategy"],
        "top_k": first["config"]["top_k"],
        "dataset": first["dataset"],
        "golden_set": {
            "path": first["dataset"]["path"],
            "total": first["dataset"]["evaluated"],
            "by_type": question_types,
        },
        "source_reports": {
            layer: str(report_paths(False)[layer]) for layer in LAYERS
        },
        "layers": {
            layer: {
                "label": reports[layer]["label"],
                "config": reports[layer]["config"],
                "metrics": reports[layer]["metrics"],
            }
            for layer in LAYERS
        },
    }
    write_json(path, matrix)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the 120-question routing ablation")
    parser.add_argument("--limit", type=int, default=0,
                        help="debug only; formal output requires all 120")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--route-candidate-limit", type=int, default=50)
    parser.add_argument("--resume", action="store_true",
                        help="reuse completed, contract-matching layer reports")
    args = parser.parse_args()

    use_project_root_as_cwd()
    load_env()
    manifest = validate_dataset()
    questions = load_questions(max(0, args.limit))
    if not questions:
        raise SystemExit("dataset is empty")

    temporary = len(questions) != FORMAL_QUESTION_COUNT
    print("routing ablation: {} questions, temporary={}".format(
        len(questions), temporary))
    paths = report_paths(temporary)
    started = time.time()
    reports: Dict[str, Dict[str, Any]] = {}
    pending: List[str] = []
    for layer in LAYERS:
        existing = (
            reusable_report(
                paths[layer], layer, manifest, args.top_k,
                args.route_candidate_limit, temporary, len(questions))
            if args.resume else None
        )
        if existing is not None:
            reports[layer] = existing
            print("{} report reused -> {}".format(layer, paths[layer]))
        else:
            pending.append(layer)

    if pending:
        # One service instance is reused across every pending layer.
        service = rag.build_service(
            strategy="recursive", mode="hybrid", use_rerank=True)
        for layer in pending:
            print("\n== {} {} ==".format(layer, LAYERS[layer]["label"]),
                  flush=True)
            reports[layer] = evaluate_layer(
                layer,
                questions,
                service,
                manifest,
                top_k=args.top_k,
                route_candidate_limit=args.route_candidate_limit,
                temporary=temporary,
            )
            write_json(paths[layer], reports[layer])
            print("{} report -> {}".format(layer, paths[layer]), flush=True)
            print(json.dumps(reports[layer]["metrics"], ensure_ascii=False,
                             indent=2), flush=True)

    reports = {layer: reports[layer] for layer in LAYERS}
    summary = build_summary(reports, top_k=args.top_k)
    summary["total_elapsed_seconds"] = round(time.time() - started, 2)
    write_json(paths["summary"], summary)
    print("summary -> {}".format(paths["summary"]))

    if not temporary:
        matrix_path = update_official_layer_matrix(reports)
        print("layer matrix -> {}".format(matrix_path))
    print(json.dumps(summary["acceptance"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
