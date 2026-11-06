"""Run P3 scaled retrieval/L3/generation reports over the 120-question set."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import run_agent_eval as agent_eval  # noqa: E402
import run_eval as retrieval_eval  # noqa: E402
from rag_adapter import rag  # noqa: E402
from rag_adapter.settings import load_env, use_project_root_as_cwd  # noqa: E402

DATASET = PROJECT_ROOT / "evals" / "golden_set_scaled_120.jsonl"
MANIFEST = PROJECT_ROOT / "evals" / "golden_set_scaled_120.manifest.json"
REPORTS = PROJECT_ROOT / "evals" / "reports"


def load_questions(limit: int | None = None) -> list[dict]:
    questions = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return questions[:limit] if limit else questions


def summarize_repeats(reports: list[dict]) -> dict:
    numeric_keys = sorted({
        key
        for report in reports
        for key, value in (report.get("metrics") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    summary = {}
    for key in numeric_keys:
        values = [
            report["metrics"][key] for report in reports
            if isinstance((report.get("metrics") or {}).get(key), (int, float))
        ]
        if not values:
            continue
        summary[key] = {
            "mean": round(statistics.mean(values), 6),
            "stddev": round(statistics.pstdev(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "runs": len(values),
        }
    return summary


def write_report(name: str, payload: dict) -> Path:
    path = REPORTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def evaluation_rows(report: dict) -> list[dict]:
    """Return per-question rows across the agent/retrieval report contracts."""
    return report.get("rows") or report.get("questions") or []


def dataset_meta(questions: list[dict]) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "path": str(DATASET),
        "manifest": manifest,
        "evaluated": len(questions),
    }


def run_retrieval(questions: list[dict]) -> dict:
    report = retrieval_eval.evaluate_layer(
        "L0", questions, strategy="recursive", top_k=5)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "{} + BAAI/bge-small-zh-v1.5".format(
            os.environ.get("LLM_MODEL", "LLM_MODEL")),
        "dataset": dataset_meta(questions),
        "model": "BAAI/bge-small-zh-v1.5",
        "metrics": report["metrics"],
        "questions": report["questions"],
    }
    write_report("scaled_retrieval.json", payload)
    return payload


def run_l3_retrieval(questions: list[dict], top_k: int = 5) -> dict:
    args = SimpleNamespace(
        generate=False,
        rule_rewrite=True,
        max_retries=2,
        no_scope=False,
        no_pre_retrieval_routing=False,
        no_strict_route_filter=False,
        route_candidate_limit=50,
        no_llm_grade=True,
        strategy="recursive",
        top_k=top_k,
    )
    report = agent_eval.evaluate(questions, args)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_meta(questions),
        "model": "BAAI/bge-small-zh-v1.5 + BAAI/bge-reranker-base",
        "metrics": report.get("metrics") or {},
        "questions": evaluation_rows(report),
    }
    write_report("scaled_l3.json", payload)
    return payload


def run_generation(questions: list[dict], repeats: int,
                   top_k: int = 5,
                   generation_mode: str = "vector") -> dict:
    started = time.time()
    runs = []
    service = None
    if generation_mode == "vector":
        service = rag.build_service(
            strategy="recursive", mode="vector", use_rerank=False)
    for repeat in range(1, repeats + 1):
        print("generation repeat {}/{}".format(repeat, repeats), flush=True)
        args = SimpleNamespace(
            generate=True,
            rule_rewrite=True,
            max_retries=0,
            no_scope=False,
            no_pre_retrieval_routing=False,
            no_strict_route_filter=False,
            route_candidate_limit=50,
            no_llm_grade=True,
            strategy="recursive",
            top_k=top_k,
        )
        runs.append(agent_eval.evaluate(questions, args, service=service))
    summary = summarize_repeats(runs)
    for alias, source in (
        ("answer_correctness", "answer_value_rate"),
        ("citation_full_coverage", "citation_full_coverage_rate"),
        ("refusal_correctness", "refuse_correct_rate"),
    ):
        if source in summary:
            summary[alias] = dict(summary[source])
    failures = {}
    for repeat, report in enumerate(runs, start=1):
        failures[str(repeat)] = [
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "graded_reason": row.get("graded_reason"),
                "missing_obligations": [
                    item.get("obligation_id")
                    for item in row.get("missing_obligations") or []
                ],
            }
            for row in evaluation_rows(report)
            if row.get("status") in {
                "error", "answer_hard_gate_failed",
                "answered_unverified", "refused_no_evidence",
            }
        ]
    claim_report = {}
    claim_path = REPORTS / "claim_audit.json"
    if claim_path.exists():
        claim_report = json.loads(claim_path.read_text(encoding="utf-8"))
    claim_metrics = claim_report.get("metrics") or {}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 2),
        "model": "BAAI/bge-small-zh-v1.5 + {}".format(
            os.environ.get("LLM_MODEL", "LLM_MODEL")),
        "dataset": dataset_meta(questions),
        "config": {
            "repeats": repeats,
            "top_k": top_k,
            "rule_rewrite": True,
            "llm_grader": False,
            "max_retries": 0,
            "generate": True,
            "generation_mode": generation_mode,
        },
        "hard_gate": {
            "source": "claim_audit.json",
            "precision": claim_metrics.get("hard_gate_precision"),
            "recall": claim_metrics.get("hard_violation_recall"),
            "false_positive_rate": claim_metrics.get(
                "clean_answer_false_positive_rate"),
        },
        "metrics": summary,
        "runs": [
            {"metrics": report.get("metrics") or {},
             "elapsed_seconds": report.get("elapsed_seconds")}
            for report in runs
        ],
        "failures": failures,
    }
    write_report("scaled_generation.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run P3 scaled evaluation")
    parser.add_argument(
        "--mode", choices=["retrieval", "l3", "generation", "all"],
        default="retrieval",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--generation-mode", choices=["vector", "hybrid"], default="vector")
    args = parser.parse_args()
    use_project_root_as_cwd()
    load_env()
    questions = load_questions(args.limit)
    if args.mode in ("retrieval", "all"):
        report = run_retrieval(questions)
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    if args.mode in ("l3", "all"):
        report = run_l3_retrieval(questions, top_k=args.top_k)
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    if args.mode in ("generation", "all"):
        report = run_generation(
            questions, repeats=max(1, args.repeats), top_k=args.top_k,
            generation_mode=args.generation_mode)
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
