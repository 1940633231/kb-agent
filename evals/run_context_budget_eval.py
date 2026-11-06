"""P2 context-budget regression over the 50-question golden set."""
from __future__ import annotations

import json
import hashlib
import statistics
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.context.manager import ContextManager  # noqa: E402
from app.generation.prompt import PromptBuilder  # noqa: E402

from agent.context_budget import build_context_with_budget  # noqa: E402
from agent.evidence_contract import (  # noqa: E402
    assess_obligations,
    build_obligations,
)

GOLDEN_SET = PROJECT_ROOT / "evals" / "golden_set_scaled_120.jsonl"
SCALED_MANIFEST = PROJECT_ROOT / "evals" / "golden_set_scaled_120.manifest.json"
INDEX_METADATA = PROJECT_ROOT / "data" / "index" / "recursive" / "metadata.json"
REPORT_PATH = PROJECT_ROOT / "evals" / "reports" / "context_budget.json"
GENERATION_REPORT = (
    PROJECT_ROOT / "evals" / "reports" / "stage0_expanded_generation.json"
)
MAX_CONTEXT_TOKENS = 4096
OUTPUT_RESERVED_TOKENS = 1024


class CharCounter:
    @staticmethod
    def count(text: str) -> int:
        return len(text or "")


def percentile(values, ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio)))
    return round(ordered[index], 2)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDEN_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_chunks() -> dict[str, list[dict]]:
    metadata = json.loads(INDEX_METADATA.read_text(encoding="utf-8"))
    by_document: dict[str, list[dict]] = {}
    for item in metadata.values():
        document_id = str(
            item.get("document_id")
            or (item.get("metadata") or {}).get("document_id")
            or "")
        if document_id:
            by_document.setdefault(document_id, []).append(item)
    return by_document


def select_hits(item: dict, by_document: dict[str, list[dict]]) -> list[dict]:
    literals = [str(value) for value in item.get("evidence_literals") or []]
    keywords = [str(value) for value in item.get("expected_keywords") or []]
    selected = []
    for doc in item.get("expected_docs") or []:
        document_id = "{}-{}".format(doc["ts_code"], doc["year"])
        candidates = by_document.get(document_id) or []
        preferred = [
            chunk for chunk in candidates
            if any(value in (chunk.get("content") or "")
                   for value in literals + keywords)
        ]
        pool = preferred or candidates
        for index, chunk in enumerate(pool[:2]):
            copied = dict(chunk)
            copied["score"] = 1.0 - index * 0.01
            selected.append(copied)
    return selected


def long_history(index: int) -> list[dict]:
    history = []
    for turn in range(5):
        history.append({
            "role": "user",
            "content": "第{}轮历史提问{}".format(turn, "甲" * (100 + index % 50)),
        })
        history.append({
            "role": "assistant",
            "content": "第{}轮历史回答{}".format(turn, "乙" * (140 + index % 50)),
        })
    return history


def main() -> int:
    global GOLDEN_SET
    parser = argparse.ArgumentParser(description="P2/P3 context budget evaluation")
    parser.add_argument("--golden-set", default=str(GOLDEN_SET))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    golden_path = Path(args.golden_set)
    questions = load_questions()
    if golden_path != GOLDEN_SET:
        GOLDEN_SET = golden_path
        questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]
    by_document = load_chunks()
    counter = CharCounter()
    baseline_manager = ContextManager(
        token_counter=counter,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        reserved_tokens=OUTPUT_RESERVED_TOKENS,
    )
    prompt_builder = PromptBuilder()
    pipeline = SimpleNamespace(
        context_manager=baseline_manager,
        prompt_builder=prompt_builder,
    )
    rows = []
    for index, item in enumerate(questions):
        hits = select_hits(item, by_document)
        if not hits:
            continue
        history = long_history(index)
        obligations = build_obligations(item["question"])
        started = time.perf_counter()
        baseline_manager.build(item["question"], hits)
        baseline_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        result = build_context_with_budget(
            pipeline,
            item["question"],
            hits,
            history=history,
            obligations=obligations,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            output_reserved_tokens=OUTPUT_RESERVED_TOKENS,
        )
        budget_ms = (time.perf_counter() - started) * 1000
        obligation_audit = assess_obligations(
            result["chunks"], obligations, item["question"])
        rows.append({
            "id": item["id"],
            "baseline_ms": baseline_ms,
            "budget_ms": budget_ms,
            "overhead_pct": (
                max(0.0, (budget_ms - baseline_ms) / baseline_ms * 100)
                if baseline_ms else 0.0
            ),
            "prompt_token_estimate": result["budget"][
                "prompt_token_estimate"],
            "budget_overflow": result["budget"]["budget_overflow"],
            "history_tokens": result["budget"]["history_tokens"],
            "evidence_tokens": result["budget"]["evidence_tokens"],
            "skipped_history_turns": result["budget"][
                "skipped_history_turns"],
            "obligation_coverage": obligation_audit["coverage"],
            "missing_obligations": obligation_audit["missing"],
        })

    metrics = {
        "cases": len(rows),
        "budget_overflow_rate": round(
            sum(1 for row in rows if row["budget_overflow"]) / len(rows), 4
        ) if rows else None,
        "evidence_obligation_context_coverage": round(
            statistics.mean(row["obligation_coverage"] for row in rows), 4
        ) if rows else None,
        "mean_prompt_tokens": round(
            statistics.mean(row["prompt_token_estimate"] for row in rows), 2
        ) if rows else None,
        "mean_skipped_history_turns": round(
            statistics.mean(row["skipped_history_turns"] for row in rows), 2
        ) if rows else None,
        "latency_p50_ms": {
            "baseline": percentile([row["baseline_ms"] for row in rows], 0.5),
            "with_budget": percentile([row["budget_ms"] for row in rows], 0.5),
            "overhead_pct": percentile(
                [row["overhead_pct"] for row in rows], 0.5),
        },
        "latency_p95_ms": {
            "baseline": percentile([row["baseline_ms"] for row in rows], 0.95),
            "with_budget": percentile([row["budget_ms"] for row in rows], 0.95),
            "overhead_pct": percentile(
                [row["overhead_pct"] for row in rows], 0.95),
        },
    }
    generation_mean_seconds = None
    if GENERATION_REPORT.exists():
        generation_report = json.loads(
            GENERATION_REPORT.read_text(encoding="utf-8"))
        generation_mean_seconds = (
            generation_report.get("metrics") or {}
        ).get("mean_seconds_per_query")
    if generation_mean_seconds:
        p95_overhead_ms = (
            max(
                0.0,
                metrics["latency_p95_ms"]["with_budget"]
                - metrics["latency_p95_ms"]["baseline"],
            )
        )
        metrics["latency_p95_overhead_ms"] = round(p95_overhead_ms, 3)
        metrics["estimated_end_to_end_overhead_pct"] = round(
            p95_overhead_ms / (generation_mean_seconds * 1000) * 100, 6
        )
        metrics["generation_baseline_mean_seconds"] = generation_mean_seconds
    else:
        metrics["estimated_end_to_end_overhead_pct"] = None
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(GOLDEN_SET),
        "dataset_version": (
            "kb-agent-scaled-120-v1"
            if GOLDEN_SET == SCALED_MANIFEST.with_name(
                "golden_set_scaled_120.jsonl")
            else "legacy-golden-set"
        ),
        "model": "char-counter + production-rag ContextManager",
        "tokenizer": "char-counter",
        "corpus": str(INDEX_METADATA),
        "corpus_metadata_sha256": sha256_of(INDEX_METADATA),
        "config": {
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "reserved_output_tokens": OUTPUT_RESERVED_TOKENS,
        },
        "metrics": metrics,
        "questions": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("report -> {}".format(REPORT_PATH))
    return 0 if (
        metrics["budget_overflow_rate"] == 0.0
        and metrics["estimated_end_to_end_overhead_pct"] is not None
        and metrics["estimated_end_to_end_overhead_pct"] <= 5.0
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
