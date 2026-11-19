"""Run and score the frozen/candidate held-out set.

The scorer consumes `unsupported_obligations`, which the existing 120-question
runners do not model. Retrieval page matching uses `chunk_start_page` because
`describe_hit()` reports the page of the chunk start, not the answer literal.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = PROJECT_ROOT / "evals"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVALS_DIR))

import run_eval as base  # noqa: E402
from agent.answer_audit import audit_answer  # noqa: E402
from agent.graph import ask, build_graph, default_runtime  # noqa: E402
from agent.grounding import extract_numbers, number_equivalents  # noqa: E402
from rag_adapter import rag  # noqa: E402
from rag_adapter.settings import load_env, use_project_root_as_cwd  # noqa: E402

DATASET = EVALS_DIR / "heldout_40.jsonl"
MANIFEST = EVALS_DIR / "heldout_40.manifest.json"
DEFAULT_OUTPUT = EVALS_DIR / "reports" / "heldout_40_baseline.json"
CHUNK_METADATA = PROJECT_ROOT / "data" / "index" / "recursive" / "metadata.json"
_CHUNK_INDEX: dict[str, dict[str, Any]] | None = None

REFUSAL_MARKERS = (
    "无法回答", "无法确认", "无法提供", "无法获取", "没有相关", "没有找到",
    "未提及", "未披露", "不足以", "不包含", "未包含", "不在知识库",
    "知识库未包含", "知识库中没有", "不支持", "超出范围", "暂无",
)
SENTENCE_SPLIT = re.compile(r"[。！？；\n]+")
DATE_PATTERNS = (
    re.compile(r"20\d{2}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?"),
    re.compile(r"20\d{2}-\d{1,2}-\d{1,2}"),
)
CITATION_REF = re.compile(r"\[\d+\]")
CLAIM_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
NUMBER_UNIT = re.compile(
    r"-?\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|百万元|千元|元|%|吨)")
QUOTED_EVIDENCE_MARKERS = (
    "上下文未提供", "上下文未", "上下文中没有", "仅提到", "不能作为",
    "不能替代", "未单独列出", "未单独", "未明确列出", "并非", "不属于",
)
ASSERTION_PATTERN = re.compile(
    r"(?:为|是|达到|约为|金额为|数值为)\s*\**\s*-?\d")


def _norm(text: str) -> str:
    return base.normalize(text or "")


def _value_variants(value: dict[str, Any]) -> set[str]:
    return {_norm(item) for item in base.answer_variants(value) if item}


def _all_supported_variants(record: dict[str, Any]) -> set[str]:
    variants: set[str] = set()
    for value in record.get("expected_values") or []:
        variants.update(_value_variants(value))
    return variants


def load_chunk_index() -> dict[str, dict[str, Any]]:
    global _CHUNK_INDEX
    if _CHUNK_INDEX is None:
        raw = json.loads(CHUNK_METADATA.read_text(encoding="utf-8"))
        _CHUNK_INDEX = {item["chunk_id"]: item for item in raw.values()}
    return _CHUNK_INDEX


def expected_value_results(record: dict[str, Any],
                           answer: str) -> list[dict[str, Any]]:
    normalized = _norm(answer)
    results = []
    for value in record.get("expected_values") or []:
        variants = _value_variants(value)
        present = any(variant and variant in normalized for variant in variants)
        results.append({
            "label": value.get("label"),
            "company": value.get("company"),
            "year": value.get("year"),
            "display": value.get("display"),
            "present": present,
        })
    return results


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in SENTENCE_SPLIT.split(text or "")
            if item.strip()]


def _date_terms(value: str) -> set[str]:
    match = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", value or "")
    if not match:
        return {value} if value else set()
    year, month, day = match.groups()
    terms = {value, "{}年{}月".format(year, int(month))}
    if day:
        terms.add("{}年{}月{}日".format(year, int(month), int(day)))
    return terms


def _metric_terms(metric: str) -> set[str]:
    if not metric:
        return set()
    terms = {metric}
    for suffix in ("股票", "价格"):
        terms.add(metric.replace(suffix, ""))
    if "收盘价" in metric:
        terms.add("收盘价")
    if "总市值" in metric:
        terms.add("总市值")
    if "现货价" in metric:
        terms.add("现货价")
    return {item for item in terms if item}


def _unsupported_terms(item: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    if item.get("date"):
        terms.update(_date_terms(str(item["date"])))
        # Partial questions can refer to the same supported company twice.
        # Date/metric terms distinguish the unsupported market-data clause.
    elif item.get("entity"):
        terms.add(str(item["entity"]))
    terms.update(_metric_terms(str(item.get("metric") or "")))
    return terms


def _claim_numbers(text: str) -> list[str]:
    cleaned = text or ""
    for pattern in DATE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = CITATION_REF.sub(" ", cleaned)
    numbers = list(NUMBER_UNIT.findall(cleaned))
    numbers.extend(CLAIM_NUMBER.findall(cleaned))
    filtered = []
    for number in numbers:
        plain = number.replace(",", "").rstrip(".")
        if re.fullmatch(r"(?:19|20)\d{2}", plain):
            continue
        filtered.append(number)
    return list(dict.fromkeys(filtered))


def _is_supported_number(number: str, supported_variants: set[str]) -> bool:
    plain = re.sub(
        r"(亿元|万元|百万元|千元|元|%|吨)$", "", number.strip())
    forms = [plain] + number_equivalents(plain.replace(" ", ""))
    return any(_norm(item) in supported_variants for item in forms if item)


def unsupported_results(record: dict[str, Any], answer: str,
                        supported_variants: set[str] | None = None
                        ) -> list[dict[str, Any]]:
    supported_variants = supported_variants or _all_supported_variants(record)
    sentences = _sentences(answer)
    results = []
    for obligation in record.get("unsupported_obligations") or []:
        terms = _unsupported_terms(obligation)
        matched = [
            sentence for sentence in sentences
            if any(term in sentence for term in terms)
        ]
        leaked: list[str] = []
        for sentence in matched:
            quoted_without_assertion = (
                any(marker in sentence for marker in QUOTED_EVIDENCE_MARKERS)
                and ASSERTION_PATTERN.search(sentence) is None
            )
            if quoted_without_assertion:
                continue
            for number in _claim_numbers(sentence):
                if not _is_supported_number(number, supported_variants):
                    leaked.append(number)
        refused = bool(matched) and any(
            marker in answer for marker in REFUSAL_MARKERS)
        if leaked:
            status = "leaked_value"
        elif refused:
            status = "refused"
        elif matched:
            status = "not_addressed"
        else:
            status = "not_addressed"
        results.append({
            "obligation_id": obligation.get("obligation_id"),
            "entity": obligation.get("entity"),
            "year": obligation.get("year"),
            "date": obligation.get("date"),
            "metric": obligation.get("metric"),
            "status": status,
            "matched_sentences": matched,
            "leaked_numbers": list(dict.fromkeys(leaked)),
        })
    return results


def citation_chunk_results(record: dict[str, Any],
                           citations: list[dict[str, Any]]
                           ) -> dict[str, Any]:
    expected = {
        str(item["chunk_id"])
        for item in record.get("source_pages") or []
        if item.get("chunk_id")
    }
    cited = {
        str(item.get("chunk_id"))
        for item in citations or []
        if item.get("chunk_id")
    }
    missing = sorted(expected - cited)
    return {
        "expected_chunks": sorted(expected),
        "cited_chunks": sorted(cited),
        "missing_chunks": missing,
        "complete": not missing if expected else True,
    }


def score_answer(record: dict[str, Any], answer: str,
                 citations: list[dict[str, Any]] | None = None,
                 chunks: list[dict[str, Any]] | None = None,
                 ) -> dict[str, Any]:
    answer = answer or ""
    citations = citations or []
    value_results = expected_value_results(record, answer)
    values_complete = bool(value_results) and all(
        item["present"] for item in value_results)
    supported_variants = _all_supported_variants(record)
    missing_unsupported = unsupported_results(
        record, answer, supported_variants=supported_variants)
    citation_result = citation_chunk_results(record, citations)
    audit_result = (
        audit_answer(
            answer, citations, chunks or [], record["question"],
            obligations=record.get("required_evidence") or [],
        )
        if chunks is not None else None
    )
    binding_complete = (
        bool(audit_result.get("hard_gate_passed")) if audit_result else True
    )
    binding_violations = [
        str(item.get("code"))
        for item in (audit_result or {}).get("hard_violations") or []
    ]
    unsupported_refused = all(
        item["status"] == "refused" for item in missing_unsupported)
    unsupported_leaked = any(
        item["status"] == "leaked_value" for item in missing_unsupported)
    unsupported_not_addressed = any(
        item["status"] == "not_addressed" for item in missing_unsupported)
    refusal_only = (
        any(marker in answer for marker in REFUSAL_MARKERS)
        and not any(item["present"] for item in value_results)
    )
    generic_refusal = (
        any(marker in answer for marker in REFUSAL_MARKERS)
        and not _claim_numbers(answer)
    )

    if record.get("expected_behavior") == "refuse" and generic_refusal:
        # A pure unsupported-scope question can be refused without restating
        # company/year/metric, e.g. "no retrieved evidence, so cannot answer".
        for item in missing_unsupported:
            item["status"] = "refused"
            item["refused_generically"] = True
        unsupported_refused = True
        unsupported_leaked = False
        unsupported_not_addressed = False

    reasons: list[str] = []
    behavior = record.get("expected_behavior")
    if behavior == "refuse":
        if unsupported_refused and not unsupported_leaked:
            classification = "correctly_refused"
        else:
            classification = "incorrect"
            reasons.append("unsupported_scope_answered")
    elif unsupported_leaked:
        classification = "incorrect"
        reasons.append("unsupported_value_leak")
    elif not values_complete:
        classification = "wrongly_refused" if refusal_only else "incorrect"
        reasons.append("expected_value_missing")
    elif not binding_complete:
        classification = "partially_correct"
        reasons.append("evidence_binding_failed")
    elif behavior == "partial_answer":
        classification = "partially_correct"
        if unsupported_not_addressed:
            reasons.append("unsupported_not_addressed")
        if not citation_result["complete"]:
            reasons.append("citation_chunk_incomplete")
    else:
        classification = "fully_correct"

    return {
        "classification": classification,
        "reasons": reasons,
        "expected_values": value_results,
        "values_complete": values_complete,
        "citations": citation_result,
        "binding_complete": binding_complete,
        "binding_violations": binding_violations,
        "answer_contract": (audit_result or {}).get("answer_contract"),
        "unsupported": missing_unsupported,
        "unsupported_refused": unsupported_refused,
        "unsupported_leaked": unsupported_leaked,
        "unsupported_not_addressed": unsupported_not_addressed,
    }


def expected_retrieval_pairs(record: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (str(item["doc"]), int(item["chunk_start_page"]))
        for item in record.get("source_pages") or []
    }


def retrieval_page_results(record: dict[str, Any],
                           retrieved: list[dict[str, Any]]
                           ) -> dict[str, Any]:
    expected = expected_retrieval_pairs(record)
    actual = {
        (str(item.get("document_id")), int(item["chunk_start_page"]))
        for item in retrieved or []
        if item.get("document_id") and item.get("chunk_start_page") is not None
    }
    return {
        "expected_pairs": sorted(expected),
        "retrieved_pairs": sorted(actual),
        "complete": expected <= actual if expected else True,
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        row["score"]["classification"]
        for row in rows if row.get("score")
    )
    scored = [row for row in rows if row.get("score")]
    expected_refuse = [
        row for row in scored
        if row.get("expected_behavior") == "refuse"
    ]
    expected_partial = [
        row for row in scored
        if row.get("expected_behavior") == "partial_answer"
    ]
    should_answer = [
        row for row in scored
        if row.get("expected_behavior") != "refuse"
    ]
    unsupported = [
        item for row in scored for item in row["score"]["unsupported"]
    ]
    expected_values = [
        item for row in scored for item in row["score"]["expected_values"]
    ]
    citation_rows = [
        row["score"]["citations"] for row in scored
        if row["score"]["citations"]["expected_chunks"]
    ]
    retrieval_rows = [
        row["retrieval_pages"] for row in rows if row.get("retrieval_pages")
    ]
    binding_rows = [
        row for row in scored if row.get("expected_behavior") != "refuse"
    ]
    violation_counts: Counter = Counter()
    for row in binding_rows:
        violation_counts.update(row["score"].get("binding_violations") or [])

    def count_rate(items: list[dict[str, Any]], key: str,
                   value: str) -> float | None:
        if not items:
            return None
        return round(sum(item[key] == value for item in items) / len(items), 4)

    return {
        "scored_questions": len(scored),
        "classification_counts": dict(sorted(counts.items())),
        "fully_correct_rate": _mean([
            row["score"]["classification"] == "fully_correct"
            for row in scored
        ]) if scored else None,
        "partial_success_rate": _mean([
            row["score"]["classification"] == "partially_correct"
            for row in expected_partial
        ]) if expected_partial else None,
        "correct_refusal_rate": _mean([
            row["score"]["classification"] == "correctly_refused"
            for row in expected_refuse
        ]) if expected_refuse else None,
        "wrong_refusal_rate": _mean([
            row["score"]["classification"] == "wrongly_refused"
            for row in should_answer
        ]) if should_answer else None,
        "expected_value_presence_rate": _mean([
            item["present"] for item in expected_values
        ]) if expected_values else None,
        "citation_chunk_coverage_rate": _mean([
            item["complete"] for item in citation_rows
        ]) if citation_rows else None,
        "evidence_binding_rate": _mean([
            row["score"]["binding_complete"] for row in binding_rows
        ]) if binding_rows else None,
        "binding_violation_counts": dict(sorted(violation_counts.items())),
        "unsupported_refusal_rate": count_rate(
            unsupported, "status", "refused"),
        "unsupported_not_addressed_rate": count_rate(
            unsupported, "status", "not_addressed"),
        "unsupported_value_leak_rate": count_rate(
            unsupported, "status", "leaked_value"),
        "retrieval_chunk_start_page_rate": _mean([
            item["complete"] for item in retrieval_rows
        ]) if retrieval_rows else None,
    }


def run_questions(records: list[dict[str, Any]], *, generate: bool,
                  strategy: str, mode: str, max_retries: int
                  ) -> list[dict[str, Any]]:
    use_project_root_as_cwd()
    load_env()
    service = rag.build_service(
        strategy=strategy, mode=mode, use_rerank=(mode == "hybrid"))
    runtime = default_runtime(
        service=service,
        strategy=strategy,
        generate_answers=generate,
        use_llm_rewrite=False,
        max_retries=max_retries,
        enable_pre_retrieval_routing=True,
        strict_route_filter=True,
        route_candidate_limit=50,
        use_llm_grader=False,
        interrupt_on_low_confidence=False,
    )
    graph = build_graph(runtime)
    offsets = rag.load_page_offsets(strategy)
    rows = []
    for record in records:
        started = time.time()
        try:
            result = ask(
                graph, record["question"],
                thread_id="heldout-{}".format(record["id"]))
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "id": record["id"],
                "scenario": record["scenario"],
                "expected_behavior": record["expected_behavior"],
                "error": "{}: {}".format(type(exc).__name__, str(exc)[:300]),
                "seconds": round(time.time() - started, 2),
            })
            continue
        evidence = result.get("retrieved_chunks") or []
        retrieved = [rag.describe_hit(item, offsets) for item in evidence]
        answer = result.get("answer") or ""
        citations = result.get("citations") or []
        context_chunks = result.get("context_chunks") or evidence
        score = score_answer(
            record, answer, citations, context_chunks) if generate else None
        rows.append({
            "id": record["id"],
            "scenario": record["scenario"],
            "expected_behavior": record["expected_behavior"],
            "answerability_status": record["answerability_status"],
            "question": record["question"],
            "answer": answer,
            "status": result.get("status"),
            "needs_human_review": result.get("needs_human_review"),
            "retry_count": result.get("retry_count", 0),
            "retrieved": retrieved,
            "citations": citations,
            "context_chunk_ids": [
                item.get("chunk_id") for item in context_chunks
                if item.get("chunk_id")
            ],
            "answer_audit": result.get("answer_audit") or {},
            "retrieval_pages": retrieval_page_results(record, retrieved),
            "score": score,
            "seconds": round(time.time() - started, 2),
        })
    return rows


def rescore_rows(records: list[dict[str, Any]],
                 rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunk_index = load_chunk_index()
    by_id = {record["id"]: record for record in records}
    rescored = []
    for row in rows:
        record = by_id.get(row.get("id"))
        if record is None or row.get("error"):
            rescored.append(row)
            continue
        updated = dict(row)
        chunk_ids = list(dict.fromkeys(
            list(row.get("context_chunk_ids") or [])
            + [item.get("chunk_id") for item in row.get("citations") or []]
            + [item.get("chunk_id") for item in row.get("retrieved") or []]
        ))
        chunks = [
            chunk_index[chunk_id] for chunk_id in chunk_ids
            if chunk_id in chunk_index
        ]
        updated["score"] = score_answer(
            record, row.get("answer") or "", row.get("citations") or [],
            chunks)
        rescored.append(updated)
    return rescored


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the held-out set with mixed/partial refusal scoring")
    parser.add_argument("--generate", action="store_true",
                        help="call the generation chain; otherwise retrieval only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strategy", default="recursive")
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--rescore-existing", action="store_true",
        help="rescore answers in --output without calling retrieval or generation")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        records = records[:args.limit]
    started = time.time()
    output = Path(args.output)
    existing = None
    if args.rescore_existing:
        existing = json.loads(output.read_text(encoding="utf-8"))
        rows = rescore_rows(records, existing.get("rows") or [])
    else:
        rows = run_questions(
            records, generate=args.generate, strategy=args.strategy,
            mode=args.mode, max_retries=args.max_retries)
    dataset_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    summary = summarize(rows) if (args.generate or args.rescore_existing) else {
        "scored_questions": 0,
        "retrieval_chunk_start_page_rate": _mean([
            row["retrieval_pages"]["complete"]
            for row in rows if row.get("retrieval_pages")
        ]),
    }
    now = datetime.now(timezone.utc).isoformat()
    config = {
        "generate": args.generate,
        "strategy": args.strategy,
        "mode": args.mode,
        "max_retries": args.max_retries,
        "limit": args.limit,
        "rescored_existing": args.rescore_existing,
    }
    if existing is not None:
        config = dict(existing.get("config") or {}, rescored_existing=True)
        if any(row.get("answer") for row in rows):
            config["generate"] = True
    payload = {
        "generated_at": (
            existing.get("generated_at") if existing is not None else now
        ),
        "rescored_at": now if existing is not None else None,
        "dataset": str(DATASET),
        "dataset_candidate_sha256": (
            (existing or {}).get("dataset_candidate_sha256")
            or dataset_manifest.get("candidate_dataset_sha256")
        ),
        "dataset_frozen_at": (
            (existing or {}).get("dataset_frozen_at")
            or dataset_manifest.get("frozen_at")
        ),
        "dataset_sha256": dataset_manifest["sha256"],
        "config": config,
        "elapsed_seconds": (
            existing.get("elapsed_seconds") if existing is not None
            else round(time.time() - started, 2)
        ),
        "rescore_elapsed_seconds": (
            round(time.time() - started, 2) if existing is not None else None
        ),
        "metrics": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "metrics": summary,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
