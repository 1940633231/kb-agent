"""Audit the scaled golden set against indexed source chunks and page offsets."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "evals" / "golden_set_scaled_120.jsonl"
MANIFEST = PROJECT_ROOT / "evals" / "golden_set_scaled_120.manifest.json"
REGISTRY = PROJECT_ROOT / "data" / "docs" / "annual_reports" / "metadata.json"
INDEX_METADATA = PROJECT_ROOT / "data" / "index" / "recursive" / "metadata.json"
PAGE_OFFSETS = PROJECT_ROOT / "data" / "index" / "recursive" / "page_offsets.json"
REPORT = PROJECT_ROOT / "evals" / "reports" / "scaled_golden_set_audit.json"
SHORTLIST = PROJECT_ROOT / "evals" / "reports" / "scaled_golden_set_review_shortlist.json"
_PDF_PAGE_CACHE: dict[tuple[str, int], str] = {}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def page_at_offset(offsets: dict, document_id: str, offset: int) -> int | None:
    entry = offsets.get(document_id) or {}
    for page, start, end in entry.get("pages") or []:
        if int(start) <= offset < int(end):
            return int(page)
    return None


def pdf_page_text(file_path: str, page: int) -> str:
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    key = (str(path), page)
    if key in _PDF_PAGE_CACHE:
        return _PDF_PAGE_CACHE[key]
    pdf = pdfium.PdfDocument(str(path))
    try:
        text_page = pdf[page - 1].get_textpage()
        try:
            text = text_page.get_text_range()
        finally:
            text_page.close()
    finally:
        pdf.close()
    _PDF_PAGE_CACHE[key] = text
    return text


def risk_reason(code: str, message: str, severity: str = "warning") -> dict:
    return {"code": code, "message": message, "severity": severity}


def audit_record(item: dict, registry_docs: dict, chunks: dict,
                 page_offsets: dict) -> dict:
    checked: dict[str, bool] = {}
    risks: list[dict] = []
    if item["expected_behavior"] == "refuse":
        checked["no_required_evidence"] = item["required_evidence"] == []
        checked["no_expected_docs"] = item["expected_docs"] == []
        checked["no_expected_values"] = item["expected_values"] == []
        if not all(checked.values()):
            risks.append(risk_reason(
                "invalid_refusal", "refusal case carries evidence or values",
                "error"))
        return {
            "id": item["id"],
            "question": item["question"],
            "bucket": "fail" if risks else "normal",
            "risk_score": 100 if risks else 0,
            "checks": checked,
            "risks": risks,
            "source_pages": item["source_pages"],
        }

    expected_docs = {
        "{}-{}".format(doc["ts_code"], doc["year"]): doc
        for doc in item.get("expected_docs") or []
    }
    source_pages = item.get("source_pages") or []
    source_docs = {str(source.get("doc")) for source in source_pages}
    checked["source_docs_match_expected"] = source_docs == set(expected_docs)
    if not checked["source_docs_match_expected"]:
        risks.append(risk_reason(
            "source_doc_mismatch",
            "source pages do not match expected docs",
            "error"))

    extracted_page_texts: list[str] = []
    for source in source_pages:
        document_id = str(source.get("doc"))
        page = int(source.get("page") or 0)
        chunk_id = str(source.get("chunk_id") or "")
        document = registry_docs.get(document_id)
        checked["known_document_{}".format(document_id)] = document is not None
        if document is None:
            risks.append(risk_reason(
                "unknown_document", "unknown source document {}".format(
                    document_id), "error"))
            continue
        if document.get("file"):
            try:
                page_text = pdf_page_text(str(document["file"]), page)
                checked["pdf_page_exists_{}_{}".format(
                    document_id, page)] = bool(page_text.strip())
                extracted_page_texts.append(page_text)
            except Exception as exc:  # noqa: BLE001 - audit records the failure
                risks.append(risk_reason(
                    "pdf_page_unreadable",
                    "unable to read {} page {}: {}".format(
                        document_id, page, exc),
                    "error"))
        chunk = chunks.get(chunk_id)
        if chunk_id:
            checked["known_chunk_{}".format(chunk_id)] = chunk is not None
            if chunk is None:
                risks.append(risk_reason(
                    "unknown_chunk",
                    "missing source chunk {}".format(chunk_id), "error"))
                continue
            actual_page = page_at_offset(
                page_offsets, document_id,
                int(chunk.get("start_offset") or 0))
            checked["page_matches_{}".format(chunk_id)] = actual_page == page
            if actual_page != page:
                risks.append(risk_reason(
                    "page_mismatch",
                    "{} page {} does not match chunk page {}".format(
                        document_id, page, actual_page),
                    "error"))
    joined_content = "\n".join(
        [
            chunks.get(str(source.get("chunk_id")), {}).get("content") or ""
            for source in source_pages
        ] + extracted_page_texts
    )
    question = item["question"]
    keywords = [str(value) for value in item.get("expected_keywords") or []]
    checked["keywords_in_question"] = all(value in question for value in keywords)
    if not checked["keywords_in_question"]:
        risks.append(risk_reason(
            "keyword_not_in_question",
            "one or more expected keywords are absent from the question"))

    literals = [str(value) for value in item.get("evidence_literals") or []]
    checked["evidence_literals_present_in_source"] = bool(literals) and all(
        literal in joined_content for literal in literals
    )
    if not checked["evidence_literals_present_in_source"]:
        risks.append(risk_reason(
            "literal_not_in_source",
            "expected evidence literal is absent from the declared chunks",
            "error"))

    for value_entry in item.get("expected_values") or []:
        literal = str(value_entry.get("display") or "")
        numeric_literal = re.match(r"(-?[\d,]+(?:\.\d+)?)", literal)
        if numeric_literal:
            plain = numeric_literal.group(1).replace(",", "")
            try:
                expected = float(value_entry["value"])
                checked["value_parse_{}".format(value_entry.get("label"))] = (
                    math.isclose(float(plain), expected, rel_tol=1e-8,
                                 abs_tol=1e-6)
                )
            except (KeyError, TypeError, ValueError):
                checked["value_parse_{}".format(value_entry.get("label"))] = False
            if not checked["value_parse_{}".format(value_entry.get("label"))]:
                risks.append(risk_reason(
                    "value_display_mismatch",
                    "display value does not match numeric value: {}".format(
                        value_entry.get("label"))))

        unit = value_entry.get("unit")
        if unit and unit != "%" and unit not in joined_content:
            page_visible = False
            for source in source_pages:
                document_id = str(source.get("doc"))
                page = int(source.get("page") or 0)
                document = registry_docs.get(document_id) or {}
                try:
                    if document.get("file") and unit in pdf_page_text(
                            str(document["file"]), page):
                        page_visible = True
                except Exception:  # noqa: BLE001 - already reported above
                    pass
                for chunk in chunks.values():
                    if str(chunk.get("document_id")) != document_id:
                        continue
                    chunk_page = page_at_offset(
                        page_offsets, document_id,
                        int(chunk.get("start_offset") or 0))
                    if chunk_page == page and unit in (
                            chunk.get("content") or ""):
                        page_visible = True
                        break
                if page_visible:
                    break
            checked["unit_page_visible_{}".format(
                value_entry.get("label"))] = page_visible
            if not page_visible:
                risks.append(risk_reason(
                    "unit_not_visible",
                    "unit {} is not visible on the declared source page for {}"
                    .format(unit, value_entry.get("label"))))

    for keyword in keywords:
        keyword_positions = [
            match.start() for match in re.finditer(
                re.escape(keyword), joined_content)
        ]
        if not keyword_positions:
            continue
        literal_positions = []
        for literal in literals:
            literal_positions.extend(
                match.start() for match in re.finditer(
                    re.escape(literal), joined_content)
            )
        nearest = min(
            (abs(literal_position - keyword_position)
             for literal_position in literal_positions
             for keyword_position in keyword_positions),
            default=999999,
        )
        if nearest > 240:
            risks.append(risk_reason(
                "metric_literal_far_apart",
                "{} is more than 240 chars from its evidence literal".format(
                    keyword)))

    for doc in expected_docs.values():
        checked["question_has_company_{}".format(doc["ts_code"])] = (
            doc["company"] in question)
        checked["question_has_year_{}".format(doc["year"])] = (
            str(doc["year"]) in question)
        if not checked["question_has_company_{}".format(doc["ts_code"])]:
            risks.append(risk_reason(
                "company_not_in_question",
                "company {} is absent from question".format(doc["company"])))
        if not checked["question_has_year_{}".format(doc["year"])]:
            risks.append(risk_reason(
                "year_not_in_question",
                "year {} is absent from question".format(doc["year"])))

    errors = sum(1 for risk in risks if risk["severity"] == "error")
    warnings = len(risks) - errors
    score = min(100, errors * 50 + warnings * 10)
    bucket = "fail" if errors else "high_risk" if warnings >= 2 else (
        "review" if warnings else "normal")
    for risk in risks:
        checked[risk["code"]] = False
    return {
        "id": item["id"],
        "question": item["question"],
        "bucket": bucket,
        "risk_score": score,
        "checks": checked,
        "risks": risks,
        "source_pages": source_pages,
    }


def main() -> int:
    records = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry = load_json(REGISTRY)
    index_metadata = load_json(INDEX_METADATA)
    page_offsets = load_json(PAGE_OFFSETS)
    registry_docs = {
        "{}-{}".format(item["ts_code"], item["year"]): item
        for item in registry["documents"]
    }
    chunks = {
        str(item.get("chunk_id")): item
        for item in index_metadata.values()
        if item.get("chunk_id")
    }
    results = [
        audit_record(item, registry_docs, chunks, page_offsets)
        for item in records
    ]
    buckets = Counter(item["bucket"] for item in results)
    shortlist = sorted(
        (item for item in results if item["bucket"] != "normal"),
        key=lambda item: (-item["risk_score"], item["id"]),
    )[:30]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET),
        "manifest": load_json(MANIFEST),
        "summary": {
            "records": len(results),
            "normal": buckets.get("normal", 0),
            "review": buckets.get("review", 0),
            "high_risk": buckets.get("high_risk", 0),
            "fail": buckets.get("fail", 0),
            "manual_review_shortlist": len(shortlist),
        },
        "shortlist": shortlist,
        "records": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    SHORTLIST.write_text(
        json.dumps(shortlist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("report -> {}".format(REPORT))
    print("shortlist -> {}".format(SHORTLIST))
    return 0 if report["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
