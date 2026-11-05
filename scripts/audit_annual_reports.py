"""Audit the annual-report corpus before rebuilding the retrieval index.

The audit is intentionally independent from the retrieval stack. It validates
metadata, file integrity, PDFium readability, text volume and source markers,
then writes a machine-readable report that can gate the ingest step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TARGETS = PROJECT_ROOT / "configs" / "cninfo_targets.json"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "docs" / "annual_reports" / "metadata.json"
DEFAULT_REPORT = PROJECT_ROOT / "evals" / "reports" / "corpus_audit.json"
DEFAULT_YEARS = [2023, 2024, 2025]
MIN_EXTRACTED_CHARS = 20_000
MIN_PDF_BYTES = 1_024
FINANCIAL_TERMS = (
    "营业收入", "营业总收入", "净利润", "归属于上市公司股东的净利润",
    "资产负债", "总资产", "现金流量", "经营活动产生的现金流量净额",
    "基本每股收益", "加权平均净资产收益率", "不良贷款率", "营业利润",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_targets(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_targets = payload.get("targets") if isinstance(payload, dict) else payload
    if not isinstance(raw_targets, list):
        raise ValueError("target config must be a list or contain a 'targets' list")
    targets: list[dict] = []
    for raw in raw_targets:
        if not isinstance(raw, dict) or raw.get("enabled") is False:
            continue
        company = str(raw.get("company") or "").strip()
        ts_code = str(raw.get("ts_code") or "").strip()
        if company and re.fullmatch(r"\d{6}", ts_code):
            targets.append({"company": company, "ts_code": ts_code})
    return targets


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def company_aliases(company: str) -> list[str]:
    aliases = {company, re.sub(r"[A-Za-z]$", "", company)}
    return sorted(alias for alias in aliases if alias)


def resolve_pdf_path(file_value: str, metadata_path: Path) -> Path:
    path = Path(file_value)
    if path.is_absolute():
        return path
    candidates = [PROJECT_ROOT / path, metadata_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def extract_with_pdfium(path: Path) -> tuple[str, int]:
    from rag_adapter.pdfium_loader import PdfiumLoader

    extraction = PdfiumLoader().extract(path)
    return extraction.content, len(extraction.pages)


def inspect_document(entry: dict, metadata_path: Path,
                     min_chars: int,
                     extractor: Callable[[Path], tuple[str, int]]
                     = extract_with_pdfium) -> dict:
    pdf_path = resolve_pdf_path(str(entry.get("file") or ""), metadata_path)
    checks = {
        "file_exists": pdf_path.is_file(),
        "pdf_header": False,
        "size_ok": False,
        "sha_matches_metadata": False,
        "pdf_openable": False,
        "text_extractable": False,
        "char_count_ok": False,
        "annual_report_marker": False,
        "company_marker": False,
        "financial_marker": False,
        "title_year_match": False,
    }
    details: dict = {
        "ts_code": str(entry.get("ts_code") or ""),
        "company": str(entry.get("company") or ""),
        "year": entry.get("year"),
        "title": str(entry.get("title") or ""),
        "file": str(pdf_path),
        "byte_size": None,
        "sha256": None,
        "page_count": None,
        "char_count": None,
        "financial_term_hits": [],
        "error": None,
        "checks": checks,
    }

    title_years = [
        int(value) for value in re.findall(r"(20\d{2})\s*年", details["title"])
    ]
    checks["title_year_match"] = int(entry.get("year") or 0) in title_years
    if not checks["file_exists"]:
        details["error"] = "PDF file does not exist"
        details["passed"] = False
        return details

    try:
        details["byte_size"] = pdf_path.stat().st_size
        checks["size_ok"] = details["byte_size"] >= MIN_PDF_BYTES
        with pdf_path.open("rb") as handle:
            checks["pdf_header"] = handle.read(4) == b"%PDF"
        details["sha256"] = sha256_of(pdf_path)
        checks["sha_matches_metadata"] = (
            not entry.get("sha256")
            or str(entry["sha256"]).lower() == details["sha256"]
        )
    except OSError as exc:
        details["error"] = "file read failed: {}".format(exc)

    if checks["pdf_header"]:
        try:
            content, page_count = extractor(pdf_path)
            normalized = normalize_text(content)
            details["page_count"] = page_count
            details["char_count"] = len(content)
            checks["pdf_openable"] = True
            checks["text_extractable"] = bool(content.strip())
            checks["char_count_ok"] = len(content) >= min_chars
            checks["annual_report_marker"] = (
                "年度报告" in normalized or "年度報告" in normalized
            )
            checks["company_marker"] = any(
                normalize_text(alias) in normalized
                for alias in company_aliases(details["company"])
            )
            details["financial_term_hits"] = [
                term for term in FINANCIAL_TERMS if term in content
            ]
            checks["financial_marker"] = bool(details["financial_term_hits"])
        except Exception as exc:  # noqa: BLE001 - recorded per document
            details["error"] = "PDFium extraction failed: {}".format(exc)

    details["passed"] = all(checks.values())
    return details


def audit_corpus(targets_path: Path = DEFAULT_TARGETS,
                 metadata_path: Path = DEFAULT_METADATA,
                 years: list[int] | None = None,
                 min_chars: int = MIN_EXTRACTED_CHARS,
                 expected_companies: int = 30,
                 extractor: Callable[[Path], tuple[str, int]]
                 = extract_with_pdfium) -> dict:
    years = sorted(int(year) for year in (years or DEFAULT_YEARS))
    targets = load_targets(targets_path)
    registry = json.loads(metadata_path.read_text(encoding="utf-8"))
    documents = [
        item for item in (registry.get("documents") or [])
        if isinstance(item, dict)
    ]

    key_counts: dict[tuple[str, int], int] = defaultdict(int)
    for item in documents:
        try:
            key = (str(item.get("ts_code") or ""), int(item.get("year")))
        except (TypeError, ValueError):
            continue
        key_counts[key] += 1
    duplicate_keys = sorted(
        "{}:{}".format(code, year)
        for (code, year), count in key_counts.items()
        if count > 1
    )

    inspected = []
    sha_counts: dict[str, int] = defaultdict(int)
    for item in documents:
        result = inspect_document(item, metadata_path, min_chars, extractor)
        if result.get("sha256"):
            sha_counts[result["sha256"]] += 1
        inspected.append(result)
    for result in inspected:
        digest = result.get("sha256")
        result["checks"]["sha_unique"] = bool(
            digest and sha_counts.get(digest) == 1
        )
        result["passed"] = bool(result["passed"] and result["checks"]["sha_unique"])

    target_by_code = {item["ts_code"]: item for item in targets}
    coverage = {
        item["ts_code"]: {
            "company": item["company"],
            "years": {
                str(year): key_counts.get((item["ts_code"], year), 0)
                for year in years
            },
        }
        for item in targets
    }
    missing_companies = [
        item["company"] for item in targets
        if not any(key_counts.get((item["ts_code"], year), 0)
                   for year in years)
    ]
    missing_years = [
        "{}:{}".format(item["ts_code"], year)
        for item in targets
        for year in years
        if key_counts.get((item["ts_code"], year), 0) == 0
    ]

    failed_documents = [
        result for result in inspected
        if not result["passed"] or result["ts_code"] not in target_by_code
    ]
    expected_document_count = len(targets) * len(years)
    criteria = {
        "company_count": len(targets) == expected_companies,
        "expected_document_count": len(documents) == expected_document_count,
        "unique_ts_code_year": not duplicate_keys,
        "full_year_coverage": not missing_years,
        "all_pdfs_valid": all(
            item["checks"]["file_exists"]
            and item["checks"]["pdf_header"]
            and item["checks"]["size_ok"]
            for item in inspected
        ),
        "all_pdfs_openable": all(
            item["checks"]["pdf_openable"] for item in inspected
        ),
        "all_text_extractable": all(
            item["checks"]["text_extractable"] for item in inspected
        ),
        "char_count_threshold": all(
            item["checks"]["char_count_ok"] for item in inspected
        ),
        "all_keywords_present": all(
            item["checks"]["annual_report_marker"]
            and item["checks"]["company_marker"]
            and item["checks"]["financial_marker"]
            for item in inspected
        ),
        "title_year_consistent": all(
            item["checks"]["title_year_match"] for item in inspected
        ),
        "sha256_unique": all(
            item["checks"].get("sha_unique", False) for item in inspected
        ),
        "sha256_matches_metadata": all(
            item["checks"]["sha_matches_metadata"] for item in inspected
        ),
    }
    passed = bool(inspected) and all(criteria.values()) and not failed_documents
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets_config": str(targets_path),
        "metadata": str(metadata_path),
        "thresholds": {
            "expected_companies": expected_companies,
            "expected_years": years,
            "expected_documents": expected_document_count,
            "min_extracted_chars": min_chars,
            "min_pdf_bytes": MIN_PDF_BYTES,
        },
        "summary": {
            "passed": passed,
            "company_count": len(targets),
            "metadata_document_count": len(documents),
            "unique_document_count": len(key_counts),
            "duplicate_document_count": len(documents) - len(key_counts),
            "pdf_openable_count": sum(
                item["checks"]["pdf_openable"] for item in inspected
            ),
            "text_extractable_count": sum(
                item["checks"]["text_extractable"] for item in inspected
            ),
            "total_extracted_chars": sum(
                item.get("char_count") or 0 for item in inspected
            ),
            "failed_document_count": len(failed_documents),
            "missing_companies": missing_companies,
            "missing_company_years": missing_years,
            "duplicate_keys": duplicate_keys,
        },
        "criteria": criteria,
        "coverage": coverage,
        "documents": inspected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit annual-report PDF corpus")
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--min-chars", type=int, default=MIN_EXTRACTED_CHARS)
    parser.add_argument("--expected-companies", type=int, default=30)
    args = parser.parse_args()

    report = audit_corpus(
        targets_path=Path(args.targets),
        metadata_path=Path(args.metadata),
        years=args.years,
        min_chars=args.min_chars,
        expected_companies=args.expected_companies,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = report["summary"]
    print(
        "corpus audit: {status} | companies={company_count} "
        "documents={metadata_document_count} openable={pdf_openable_count} "
        "chars={total_extracted_chars:,}".format(
            status="PASS" if summary["passed"] else "FAIL",
            **summary,
        )
    )
    print("report -> {}".format(report_path))
    if not summary["passed"]:
        print("failed criteria:")
        for name, value in report["criteria"].items():
            if not value:
                print("  - {}".format(name))
        if summary["missing_company_years"]:
            print("missing company-years: {}".format(
                ", ".join(summary["missing_company_years"])
            ))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
