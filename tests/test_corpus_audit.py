"""Corpus audit gate: coverage, file integrity and PDFium readability."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audit_annual_reports as audit  # noqa: E402


def write_config(path: Path, companies: list[tuple[str, str]]) -> None:
    path.write_text(json.dumps({
        "targets": [
            {
                "company": company,
                "ts_code": ts_code,
                "stock": "",
                "searchkey": company,
                "enabled": True,
            }
            for company, ts_code in companies
        ]
    }, ensure_ascii=False), encoding="utf-8")


def write_fake_corpus(tmp_path: Path, companies: list[tuple[str, str]],
                      year: int = 2025) -> tuple[Path, Path]:
    config = tmp_path / "targets.json"
    write_config(config, companies)
    documents = []
    for index, (company, ts_code) in enumerate(companies):
        pdf_path = tmp_path / "{}_{}.pdf".format(ts_code, year)
        pdf_path.write_bytes(
            b"%PDF-1.7\n" + bytes([65 + index]) * 2048
        )
        documents.append({
            "company": company,
            "ts_code": ts_code,
            "year": year,
            "title": "{} {}年年度报告".format(company, year),
            "file": str(pdf_path),
            "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        })
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"documents": documents}, ensure_ascii=False),
                        encoding="utf-8")
    return config, metadata


def fake_extractor(company: str):
    def extract(_path: Path):
        return (
            "{} {}年年度报告 营业收入 净利润 总资产".format(company, 2025),
            3,
        )
    return extract


def test_complete_small_corpus_passes(tmp_path):
    companies = [("公司甲", "000001"), ("公司乙", "000002")]
    config, metadata = write_fake_corpus(tmp_path, companies)
    contents = {
        "000001": fake_extractor("公司甲"),
        "000002": fake_extractor("公司乙"),
    }

    report = audit.audit_corpus(
        config,
        metadata,
        years=[2025],
        min_chars=10,
        expected_companies=2,
        extractor=lambda path: contents[path.stem.split("_")[0]](path),
    )

    assert report["summary"]["passed"] is True
    assert report["summary"]["pdf_openable_count"] == 2
    assert report["summary"]["total_extracted_chars"] > 0
    assert all(report["criteria"].values())


def test_missing_company_year_fails_gate(tmp_path):
    companies = [("公司甲", "000001"), ("公司乙", "000002")]
    config, metadata = write_fake_corpus(tmp_path, companies)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["documents"] = payload["documents"][:1]
    metadata.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = audit.audit_corpus(
        config,
        metadata,
        years=[2025],
        min_chars=10,
        expected_companies=2,
        extractor=fake_extractor("公司甲"),
    )

    assert report["summary"]["passed"] is False
    assert report["criteria"]["expected_document_count"] is False
    assert report["criteria"]["full_year_coverage"] is False
    assert report["summary"]["missing_company_years"] == ["000002:2025"]


def test_duplicate_code_year_fails_metadata_gate(tmp_path):
    companies = [("公司甲", "000001")]
    config, metadata = write_fake_corpus(tmp_path, companies)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["documents"].append(dict(payload["documents"][0]))
    metadata.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = audit.audit_corpus(
        config,
        metadata,
        years=[2025],
        min_chars=10,
        expected_companies=1,
        extractor=fake_extractor("公司甲"),
    )

    assert report["criteria"]["unique_ts_code_year"] is False
    assert report["summary"]["duplicate_keys"] == ["000001:2025"]
    assert report["summary"]["passed"] is False


def test_unreadable_pdf_is_recorded_not_silently_skipped(tmp_path):
    companies = [("公司甲", "000001")]
    config, metadata = write_fake_corpus(tmp_path, companies)

    def broken_extractor(_path: Path):
        raise RuntimeError("corrupt stream")

    report = audit.audit_corpus(
        config,
        metadata,
        years=[2025],
        min_chars=10,
        expected_companies=1,
        extractor=broken_extractor,
    )

    document = report["documents"][0]
    assert document["checks"]["pdf_openable"] is False
    assert document["checks"]["text_extractable"] is False
    assert "corrupt stream" in document["error"]
    assert report["summary"]["passed"] is False


def test_title_year_mismatch_is_a_hard_failure(tmp_path):
    companies = [("公司甲", "000001")]
    config, metadata = write_fake_corpus(tmp_path, companies)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["documents"][0]["title"] = "公司甲2024年年度报告"
    metadata.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = audit.audit_corpus(
        config,
        metadata,
        years=[2025],
        min_chars=10,
        expected_companies=1,
        extractor=fake_extractor("公司甲"),
    )

    assert report["documents"][0]["checks"]["title_year_match"] is False
    assert report["criteria"]["title_year_consistent"] is False
    assert report["summary"]["passed"] is False
