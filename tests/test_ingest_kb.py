"""Annual-report loading for full and available-only index builds."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("app.core.config", reason="需要 production-rag（app 包）")

from rag_adapter import rag  # noqa: E402


def write_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\nfixture")


def registry_entry(pdf_path: Path, code: str, year: int) -> dict:
    return {
        "ts_code": code,
        "company": "公司{}".format(code),
        "year": year,
        "title": "{}年年度报告".format(year),
        "file": str(pdf_path),
        "pdf_url": "https://example.invalid/report.pdf",
        "sha256": "fixture",
    }


def write_registry(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps({"documents": entries}, ensure_ascii=False),
                    encoding="utf-8")


class FakePdfiumLoader:
    def __init__(self, metadata):
        self.metadata = metadata

    def extract(self, path: Path):
        content = "{} {}年年度报告".format(
            self.metadata["company"], self.metadata["year"])
        return SimpleNamespace(content=content, pages=[(1, 0, len(content))])


def test_available_only_skips_registered_pdfs_that_are_not_downloaded(
        tmp_path: Path, monkeypatch) -> None:
    present = tmp_path / "reports" / "000001_2025.pdf"
    missing = tmp_path / "reports" / "000002_2025.pdf"
    write_pdf(present)
    registry = tmp_path / "metadata.json"
    write_registry(registry, [
        registry_entry(present, "000001", 2025),
        registry_entry(missing, "000002", 2025),
    ])
    monkeypatch.setattr(rag, "PdfiumLoader", FakePdfiumLoader)

    documents, offsets = rag.load_annual_reports(
        registry, available_only=True)

    assert [item.document_id for item in documents] == ["000001-2025"]
    assert set(offsets) == {"000001-2025"}


def test_default_loading_still_fails_on_missing_pdf(tmp_path: Path) -> None:
    missing = tmp_path / "reports" / "000002_2025.pdf"
    registry = tmp_path / "metadata.json"
    write_registry(registry, [registry_entry(missing, "000002", 2025)])

    with pytest.raises(FileNotFoundError, match="000002_2025.pdf"):
        rag.load_annual_reports(registry)
