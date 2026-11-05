from __future__ import annotations

from pathlib import Path

from scripts.fetch_annual_reports import registered_document_available


def test_registered_document_requires_existing_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "reports" / "sample.pdf"

    missing = {"file": "reports/sample.pdf"}
    assert not registered_document_available(missing)

    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-smoke")

    assert registered_document_available(missing)


def test_registered_document_rejects_blank_path() -> None:
    assert not registered_document_available({"file": ""})
