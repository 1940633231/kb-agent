from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

import pytest

from scripts.fetch_annual_reports import load_targets, registered_document_available

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import fetch_annual_reports as fetch  # noqa: E402


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


def test_eval_target_config_contains_p0_companies() -> None:
    targets = load_targets(Path("configs/eval_cninfo_targets.json"))

    assert {target["ts_code"] for target in targets} == {
        "600519",
        "300750",
        "600036",
    }


def test_repository_target_config_has_30_enabled_companies():
    targets = fetch.load_targets(PROJECT_ROOT / "configs" / "cninfo_targets.json")

    assert len(targets) == 30
    assert len({target["ts_code"] for target in targets}) == 30
    assert all(target["enabled"] for target in targets)


def test_disabled_replacement_candidates_are_opt_in():
    all_targets = fetch.load_targets(
        PROJECT_ROOT / "configs" / "cninfo_targets.json",
        include_disabled=True,
    )
    enabled_targets = fetch.load_targets(
        PROJECT_ROOT / "configs" / "cninfo_targets.json"
    )

    assert len(all_targets) == 35
    assert len(enabled_targets) == 30
    assert "中国中免" in {target["company"] for target in all_targets}
    assert "中国中免" not in {target["company"] for target in enabled_targets}


def test_company_filter_keeps_cli_compatibility():
    targets = fetch.load_targets(
        PROJECT_ROOT / "configs" / "cninfo_targets.json",
        companies=["宁德时代"],
    )

    assert [target["ts_code"] for target in targets] == ["300750"]


def test_blank_query_fields_fall_back_to_company_name(tmp_path):
    config = tmp_path / "targets.json"
    config.write_text(json.dumps([{
        "company": "示例公司",
        "ts_code": "000001",
        "stock": "",
        "searchkey": "",
        "enabled": True,
    }], ensure_ascii=False), encoding="utf-8")

    targets = fetch.load_targets(config)

    assert targets[0]["searchkey"] == "示例公司"


def test_query_uses_searchkey_fallback(monkeypatch):
    captured = {}

    def fake_request(url, data=None, retries=4, timeout=30):
        captured["form"] = urllib.parse.parse_qs(
            data.decode("utf-8"), keep_blank_values=True
        )
        return b'{"announcements": [], "totalAnnouncement": 0}'

    monkeypatch.setattr(fetch, "_request", fake_request)
    target = {
        "company": "示例公司",
        "ts_code": "000001",
        "stock": "",
        "searchkey": "",
    }

    rows = fetch.query_announcements(target, [2025], max_pages=3)

    assert rows == []
    assert captured["form"]["stock"] == [""]
    assert captured["form"]["searchkey"] == ["示例公司"]


def test_metadata_deduplication_keeps_one_record_per_code_year():
    documents = [
        {"ts_code": "000001", "year": 2025, "title": "old"},
        {"ts_code": "000001", "year": 2025, "title": "new"},
        {"ts_code": "000002", "year": 2024, "title": "other"},
    ]

    result = fetch.dedupe_documents(documents)

    assert [item["title"] for item in result] == ["new", "other"]


def test_pick_annual_report_rejects_summary_and_wrong_year():
    rows = [
        {"announcementTitle": "示例公司2025年年度报告摘要"},
        {"announcementTitle": "示例公司2024年年度报告"},
        {"announcementTitle": "示例公司<em>2025</em>年年度报告"},
    ]

    picked = fetch.pick_annual_report(rows, 2025)

    assert picked is rows[2]


def test_invalid_config_raises_clear_error(tmp_path):
    config = tmp_path / "bad.json"
    config.write_text(json.dumps([{
        "company": "示例公司",
        "ts_code": "1",
    }], ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="six-digit"):
        fetch.load_targets(config)
