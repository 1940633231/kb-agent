"""Regression checks for the held-out candidate builder and frozen metadata."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import build_heldout_40 as builder  # noqa: E402

LOCAL_INDEX_AVAILABLE = (
    builder.INDEX_METADATA.exists() and builder.PAGE_OFFSETS.exists()
)


def _records() -> list[dict]:
    return [
        json.loads(line)
        for line in builder.OUTPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _canonical_sha256(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def test_heldout_candidate_has_40_unique_questions_and_eight_scenarios():
    records = _records()

    assert len(records) == 40
    assert len({record["id"] for record in records}) == 40
    assert len({record["question"] for record in records}) == 40
    assert Counter(record["scenario"] for record in records) == {
        category: 5 for category in builder.CATEGORIES
    }
    assert sum(
        record["review"]["priority"] == "high" for record in records
    ) == 18


@pytest.mark.skipif(
    not LOCAL_INDEX_AVAILABLE,
    reason="本地检索索引未构建，跳过需要 chunk/page metadata 的源文审计",
)
def test_heldout_sources_still_pass_page_and_development_overlap_audit():
    raw = json.loads(builder.INDEX_METADATA.read_text(encoding="utf-8"))
    chunks = {item["chunk_id"]: item for item in raw.values()}
    offsets = json.loads(builder.PAGE_OFFSETS.read_text(encoding="utf-8"))
    records = [
        builder.build_record(spec, chunks, offsets)
        for spec in builder.SPECS
    ]

    builder.validate(records, chunks, offsets)
    assert all(
        record["review"]["human_confirmation"] == "confirmed"
        for record in _records()
    )


@pytest.mark.skipif(
    not LOCAL_INDEX_AVAILABLE,
    reason="本地检索索引未构建，跳过需要 chunk/page metadata 的页码校验",
)
def test_heldout_source_pages_point_at_the_evidence_not_the_chunk_start():
    raw = json.loads(builder.INDEX_METADATA.read_text(encoding="utf-8"))
    chunks = {item["chunk_id"]: item for item in raw.values()}
    offsets = json.loads(builder.PAGE_OFFSETS.read_text(encoding="utf-8"))
    records = [
        builder.build_record(spec, chunks, offsets)
        for spec in builder.SPECS
    ]

    for record in records:
        for value, source in zip(record["expected_values"],
                                 record["source_pages"]):
            chunk = chunks[source["chunk_id"]]
            content = chunk["content"]
            literal_at = builder.literal_index(
                content, source.get("literal") or str(value["value"]))
            assert literal_at is not None
            expected_page = builder.page_for_offset(
                source["doc"],
                int(chunk["start_offset"]) + literal_at,
                offsets)
            assert source["evidence_page"] == expected_page
            assert source["evidence_page"] >= source["chunk_start_page"]
        for conflict in record["conflicting_evidence"]:
            assert conflict["kind"] in {
                "equivalent_unit", "rounded_restatement",
                "different_caliber",
            }
    # The evidence page must differ from the chunk start page somewhere,
    # otherwise this dataset silently regresses to the old citation bug.
    assert any(
        source["evidence_page"] != source["chunk_start_page"]
        for record in records
        for source in record["source_pages"]
    )


def test_heldout_manifest_matches_frozen_dataset_and_baseline_hashes():
    manifest = json.loads(builder.MANIFEST.read_text(encoding="utf-8"))
    # The frozen hash is defined over canonical LF bytes so Windows checkouts
    # and Linux CI produce the same digest.
    digest = _canonical_sha256(builder.OUTPUT)
    baseline_digest = _canonical_sha256(builder.BASELINE)
    baseline = json.loads(builder.BASELINE.read_text(encoding="utf-8"))

    assert manifest["status"] == "frozen"
    assert manifest["version"] == "v1-frozen"
    assert manifest["sha256"] == digest
    assert manifest["counts"]["total"] == 40
    assert manifest["counts"]["source_audit_passed"] == 40
    assert manifest["counts"]["human_confirmation_pending"] == 0
    assert manifest["counts"]["human_confirmation_confirmed"] == 40
    assert manifest["baseline"]["sha256"] == baseline_digest
    assert baseline["dataset_sha256"] == digest
    assert baseline["dataset_candidate_sha256"] == manifest[
        "candidate_dataset_sha256"]


def test_review_sheet_shows_supported_and_unsupported_parts_together():
    records = _records()
    review = builder.review_markdown(records)

    assert "Behavior" in review
    assert "explicitly unsupported: 华为 2024 研发费用" in review
    assert "expected_values` are the parts that must be answered" in review
