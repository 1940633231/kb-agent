"""P3 scaled golden-set structure and evidence-contract tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATASET = PROJECT_ROOT / "evals" / "golden_set_scaled_120.jsonl"
MANIFEST = PROJECT_ROOT / "evals" / "golden_set_scaled_120.manifest.json"
REGISTRY = PROJECT_ROOT / "data" / "docs" / "annual_reports" / "metadata.json"


def load_records() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_scaled_set_has_120_unique_records_and_expected_mix():
    records = load_records()
    assert len(records) == 120
    assert len({item["id"] for item in records}) == 120
    counts = {}
    for item in records:
        counts[item["dataset_source"]] = counts.get(
            item["dataset_source"], 0) + 1
    assert counts == {"company": 90, "cross": 15, "refusal": 15}


def test_manifest_matches_dataset_and_counts():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = load_records()
    assert manifest["counts"]["total"] == len(records)
    assert manifest["companies"] == 30
    assert manifest["documents"] == 90


def test_answer_records_carry_evidence_claims_routes_and_pages():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    known_documents = {
        "{}-{}".format(item["ts_code"], item["year"])
        for item in registry["documents"]
    }
    for item in load_records():
        if item["expected_behavior"] != "answer":
            continue
        assert item["required_evidence"]
        assert item["required_claims"]
        assert item["route_expectations"]
        assert item["evidence_literals"]
        assert item["source_pages"]
        obligation_ids = {
            entry["obligation_id"] for entry in item["required_evidence"]
        }
        route_obligations = {
            obligation_id
            for route in item["route_expectations"]
            for obligation_id in route["required_obligations"]
        }
        assert obligation_ids == route_obligations
        for source in item["source_pages"]:
            assert source["doc"] in known_documents
            assert int(source["page"]) >= 1


def test_refusal_records_have_no_valid_evidence_obligation():
    refusals = [
        item for item in load_records()
        if item["expected_behavior"] == "refuse"
    ]
    assert len(refusals) == 15
    for item in refusals:
        assert item["required_evidence"] == []
        assert item["required_claims"] == []
        assert item["expected_docs"] == []
        assert item["expected_values"] == []


def test_each_company_has_three_company_questions():
    counts = {}
    for item in load_records():
        if item["dataset_source"] != "company":
            continue
        code = item["expected_docs"][0]["ts_code"]
        counts[code] = counts.get(code, 0) + 1

    assert len(counts) == 30
    assert set(counts.values()) == {3}
