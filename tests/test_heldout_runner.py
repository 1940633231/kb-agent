"""Behavior tests for held-out mixed/partial refusal scoring."""
from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import run_heldout_eval as runner  # noqa: E402
from rag_adapter import rag  # noqa: E402


def _records() -> dict[str, dict]:
    return {
        record["id"]: record
        for record in (
            json.loads(line)
            for line in runner.DATASET.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _citations(record: dict) -> list[dict]:
    return [
        {"chunk_id": item["chunk_id"], "document_id": item["doc"]}
        for item in record["source_pages"]
    ]


def test_mixed_answer_passes_only_when_unknown_part_is_explicitly_refused():
    record = _records()["heldout-mixed-001"]
    answer = (
        "贵州茅台2024年销售费用为5,639,300,059.49元[1]。"
        "华为2024年研发费用不在知识库中，无法回答。"
    )

    score = runner.score_answer(record, answer, _citations(record))

    assert score["classification"] == "partially_correct"
    assert score["unsupported_refused"] is True
    assert score["unsupported_not_addressed"] is False
    assert score["unsupported_leaked"] is False


def test_mixed_answer_is_incorrect_when_unknown_part_is_fabricated():
    record = _records()["heldout-mixed-001"]
    answer = (
        "贵州茅台2024年销售费用为5,639,300,059.49元[1]。"
        "华为2024年研发费用为100亿元。"
    )

    score = runner.score_answer(record, answer, _citations(record))

    assert score["classification"] == "incorrect"
    assert "unsupported_value_leak" in score["reasons"]
    assert score["unsupported"][0]["status"] == "leaked_value"
    assert "100亿元" in score["unsupported"][0]["leaked_numbers"]


def test_quoted_context_number_is_not_treated_as_unknown_entity_leak():
    record = _records()["heldout-mixed-002"]
    answer = (
        "美的集团2024年研发费用无法回答；上下文仅提到研发投入超过160亿元，"
        "不能作为研发费用。苹果公司2024年研发费用不在知识库中，无法回答。"
    )

    score = runner.score_answer(record, answer, _citations(record))

    assert score["unsupported_leaked"] is False
    assert score["unsupported_refused"] is True
    assert score["classification"] == "wrongly_refused"


def test_mixed_answer_is_not_full_when_unknown_part_is_silently_omitted():
    record = _records()["heldout-mixed-001"]
    answer = "贵州茅台2024年销售费用为5,639,300,059.49元[1]。"

    score = runner.score_answer(record, answer, _citations(record))

    assert score["classification"] == "partially_correct"
    assert "unsupported_not_addressed" in score["reasons"]
    assert score["unsupported"][0]["status"] == "not_addressed"


def test_unknown_entity_question_is_correctly_refused():
    record = _records()["heldout-refuse-001"]
    answer = "现有检索结果中没有覆盖该问题的证据，因此无法回答。"

    score = runner.score_answer(record, answer, [])

    assert score["classification"] == "correctly_refused"
    assert score["unsupported_refused"] is True
    assert score["unsupported"][0]["refused_generically"] is True


def test_multi_value_answer_requires_all_values_and_expected_chunks():
    record = _records()["heldout-multiyear-001"]
    answer = (
        "美的集团2023年研发费用为14,583,311千元[1]；"
        "2024年研发费用为16,232,771千元[2]。"
    )

    score = runner.score_answer(record, answer, _citations(record))

    assert score["classification"] == "fully_correct"
    assert score["citations"]["complete"] is True


def test_value_presence_without_four_tuple_binding_is_not_fully_correct():
    record = _records()["heldout-conflict-003"]
    source = record["source_pages"][0]
    chunk = dict(runner.load_chunk_index()[source["chunk_id"]])
    chunk["metadata"] = dict(chunk["metadata"], company="贵州茅台")
    answer = (
        "海康威视2024年归母净利润为11,977,327,023.54元[1]。"
    )

    score = runner.score_answer(
        record, answer,
        [{"number": 1, "chunk_id": source["chunk_id"]}],
        [chunk],
    )

    assert score["classification"] == "partially_correct"
    assert score["binding_complete"] is False
    assert "evidence_binding_failed" in score["reasons"]
    assert score["binding_violations"]


def test_partial_answer_refuses_external_market_value_by_date():
    record = _records()["heldout-partial-001"]
    answer = (
        "美的集团2024年销售费用为38,753,649千元[1]。"
        "截至2025年6月30日，收盘价无法提供。"
    )

    score = runner.score_answer(record, answer, _citations(record))

    assert score["classification"] == "partially_correct"
    assert score["unsupported_refused"] is True
    assert score["unsupported"][0]["status"] == "refused"


def test_partial_answer_detects_external_market_value_leak_by_date():
    record = _records()["heldout-partial-001"]
    answer = (
        "美的集团2024年销售费用为38,753,649千元[1]。"
        "截至2025年6月30日，收盘价为100元。"
    )

    score = runner.score_answer(record, answer, _citations(record))

    assert score["classification"] == "incorrect"
    assert score["unsupported"][0]["status"] == "leaked_value"
    assert "100元" in score["unsupported"][0]["leaked_numbers"]


def test_year_month_is_not_a_market_price_leak():
    record = _records()["heldout-partial-005"]
    answer = (
        "中国神华2024年研发费用为2,727百万元[1]。"
        "2025年7月秦皇岛动力煤现货价：现有信息无法回答，上下文未提供该价格。"
    )

    score = runner.score_answer(record, answer, _citations(record))

    assert score["classification"] == "partially_correct"
    assert score["unsupported"][0]["status"] == "refused"
    assert score["unsupported"][0]["leaked_numbers"] == []


def test_retrieval_page_matching_uses_chunk_start_page_not_evidence_page():
    record = _records()["heldout-mixed-002"]
    source = record["source_pages"][0]
    assert source["chunk_start_page"] != source["evidence_page"]

    matched = runner.retrieval_page_results(record, [{
        "document_id": source["doc"],
        "chunk_start_page": source["chunk_start_page"],
    }])
    mismatched = runner.retrieval_page_results(record, [{
        "document_id": source["doc"],
        "chunk_start_page": source["evidence_page"],
    }])

    assert matched["complete"] is True
    assert mismatched["complete"] is False


def test_describe_hit_exposes_chunk_start_page_explicitly():
    record = _records()["heldout-mixed-002"]
    source = record["source_pages"][0]
    raw = json.loads(
        runner.PROJECT_ROOT.joinpath(
            "data", "index", "recursive", "metadata.json"
        ).read_text(encoding="utf-8")
    )
    chunk = next(
        item for item in raw.values()
        if item["chunk_id"] == source["chunk_id"]
    )
    offsets = rag.load_page_offsets("recursive")
    described = rag.describe_hit({
        "document_id": chunk["document_id"],
        "chunk_id": chunk["chunk_id"],
        "start_offset": chunk["start_offset"],
        "content": chunk["content"],
        "metadata": chunk["metadata"],
    }, offsets)

    assert described["chunk_start_page"] == source["chunk_start_page"]
    assert described["page"] == described["chunk_start_page"]
