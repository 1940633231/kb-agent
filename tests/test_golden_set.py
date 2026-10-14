"""golden set 的结构校验：格式必须先可解析、可判定，才能当质量闸门。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_adapter.settings import GOLDEN_SET_PATH, REPORTS_REGISTRY  # noqa: E402

QUESTION_TYPES = {"fact", "table", "cross_section", "cross_document", "calc", "refuse"}
BEHAVIORS = {"answer", "refuse", "human_review"}


def load_items():
    return [json.loads(line) for line in
            GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def known_documents() -> set:
    registry = json.loads(REPORTS_REGISTRY.read_text(encoding="utf-8"))
    return {"{}-{}".format(d["ts_code"], d["year"]) for d in registry["documents"]}


@pytest.fixture(scope="module")
def items():
    return load_items()


def test_at_least_twenty_questions(items):
    assert len(items) >= 20, "P0 最小集应为 20 题"


def test_ids_are_unique(items):
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids))


def test_question_type_allocation(items):
    counts: dict = {}
    for item in items:
        counts[item["question_type"]] = counts.get(item["question_type"], 0) + 1
    assert set(counts) <= QUESTION_TYPES
    for required in ("fact", "table", "cross_section", "cross_document", "calc", "refuse"):
        assert counts.get(required, 0) > 0, "缺少 {} 题型".format(required)


def test_behavior_and_docs_are_consistent(items):
    for item in items:
        assert item["expected_behavior"] in BEHAVIORS
        if item["expected_behavior"] == "refuse":
            assert item["expected_docs"] == []
            assert item["expected_values"] == []
        else:
            assert item["expected_docs"], "{} 缺期望文档".format(item["id"])
            for doc in item["expected_docs"]:
                assert set(doc) >= {"company", "year", "ts_code"}


def test_numeric_answers_carry_units(items):
    for item in items:
        for value in item["expected_values"]:
            assert value["unit"], "{} 缺单位".format(item["id"])
            assert isinstance(value["value"], (int, float))
            assert value["display"]


def test_calc_questions_declare_evidence_literals(items):
    """计算题的期望值是算出来的，必须显式声明原文里真实存在的字面量。"""
    for item in items:
        literals = item.get("evidence_literals")
        if literals is not None:
            assert literals, "{} 的 evidence_literals 不能为空".format(item["id"])
            assert all(isinstance(text, str) and text.strip() for text in literals)
        if item["question_type"] == "calc":
            assert literals, "{} 是计算题，必须给出 evidence_literals".format(item["id"])


def test_source_pages_point_to_fetched_reports(items):
    documents = known_documents()
    for item in items:
        for source in item["source_pages"]:
            assert source["doc"] in documents, "{} 引用了未抓取的文档".format(item["id"])
            assert int(source["page"]) >= 1
        if item["expected_behavior"] == "answer":
            assert item["source_pages"], "{} 缺原文出处".format(item["id"])
