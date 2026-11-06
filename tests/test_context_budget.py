"""P2 unified budget tests using production-rag's real ContextManager."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.context.manager import ContextManager  # noqa: E402

from agent.context_budget import build_context_with_budget  # noqa: E402
from agent.evidence_contract import build_obligations  # noqa: E402
from fakes import FakePromptBuilder, make_hit  # noqa: E402


class CharCounter:
    def count(self, text: str) -> int:
        return len(text or "")


class HalfCounter:
    def count(self, text: str) -> int:
        return max(0, len(text or "") // 2)


def pipeline(max_tokens: int, counter=None):
    manager = ContextManager(
        token_counter=counter or CharCounter(),
        max_context_tokens=max_tokens,
        reserved_tokens=0,
    )
    return SimpleNamespace(
        context_manager=manager,
        prompt_builder=FakePromptBuilder(),
    )


def history(turns: int, chars: int = 120):
    items = []
    for index in range(turns):
        items.append({"role": "user", "content": "用户{}".format(index) + "x" * chars})
        items.append({"role": "assistant", "content": "回答{}".format(index) + "y" * chars})
    return items


def test_short_question_fits_and_reports_budget():
    hit = make_hit("600519-2023", "贵州茅台2023年营业收入 147,693,604,994.14")
    result = build_context_with_budget(
        pipeline(300),
        "贵州茅台2023年的营业收入是多少元？",
        [hit],
        history=[],
        obligations=build_obligations("贵州茅台2023年的营业收入是多少元？"),
        output_reserved_tokens=80,
    )

    assert result["chunks"]
    assert result["budget"]["budget_overflow"] is False
    assert result["budget"]["prompt_token_estimate"] <= 300


def test_five_long_history_turns_are_trimmed_to_recent_turns():
    hit = make_hit("600519-2023", "贵州茅台2023年营业收入 147,693,604,994.14")
    result = build_context_with_budget(
        pipeline(500),
        "贵州茅台2023年的营业收入是多少元？",
        [hit],
        history=history(5, 180),
        obligations=build_obligations("贵州茅台2023年的营业收入是多少元？"),
        output_reserved_tokens=80,
    )

    assert result["budget"]["skipped_history_turns"] > 0
    assert len(result["history"]) < 10
    assert result["budget"]["budget_overflow"] is False


def test_history_is_trimmed_before_critical_evidence():
    question = "贵州茅台2023年的营业收入是多少元？"
    critical = make_hit(
        "600519-2023", "贵州茅台2023年营业收入 147,693,604,994.14", score=0.1)
    noise = make_hit(
        "600519-2023", "无关键指标" + "z" * 120, score=0.99)
    result = build_context_with_budget(
        pipeline(420),
        question,
        [noise, critical],
        history=history(4, 180),
        obligations=build_obligations(question),
        output_reserved_tokens=80,
    )

    ids = {item["chunk_id"] for item in result["chunks"]}
    assert critical["chunk_id"] in ids
    assert result["budget"]["history_tokens"] >= 0


def test_multi_document_evidence_is_kept_when_budget_allows():
    question = "贵州茅台与招商银行2023年营业收入分别是多少元？"
    hits = [
        make_hit("600519-2023", "贵州茅台2023年营业收入 147,693,604,994.14", 0.8),
        make_hit("600036-2023", "招商银行2023年营业收入 339,123,000,000.00", 0.7),
    ]
    result = build_context_with_budget(
        pipeline(500),
        question,
        hits,
        history=[],
        obligations=build_obligations(question),
        output_reserved_tokens=80,
    )

    ids = {item["document_id"] for item in result["chunks"]}
    assert ids == {"600519-2023", "600036-2023"}
    assert result["budget"]["budget_overflow"] is False


def test_insufficient_output_reserve_is_reported_not_hidden():
    hit = make_hit("600519-2023", "贵州茅台2023年营业收入 147,693,604,994.14")
    result = build_context_with_budget(
        pipeline(120),
        "贵州茅台2023年的营业收入是多少元？",
        [hit],
        history=[],
        obligations=build_obligations("贵州茅台2023年的营业收入是多少元？"),
        output_reserved_tokens=100,
    )

    assert result["budget"]["budget_overflow"] is True
    assert result["budget"]["evidence_tokens"] >= 0


def test_tokenizer_backend_changes_reported_budget():
    hit = make_hit("600519-2023", "贵州茅台2023年营业收入 147,693,604,994.14")
    question = "贵州茅台2023年的营业收入是多少元？"
    char_result = build_context_with_budget(
        pipeline(400, CharCounter()), question, [hit],
        obligations=build_obligations(question),
        output_reserved_tokens=80,
    )
    half_result = build_context_with_budget(
        pipeline(400, HalfCounter()), question, [hit],
        obligations=build_obligations(question),
        output_reserved_tokens=80,
    )

    assert half_result["budget"]["query_tokens"] < char_result["budget"]["query_tokens"]
    assert half_result["budget"]["prompt_token_estimate"] <= 400


def test_prompt_components_share_one_budget():
    hit = make_hit("600519-2023", "贵州茅台2023年营业收入 147,693,604,994.14")
    question = "贵州茅台2023年的营业收入是多少元？"
    result = build_context_with_budget(
        pipeline(300),
        question,
        [hit],
        history=history(3, 80),
        obligations=build_obligations(question),
        output_reserved_tokens=90,
    )
    budget = result["budget"]

    assert budget["system_tokens"] > 0
    assert budget["query_tokens"] > 0
    assert budget["reserved_output_tokens"] == 90
    assert budget["prompt_token_estimate"] == (
        budget["fixed_prompt_tokens"]
        + budget["query_tokens"]
        + budget["history_tokens"]
        + budget["evidence_tokens"]
        + budget["reserved_output_tokens"]
    )
    assert budget["prompt_token_estimate"] <= 300
