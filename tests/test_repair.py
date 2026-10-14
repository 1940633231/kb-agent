"""生成侧落地复核与修复的测试（假生成器脚本化回答，不调 LLM）。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# kb-agent 同进程 import production-rag（app 包）；CI 里没有它时整模块跳过
pytest.importorskip("app.core.config", reason="需要 production-rag（app 包）")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.graph import ask, build_graph, default_runtime  # noqa: E402
from agent.grounding import (detect_ambiguous_metric,  # noqa: E402
                             has_statement_label, verify_grounding)
from fakes import FakeService, make_hit  # noqa: E402


def hit(document_id: str, content: str) -> Dict:
    return make_hit(document_id, content)


class ScriptedGenerator:
    """按调用顺序返回脚本化的回答，用来驱动"初稿错误 → 修复"的路径。"""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls: List[str] = []

    def generate(self, messages) -> str:
        self.calls.append(messages[-1]["content"])
        if len(self.answers) > 1:
            return self.answers.pop(0)
        return self.answers[0]

    def stream_generate(self, messages):
        yield self.generate(messages)


def build(resolver, *answers: str, **runtime_kwargs):
    generator = ScriptedGenerator(*answers)
    service = FakeService(resolver, generator=generator)
    # 修复环节默认关闭（离线分析显示触发过宽），这些用例显式打开来验证机制本身
    runtime_kwargs.setdefault("enable_answer_repair", True)
    runtime = default_runtime(service=service, use_llm_rewrite=False,
                              **runtime_kwargs)
    return build_graph(runtime), generator


QUESTION = "贵州茅台2023年的营业收入是多少元？"


def test_grounded_answer_needs_no_repair():
    content = "主要会计数据 营业收入 147,693,604,994.14 归属于上市公司股东的净利润"
    resolver = lambda q: [hit("600519-2023", content)]
    graph, generator = build(resolver, "贵州茅台2023年营业收入为 147,693,604,994.14 元 [1]。")
    result = ask(graph, QUESTION, thread_id="rep-ok")

    assert result["status"] == "answered"
    assert len(generator.calls) == 1, "落地通过时不该触发修复"
    assert result["grounding"]["verified"] is True


def test_ungrounded_number_triggers_repair_and_succeeds():
    """实测错例：初稿用了母公司口径的数，修复后改成片段里逐字出现的值。"""
    content = "主要会计数据 营业收入 147,693,604,994.14"
    resolver = lambda q: [hit("600519-2023", content)]
    graph, generator = build(
        resolver,
        "贵州茅台2023年营业收入为 81,415,570,788.59 元 [1]。",
        "贵州茅台2023年营业收入为 147,693,604,994.14 元（主要会计数据）[1]。")
    result = ask(graph, QUESTION, thread_id="rep-fix")

    assert result["status"] == "answered_repaired"
    assert len(generator.calls) == 2, "应触发一次修复"
    assert "修复" in generator.calls[1] or "校订" in generator.calls[1]
    assert "147,693,604,994.14" in result["answer"]
    assert result["grounding"]["verified"] is True
    assert result["repair_history"][0]["fixed"] is True


def test_unfixable_answer_gets_caveat_and_review_flag():
    content = "主要会计数据 营业收入 147,693,604,994.14"
    resolver = lambda q: [hit("600519-2023", content)]
    bogus = "贵州茅台2023年营业收入为 999,888,777,666.55 元 [1]。"
    graph, _ = build(resolver, bogus, bogus)
    result = ask(graph, QUESTION, thread_id="rep-bad")

    assert result["status"] == "answered_unverified"
    assert result["needs_human_review"] is True
    assert "无法在引用片段中逐字定位" in result["answer"]
    assert result["repair_history"][0]["fixed"] is False


def test_derived_value_is_accepted():
    """由片段内若干数字相加得到的值应当算落地（ar-fact-004 的四季度之和）。"""
    content = ("分季度主要财务数据 归属于上市公司股东的净利润 "
               "26,847,474,238.76 18,555,488,059.34 19,223,784,414.08 17,693,320,389.50")
    chunk = hit("600519-2025", content)
    citation = {"number": 1, "chunk_id": chunk["chunk_id"],
                "document_id": "600519-2025"}
    outcome = verify_grounding(
        "2025年归母净利润为四季度之和 82,320,067,101.68 元 [1]",
        [citation], [chunk],
        "贵州茅台2025年归属于上市公司股东的净利润是多少元？")
    assert outcome["verified"] is True
    assert outcome["derived"] == ["82,320,067,101.68"] or "82,320,067,101.68" in outcome["grounded"]


def test_ambiguity_detected_within_one_document_only():
    """同一份文档里同一指标两个口径 → 歧义；跨文档各自一个值 → 不算歧义。"""
    merged = "主要会计数据 营业收入 147,693,604,994.14"
    parent = "母公司利润表 营业收入 81,415,570,788.59"
    same_doc = [hit("600519-2023", merged), hit("600519-2023", parent)]
    flagged = detect_ambiguous_metric(QUESTION, same_doc)
    assert flagged["ambiguous"] is True
    assert "147,693,604,994.14" in flagged["values"]

    cross_doc = [hit("600519-2023", merged), hit("600519-2024", parent)]
    assert detect_ambiguous_metric(QUESTION, cross_doc)["ambiguous"] is False


def test_ambiguous_without_label_triggers_repair():
    merged = "主要会计数据 营业收入 147,693,604,994.14"
    parent = "母公司利润表 营业收入 81,415,570,788.59"
    resolver = lambda q: [hit("600519-2023", merged), hit("600519-2023", parent)]
    graph, generator = build(
        resolver,
        "营业收入为 81,415,570,788.59 元 [2]。",
        "营业收入为 147,693,604,994.14 元（主要会计数据）[1]。")
    result = ask(graph, QUESTION, thread_id="rep-ambig")

    assert len(generator.calls) == 2
    assert result["ambiguity"]["ambiguous"] is True
    assert has_statement_label(result["answer"]) is True
