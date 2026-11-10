"""Answer prompt contract for auditable numeric responses."""
from __future__ import annotations

import pytest

pytest.importorskip("app.generation.prompt", reason="需要 production-rag（app 包）")

from rag_adapter.prompting import GroundedAnswerPromptBuilder  # noqa: E402


def test_answer_prompt_requires_verbatim_metric_and_original_unit():
    messages = GroundedAnswerPromptBuilder().build(
        "美的集团2023年的营业收入是多少千元？",
        "[1] 美的集团2023年年度报告 营业总收入 373,709,804",
    )
    system = messages[0]["content"]

    assert "逐字写出题目中的指标名称" in system
    assert "原文口径" in system
    assert "原文数值和原文单位" in system
    assert "必须优先引用精确表格值" in system
    assert "多公司、多年份或多指标题必须逐项回答" in system
    assert "引用必须指向包含该指标和数值的原表或原句" in system
    assert "不能自行推导" in system
