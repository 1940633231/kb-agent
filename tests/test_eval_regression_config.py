"""Regression gates only fail when higher-is-better metrics degrade."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("rag_eval.config", reason="需要 rag-eval")

from rag_eval.config import load_eval_config  # noqa: E402

BASELINE = (
    PROJECT_ROOT / "evals" / "rag_eval" / "baselines"
    / "kb_agent_l3_50.json"
)
CONFIG = PROJECT_ROOT / "configs" / "eval.kb_agent.yaml"
LOWER_IS_BETTER = {
    "retry_count",
    "retry_rate",
    "rewrite_sources",
    "foreign_evidence_rate",
    "foreign_citation_rate",
    "human_review_rate",
}


def test_l3_gate_baseline_excludes_lower_is_better_metrics() -> None:
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert not LOWER_IS_BETTER.intersection(payload["metrics"]["values"])


def test_l3_gate_thresholds_only_cover_higher_is_better_metrics() -> None:
    config = load_eval_config(CONFIG)

    assert set(config.thresholds) == {"recall@5", "evidence_hit_final"}
