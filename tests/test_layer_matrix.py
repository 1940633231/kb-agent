"""Layer-matrix report writing regressions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

from run_agent_eval import update_layer_matrix  # noqa: E402


def write_layer(path: Path, layer: str, value: float) -> None:
    update_layer_matrix(
        path,
        layer,
        "label-{}".format(layer),
        {"layer": layer},
        {"doc_recall@5": value},
    )


def test_l2_l3a_l3b_coexist_in_layer_matrix(tmp_path):
    path = tmp_path / "layer_matrix.json"

    write_layer(path, "L2", 0.91)
    write_layer(path, "L3a", 0.96)
    write_layer(path, "L3b", 0.97)

    matrix = json.loads(path.read_text(encoding="utf-8"))
    assert set(matrix["layers"]) == {"L2", "L3a", "L3b"}
    assert matrix["layers"]["L2"]["metrics"]["doc_recall@5"] == 0.91
    assert matrix["layers"]["L3a"]["metrics"]["doc_recall@5"] == 0.96
    assert matrix["layers"]["L3b"]["metrics"]["doc_recall@5"] == 0.97


def test_rerunning_l3a_only_updates_l3a(tmp_path):
    path = tmp_path / "layer_matrix.json"
    write_layer(path, "L2", 0.91)
    write_layer(path, "L3a", 0.96)
    write_layer(path, "L3b", 0.97)

    write_layer(path, "L3a", 0.94)

    matrix = json.loads(path.read_text(encoding="utf-8"))
    assert matrix["layers"]["L2"]["metrics"]["doc_recall@5"] == 0.91
    assert matrix["layers"]["L3a"]["metrics"]["doc_recall@5"] == 0.94
    assert matrix["layers"]["L3b"]["metrics"]["doc_recall@5"] == 0.97


def test_rerunning_l3b_only_updates_l3b(tmp_path):
    path = tmp_path / "layer_matrix.json"
    write_layer(path, "L2", 0.91)
    write_layer(path, "L3a", 0.96)
    write_layer(path, "L3b", 0.97)

    write_layer(path, "L3b", 0.93)

    matrix = json.loads(path.read_text(encoding="utf-8"))
    assert matrix["layers"]["L2"]["metrics"]["doc_recall@5"] == 0.91
    assert matrix["layers"]["L3a"]["metrics"]["doc_recall@5"] == 0.96
    assert matrix["layers"]["L3b"]["metrics"]["doc_recall@5"] == 0.93
