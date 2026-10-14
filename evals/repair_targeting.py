"""离线评估"生成侧修复"打不打得到真正的错（不需要调用 LLM）。

用途：修复环节要不要开、会不会误伤，可以在已有报告上先算清楚——
    对每条答案算 (a) 有没有引不到出处的数字 (b) 是否口径歧义且未标注，
    再与 LLM 裁判的裁决交叉，得到一张"命中 / 误伤 / 漏网 / 正确放过"的表。

用法：
    python evals/repair_targeting.py --report evals/reports/l3_report_gen_round2_judged.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import run_eval as base  # noqa: E402

from agent.grounding import (detect_ambiguous_metric, has_statement_label,  # noqa: E402
                             verify_grounding)


def load_chunks(strategy: str = "recursive") -> Dict[str, Dict]:
    meta = json.loads((PROJECT_ROOT / "data" / "index" / strategy /
                       "metadata.json").read_text(encoding="utf-8"))
    return {value["chunk_id"]: value for value in meta.values()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair-loop targeting analysis")
    parser.add_argument("--report",
                        default="evals/reports/l3_report_gen_round2_judged.json")
    parser.add_argument("--strategy", default="recursive")
    args = parser.parse_args()

    report = json.loads((PROJECT_ROOT / args.report).read_text(encoding="utf-8"))
    chunks = load_chunks(args.strategy)
    gold = {json.loads(line)["id"]: json.loads(line)
            for line in base.GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()}

    table = {"hit": 0, "false_alarm": 0, "missed": 0, "pass": 0}
    print("{:16s} {:10s} {:9s} {:9s} {}".format("id", "裁判", "会触发", "歧义",
                                                "引不到出处的数字"))
    for row in report["questions"]:
        if row.get("expected_behavior") != "answer" or not row.get("answer"):
            continue
        item = gold[row["id"]]
        cited = [chunks.get(c.get("chunk_id")) for c in row.get("citations") or []]
        cited = [c for c in cited if c]
        grounding = verify_grounding(row["answer"], row.get("citations") or [],
                                     cited, item["question"])
        ambiguity = detect_ambiguous_metric(item["question"], cited)
        ambiguous_unlabeled = (ambiguity.get("ambiguous")
                               and not has_statement_label(row["answer"]))
        would_repair = bool(grounding["ungrounded"]) or ambiguous_unlabeled
        verdict = (row.get("judged") or {}).get("verdict", "unknown")
        wrong = verdict in ("incorrect", "partial")
        if wrong and would_repair:
            table["hit"] += 1
        elif wrong and not would_repair:
            table["missed"] += 1
        elif not wrong and would_repair:
            table["false_alarm"] += 1
        else:
            table["pass"] += 1
        print("{:16s} {:10s} {:9s} {:9s} {}".format(
            row["id"], verdict, "是" if would_repair else "否",
            "是" if ambiguity.get("ambiguous") else "否",
            "、".join(grounding["ungrounded"][:2]) or "-"))

    print()
    print("裁判判错且会被修复命中: {} 题".format(table["hit"]))
    print("裁判判错但修复漏网  : {} 题".format(table["missed"]))
    print("裁判判对却会被改动  : {} 题（误伤风险）".format(table["false_alarm"]))
    print("裁判判对且正确放过  : {} 题".format(table["pass"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
