"""离线复核"答案落地校验"：不重跑图，直接用报告 + 索引重算指标。

用途：判定规则（要求被引片段含哪些指标词）改一次就要跑一遍 40 分钟的全量评估，
代价太高。这个脚本用已有的 l3_report 与 data/index 重新算，用来先比规则再定稿。

    python evals/recheck_grounding.py --report evals/reports/l3_report_gen.json
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

from agent.grounding import _equivalents, extract_numbers  # noqa: E402
from agent.grading import extract_facets, normalize  # noqa: E402


def check(answer: str, citations: List[Dict], chunks: Dict[str, Dict],
          question: str, rule: str) -> Dict[str, Any]:
    numbers = extract_numbers(answer or "")
    tokens = extract_facets(question)["keywords"]
    texts: List[str] = []
    for citation in citations:
        chunk = chunks.get(citation.get("chunk_id"))
        if chunk is None:
            continue
        content = normalize(chunk.get("content") or "")
        if not tokens:
            texts.append(content)
        elif rule == "all":
            if all(normalize(t) in content for t in tokens):
                texts.append(content)
        else:  # longest：只要含"最长的那个指标词"就算相关片段
            longest = max(len(t) for t in tokens)
            if any(normalize(t) in content for t in tokens if len(t) == longest):
                texts.append(content)
    ungrounded = [n for n in numbers
                  if not any(any(v and v in text for v in _equivalents(n))
                             for text in texts)]
    return {"checked": len(numbers), "ungrounded": ungrounded,
            "verified": not ungrounded}


def main() -> int:
    parser = argparse.ArgumentParser(description="Recheck grounding offline")
    parser.add_argument("--report", default="evals/reports/l3_report_gen.json")
    parser.add_argument("--strategy", default="recursive")
    args = parser.parse_args()

    report = json.loads((PROJECT_ROOT / args.report).read_text(encoding="utf-8"))
    meta = json.loads((PROJECT_ROOT / "data" / "index" / args.strategy /
                       "metadata.json").read_text(encoding="utf-8"))
    chunks = {v["chunk_id"]: v for v in meta.values()}
    gold = {json.loads(line)["id"]: json.loads(line)
            for line in base.GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()}

    print("{:16s} {:>8s} {:>8s}  {}".format("id", "all规则", "longest", "ungrounded(longest)"))
    verified = {"all": 0, "longest": 0}
    counted = 0
    for row in report["questions"]:
        if not row.get("answer"):
            continue
        question = gold[row["id"]]["question"]
        counted += 1
        results = {rule: check(row["answer"], row.get("citations") or [], chunks,
                               question, rule) for rule in ("all", "longest")}
        for rule in results:
            verified[rule] += 1 if results[rule]["verified"] else 0
        print("{:16s} {:>8s} {:>8s}  {}".format(
            row["id"], "OK" if results["all"]["verified"] else "flagged",
            "OK" if results["longest"]["verified"] else "flagged",
            "，".join(results["longest"]["ungrounded"][:2]) or "-"))
    print()
    for rule in ("all", "longest"):
        print("{} 规则：全部数字可落地 {}/{} = {:.3f}".format(
            rule, verified[rule], counted, verified[rule] / counted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
