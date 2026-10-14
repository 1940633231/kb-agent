"""对"证据未命中"的题做归因：期望数值所在 chunk 到底排在第几。

用途：分层矩阵里 evidence_hit 不达标时，区分两种完全不同的原因——
    (a) 排序问题：含答案的 chunk 落在 top-5 之外（放宽 top_k 就能拿到）；
    (b) 召回问题：即便放宽到池子也找不到（切分/表述/结构化缺失）。
结论直接决定下一步是"调证据预算/加查询改写"还是"改解析与切分"。

用法：
    python evals/diagnose_missing_evidence.py --ids ar-fact-001 ar-multi-002
    python evals/diagnose_missing_evidence.py --all-failing --top-k 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import run_eval as base  # noqa: E402

from rag_adapter import rag  # noqa: E402
from rag_adapter.settings import load_env, use_project_root_as_cwd  # noqa: E402


def find_value_rank(hits: List[Dict], value_entry: Dict) -> Dict:
    for position, hit in enumerate(hits, 1):
        text = base.normalize(hit.get("content") or "")
        for variant in base.text_variants(value_entry):
            if variant and base.normalize(variant) in text:
                return {"found": True, "rank": position,
                        "document_id": hit.get("document_id")}
    return {"found": False, "rank": None, "document_id": None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose missing evidence ranks")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--all-failing", action="store_true",
                        help="诊断 L3 报告里 evidence_hit_final 为假的题")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--strategy", default="recursive")
    args = parser.parse_args()

    use_project_root_as_cwd()
    load_env()

    questions = {item["id"]: item
                 for item in base.load_golden_set(base.GOLDEN_SET_PATH)}
    target_ids = list(args.ids)
    if args.all_failing:
        report = json.loads((PROJECT_ROOT / "evals" / "reports" /
                             "l3_report.json").read_text(encoding="utf-8"))
        target_ids.extend(row["id"] for row in report["questions"]
                          if row["evidence_hit_final"] is False)
    if not target_ids:
        parser.error("需要 --ids 或 --all-failing")

    service = rag.build_service(strategy=args.strategy, mode="hybrid",
                                use_rerank=True)
    for item_id in target_ids:
        item = questions[item_id]
        hits = rag.retrieve_multi(service, [item["question"]],
                                  top_k=args.top_k, anchor=item["question"])
        print("== {} (top_k={}) ==".format(item_id, args.top_k))
        print("   {}".format(item["question"]))
        for entry in item["expected_values"]:
            outcome = find_value_rank(hits, entry)
            print("   {} = {} -> {}".format(
                entry["label"], entry["display"],
                "rank {} in {}".format(outcome["rank"], outcome["document_id"])
                if outcome["found"] else "池内未找到（召回层面的问题）"))
        print("   检索到的文档: {}".format(
            sorted({hit.get("document_id") for hit in hits})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
