"""LLM 裁判：按方案 §6.2 的"答案正确性"口径判答案对不对。

为什么需要它（实测）：字符串匹配已经到极限——
    `ar-fact-001` 第二轮的回答把"母公司营业收入 81,415,570,788.59 元"当成答案，
    但它同时把含正确值的表格行也贴进了正文，于是"期望值是否出现在答案里"判成通过（**过宽**）；
    而模型把 147,693,604,994.14 元正确换算成"14,769,360.50 万元"时，粗粒度匹配又可能判不中（**过窄**）。
    这类"数值是否真的一致"的判定，只能交给会读语义的裁判。

用法：
    python evals/judge_answers.py --report evals/reports/l3_report_gen.json
    python evals/judge_answers.py --report evals/reports/l3_report_gen_before_optimization.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import run_eval as base  # noqa: E402

from rag_adapter import rag  # noqa: E402
from rag_adapter.settings import load_env, use_project_root_as_cwd  # noqa: E402

JUDGE_SYSTEM_PROMPT = (
    "你是严格的答案裁判。给定问题、参考答案（含单位）与模型答案，判断模型答案是否正确。\n"
    "判为 correct：给出的数值与参考答案一致——允许等价的单位换算（元/千元/万元/亿元）、"
    "允许合理舍入，且实体（公司）、年份、指标名都与问题一致。\n"
    "判为 partial：部分正确（多值问题只答对一部分，或数值对但漏了单位/口径说明）。\n"
    "判为 incorrect：数值不符、单位换算错误、答成其它年份/公司/指标、答非所问、"
    "或应当给出数值却只回答“无法回答”。\n"
    "只输出 JSON：{\"verdict\": \"correct\"|\"partial\"|\"incorrect\", "
    "\"reason\": \"一句话理由\", "
    "\"category\": \"none|value|unit|entity_year|metric|no_answer\"}"
)


def reference_block(item: Dict[str, Any]) -> str:
    lines = []
    for entry in item.get("expected_values") or []:
        lines.append("- {} = {} {}".format(entry.get("label", ""),
                                           entry.get("display", ""),
                                           entry.get("unit", "")))
    docs = ["{}{}".format(doc.get("company", ""), doc.get("year", ""))
            for doc in item.get("expected_docs") or []]
    if docs:
        lines.append("- 应来自: {}".format("、".join(docs)))
    return "\n".join(lines) or "-（无具体数值）"


class AnswerJudge:
    def __init__(self, generator: Any) -> None:
        self.generator = generator
        self.errors = 0

    def judge(self, item: Dict[str, Any], answer: str) -> Dict[str, Any]:
        prompt = (
            "问题：{question}\n\n参考答案：\n{reference}\n\n模型答案：\n{answer}"
        ).format(question=item["question"], reference=reference_block(item),
                 answer=(answer or "")[:1500])
        try:
            raw = self.generator.generate([
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
        except Exception as exc:  # noqa: BLE001 - 裁判失败不影响其它题
            self.errors += 1
            return {"verdict": "unknown", "reason": "judge 调用失败: {}".format(exc),
                    "category": "none", "source": "error"}
        parsed = self._parse(raw)
        if parsed is None:
            self.errors += 1
            return {"verdict": "unknown", "reason": "裁判输出解析失败",
                    "category": "none", "source": "error"}
        parsed["source"] = "llm"
        return parsed

    @staticmethod
    def _parse(raw: Optional[str]) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        verdict = str(payload.get("verdict", "")).strip().lower()
        if verdict not in ("correct", "partial", "incorrect"):
            return None
        return {"verdict": verdict, "reason": str(payload.get("reason", ""))[:200],
                "category": str(payload.get("category", "none"))[:32]}


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM judge for answer correctness")
    parser.add_argument("--report", default="evals/reports/l3_report_gen.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    use_project_root_as_cwd()
    load_env()

    report_path = PROJECT_ROOT / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gold = {json.loads(line)["id"]: json.loads(line)
            for line in base.GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()}

    service = rag.build_service(strategy="recursive", mode="hybrid", use_rerank=True)
    judge = AnswerJudge(service._pipeline.generator)  # noqa: SLF001

    scores: List[float] = []
    categories: Dict[str, int] = {}
    verdicts: Dict[str, int] = {}
    for row in report["questions"]:
        if row.get("expected_behavior") != "answer":
            continue
        item = gold[row["id"]]
        outcome = judge.judge(item, row.get("answer") or "")
        row["judged"] = outcome
        verdicts[outcome["verdict"]] = verdicts.get(outcome["verdict"], 0) + 1
        if outcome["verdict"] == "incorrect":
            categories[outcome["category"]] = categories.get(outcome["category"], 0) + 1
        scores.append({"correct": 1.0, "partial": 0.5}.get(outcome["verdict"], 0.0))
        print("{:16s} {:9s} {:14s} {}".format(row["id"], outcome["verdict"],
                                              outcome["category"],
                                              outcome["reason"][:60]))

    report["judged_summary"] = {
        "answerable": len(scores),
        "answer_correct_rate": round(sum(scores) / len(scores), 4) if scores else None,
        "verdicts": verdicts,
        "incorrect_categories": categories,
        "judge": "llm (qwen)",
    }
    out_path = PROJECT_ROOT / (args.out or str(report_path).replace(
        PROJECT_ROOT.as_posix(), "").lstrip("/").replace(".json", "_judged.json"))
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print()
    print(json.dumps(report["judged_summary"], ensure_ascii=False))
    print("报告 -> {}".format(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
