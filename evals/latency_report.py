"""首 token 与分阶段延迟（S4）。

分层矩阵里的"每题耗时"是端到端数字，看不出瓶颈在哪。这里借 LangGraph 的
custom 流事件把一次问答切成三段，给出 P50/P95：

    检索（含重试轮）→ 首 token（TTFT）→ 生成剩余 → 收尾

token 用量用 tiktoken 估算（cl100k 不是 Qwen 分词器，属估算）；
单价不写死在代码里，用 KB_AGENT_INPUT_PRICE_PER_M / KB_AGENT_OUTPUT_PRICE_PER_M 传入。

用法：
    python evals/latency_report.py --limit 6
    python evals/latency_report.py --limit 6 --no-generate   # 只测检索段
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import run_eval as base  # noqa: E402
import run_agent_eval as agent_eval  # noqa: E402

from agent.graph import build_graph, default_runtime  # noqa: E402
from agent.state import initial_state  # noqa: E402
from rag_adapter.settings import load_env, use_project_root_as_cwd  # noqa: E402


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * pct
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    weight = position - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 2)


def measure(graph: Any, question: str, thread_id: str) -> Dict[str, Any]:
    started = time.time()
    retrieval_end = first_token = generation_end = None
    tokens = 0
    final: Dict[str, Any] = {}
    for mode, payload in graph.stream(
            initial_state(question, {"thread_id": thread_id}),
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["updates", "custom"]):
        if mode == "custom":
            if isinstance(payload, dict) and payload.get("type") == "token":
                tokens += 1
                if first_token is None:
                    first_token = time.time()
            continue
        for node, update in (payload or {}).items():
            if node == "retrieve_knowledge":
                retrieval_end = time.time()
            elif node == "generate_answer":
                generation_end = time.time()
            elif node == "finalize_response":
                final = dict(update or {})
    finished = time.time()
    return {
        "retrieval_s": None if retrieval_end is None else round(retrieval_end - started, 2),
        "ttft_s": None if not (retrieval_end and first_token) else round(first_token - retrieval_end, 2),
        "generation_s": None if not (first_token and generation_end) else round(generation_end - first_token, 2),
        "total_s": round(finished - started, 2),
        "stream_events": tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency / TTFT report")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--strategy", default="recursive")
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--out", default="evals/reports/latency.json")
    args = parser.parse_args()

    use_project_root_as_cwd()
    load_env()

    questions = base.load_golden_set(base.GOLDEN_SET_PATH)
    answerable = [q for q in questions if q.get("expected_behavior") == "answer"]
    sample = answerable[:args.limit]

    runtime = default_runtime(generate_answers=not args.no_generate,
                              use_llm_rewrite=False, enable_token_stream=True)
    graph = build_graph(runtime)

    rows: List[Dict[str, Any]] = []
    for item in sample:
        row = measure(graph, item["question"], "lat-{}".format(item["id"]))
        row["id"] = item["id"]
        row["question_type"] = item["question_type"]
        rows.append(row)
        print("{:16s} 检索={retrieval_s} 首token={ttft_s} 生成={generation_s} "
              "总={total_s}s".format(item["id"], **row))

    summary = {
        "samples": len(rows),
        "generate": not args.no_generate,
        "model": "qwen-flash",
        "retrieval_p50": percentile([r["retrieval_s"] for r in rows if r["retrieval_s"]], 0.5),
        "retrieval_p95": percentile([r["retrieval_s"] for r in rows if r["retrieval_s"]], 0.95),
        "ttft_p50": percentile([r["ttft_s"] for r in rows if r["ttft_s"]], 0.5),
        "ttft_p95": percentile([r["ttft_s"] for r in rows if r["ttft_s"]], 0.95),
        "total_p50": percentile([r["total_s"] for r in rows], 0.5),
        "total_p95": percentile([r["total_s"] for r in rows], 0.95),
        "mean_stream_events": round(statistics.fmean(
            [r["stream_events"] for r in rows]), 1) if rows else None,
    }
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告 -> {}".format(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
