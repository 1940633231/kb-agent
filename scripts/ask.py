"""命令行演示：跑一次主图，打印答案、引用与执行痕迹。

用法：
    python scripts/ask.py "贵州茅台2023年的营业收入是多少元？"
    python scripts/ask.py "宁德时代2023年与2025年的营业收入分别是多少？" --no-generate
    python scripts/ask.py "记住我以后都用表格回答" --postgres          # 写入长期记忆
    python scripts/ask.py "招商银行2024年不良贷款率是多少？" --postgres --interrupt

--thread 相同即同一会话；--postgres 让线程状态与长期记忆落在 Postgres，
因此可以跨进程恢复（配合 scripts/review_cli.py 做人工审核演示）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import ask, build_graph, default_runtime  # noqa: E402
from rag_adapter.settings import load_env, use_project_root_as_cwd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the kb-agent graph")
    parser.add_argument("question")
    parser.add_argument("--thread", default="cli")
    parser.add_argument("--strategy", default="recursive")
    parser.add_argument("--no-generate", action="store_true",
                        help="只跑检索与判定，不调用 LLM 生成")
    parser.add_argument("--rule-rewrite", action="store_true",
                        help="改写走规则兜底，不调用 LLM")
    parser.add_argument("--interrupt", action="store_true",
                        help="低置信度时挂起等待人工审核（需配合 --postgres）")
    parser.add_argument("--postgres", action="store_true",
                        help="用 Postgres 做 Checkpointer 与 Store（可跨进程恢复）")
    parser.add_argument("--user", default="demo-user", help="长期记忆的用户标识")
    parser.add_argument("--verbose", action="store_true", help="打印执行痕迹")
    args = parser.parse_args()

    use_project_root_as_cwd()
    load_env()

    bundle = None
    checkpointer = store = None
    if args.postgres:
        from persistence.postgres import build_postgres_bundle
        bundle = build_postgres_bundle()
        if bundle is None:
            print("Postgres 不可用：检查 .env 的 POSTGRES_* 与容器是否启动")
            return 2
        checkpointer, store = bundle.checkpointer, bundle.store

    runtime = default_runtime(strategy=args.strategy,
                              use_llm_rewrite=not args.rule_rewrite,
                              generate_answers=not args.no_generate,
                              interrupt_on_low_confidence=args.interrupt,
                              memory_store=store, memory_user_id=args.user)
    graph = build_graph(runtime, checkpointer=checkpointer, store=store)

    started = time.time()
    result = ask(graph, args.question, thread_id=args.thread)
    elapsed = time.time() - started

    try:
        if args.interrupt:
            from agent.review import pending_review
            payload = pending_review(graph, args.thread)
            if payload:
                print("已挂起等待人工审核:")
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                print()
                print("恢复: python scripts/review_cli.py --thread {} "
                      "--review-id {} --decision approve".format(
                          args.thread, payload["review_id"]))
                return 0

        print("问题: {}".format(result["question"]))
        print("状态: {} | 重试: {} | 是否标记待审: {} | 记忆命中: {} 条".format(
            result["status"], result["retry_count"],
            result["needs_human_review"], len(result.get("memories") or [])))
        print("检索证据: {} 个片段 / {} 份年报".format(
            len(result["retrieved_chunks"]),
            (result["retrieval_score"] or {}).get("document_count", 0)))
        score = result["retrieval_score"] or {}
        print("证据判定: {} ({})".format(
            "通过" if score.get("passed") else "不通过", score.get("reason", "")))
        print()
        print("答案: {}".format(result.get("answer") or "(未生成)"))
        if result.get("citations"):
            print()
            print("引用:")
            for citation in result["citations"]:
                print("  [{}] {} p.{} {}".format(
                    citation.get("number"), citation.get("file_name"),
                    citation.get("page", "-"), citation.get("content_preview", "")))
        if result.get("memory_written"):
            print()
            print("已写入长期记忆:")
            print(json.dumps(result["memory_written"], ensure_ascii=False, indent=2))
        if args.verbose:
            print()
            print("执行痕迹:")
            print(json.dumps(result.get("trace") or [], ensure_ascii=False, indent=2))
            print()
            print("改写历史:")
            print(json.dumps(result.get("rewrite_history") or [], ensure_ascii=False,
                             indent=2))
        print()
        print("耗时 {:.2f}s".format(elapsed))
        return 0
    finally:
        if bundle:
            bundle.close()


if __name__ == "__main__":
    sys.exit(main())
