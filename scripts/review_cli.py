"""人工审核 CLI：查看待审项、提交审核结果（approve / reject / supplement）。

必须配合 Postgres：审核天然跨进程——挂起发生在服务进程，提交来自另一个请求或
进程，只有持久化 checkpointer 才能恢复执行。

用法：
    python scripts/review_cli.py --thread demo --list
    python scripts/review_cli.py --thread demo --review-id abc123 --decision approve --reviewer alice
    python scripts/review_cli.py --thread demo --review-id abc123 --decision supplement --note "口径用万元"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import build_graph, default_runtime  # noqa: E402
from agent.review import pending_review, resume_review  # noqa: E402
from persistence.postgres import build_postgres_bundle  # noqa: E402
from rag_adapter.settings import load_env, use_project_root_as_cwd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Human review CLI")
    parser.add_argument("--thread", required=True)
    parser.add_argument("--list", action="store_true", help="只查看待审项")
    parser.add_argument("--review-id")
    parser.add_argument("--decision", choices=["approve", "reject", "supplement"])
    parser.add_argument("--reviewer", default=os.environ.get("REVIEWER", "reviewer"))
    parser.add_argument("--note", default="")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0,
                        help="超过该时长未审核判定过期（0 表示不判超时）")
    parser.add_argument("--user", default="demo-user")
    args = parser.parse_args()

    use_project_root_as_cwd()
    load_env()

    bundle = build_postgres_bundle()
    if bundle is None:
        print("Postgres 不可用：审核流程需要持久化 checkpointer")
        return 2
    try:
        runtime = default_runtime(generate_answers=False,
                                  interrupt_on_low_confidence=True,
                                  memory_store=bundle.store,
                                  memory_user_id=args.user)
        graph = build_graph(runtime, checkpointer=bundle.checkpointer,
                            store=bundle.store)
        if args.list or not args.decision:
            payload = pending_review(graph, args.thread)
            if payload is None:
                print("该线程没有待审项")
                return 0
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if not args.review_id:
            parser.error("提交审核需要 --review-id（先用 --list 拿到）")
        result = resume_review(graph, args.thread, args.review_id,
                               decision=args.decision, reviewer=args.reviewer,
                               note=args.note, store=bundle.store,
                               timeout_seconds=args.timeout_seconds or None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in ("resumed", "already_processed") else 1
    finally:
        bundle.close()


if __name__ == "__main__":
    sys.exit(main())
