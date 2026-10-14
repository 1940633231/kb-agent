"""长期记忆 CLI：查看/写入/删除某个用户的记忆（演示跨会话与 namespace 隔离）。

用法：
    python scripts/memories.py --user alice --add "记住我以后都用表格回答"
    python scripts/memories.py --user alice --list
    python scripts/memories.py --user bob --list          # 看不到 alice 的记忆
    python scripts/memories.py --user alice --delete <memory_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.memory import forget, recall, remember  # noqa: E402
from persistence.postgres import build_postgres_bundle  # noqa: E402
from rag_adapter.settings import load_env, use_project_root_as_cwd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Long-term memory CLI")
    parser.add_argument("--user", required=True)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--add")
    parser.add_argument("--type", default="preference",
                        choices=["preference", "fact", "task"])
    parser.add_argument("--delete")
    args = parser.parse_args()

    use_project_root_as_cwd()
    load_env()

    bundle = build_postgres_bundle()
    if bundle is None:
        print("Postgres 不可用：长期记忆需要 Store")
        return 2
    try:
        if args.add:
            record = remember(bundle.store, args.user, args.add,
                              memory_type=args.type, source="cli")
            print("已写入: {}".format(json.dumps(record, ensure_ascii=False)))
        if args.delete:
            forget(bundle.store, args.user, args.delete)
            print("已删除: {}".format(args.delete))
        memories = recall(bundle.store, args.user, limit=20)
        print("{} 的记忆（{} 条）:".format(args.user, len(memories)))
        for memory in memories:
            print("  [{}] {} | {}".format(memory.get("memory_id"),
                                          memory.get("memory_type"),
                                          memory.get("content")))
        return 0
    finally:
        bundle.close()


if __name__ == "__main__":
    sys.exit(main())
