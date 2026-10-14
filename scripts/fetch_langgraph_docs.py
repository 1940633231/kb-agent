"""拉取 LangGraph 官方文档（辅库：技术问答、文档类型对照）。

踩过的坑（2026-10-13 实测）：
  1. `langchain-ai/langgraph` 仓库的 docs/ 现在只剩重定向配置，正文文档已迁到
     独立的 `langchain-ai/docs` 仓库，所以按老办法 clone langgraph 拿不到 markdown。
  2. 本机直连 github.com 超时，改用镜像；浅克隆 + 稀疏检出即可，只取 langgraph 部分
     （43 篇 md/mdx，约 2 MB，而不是整个文档 monorepo）。

产物：
    data/docs/langgraph_docs/                 稀疏检出（.gitignore 忽略）
    data/docs/langgraph_docs_registry.json    来源登记（进 git）

用法：
    python scripts/fetch_langgraph_docs.py                 # 默认镜像
    python scripts/fetch_langgraph_docs.py --upstream      # 网络能直连 GitHub 时
    python scripts/fetch_langgraph_docs.py --register-only # 只补登记表
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST = PROJECT_ROOT / "data" / "docs" / "langgraph_docs"
REGISTRY = PROJECT_ROOT / "data" / "docs" / "langgraph_docs_registry.json"

UPSTREAM = "https://github.com/langchain-ai/docs.git"
MIRROR = "https://ghfast.top/" + UPSTREAM
DOC_PATH = "src/oss/langgraph"


def run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(args, cwd=str(cwd) if cwd else None,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("{} 失败: {}".format(" ".join(args),
                                               (result.stderr or "").strip()[:500]))
    return result


def count_markdown(root: Path) -> tuple[int, int]:
    files = [p for p in root.rglob("*") if p.suffix in {".md", ".mdx"}]
    return len(files), sum(p.stat().st_size for p in files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch LangGraph docs (auxiliary corpus)")
    parser.add_argument("--upstream", action="store_true",
                        help="直接使用 github.com（默认走镜像）")
    parser.add_argument("--register-only", action="store_true",
                        help="跳过 clone，只按当前工作区写登记表")
    args = parser.parse_args()

    source = UPSTREAM if args.upstream else MIRROR
    doc_dir = DEST / DOC_PATH

    if not args.register_only and not DEST.exists():
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "--quiet", source, str(DEST)])
        run(["git", "sparse-checkout", "set", DOC_PATH], cwd=DEST)

    if not doc_dir.exists():
        print("未找到 {}，先执行不带 --register-only 的抓取".format(doc_dir))
        return 1

    commit = run(["git", "rev-parse", "HEAD"], cwd=DEST).stdout.strip()
    files, size = count_markdown(DEST)
    registry = {
        "upstream": UPSTREAM,
        "source_used": source,
        "sparse_path": DOC_PATH,
        "commit": commit,
        "markdown_files": files,
        "markdown_bytes": size,
        "local_dir": str(DEST.relative_to(PROJECT_ROOT).as_posix()),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "note": "辅库暂未灌入索引：L0/L1/L2 基线固定在 9 份年报语料上，"
                "避免语料变动导致基线不可比。",
    }
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("commit={} markdown={} 大小={:.1f}MB".format(commit[:8], files, size / 1e6))
    print("登记表 -> {}".format(REGISTRY))
    return 0


if __name__ == "__main__":
    sys.exit(main())
