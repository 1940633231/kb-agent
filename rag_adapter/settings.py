"""kb-agent 的路径与环境设置。

production-rag 的索引目录、原始文档目录都是相对 CWD 的
（Config.index_dir_for 返回 Path("data/index")/{strategy}），所以所有入口脚本
都必须以项目根为工作目录运行：用 use_project_root_as_cwd() 显式切换，
避免"在别处运行就写到别处去"。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "kb_agent.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

ANNUAL_REPORTS_DIR = PROJECT_ROOT / "data" / "docs" / "annual_reports"
REPORTS_REGISTRY = ANNUAL_REPORTS_DIR / "metadata.json"

EVALS_DIR = PROJECT_ROOT / "evals"
GOLDEN_SET_PATH = EVALS_DIR / "golden_set.jsonl"
BASELINE_PATH = EVALS_DIR / "baseline.json"


def load_env(path: Optional[Path] = None) -> None:
    """把 .env 读进 os.environ；已存在的环境变量不覆盖。"""
    env_path = path or ENV_PATH
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def use_project_root_as_cwd() -> Path:
    """切换到项目根目录（production-rag 的相对路径依赖 CWD）。"""
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT


def index_dir(strategy: str = "recursive") -> Path:
    return PROJECT_ROOT / "data" / "index" / strategy


def page_offsets_path(strategy: str = "recursive") -> Path:
    return index_dir(strategy) / "page_offsets.json"
