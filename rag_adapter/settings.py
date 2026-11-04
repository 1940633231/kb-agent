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

DASHSCOPE_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_COMPAT_MODEL = "qwen-flash"


def load_env(path: Optional[Path] = None) -> None:
    """把 .env 读进 os.environ；已存在的环境变量不覆盖。"""
    env_path = path or ENV_PATH
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(
                key.strip(), value.strip().strip('"').strip("'")
            )
    _bridge_legacy_llm_env()


def _bridge_legacy_llm_env() -> None:
    """兼容旧 DashScope 配置，运行时统一使用 LLM_* 变量。

    显式配置的 LLM_* 始终优先；只有旧变量存在且新变量为空时才迁移。
    """
    legacy_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not legacy_api_key:
        return

    def set_if_blank(name: str, value: str) -> None:
        if not os.getenv(name, "").strip():
            os.environ[name] = value

    set_if_blank("LLM_API_KEY", legacy_api_key)
    set_if_blank("LLM_BASE_URL", DASHSCOPE_COMPAT_BASE_URL)
    set_if_blank(
        "LLM_MODEL",
        os.getenv("DASHSCOPE_MODEL", "").strip() or DASHSCOPE_COMPAT_MODEL,
    )


def use_project_root_as_cwd() -> Path:
    """切换到项目根目录（production-rag 的相对路径依赖 CWD）。"""
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT


def index_dir(strategy: str = "recursive") -> Path:
    return PROJECT_ROOT / "data" / "index" / strategy


def page_offsets_path(strategy: str = "recursive") -> Path:
    return index_dir(strategy) / "page_offsets.json"
