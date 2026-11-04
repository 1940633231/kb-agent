"""kb-agent 配置与 production-rag 统一 LLM 接口的兼容性测试。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_adapter.settings import (DASHSCOPE_COMPAT_BASE_URL,  # noqa: E402
                                  load_env)


def test_legacy_dashscope_env_is_bridged(monkeypatch, tmp_path):
    env_file = tmp_path / "legacy.env"
    env_file.write_text(
        "DASHSCOPE_API_KEY=legacy-key\nDASHSCOPE_MODEL=qwen-flash\n",
        encoding="utf-8",
    )
    for name in ("DASHSCOPE_API_KEY", "DASHSCOPE_MODEL",
                 "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    load_env(env_file)

    assert os.environ["LLM_API_KEY"] == "legacy-key"
    assert os.environ["LLM_BASE_URL"] == DASHSCOPE_COMPAT_BASE_URL
    assert os.environ["LLM_MODEL"] == "qwen-flash"


def test_explicit_llm_env_wins_over_legacy(monkeypatch, tmp_path):
    env_file = tmp_path / "legacy.env"
    env_file.write_text(
        "DASHSCOPE_API_KEY=legacy-key\nDASHSCOPE_MODEL=legacy-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_MODEL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "explicit-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://explicit.example/v1")
    monkeypatch.setenv("LLM_MODEL", "explicit-model")

    load_env(env_file)

    assert os.environ["LLM_API_KEY"] == "explicit-key"
    assert os.environ["LLM_BASE_URL"] == "https://explicit.example/v1"
    assert os.environ["LLM_MODEL"] == "explicit-model"


def test_kb_config_builds_upstream_unified_llm(monkeypatch):
    pytest.importorskip("app.core.config", reason="需要 production-rag（app 包）")
    pytest.importorskip("app.generation.generator", reason="需要 production-rag（app 包）")

    from app.generation.generator import create_generator_from_config
    from rag_adapter.rag import build_config

    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    config = build_config()
    assert config.generation_backend == "llm"
    assert config.generation_llm_base_url == "https://llm.example/v1"
    assert config.generation_llm_model == "test-model"

    generator = create_generator_from_config(config)
    assert type(generator).__name__ == "LLMGenerator"
