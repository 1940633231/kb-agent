"""接口层配置（环境变量驱动，默认值面向本地演示）。"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8300
    #: 默认是否走生成链路（关掉可用于只演示检索层）
    generate: bool = True
    #: 是否在低置信度时挂起等人工审核
    interrupt_on_low_confidence: bool = True
    #: 单进程内允许并发的图执行数（CPU 重排很重，避免把机器打满）
    max_concurrency: int = 2
    #: 人工审核超时（秒）
    review_timeout_seconds: float = 3600.0
    #: 默认记忆用户（演示用；真实系统来自鉴权上下文）
    default_user: str = "demo-user"
    #: 生成过程的 token 级流式（SSE）
    token_stream: bool = True
    #: 规则评分通过后再做一次 LLM 语义复核（抓"规则假通过"）
    llm_grade: bool = True

    @classmethod
    def from_env(cls) -> "ApiSettings":
        def flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        return cls(
            host=os.environ.get("KB_AGENT_HOST", "127.0.0.1"),
            port=int(os.environ.get("KB_AGENT_PORT", "8300")),
            generate=flag("KB_AGENT_GENERATE", True),
            interrupt_on_low_confidence=flag("KB_AGENT_INTERRUPT", True),
            max_concurrency=int(os.environ.get("KB_AGENT_MAX_CONCURRENCY", "2")),
            review_timeout_seconds=float(os.environ.get("KB_AGENT_REVIEW_TIMEOUT",
                                                        "3600")),
            default_user=os.environ.get("KB_AGENT_USER", "demo-user"),
            token_stream=flag("KB_AGENT_TOKEN_STREAM", True),
            llm_grade=flag("KB_AGENT_LLM_GRADE", True),
        )
