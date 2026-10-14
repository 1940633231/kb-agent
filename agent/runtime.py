"""主图运行时依赖：检索服务、改写器、阈值等，集中注入便于测试替身。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentRuntime:
    """节点用到的外部依赖与可调参数。"""

    service: Any                      # RAGService（L2 配置：hybrid + rerank）
    strategy: str = "recursive"
    top_k: int = 5                    # 与 L0/L1/L2 对齐，保证可比
    max_retries: int = 2              # v2 §4.1：重试上限 2 次
    # 证据预算：每轮最多新增 top_k 条，重试只是追加，因此上限 = top_k × (1+重试)
    # 默认 20 保证"第一轮证据永不因追加被挤掉"，L3 的 top-5 与 L2 逐题一致。
    evidence_limit: int = 20
    # 重试轮的检索预算：方案 §4.3 允许的重试动作包含"放宽检索条件"。
    # 实测依据：茅台两道元级数字题，含答案的片段排在 rerank 第 6 / 第 16 位，
    # top-5 拿不到，放宽到 15 就能覆盖。
    widened_top_k: int = 15
    min_coverage: float = 0.6         # 规则评分的关键词覆盖率下限
    rewriter: Optional[Any] = None    # 有 rewrite(question, reason, prev) -> List[str]
    generate_answers: bool = True     # 评估检索层时置 False，跳过 LLM 生成
    memory_store: Optional[Any] = None   # LangGraph Store（跨会话记忆）
    memory_user_id: str = ""             # 记忆 namespace 里的用户标识
    review_timeout_seconds: Optional[float] = 3600.0  # 审核超时（秒）
    # document_id → {title, company, year, pages}，用于把引用的字符偏移映射回原文页码
    page_offsets: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def pipeline(self):
        return self.service._pipeline  # noqa: SLF001 - 上游只暴露 query()

    @property
    def mode(self) -> str:
        return getattr(self.service, "mode", "vector")

    @property
    def rerank_enabled(self) -> bool:
        return bool(getattr(self.service, "use_rerank", False))
