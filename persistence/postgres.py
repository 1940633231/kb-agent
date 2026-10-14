"""Postgres Checkpointer / Store 的装配与生命周期管理。

Checkpointer 与 Store 的区别（v2 §4.4 / §4.5，面试要能一句话说清）：
    - Checkpointer：保存**某个线程执行到哪了**，服务于中断恢复与继续执行；
      它不是聊天记录表，也不是知识库。
    - Store：保存**跨线程仍然成立的记忆**（用户偏好、业务约定），按 namespace 隔离。
    两者可以共用同一套 Postgres 基础设施，但承担完全不同的职责。

连接串来自 .env：
    POSTGRES_HOST=127.0.0.1
    POSTGRES_PORT=5433
    POSTGRES_USER=kb_agent
    POSTGRES_PASSWORD=kb_agent
    POSTGRES_DB=kb_agent

没配置或连不上时返回 None，调用方退回 InMemorySaver —— 保证没有数据库的环境
（例如 CI 里的纯单测）仍能跑图。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


def postgres_dsn() -> Optional[str]:
    """从环境变量拼 DSN；缺少必要变量时返回 None。"""
    host = os.environ.get("POSTGRES_HOST")
    db = os.environ.get("POSTGRES_DB")
    if not host or not db:
        return None
    user = os.environ.get("POSTGRES_USER", "kb_agent")
    password = os.environ.get("POSTGRES_PASSWORD", "kb_agent")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return "postgresql://{}:{}@{}:{}/{}".format(user, password, host, port, db)


@dataclass
class PostgresBundle:
    """一个连接池 + 两个 saver/store，进程内复用。"""

    pool: Any
    checkpointer: Any
    store: Any
    dsn: str

    def close(self) -> None:
        try:
            self.pool.close()
        except Exception:  # noqa: BLE001 - 关闭失败不影响主流程
            pass


def build_postgres_bundle(setup: bool = True,
                          pool_size: int = 5) -> Optional[PostgresBundle]:
    """建连接池并初始化 Checkpointer 与 Store 的表结构。

    连不上或缺少依赖时返回 None（调用方降级到内存实现）。
    """
    dsn = postgres_dsn()
    if not dsn:
        return None

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore
        from psycopg_pool import ConnectionPool
    except ImportError:
        return None

    pool = ConnectionPool(dsn, min_size=1, max_size=pool_size, open=True,
                          kwargs={"autocommit": True})
    try:
        checkpointer = PostgresSaver(pool)
        store = PostgresStore(pool)
        if setup:
            # 建表是幂等的（CREATE TABLE IF NOT EXISTS / migration），可反复调用
            checkpointer.setup()
            store.setup()
    except Exception:  # noqa: BLE001 - 初始化失败时释放连接池并降级
        pool.close()
        return None
    return PostgresBundle(pool=pool, checkpointer=checkpointer, store=store, dsn=dsn)
