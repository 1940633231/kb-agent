"""第二阶段：Checkpointer 多轮/重启恢复、Store 跨会话记忆与 namespace 隔离。

没有配置 Postgres（或连不上）时，Postgres 相关的用例会 skip，
纯内存部分照常执行，保证 CI 里没有数据库也能跑。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# kb-agent 同进程 import production-rag（app 包）；CI 里没有它时整模块跳过
pytest.importorskip("app.core.config", reason="需要 production-rag（app 包）")

from agent.memory import (extract_memory, format_memories,  # noqa: E402
                          memory_namespace, recall, remember)
from agent.state import initial_state  # noqa: E402
from persistence.postgres import build_postgres_bundle, postgres_dsn  # noqa: E402
from rag_adapter.settings import load_env  # noqa: E402

load_env()
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "5433")
os.environ.setdefault("POSTGRES_USER", "kb_agent")
os.environ.setdefault("POSTGRES_PASSWORD", "kb_agent")
os.environ.setdefault("POSTGRES_DB", "kb_agent")


@pytest.fixture(scope="module")
def bundle():
    built = build_postgres_bundle()
    if built is None:
        pytest.skip("未配置或连不上 Postgres，跳过持久化用例")
    yield built
    built.close()


class InMemoryStore:
    """够用的 Store 替身：验证记忆读写与 namespace 隔离，不需要数据库。"""

    def __init__(self):
        self.data = {}

    def put(self, namespace, key, value):
        self.data.setdefault(tuple(namespace), {})[key] = dict(value)

    def get(self, namespace, key):
        item = self.data.get(tuple(namespace), {}).get(key)
        if item is None:
            return None
        return type("Item", (), {"key": key, "value": item, "namespace": tuple(namespace)})()

    def search(self, namespace_prefix, filter=None, limit=10, query=None, offset=0):
        prefix = tuple(namespace_prefix)
        results = []
        for namespace, items in self.data.items():
            if namespace[:len(prefix)] != prefix:
                continue
            for key, value in items.items():
                if filter and any(value.get(k) != v for k, v in filter.items()):
                    continue
                results.append(type("Item", (), {
                    "key": key, "value": value, "namespace": namespace,
                    "created_at": value.get("created_at"), "score": None})())
        return results[:limit]

    def delete(self, namespace, key):
        self.data.get(tuple(namespace), {}).pop(key, None)


def test_extract_memory_requires_durable_marker():
    assert extract_memory("贵州茅台2023年的营业收入是多少元？") is None
    assert extract_memory("这次先看2023年") is None
    durable = extract_memory("记住我以后回答都用表格，先给结论")
    assert durable and durable["memory_type"] == "preference"
    assert extract_memory("我是消费品行业分析师，我们公司口径用万元")["memory_type"] == "fact"


def test_store_isolates_users_by_namespace():
    store = InMemoryStore()
    remember(store, "alice", "记住我以后都用表格回答")
    remember(store, "bob", "记住我以后都用要点回答")

    alice = recall(store, "alice")
    bob = recall(store, "bob")
    assert [m["content"] for m in alice] == ["记住我以后都用表格回答"]
    assert [m["content"] for m in bob] == ["记住我以后都用要点回答"]
    assert memory_namespace("alice") != memory_namespace("bob")
    assert "表格" in format_memories(alice)


def test_interrupted_state_keeps_progress():
    """Thread 状态里保留的是执行进度：初始 State 有线程标识与空的证据/记忆。"""
    state = initial_state("问题", {"thread_id": "t-1"})
    assert state["thread_id"] == "t-1"
    assert state["retrieved_chunks"] == []
    assert state["memories"] == []


def test_default_runtime_keeps_memory_configuration(monkeypatch):
    """回归：memory_store 曾被 default_runtime 吞进 **extra，导致记忆静默写不进去。"""
    from agent import graph as agent_graph
    from agent.runtime import AgentRuntime

    store = InMemoryStore()
    # 打桩掉真实检索服务，避免为了一个"装配参数"的回归测试加载 embedding/rerank 模型
    monkeypatch.setattr(agent_graph.rag, "build_service",
                        lambda **kwargs: object())
    runtime = agent_graph.default_runtime(memory_store=store, memory_user_id="alice",
                                          use_llm_rewrite=False)
    assert isinstance(runtime, AgentRuntime)
    assert runtime.memory_store is store
    assert runtime.memory_user_id == "alice"


def test_checkpointer_persists_thread_across_graph_instances(bundle, tmp_path):
    """同一 thread 在**另一个进程**里恢复：状态真的落在 Postgres，而不是进程内存。"""
    script = textwrap.dedent(
        """
        import os, sys, json
        sys.path.insert(0, {project!r})
        from rag_adapter.settings import load_env, use_project_root_as_cwd
        load_env(); use_project_root_as_cwd()
        from langgraph.graph import START, END, StateGraph
        from typing import TypedDict
        from persistence.postgres import build_postgres_bundle

        class S(TypedDict, total=False):
            question: str
            turns: int

        def node(state):
            return {{"turns": int(state.get("turns") or 0) + 1}}

        bundle = build_postgres_bundle()
        builder = StateGraph(S)
        builder.add_node("step", node)
        builder.add_edge(START, "step")
        builder.add_edge("step", END)
        graph = builder.compile(checkpointer=bundle.checkpointer)
        config = {{"configurable": {{"thread_id": sys.argv[2]}}}}
        result = graph.invoke({{"question": sys.argv[1]}}, config=config)
        print(json.dumps({{"turns": result["turns"]}}))
        bundle.close()
        """
    ).format(project=str(PROJECT_ROOT))
    script_path = tmp_path / "child_graph.py"
    script_path.write_text(script, encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    # 每次跑用唯一 thread_id，避免上一轮遗留的 checkpoint 影响断言
    thread_id = "restart-{}".format(uuid.uuid4().hex[:8])
    first = subprocess.run([sys.executable, str(script_path), "第一轮", thread_id],
                           capture_output=True, text=True, env=env, timeout=180)
    assert first.returncode == 0, first.stderr[-800:]
    assert json.loads(first.stdout.strip().splitlines()[-1])["turns"] == 1

    second = subprocess.run([sys.executable, str(script_path), "第二轮", thread_id],
                            capture_output=True, text=True, env=env, timeout=180)
    assert second.returncode == 0, second.stderr[-800:]
    assert json.loads(second.stdout.strip().splitlines()[-1])["turns"] == 2, \
        "新进程应从 Postgres 读回上一轮状态"


def test_psycopg_dsn_from_env():
    assert postgres_dsn() is not None
