"""接口层的应用上下文：装配一次性依赖（检索服务、图、Checkpointer、Store）。

模型加载很贵（embedding + rerank 约 40s），所以进程内有且只有一个 context；
同步接口与流式接口共用同一个 RAGService，只各建一份 runtime（差别仅在
是否开启 token 流式）。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

from agent.graph import build_graph, default_runtime
from agent.review import pending_review, resume_review
from api.logging import log_event
from api.settings import ApiSettings
from persistence.postgres import build_postgres_bundle
from rag_adapter import rag
from rag_adapter.settings import load_env, use_project_root_as_cwd


@dataclass
class AppContext:
    settings: ApiSettings
    service: Any
    graph: Any
    store: Any = None
    checkpointer: Any = None
    bundle: Any = None
    semaphore: threading.BoundedSemaphore = field(
        default_factory=lambda: threading.BoundedSemaphore(2))

    def close(self) -> None:
        if self.bundle is not None:
            self.bundle.close()

    # ---- 问答 ----
    def ask(self, question: str, thread_id: str, user_id: str,
            generate: Optional[bool] = None) -> Dict[str, Any]:
        """同步问答：一次跑完，返回最终 State。"""
        from agent.state import initial_state

        runtime = self._runtime(generate=generate, streaming=False)
        graph = build_graph(runtime, checkpointer=self.checkpointer, store=self.store)
        config = {"configurable": {"thread_id": thread_id}}
        with self.semaphore:
            return graph.invoke(
                initial_state(question, {"thread_id": thread_id,
                                         "user_id": user_id}), config=config)

    def ask_stream(self, question: str, thread_id: str, user_id: str,
                   generate: Optional[bool] = None) -> Iterator[Dict[str, Any]]:
        """SSE 事件流：节点级进度 → token → 最终结果。"""
        from agent.state import initial_state

        runtime = self._runtime(generate=generate, streaming=True)
        graph = build_graph(runtime, checkpointer=self.checkpointer, store=self.store)
        config = {"configurable": {"thread_id": thread_id}}
        state = initial_state(question, {"thread_id": thread_id,
                                        "user_id": user_id})

        yield {"event": "progress",
               "data": {"node": "start", "thread_id": thread_id,
                        "question": question}}
        with self.semaphore:
            for mode, payload in graph.stream(state, config=config,
                                              stream_mode=["updates", "custom"]):
                if mode == "custom":
                    if isinstance(payload, dict) and payload.get("type") == "token":
                        yield {"event": "token", "data": payload}
                    continue
                for node, update in (payload or {}).items():
                    update = update or {}
                    trace = update.get("trace") or []
                    entry = trace[0] if trace else {}
                    yield {"event": "progress",
                           "data": {"node": node,
                                    "status": update.get("status"),
                                    "retry": entry.get("retry",
                                                       update.get("retry_count")),
                                    "budget": entry.get("budget")
                                              or (update.get("retrieval_stats")
                                                  or {}).get("budget"),
                                    "action": update.get("next_action")
                                              or entry.get("action"),
                                    "passed": (update.get("retrieval_score")
                                               or {}).get("passed", entry.get("passed")),
                                    "written": entry.get("written"),
                                    "decision": entry.get("decision")}}
            snapshot = graph.get_state(config)

        values = dict(snapshot.values or {})
        answer = self.result_payload(values)
        answer["pending_review"] = pending_review(graph, thread_id)
        yield {"event": "answer", "data": answer}

    # ---- 人工审核 ----
    def get_pending_review(self, thread_id: str) -> Optional[Dict[str, Any]]:
        return pending_review(self.graph, thread_id)

    def submit_review(self, thread_id: str, review_id: str, decision: str,
                      reviewer: str = "", note: str = "") -> Dict[str, Any]:
        return resume_review(self.graph, thread_id, review_id, decision=decision,
                             reviewer=reviewer, note=note, store=self.store,
                             timeout_seconds=self.settings.review_timeout_seconds)

    # ---- 状态查询 ----
    def thread_state(self, thread_id: str) -> Dict[str, Any]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        values = dict(snapshot.values or {})
        payload = self.result_payload(values)
        payload["pending_review"] = pending_review(self.graph, thread_id)
        payload["has_checkpoint"] = bool(values)
        return payload

    @staticmethod
    def result_payload(values: Dict[str, Any]) -> Dict[str, Any]:
        score = values.get("retrieval_score") or {}
        return {
            "thread_id": values.get("thread_id"),
            "question": values.get("question"),
            "answer": values.get("answer"),
            "status": values.get("status"),
            "retry_count": values.get("retry_count", 0),
            "needs_human_review": values.get("needs_human_review", False),
            "citations": values.get("citations") or [],
            "evidence_count": len(values.get("retrieved_chunks") or []),
            "retrieval": {"passed": score.get("passed"),
                          "reason": score.get("reason"),
                          "documents": score.get("documents") or []},
            "memories": [m.get("content") for m in values.get("memories") or []],
            "memory_written": values.get("memory_written") or [],
            "review_result": values.get("review_result"),
        }

    # ---- 内部 ----
    def _runtime(self, generate: Optional[bool], streaming: bool):
        return default_runtime(
            service=self.service,
            generate_answers=self.settings.generate if generate is None else generate,
            memory_store=self.store,
            memory_user_id=self.settings.default_user,
            review_timeout_seconds=self.settings.review_timeout_seconds,
            interrupt_on_low_confidence=self.settings.interrupt_on_low_confidence,
            enable_token_stream=bool(streaming and self.settings.token_stream),
            use_llm_grader=self.settings.llm_grade,
        )


_context: Optional[AppContext] = None
_context_lock = threading.Lock()


def build_context(settings: Optional[ApiSettings] = None,
                  service: Optional[Any] = None) -> AppContext:
    """装配上下文（模型只加载一次；Postgres 不可用时降级内存）。"""
    use_project_root_as_cwd()
    load_env()
    settings = settings or ApiSettings.from_env()

    service = service or rag.build_service(strategy="recursive", mode="hybrid",
                                          use_rerank=True)
    bundle = build_postgres_bundle()
    checkpointer = bundle.checkpointer if bundle else None
    store = bundle.store if bundle else None
    if bundle is None:
        log_event("postgres_unavailable",
                  detail="线程状态退化为进程内存，人工审核无法跨进程恢复")

    runtime = default_runtime(
        service=service, memory_store=store, memory_user_id=settings.default_user,
        review_timeout_seconds=settings.review_timeout_seconds,
        interrupt_on_low_confidence=settings.interrupt_on_low_confidence)
    graph = build_graph(runtime, checkpointer=checkpointer, store=store)

    log_event("context_ready", generate=settings.generate,
              interrupt=settings.interrupt_on_low_confidence,
              checkpointer=type(checkpointer).__name__ if checkpointer else "InMemorySaver",
              store=type(store).__name__ if store else "None")
    return AppContext(settings=settings, service=service, graph=graph,
                      store=store, checkpointer=checkpointer, bundle=bundle,
                      semaphore=threading.BoundedSemaphore(settings.max_concurrency))


def get_context() -> AppContext:
    global _context
    with _context_lock:
        if _context is None:
            _context = build_context()
        return _context


def set_context(context: AppContext) -> None:
    """测试注入用：替换进程内单例。"""
    global _context
    with _context_lock:
        _context = context
