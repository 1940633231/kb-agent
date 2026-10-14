"""人工审核：Interrupt 暂停 → Command 恢复，含幂等与超时。

为什么需要幂等：审核提交来自网络（可能重试、可能重复点击）。同一条待审项
重复提交必须返回同一个结果，不能推进图状态两次。做法是以 review_id 为键把
处理结果写进 Store，恢复前先查这张表。

为什么校验 review_id：待审项可能已经不是当前挂起的这一条（用户看的是旧页面）。
不一致时拒绝恢复，避免把审核结果套到另一个问题上。

并发：进程内用锁串行化同一 thread 的提交；多进程/多副本需要数据库层互斥
（SELECT ... FOR UPDATE 或审核表唯一约束），属于"明确不达标"范围。
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

APP_NS = "kb-agent"
REVIEW_NS = (APP_NS, "reviews")

_submit_lock = threading.Lock()


def review_namespace() -> Tuple[str, ...]:
    return REVIEW_NS


def build_review_payload(state: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
    """构造待审 payload（随 interrupt 一起暴露给调用方）。"""
    score = state.get("retrieval_score") or {}
    return {
        "review_id": uuid.uuid4().hex[:12],
        "thread_id": thread_id,
        "question": state.get("question"),
        "reason": score.get("reason", ""),
        "retry_count": state.get("retry_count", 0),
        "evidence": [
            {"document_id": hit.get("document_id"), "score": hit.get("score"),
             "snippet": (hit.get("content") or "")[:120]}
            for hit in (state.get("retrieved_chunks") or [])[:5]
        ],
        "draft_answer": state.get("answer"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _pending_interrupt(graph: Any, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """读取当前线程挂起的 interrupt payload（没有则返回 None）。"""
    snapshot = graph.get_state(config)
    for task in (getattr(snapshot, "tasks", None) or []):
        for item in (getattr(task, "interrupts", None) or []):
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                return value
    return None


def pending_review(graph: Any, thread_id: str) -> Optional[Dict[str, Any]]:
    """外部查询接口：这个线程是否在等人工审核、等的是什么。"""
    return _pending_interrupt(graph, {"configurable": {"thread_id": thread_id}})


def _load_record(store: Any, review_id: str) -> Optional[Dict[str, Any]]:
    if store is None:
        return None
    try:
        item = store.get(REVIEW_NS, review_id)
    except Exception:  # noqa: BLE001
        return None
    if item is None:
        return None
    return dict(getattr(item, "value", {}) or {})


def _save_record(store: Any, review_id: str, record: Dict[str, Any]) -> None:
    if store is None:
        return
    try:
        store.put(REVIEW_NS, review_id, record)
    except Exception:  # noqa: BLE001 - 幂等表写失败不应阻塞流程
        pass


def resume_review(graph: Any, thread_id: str, review_id: str,
                  decision: str, reviewer: str = "", note: str = "",
                  store: Any = None,
                  timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
    """提交人工审核结果并恢复图执行。

    decision: approve（采纳草稿继续）| reject（驳回）| supplement（补充信息）
    返回 status ∈ resumed / already_processed / no_pending_review /
                  review_id_mismatch / expired
    """
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}

    with _submit_lock:
        previous = _load_record(store, review_id)
        if previous:
            return dict(previous, status="already_processed")

        payload = _pending_interrupt(graph, config)
        if payload is None:
            return {"status": "no_pending_review", "thread_id": thread_id}
        if payload.get("review_id") != review_id:
            return {"status": "review_id_mismatch",
                    "expected": payload.get("review_id"), "received": review_id}

        if timeout_seconds is not None and payload.get("created_at"):
            try:
                age = (datetime.now(timezone.utc)
                       - datetime.fromisoformat(payload["created_at"])).total_seconds()
            except ValueError:
                age = 0.0
            if age > timeout_seconds:
                result = {"status": "expired", "review_id": review_id,
                          "age_seconds": round(age, 1), "reviewer": reviewer}
                _save_record(store, review_id, result)
                return result

        submission = {
            "review_id": review_id,
            "decision": decision,
            "reviewer": reviewer,
            "note": note,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        state = graph.invoke(Command(resume=submission), config=config)
        result = {
            "status": "resumed",
            "review_id": review_id,
            "thread_id": thread_id,
            "decision": decision,
            "reviewer": reviewer,
            "answer": state.get("answer"),
            "final_status": state.get("status"),
            "citations": state.get("citations") or [],
        }
        _save_record(store, review_id, result)
        return result
