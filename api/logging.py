"""结构化日志与请求上下文。

为什么单独做一层：production-rag 的 logger 是面向"单次检索"的可读文本日志，
接口层还需要**请求级**的可检索字段（request_id / thread_id / 耗时 / 状态），
才能把一次 SSE 流的多个事件串成一条链路。
输出一行一个 JSON，方便采集（也方便演示时直接看）。
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

_request_id: ContextVar[str] = ContextVar("request_id", default="")

logger = logging.getLogger("kb_agent.api")


def configure_logging(level: str = "INFO") -> None:
    """配置一行一个 JSON 的 stdout 日志（重复调用安全）。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("kb_agent")
    root.handlers = [handler]
    root.setLevel(level.upper())
    root.propagate = False


def new_request_id() -> str:
    request_id = uuid.uuid4().hex[:12]
    _request_id.set(request_id)
    return request_id


def current_request_id() -> str:
    return _request_id.get()


def log_event(event: str, level: str = "info", **fields: Any) -> Dict[str, Any]:
    """写一条结构化日志，返回落库/返回给调用方的同一条记录。"""
    record: Dict[str, Any] = {
        "ts": round(time.time(), 3),
        "event": event,
        "request_id": current_request_id(),
    }
    record.update(fields)
    getattr(logger, level, logger.info)(json.dumps(record, ensure_ascii=False))
    return record


def error_payload(message: str, error_type: str,
                  detail: Optional[str] = None) -> Dict[str, Any]:
    payload = {"error": error_type, "message": message,
               "request_id": current_request_id()}
    if detail:
        payload["detail"] = detail
    return payload
