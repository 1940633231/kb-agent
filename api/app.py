"""FastAPI 应用：问答（同步 / SSE）、线程状态、人工审核、健康检查。

职责边界：这一层只做 HTTP 语义（参数校验、错误映射、SSE 编码、请求级日志），
Agent 逻辑仍在 LangGraph 主图里——接口层不复制判定与重试。

SSE 事件类型：
    progress  节点级进度（understand / retrieve / grade / decide / rewrite /
              generate / remember / human_review / finalize）
    token     生成过程中的文本片段
    answer    最终答案 + 引用 + 检索判定 + 待审信息
    error     失败（同样走事件流，便于前端统一渲染）

为什么进度事件和 token 一起推：本机瓶颈是 CPU 上的 Cross-Encoder 重排
（单题 60~75s），用户最需要看到的是"检索在重试"这类进度，而不是等到最后
才知道结果。
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.logging import (configure_logging, error_payload, log_event,
                         new_request_id)
from api.service import AppContext, get_context, set_context


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    thread_id: str = Field(default="", max_length=64)
    user_id: str = Field(default="", max_length=64)
    generate: Optional[bool] = None


class ReviewRequest(BaseModel):
    review_id: str = Field(min_length=4, max_length=64)
    decision: str = Field(pattern="^(approve|reject|supplement)$")
    reviewer: str = Field(default="", max_length=64)
    note: str = Field(default="", max_length=500)


def _thread_id(requested: str) -> str:
    if requested:
        return requested
    return "t-{}".format(new_request_id())


def _sse(event: str, data: Dict[str, Any]) -> str:
    return "event: {}\ndata: {}\n\n".format(
        event, json.dumps(data, ensure_ascii=False, default=str))


def create_app(context: Optional[AppContext] = None) -> FastAPI:
    configure_logging()
    if context is not None:
        set_context(context)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx = get_context()          # 启动时加载模型并建表（只做一次）
        app.state.context = ctx
        try:
            yield
        finally:
            ctx.close()

    app = FastAPI(title="kb-agent API", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = new_request_id()
        started = time.time()
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - 统一映射为 500 + 结构化错误
            log_event("request_failed", level="error", path=str(request.url.path),
                      error=type(exc).__name__, detail=str(exc)[:300])
            return JSONResponse(status_code=500,
                                content=error_payload("内部错误", type(exc).__name__,
                                                      str(exc)[:300]))
        response.headers["X-Request-Id"] = request_id
        log_event("request_done", path=str(request.url.path),
                  method=request.method, status_code=response.status_code,
                  duration_ms=round((time.time() - started) * 1000, 1))
        return response

    @app.get("/health")
    def health() -> Dict[str, Any]:
        ctx = get_context()
        return {
            "status": "ok",
            "mode": "hybrid+rerank",
            "generate": ctx.settings.generate,
            "interrupt_on_low_confidence": ctx.settings.interrupt_on_low_confidence,
            "checkpointer": type(ctx.checkpointer).__name__ if ctx.checkpointer
            else "InMemorySaver",
            "store": type(ctx.store).__name__ if ctx.store else None,
        }

    @app.post("/ask")
    def ask(payload: AskRequest) -> Dict[str, Any]:
        ctx = get_context()
        thread_id = _thread_id(payload.thread_id)
        started = time.time()
        try:
            state = ctx.ask(payload.question, thread_id,
                            payload.user_id or ctx.settings.default_user,
                            generate=payload.generate)
        except Exception as exc:  # noqa: BLE001
            log_event("ask_failed", level="error", thread_id=thread_id,
                      error=type(exc).__name__, detail=str(exc)[:300])
            raise HTTPException(status_code=500,
                                detail="问答执行失败: {}".format(type(exc).__name__))
        result = ctx.result_payload(dict(state))
        result["pending_review"] = ctx.get_pending_review(thread_id)
        log_event("ask_done", thread_id=thread_id, status=result["status"],
                  retry_count=result["retry_count"],
                  duration_ms=round((time.time() - started) * 1000, 1))
        return result

    @app.post("/ask/stream")
    def ask_stream(payload: AskRequest) -> StreamingResponse:
        ctx = get_context()
        thread_id = _thread_id(payload.thread_id)
        user_id = payload.user_id or ctx.settings.default_user

        def event_stream() -> Iterator[str]:
            started = time.time()
            try:
                for event in ctx.ask_stream(payload.question, thread_id, user_id,
                                            generate=payload.generate):
                    yield _sse(event["event"], event["data"])
            except Exception as exc:  # noqa: BLE001 - 错误也走事件流
                log_event("stream_failed", level="error", thread_id=thread_id,
                          error=type(exc).__name__, detail=str(exc)[:300])
                yield _sse("error", error_payload("流式问答失败",
                                                 type(exc).__name__,
                                                 str(exc)[:300]))
            finally:
                log_event("stream_done", thread_id=thread_id,
                          duration_ms=round((time.time() - started) * 1000, 1))

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no",
                                          "Connection": "keep-alive"})

    @app.get("/threads/{thread_id}")
    def thread_state(thread_id: str) -> Dict[str, Any]:
        ctx = get_context()
        state = ctx.thread_state(thread_id)
        if not state.get("has_checkpoint"):
            raise HTTPException(status_code=404,
                                detail="该线程没有可恢复的状态")
        return state

    @app.get("/threads/{thread_id}/review")
    def get_review(thread_id: str) -> Dict[str, Any]:
        ctx = get_context()
        payload = ctx.get_pending_review(thread_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="该线程没有待审项")
        return payload

    @app.post("/threads/{thread_id}/review")
    def submit_review(thread_id: str, payload: ReviewRequest) -> Dict[str, Any]:
        ctx = get_context()
        result = ctx.submit_review(thread_id, payload.review_id, payload.decision,
                                   reviewer=payload.reviewer, note=payload.note)
        log_event("review_submitted", thread_id=thread_id,
                  review_id=payload.review_id, decision=payload.decision,
                  status=result.get("status"))
        if result.get("status") == "no_pending_review":
            raise HTTPException(status_code=409,
                                detail="该线程没有待审项（可能已被处理）")
        if result.get("status") == "review_id_mismatch":
            raise HTTPException(status_code=409,
                                detail="review_id 与当前待审项不一致")
        if result.get("status") == "expired":
            raise HTTPException(status_code=410, detail="审核已超时")
        return result

    return app


app = create_app()
