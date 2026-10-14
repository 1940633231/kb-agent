"""接口层测试：同步问答、SSE 事件流、人工审核、错误映射（全部用假服务）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# kb-agent 同进程 import production-rag（app 包）；CI 里没有它时整模块跳过
pytest.importorskip("app.core.config", reason="需要 production-rag（app 包）")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.graph import build_graph, default_runtime  # noqa: E402
from api.app import create_app  # noqa: E402
from api.service import AppContext  # noqa: E402
from api.settings import ApiSettings  # noqa: E402
from fakes import DictStore, FakeGenerator, FakeService, maotai_hit  # noqa: E402

QUESTION = "贵州茅台2023年的营业收入是多少元？"


def parse_sse(body: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for block in body.split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        event = {"event": None, "data": None}
        for line in lines:
            if line.startswith("event: "):
                event["event"] = line[len("event: "):]
            elif line.startswith("data: "):
                event["data"] = json.loads(line[len("data: "):])
        events.append(event)
    return events


def build_context(resolver=None, interrupt: bool = False, token_stream: bool = True
                  ) -> AppContext:
    resolver = resolver or (lambda query: [maotai_hit(2023)])
    service = FakeService(resolver, generator=FakeGenerator())
    settings = ApiSettings(generate=True, interrupt_on_low_confidence=interrupt,
                           token_stream=token_stream)
    runtime = default_runtime(service=service, memory_user_id="tester",
                              interrupt_on_low_confidence=interrupt,
                              review_timeout_seconds=None)
    # 关键：AppContext.ask() 会用 self.checkpointer 另建图，
    # 测试里必须让两处共用同一个 saver，否则挂起的审核读不到
    saver = InMemorySaver()
    store = DictStore()
    return AppContext(settings=settings, service=service, store=store,
                      graph=build_graph(runtime, checkpointer=saver),
                      checkpointer=saver)


@pytest.fixture
def client():
    with TestClient(create_app(build_context())) as test_client:
        yield test_client


def test_health_reports_checkpointer_and_store(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["checkpointer"] == "InMemorySaver"
    assert body["interrupt_on_low_confidence"] is False


def test_ask_returns_answer_with_citations(client):
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 200
    body = response.json()

    assert "147,693,604,994.14" in body["answer"]
    assert body["status"] == "answered"
    assert body["retrieval"]["passed"] is True
    assert body["citations"], "答案里的 [1] 应被解析成引用"
    assert body["citations"][0]["document_id"] == "600519-2023"
    assert body["thread_id"], "未指定 thread_id 时由服务端生成"
    assert response.headers["X-Request-Id"]


def test_ask_stream_emits_progress_token_and_answer(client):
    with client.stream("POST", "/ask/stream", json={"question": QUESTION,
                                                    "thread_id": "s-1"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in resp.iter_text())

    events = parse_sse(body)
    kinds = [event["event"] for event in events]
    assert kinds[0] == "progress"
    assert "token" in kinds
    assert kinds[-1] == "answer"

    nodes = [event["data"]["node"] for event in events
             if event["event"] == "progress" and event["data"].get("node")]
    assert "understand_query" in nodes
    assert "retrieve_knowledge" in nodes
    assert "finalize_response" in nodes

    tokens = "".join(event["data"]["text"] for event in events
                     if event["event"] == "token")
    answer_event = events[-1]["data"]
    assert tokens.strip() == (answer_event["answer"] or "").strip()
    assert answer_event["citations"]


def test_stream_without_token_streaming_still_answers():
    context = build_context(token_stream=False)
    with TestClient(create_app(context)) as test_client:
        with test_client.stream("POST", "/ask/stream",
                                json={"question": QUESTION}) as resp:
            events = parse_sse("".join(chunk for chunk in resp.iter_text()))
    assert events[-1]["event"] == "answer"
    assert events[-1]["data"]["answer"]
    assert not [e for e in events if e["event"] == "token"]


def test_review_flow_over_http():
    # 让检索结果与问题不匹配，触发低置信度 → 挂起人工审核
    context = build_context(resolver=lambda q: [maotai_hit(2025)], interrupt=True)
    with TestClient(create_app(context)) as test_client:
        body = test_client.post("/ask", json={"question": QUESTION,
                                             "thread_id": "rev-1"}).json()
        pending = body["pending_review"]
        assert pending and pending["review_id"]
        assert body["needs_human_review"] is True
        assert pending["draft_answer"], "挂起前应有草稿答案"

        fetched = test_client.get("/threads/rev-1/review").json()
        assert fetched["review_id"] == pending["review_id"]

        resumed = test_client.post("/threads/rev-1/review", json={
            "review_id": pending["review_id"], "decision": "approve",
            "reviewer": "alice"}).json()
        assert resumed["status"] == "resumed"
        assert resumed["final_status"] == "review_approved"

        # 幂等：同一 review_id 再提交返回同一结果，不再是待审状态
        again = test_client.post("/threads/rev-1/review", json={
            "review_id": pending["review_id"], "decision": "reject"}).json()
        assert again["status"] == "already_processed"
        assert test_client.get("/threads/rev-1/review").status_code == 404

        state = test_client.get("/threads/rev-1").json()
        assert state["status"] == "review_approved"
        assert state["has_checkpoint"] is True


def test_review_id_mismatch_returns_409():
    context = build_context(resolver=lambda q: [maotai_hit(2025)], interrupt=True)
    with TestClient(create_app(context)) as test_client:
        test_client.post("/ask", json={"question": QUESTION, "thread_id": "rev-2"})
        response = test_client.post("/threads/rev-2/review", json={
            "review_id": "deadbeef", "decision": "approve"})
    assert response.status_code == 409


def test_unknown_thread_returns_404(client):
    assert client.get("/threads/nope").status_code == 404
    assert client.get("/threads/nope/review").status_code == 404


def test_invalid_request_returns_422(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/threads/t-1/review", json={
        "review_id": "abcdef", "decision": "later"}).status_code == 422
