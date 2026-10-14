"""接口层演示客户端：跑通方案 §6.3 要求的四个场景。

前提：服务已启动（python -m api），且 Postgres 在跑（人工审核要跨进程恢复）。

    python scripts/demo_api.py
    python scripts/demo_api.py --base-url http://127.0.0.1:8300 --skip-review

四个场景：
    1 正常问答（同步）       POST /ask
    2 检索重试（SSE 进度）    POST /ask/stream（能看到 rewrite / retry 进度事件）
    3 重启后恢复             GET  /threads/{id}（服务重启后线程状态仍在）
    4 人工审核               POST /ask --interrupt → GET/POST /threads/{id}/review
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, Iterator

import requests

NORMAL_QUESTION = "贵州茅台2024年的销售费用是多少元？"
MULTI_QUESTION = "宁德时代2023年与2025年的营业收入分别是多少？"
RETRY_QUESTION = "比亚迪2024年的营业收入是多少？"
REFUSE_QUESTION = "华为2024年的研发投入是多少？"


def sse_events(url: str, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """读取 SSE 流并逐事件 yield（不依赖第三方 SSE 库）。"""
    with requests.post(url, json=payload, stream=True, timeout=900) as response:
        response.raise_for_status()
        event_name, data_lines = None, []
        for raw in response.iter_lines(decode_unicode=True):
            line = raw or ""
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: "):])
            elif not line.strip() and event_name:
                body = "\n".join(data_lines)
                try:
                    yield {"event": event_name, "data": json.loads(body)}
                except json.JSONDecodeError:
                    yield {"event": event_name, "data": {"raw": body}}
                event_name, data_lines = None, []


def show_answer(result: Dict[str, Any]) -> None:
    print("  状态: {} | 重试: {} | 证据: {} 片段".format(
        result.get("status"), result.get("retry_count"), result.get("evidence_count")))
    print("  检索判定: {} ({})".format(
        (result.get("retrieval") or {}).get("passed"),
        (result.get("retrieval") or {}).get("reason")))
    answer = (result.get("answer") or "").strip().replace("\n", " ")
    print("  答案: {}".format(answer[:160]))
    for citation in result.get("citations") or []:
        print("  引用 [{}] {} p.{}".format(citation.get("number"),
                                          citation.get("document_title") or
                                          citation.get("file_name"),
                                          citation.get("page")))


def main() -> int:
    parser = argparse.ArgumentParser(description="kb-agent API demo")
    parser.add_argument("--base-url", default="http://127.0.0.1:8300")
    parser.add_argument("--thread", default="demo-{}".format(int(time.time())))
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--skip-stream", action="store_true")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print("== 0. 健康检查 ==")
    health = requests.get(base + "/health", timeout=30).json()
    print("  {}".format(json.dumps(health, ensure_ascii=False)))

    print("\n== 1. 正常问答（同步）==")
    started = time.time()
    result = requests.post(base + "/ask",
                           json={"question": NORMAL_QUESTION,
                                 "thread_id": args.thread},
                           timeout=900).json()
    show_answer(result)
    print("  耗时 {:.1f}s".format(time.time() - started))

    if not args.skip_stream:
        print("\n== 2. 检索重试（SSE 进度 + token）==")
        print("  问题: {}".format(MULTI_QUESTION))
        tokens, retries, final = 0, [], None
        for event in sse_events(base + "/ask/stream",
                                {"question": MULTI_QUESTION,
                                 "thread_id": args.thread + "-sse"}):
            if event["event"] == "progress":
                node = event["data"].get("node")
                extra = ""
                if node == "decide_next_step":
                    extra = " action={}".format(event["data"].get("action"))
                if node == "retrieve_knowledge":
                    retries.append(event["data"].get("retry"))
                    extra = " retry={} budget={}".format(event["data"].get("retry"),
                                                         event["data"].get("budget"))
                print("  [progress] {}{}".format(node, extra))
            elif event["event"] == "token":
                tokens += 1
            elif event["event"] == "answer":
                final = event["data"]
            elif event["event"] == "error":
                print("  [error] {}".format(event["data"]))
        if final:
            show_answer(final)
        print("  收到 {} 个 token 事件，检索轮次: {}".format(tokens, retries))

        print("\n  同一接口再跑一道证据不足的题，看重试与放宽预算的事件：")
        retry_seen = []
        for event in sse_events(base + "/ask/stream",
                                {"question": RETRY_QUESTION,
                                 "thread_id": args.thread + "-retry"}):
            if event["event"] == "progress" and event["data"].get("node") in (
                    "retrieve_knowledge", "rewrite_query", "decide_next_step"):
                print("  [progress] {} retry={} budget={} action={} decision={}".format(
                    event["data"].get("node"), event["data"].get("retry"),
                    event["data"].get("budget"), event["data"].get("action"),
                    event["data"].get("decision")))
                if event["data"].get("node") == "retrieve_knowledge":
                    retry_seen.append(event["data"].get("budget"))
            elif event["event"] == "answer":
                print("  最终状态: {} | 待审: {}".format(
                    event["data"].get("status"),
                    bool(event["data"].get("pending_review"))))
        print("  各轮检索预算: {}".format(retry_seen))

    print("\n== 3. 线程状态查询（模拟重启后恢复）==")
    state = requests.get("{}/threads/{}".format(base, args.thread), timeout=60)
    if state.status_code == 200:
        payload = state.json()
        print("  可恢复: {} | 状态: {} | 证据: {}".format(
            payload.get("has_checkpoint"), payload.get("status"),
            payload.get("evidence_count")))
    else:
        print("  {} {}".format(state.status_code, state.text[:120]))

    if args.skip_review:
        return 0

    print("\n== 4. 人工审核（低置信度 → 挂起 → 提交 → 恢复）==")
    review_thread = args.thread + "-review"
    body = requests.post(base + "/ask",
                         json={"question": REFUSE_QUESTION,
                               "thread_id": review_thread}, timeout=900).json()
    pending = body.get("pending_review")
    if not pending:
        print("  该问题没有挂起（可能未开启 KB_AGENT_INTERRUPT）")
        return 0
    print("  挂起: review_id={} reason={}".format(pending["review_id"],
                                                 pending["reason"]))
    print("  草稿: {}".format((pending.get("draft_answer") or "")[:100]))

    fetched = requests.get("{}/threads/{}/review".format(base, review_thread),
                           timeout=60).json()
    assert fetched["review_id"] == pending["review_id"]

    resumed = requests.post("{}/threads/{}/review".format(base, review_thread),
                            json={"review_id": pending["review_id"],
                                  "decision": "reject", "reviewer": "demo",
                                  "note": "语料确实不含该公司"}, timeout=900).json()
    print("  审核结果: {} → {}".format(resumed.get("status"),
                                      resumed.get("final_status")))
    again = requests.post("{}/threads/{}/review".format(base, review_thread),
                          json={"review_id": pending["review_id"],
                                "decision": "approve"}, timeout=900)
    print("  重复提交: HTTP {} {}".format(
        again.status_code, (again.json() or {}).get("status")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
