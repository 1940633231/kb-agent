"""主图 State：只保留后续节点真正需要的字段，避免 State 无限膨胀（v2 §4.2）。

Reducer 约定：
    - 累积字段用 Annotated[...] 显式声明 reducer（messages / trace /
      rewrite_history），否则 LangGraph 会用后写覆盖前值，历史丢失。
    - 其余字段是"当前轮的结论"（retrieved_chunks / retrieval_score /
      retry_count / status ...），按覆盖语义更新，语义更清晰、体积不增长。
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # 当前线程的对话消息（多轮上下文；由 add_messages 累积，不覆盖历史）
    messages: Annotated[List[Any], add_messages]

    # 本轮问题与检索用的查询
    question: str
    thread_id: str
    rewritten_query: str
    sub_queries: List[str]
    rewrite_history: Annotated[List[Dict[str, Any]], operator.add]

    # 当前证据与评分（覆盖写：只反映"当前这轮"）
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]
    retrieval_score: Dict[str, Any]
    retry_count: int
    next_action: str

    # 问题理解结果（facets 参与评分与改写，needs_retrieval 决定是否检索）
    facets: Dict[str, Any]
    needs_retrieval: bool

    # 跨会话长期记忆（第二阶段）：召回到的偏好 + 本次写入的记忆
    memories: List[Dict[str, Any]]
    memory_written: List[Dict[str, Any]]

    # 答案与引用
    answer: Optional[str]
    citations: List[Dict[str, Any]]
    # 生成时真正送进 prompt 的 chunks（[1][2] 编号与它一一对应）
    context_chunks: List[Dict[str, Any]]
    # 答案落地校验（数字是否能在"含问题指标词"的被引片段里找到）
    grounding: Dict[str, Any]

    # 人工介入（第二阶段接 Interrupt/Command，本阶段只记录待审标记）
    needs_human_review: bool
    review_result: Optional[Dict[str, Any]]

    status: str
    trace: Annotated[List[Dict[str, Any]], operator.add]


def initial_state(question: str,
                  thread_extra: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """构造一轮问答的初始 State。"""
    state: Dict[str, Any] = {
        "messages": [{"role": "user", "content": question}],
        "question": question,
        "thread_id": "",
        "rewritten_query": question,
        "sub_queries": [],
        "rewrite_history": [],
        "retrieved_chunks": [],
        "retrieval_stats": {},
        "retrieval_score": {},
        "retry_count": 0,
        "next_action": "",
        "facets": {},
        "needs_retrieval": True,
        "memories": [],
        "memory_written": [],
        "answer": None,
        "citations": [],
        "context_chunks": [],
        "grounding": {},
        "needs_human_review": False,
        "review_result": None,
        "status": "received",
        "trace": [],
    }
    if thread_extra:
        state.update(thread_extra)
    return state
