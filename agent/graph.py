"""主图装配：State + 节点 + 普通边 + 条件边（v2 §4.1）。

流程：
    START → understand_query ─(需要检索?)→ retrieve_knowledge → grade_retrieval
          → decide_next_step ─(证据不足且未超限)→ rewrite_query → retrieve_knowledge
                             ─(证据充分 / 超限后带保留作答)→ generate_answer
                             ─(无任何证据)→ finalize_response
                             ─(开启人工审核)→ human_review → finalize_response
          → finalize_response → END

Checkpointer：本阶段用 InMemorySaver；第二阶段换成 Postgres 实现即可，
build_graph 的 checkpointer 参数就是为此留的注入点。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (build_nodes, route_after_decide,
                         route_after_generate, route_after_understand)
from agent.rewriter import LLMRewriter
from agent.runtime import AgentRuntime
from agent.state import AgentState, initial_state
from rag_adapter import rag
from rag_adapter.settings import load_env


def build_graph(runtime: AgentRuntime, checkpointer: Optional[Any] = None,
                store: Optional[Any] = None):
    """编译主图。

    checkpointer：线程状态（默认 InMemorySaver；第二阶段用 PostgresSaver）；
    store：跨会话长期记忆（默认 None；第二阶段用 PostgresStore，通过
    runtime.memory_store 注入节点，compile(store=...) 让图本身也持有它）。
    """
    nodes = build_nodes(runtime)

    builder = StateGraph(AgentState)
    for name, node in nodes.items():
        builder.add_node(name, node)

    builder.add_edge(START, "understand_query")
    builder.add_conditional_edges("understand_query", route_after_understand, {
        "retrieve_knowledge": "retrieve_knowledge",
        "remember_preference": "remember_preference",
        "finalize_response": "finalize_response",
    })
    builder.add_edge("retrieve_knowledge", "grade_retrieval")
    builder.add_edge("grade_retrieval", "decide_next_step")
    builder.add_conditional_edges("decide_next_step", route_after_decide, {
        "rewrite_query": "rewrite_query",
        "generate_answer": "generate_answer",
        "finalize_response": "finalize_response",
    })
    builder.add_edge("rewrite_query", "retrieve_knowledge")
    builder.add_conditional_edges("generate_answer", route_after_generate, {
        "human_review": "human_review",
        "remember_preference": "remember_preference",
    })
    # human_review 里用 interrupt() 暂停；Command(resume=...) 恢复后继续走到 finalize
    builder.add_edge("human_review", "finalize_response")
    builder.add_edge("remember_preference", "finalize_response")
    builder.add_edge("finalize_response", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver(),
                           store=store)


def default_runtime(strategy: str = "recursive",
                    use_llm_rewrite: bool = True,
                    generate_answers: bool = True,
                    max_retries: int = 2,
                    service: Optional[Any] = None,
                    memory_store: Optional[Any] = None,
                    memory_user_id: str = "",
                    review_timeout_seconds: Optional[float] = 3600.0,
                    **extra: Any) -> AgentRuntime:
    """默认运行时：L2 检索配置（hybrid + rerank）+ LLM 改写。

    用 L2 配置是因为 §6.4 的 L3 = L2 + 自纠错，只有检索配置一致，
    增量才能归因到编排层。
    """
    load_env()
    # service 可注入：同一进程里要为同步接口与流式接口各建一份运行时，
    # 不该因此重复加载 embedding / rerank 模型
    service = service or rag.build_service(strategy=strategy, mode="hybrid",
                                          use_rerank=True)
    page_offsets = rag.load_page_offsets(strategy)
    rewriter = None
    if use_llm_rewrite:
        generator = service._pipeline.generator  # noqa: SLF001
        if generator is not None:
            rewriter = LLMRewriter(generator)
    return AgentRuntime(service=service, strategy=strategy, max_retries=max_retries,
                        rewriter=rewriter, generate_answers=generate_answers,
                        memory_store=memory_store, memory_user_id=memory_user_id,
                        review_timeout_seconds=review_timeout_seconds,
                        page_offsets=page_offsets,
                        extra=dict(extra))


def ask(graph: Any, question: str, thread_id: str = "demo",
        **kwargs: Any) -> Dict[str, Any]:
    """跑一问：同一 thread_id 下多次调用即多轮对话（依赖 checkpointer）。"""
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(initial_state(question, {"thread_id": thread_id}),
                        config=config, **kwargs)
