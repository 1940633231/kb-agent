"""检索子图：只做"检索 + 合并去重 + 返回候选与检索统计"。

边界（v2 §2.2 / §4.1，已定死）：
    - 子图**不做**相关性判定、**不自环重试**，因此它是无状态、可独立单测的；
    - 证据是否足够由主图的 grade_retrieval 判定，重试由 decide_next_step 决定；
    - 子图返回的 stats 只是"每路查了几个、合并后几个"这类检索统计，
      不是"够不够"的结论。

多路输入：主图第一次调用只传原始查询；重试时传改写后的多条子查询，
子图逐条检索后按 chunk_id 合并（同 chunk 取最高分）。
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_adapter import rag


class RetrievalState(TypedDict, total=False):
    """子图自己的状态，与主图 State 隔离，便于单独测试。"""

    queries: List[str]
    anchor: str          # 用于统一重排的锚点查询（通常是用户原始问题）
    top_k: int
    candidates: List[Dict[str, Any]]
    stats: Dict[str, Any]


def merge_candidates(candidates: List[Dict]) -> List[Dict]:
    """按 chunk_id 去重并保留最高分，按分数降序返回。"""
    best: Dict[str, Dict] = {}
    for hit in candidates:
        key = str(hit.get("chunk_id") or hit.get("vector_id"))
        current = best.get(key)
        if current is None or float(hit.get("score", 0.0)) > float(
                current.get("score", 0.0)):
            best[key] = hit
    return sorted(best.values(), key=lambda h: float(h.get("score", 0.0)),
                  reverse=True)


def build_retrieval_subgraph(runtime: Any):
    """构建并编译检索子图（返回可直接 invoke 的 CompiledStateGraph）。"""

    def plan_queries(state: RetrievalState) -> Dict[str, Any]:
        """规整查询列表：去空、去重、保序。"""
        queries: List[str] = []
        for query in state.get("queries") or []:
            text = (query or "").strip()
            if text and text not in queries:
                queries.append(text)
        return {"queries": queries}

    def search(state: RetrievalState) -> Dict[str, Any]:
        """多路查询一次检索（合并去重 + 单次重排），记录检索统计。"""
        top_k = int(state.get("top_k") or runtime.top_k)
        queries = state.get("queries") or []
        hits = rag.retrieve_multi(runtime.service, queries, top_k=top_k,
                                  anchor=state.get("anchor"))
        return {
            "candidates": hits,
            "stats": {"queries": list(queries), "returned": len(hits),
                      "top_k": top_k,
                      "top_score": float(hits[0].get("score", 0.0)) if hits else None},
        }

    builder = StateGraph(RetrievalState)
    builder.add_node("plan_queries", plan_queries)
    builder.add_node("search", search)
    builder.add_edge(START, "plan_queries")
    builder.add_edge("plan_queries", "search")
    builder.add_edge("search", END)
    return builder.compile()
