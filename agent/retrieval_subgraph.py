"""检索子图：只做"检索 + 合并去重 + 返回候选与检索统计"。

边界（v2 §2.2 / §4.1，已定死）：
    - 子图**不做**相关性判定、**不自环重试**，因此它是无状态、可独立单测的；
    - 证据是否足够由主图的 grade_retrieval 判定，重试由 decide_next_step 决定；
    - 子图返回的 stats 只是"每路查了几个、合并后几个"这类检索统计，
      不是"够不够"的结论。

多路输入：首轮传检索前公司/年份路由，重试只传缺失义务对应的路由；
关闭预路由时仍兼容旧的多条字符串子查询。子图逐路检索后按 chunk_id
合并（同 chunk 取最高分）。
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.evidence_contract import (
    assess_route_coverage,
    preserve_obligation_evidence,
)
from rag_adapter import rag


class RetrievalState(TypedDict, total=False):
    """子图自己的状态，与主图 State 隔离，便于单独测试。"""

    queries: List[str]
    routes: List[Dict[str, Any]]
    routing_enabled: bool
    strict_route_filter: bool
    route_candidate_limit: int
    route_document_ids: Dict[str, List[str]]
    route_obligations: Dict[str, List[str]]
    anchor: str          # 用于统一重排的锚点查询（通常是用户原始问题）
    top_k: int
    obligations: List[Dict[str, Any]]
    route_prefix: str
    candidates: List[Dict[str, Any]]
    route_hits: List[Dict[str, Any]]
    route_coverage: List[Dict[str, Any]]
    stats: Dict[str, Any]


def merge_candidates(candidates: List[Dict]) -> List[Dict]:
    """按 chunk_id 去重并保留最高分，按分数降序返回。"""
    best: Dict[str, Dict] = {}
    for hit in candidates:
        key = str(hit.get("chunk_id") or hit.get("vector_id"))
        current = best.get(key)
        if current is None or float(hit.get("score", 0.0)) > float(
                current.get("score", 0.0)):
            merged = dict(hit)
            if current is not None:
                merged["retrieval_routes"] = sorted(set(
                    current.get("retrieval_routes") or []
                ) | set(hit.get("retrieval_routes") or []))
                merged["route_obligations"] = sorted(set(
                    current.get("route_obligations") or []
                ) | set(hit.get("route_obligations") or []))
            best[key] = merged
        else:
            current["retrieval_routes"] = sorted(set(
                current.get("retrieval_routes") or []
            ) | set(hit.get("retrieval_routes") or []))
            current["route_obligations"] = sorted(set(
                current.get("route_obligations") or []
            ) | set(hit.get("route_obligations") or []))
    return sorted(best.values(), key=lambda h: float(h.get("score", 0.0)),
                  reverse=True)


def build_retrieval_subgraph(runtime: Any):
    """构建并编译检索子图（返回可直接 invoke 的 CompiledStateGraph）。"""

    def plan_queries(state: RetrievalState) -> Dict[str, Any]:
        """规整查询列表：去空、去重、保序。"""
        queries: List[str] = []
        raw_queries = list(state.get("queries") or [])
        for route in state.get("routes") or []:
            if route.get("query"):
                raw_queries.append(route["query"])
        for query in raw_queries:
            text = (query or "").strip()
            if text and text not in queries:
                queries.append(text)
        return {"queries": queries}

    def search(state: RetrievalState) -> Dict[str, Any]:
        """多路查询一次检索（合并去重 + 单次重排），记录检索统计。"""
        top_k = int(state.get("top_k") or runtime.top_k)
        queries = state.get("queries") or []
        obligations = state.get("obligations") or []
        routing_enabled = bool(state.get("routing_enabled", False))
        route_specs = list(state.get("routes") or [])
        if not routing_enabled:
            route_specs = [{
                "query_id": "route-{}".format(index),
                "query": query,
                "obligation_ids": [],
                "document_ids": [],
                "unrouted": True,
            } for index, query in enumerate(queries, start=1)]
        if routing_enabled and not route_specs:
            hits, route_hits = [], []
        else:
            default_limit = max(20, min(50, top_k * 5))
            hits, route_hits = rag.retrieve_routes_with_evidence(
                runtime.service,
                route_specs,
                top_k=top_k,
                anchor=state.get("anchor"),
                route_candidate_limit=int(
                    state.get("route_candidate_limit")
                    or getattr(runtime, "route_candidate_limit", default_limit)
                ),
                strict_route_filter=bool(
                    state.get("strict_route_filter", True)
                    if routing_enabled
                    else False
                ),
            )
        prefix = str(state.get("route_prefix") or "")
        if prefix:
            for route in route_hits:
                old_id = str(route.get("query_id"))
                route["query_id"] = prefix + old_id
                for hit in route.get("hits") or []:
                    hit["retrieval_routes"] = [
                        prefix + str(item)
                        for item in hit.get("retrieval_routes") or []
                    ]
            for hit in hits:
                hit["retrieval_routes"] = [
                    prefix + str(item)
                    for item in hit.get("retrieval_routes") or []
                ]
        hits = merge_candidates(hits)
        hits, obligation_fills = preserve_obligation_evidence(
            hits, route_hits, obligations)
        selected_route_hits = []
        for route in route_hits:
            route_id = str(route.get("query_id"))
            selected = dict(route)
            selected["hits"] = [
                hit for hit in hits
                if route_id in (hit.get("retrieval_routes") or [])
            ]
            selected_route_hits.append(selected)
        route_coverage = assess_route_coverage(
            selected_route_hits, obligations)
        coverage_by_route = {
            str(route.get("query_id")): route
            for route in route_coverage
        }
        per_query_returned = {
            route["query_id"]: route.get("returned", 0)
            for route in route_hits
        }
        per_query_candidates = {
            route["query_id"]: route.get("candidate_count", 0)
            for route in route_hits
        }
        per_query_documents = {
            route["query_id"]: sorted({
                str(hit.get("document_id")
                    or (hit.get("metadata") or {}).get("document_id") or "")
                for hit in route.get("hits") or []
                if hit.get("document_id")
                or (hit.get("metadata") or {}).get("document_id")
            })
            for route in route_hits
        }
        per_route_obligations_covered = {
            route["query_id"]: (
                coverage_by_route.get(str(route["query_id"]), {}).get("covered")
                or []
            )
            for route in route_hits
        }
        per_route_obligations_missing = {
            route["query_id"]: (
                coverage_by_route.get(str(route["query_id"]), {}).get("missing")
                or []
            )
            for route in route_hits
        }
        route_filter_degraded = any(
            bool(route.get("route_filter_degraded"))
            for route in route_hits
        )
        route_filter_missing = sorted({
            obligation_id
            for route in route_hits
            if not route.get("unrouted") and not route.get("document_ids")
            for obligation_id in route.get("obligation_ids") or []
        })
        return {
            "candidates": hits,
            "route_hits": route_hits,
            "route_coverage": route_coverage,
            "stats": {
                "queries": list(queries),
                "returned": len(hits),
                "top_k": top_k,
                "routing_enabled": routing_enabled,
                "routes_planned": len(route_hits),
                "routes_with_hits": sum(
                    1 for route in route_hits if route.get("returned")
                ),
                "routes_empty": sum(
                    1 for route in route_hits if not route.get("returned")
                ),
                "route_filter_degraded": route_filter_degraded,
                "route_filter_missing": route_filter_missing,
                "obligation_fills": obligation_fills,
                "per_query_returned": per_query_returned,
                "per_query_candidates": per_query_candidates,
                "per_query_documents": per_query_documents,
                "per_route_candidates": per_query_candidates,
                "per_route_documents": per_query_documents,
                "per_route_obligations_covered":
                    per_route_obligations_covered,
                "per_route_obligations_missing":
                    per_route_obligations_missing,
                "route_coverage": route_coverage,
                "top_score": (
                    float(hits[0].get("score", 0.0)) if hits else None
                ),
            },
        }

    builder = StateGraph(RetrievalState)
    builder.add_node("plan_queries", plan_queries)
    builder.add_node("search", search)
    builder.add_edge(START, "plan_queries")
    builder.add_edge("plan_queries", "search")
    builder.add_edge("search", END)
    return builder.compile()
