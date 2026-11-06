"""主图节点实现。

职责边界（v2 §4.1）：判定（grade_retrieval）与重试（decide_next_step /
rewrite_query）全部在主图，检索子图只负责取回候选；所有重试痕迹
（retry_count、改写后的查询、每轮评分）都写进 State，随 checkpointer 落库，
便于回放与调试。
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

from app.citation.citation import CitationExtractor
from langgraph.types import interrupt

from agent.answer_audit import audit_answer
from agent.context_budget import build_context_with_budget
from agent.grading import extract_facets, grade_evidence, needs_knowledge_base
from agent.evidence_contract import (
    assess_obligations,
    bind_routes_to_obligations,
    build_evidence_contract,
    build_initial_routes,
    obligation_query,
    preserve_obligation_evidence,
    route_for_missing_obligations,
)
from agent.evidence import scope_evidence
from agent.grounding import verify_grounding
from agent.repair import repair_claims
from agent.memory import extract_memory, format_memories, recall, remember
from agent.review import build_review_payload
from agent.retrieval_subgraph import build_retrieval_subgraph
from agent.rewriter import RuleRewriter
from rag_adapter.rag import (
    accumulate_evidence,
    cap_evidence_preserving,
    enrich_citation,
)

REFUSAL_NO_EVIDENCE = (
    "根据现有检索结果无法回答该问题：知识库中没有检索到覆盖该问题的证据，"
    "因此不做推测性回答。"
)
OUT_OF_SCOPE = (
    "该问题没有可用于检索的实体或指标（如公司名、年份、财务指标），"
    "知识库检索未执行，无法基于证据回答。"
)
PENDING_REVIEW = "该问题证据不足，已标记为待人工审核。"
REJECTED_ANSWER = "该回答未通过人工审核，已驳回，不作为最终结论。"
MEMORY_ACK = "已记录这条长期偏好，后续会话会默认遵循。"
MEMORY_UNAVAILABLE = ("识别到这是一条长期偏好，但当前线程没有配置长期记忆存储（Store），"
                      "本次未写入。")

UNCERTAIN_CLAUSE = (
    "本次检索证据未通过完整性校验。如果上下文不足以回答，请明确说明"
    "“根据现有信息无法回答”，不要给出未在上下文中出现的数字。"
)
# 数字纪律：针对实测的两类错答——模型自行换算单位后给出错值、
# 以及跨公司题把两家的数字混配（ar-crossco-002 出现过 18,606,756,000 / 186.07）
NUMERIC_DISCIPLINE = (
    "数字纪律：引用的数值必须与上下文逐字一致；需要换算单位时，先写上下文里的原值，"
    "再在括号里给出换算结果；上下文没有出现的数值一律不写；"
    "问题涉及多家公司或多个年份时，逐条标注“公司+年份=数值”，不要混用。"
)


def _history(messages: List[Any]) -> List[Dict[str, str]]:
    """把累计的 messages 转成 PromptBuilder 需要的 history（去掉当前这一轮）。"""
    items: List[Dict[str, str]] = []
    for message in messages or []:
        if isinstance(message, dict):
            role, content = message.get("role"), message.get("content")
        else:
            role = getattr(message, "type", None)
            content = getattr(message, "content", None)
        role = {"human": "user", "ai": "assistant"}.get(role, role)
        items.append({"role": role or "", "content": content or ""})
    return items[:-1]


def _stream_writer(runtime: Any):
    """拿到 LangGraph 的自定义流写入器（用于 SSE 的 token 事件）。

    只在显式开启（enable_token_stream）时尝试，非流式调用下返回 None，
    保证同步接口的行为与质量完全不受影响。
    """
    if not runtime.extra.get("enable_token_stream"):
        return None
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except Exception:  # noqa: BLE001 - 无流式上下文时退回非流式生成
        return None


def _accepts_parameter(function: Any, name: str) -> bool:
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def build_nodes(runtime: Any) -> Dict[str, Any]:
    """按运行时依赖构造节点函数集合。"""

    subgraph = build_retrieval_subgraph(runtime)
    rewriter = runtime.rewriter or RuleRewriter()

    def understand_query(state: Dict[str, Any]) -> Dict[str, Any]:
        question = (state.get("question") or "").strip()
        facets = extract_facets(question)
        contract = build_evidence_contract(question, facets=facets)
        obligations = contract["obligations"]
        routes: List[Dict[str, Any]] = []
        if runtime.enable_pre_retrieval_routing:
            planned_routes = build_initial_routes(
                question, obligations, facets=facets)
            obligations = bind_routes_to_obligations(
                obligations, planned_routes)
            routes = [route.as_dict() for route in planned_routes]
        # 跨会话记忆：先召回该用户已存的偏好/约定（namespace 隔离在 memory.py）
        memories: List[Dict[str, Any]] = []
        if runtime.memory_store is not None and runtime.memory_user_id:
            memories = recall(runtime.memory_store, runtime.memory_user_id)
        # "记住我以后都……"这类输入是记忆写入请求，不是知识库问题，不做检索
        memory_request = extract_memory(question)
        needs_retrieval = needs_knowledge_base(question) and not memory_request
        if memory_request:
            status = "memory_only"
        elif needs_retrieval:
            status = "understanding"
        else:
            status = "out_of_scope"
        return {
            "question": question,
            "rewritten_query": question,
            "sub_queries": (
                [route["query"] for route in routes] if routes else (
                    [question] if not runtime.enable_pre_retrieval_routing else []
                )
            ),
            "routes": routes,
            "route_history": ([{
                "retry": 0,
                "phase": "planned",
                "routes": routes,
            }] if runtime.enable_pre_retrieval_routing else []),
            "facets": facets,
            "obligations": obligations,
            "needs_retrieval": needs_retrieval,
            "memories": memories,
            "status": status,
            "trace": [{"node": "understand_query", "facets": facets,
                       "needs_retrieval": needs_retrieval,
                       "obligations": len(contract["obligations"]),
                       "routes": len(routes),
                       "route_documents": sum(
                           len(route.get("document_ids") or [])
                           for route in routes
                       ),
                       "refusal_expected": contract["refusal_expected"],
                       "memory_request": bool(memory_request),
                       "recalled_memories": len(memories)}],
        }

    def retrieve_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
        routing_enabled = bool(runtime.enable_pre_retrieval_routing)
        routes = state.get("routes") or []
        queries = state.get("sub_queries") or [state.get("rewritten_query")
                                               or state["question"]]
        retry = int(state.get("retry_count") or 0)
        budget = runtime.top_k if retry == 0 else runtime.widened_top_k
        obligations = state.get("obligations") or []
        result = subgraph.invoke({
            "queries": queries,
            "routes": routes,
            "routing_enabled": routing_enabled,
            "strict_route_filter": bool(runtime.strict_route_filter),
            "route_candidate_limit": int(runtime.route_candidate_limit),
            "route_document_ids": {
                str(route.get("query_id")): list(
                    route.get("document_ids") or [])
                for route in routes
            },
            "route_obligations": {
                str(route.get("query_id")): list(
                    route.get("obligation_ids") or [])
                for route in routes
            },
            "top_k": budget,
            "anchor": state["question"],
            "obligations": obligations,
            "route_prefix": "r{}-".format(retry),
        })
        # 自纠错只做加法：重试把新候选追加到已有证据之后，不推翻上一轮（见 rag.py）
        previous_candidates = state.get("retrieved_chunks") or []
        incoming_candidates = result.get("candidates") or []
        current_route_hits = result.get("route_hits") or []
        route_hits = (list(state.get("route_hits") or [])
                      + list(current_route_hits))
        candidates = accumulate_evidence(
            previous_candidates, incoming_candidates,
            len(previous_candidates) + len(incoming_candidates) + 1)
        candidates, obligation_fills = preserve_obligation_evidence(
            candidates, route_hits, obligations)
        pre_cap_obligations = assess_obligations(
            candidates, obligations, state["question"])
        protected_keys = {
            str(hit.get("chunk_id") or hit.get("vector_id"))
            for hit in candidates if hit.get("obligation_fill")
        }
        protected_keys.update(
            str(chunk_id)
            for item in pre_cap_obligations["results"]
            if item["status"] == "covered"
            for chunk_id in (item.get("matched_chunks") or [])[:1]
        )
        candidates = cap_evidence_preserving(
            candidates, runtime.evidence_limit, protected_keys)
        pre_scope_obligations = assess_obligations(
            candidates, obligations, state["question"])
        # facet 裁剪：去掉问题没点名的公司/年份片段（重试只追加的代价是污染也会留）
        if runtime.scope_evidence_enabled:
            candidates, scope_stats = scope_evidence(
                state["question"], candidates,
                year_window=int(runtime.extra.get("year_window", 0)),
                allow_fallback=not (
                    routing_enabled and runtime.strict_route_filter),
            )
        else:
            scope_stats = {"scoped": False, "reason": "disabled",
                           "kept": len(candidates), "dropped": 0}
        post_scope_obligations = assess_obligations(
            candidates, obligations, state["question"])
        route_coverage = (list(state.get("route_coverage") or [])
                          + list(result.get("route_coverage") or []))
        covered_obligations = post_scope_obligations["covered"]
        missing_obligations = post_scope_obligations["missing"]
        route_filter_missing = sorted(set(
            result.get("stats", {}).get("route_filter_missing") or []
        ) | {
            str(obligation_id)
            for route in routes
            if not route.get("unrouted") and not route.get("document_ids")
            for obligation_id in route.get("obligation_ids") or []
        })
        return {
            "retrieved_chunks": candidates,
            "route_hits": route_hits,
            "route_coverage": route_coverage,
            "retrieval_stats": dict(result.get("stats") or {},
                                    accumulated=len(candidates), budget=budget,
                                    scope=scope_stats,
                                    scope_degraded=bool(
                                        scope_stats.get("scope_degraded")),
                                    obligations_covered=covered_obligations,
                                    obligations_missing=missing_obligations,
                                    obligation_fills=obligation_fills,
                                    obligation_fill_chunks=[
                                        hit.get("chunk_id")
                                        for hit in candidates
                                        if hit.get("obligation_fill")
                                    ],
                                    pre_scope_missing=pre_scope_obligations[
                                        "missing"],
                                    post_scope_missing=post_scope_obligations[
                                        "missing"],
                                    route_filter_missing=route_filter_missing,
                                    strict_route_filter=(
                                        routing_enabled
                                        and runtime.strict_route_filter
                                    )),
            "status": "retrieving",
            "trace": [{"node": "retrieve_knowledge", "queries": list(queries),
                       "routes": result.get("stats", {}).get(
                           "routes_planned", 0),
                       "candidate_count": len(candidates),
                       "budget": budget,
                       "scoped_out": scope_stats.get("dropped", 0),
                       "retry": int(state.get("retry_count") or 0)}],
        }

    def grade_retrieval(state: Dict[str, Any]) -> Dict[str, Any]:
        hits = state.get("retrieved_chunks") or []
        score = grade_evidence(state["question"], hits,
                               min_coverage=runtime.min_coverage,
                               obligations=state.get("obligations") or [],
                               route_coverage=state.get("route_coverage") or [])
        # 规则说"够"才值得复核："规则假通过"会变成沉默的错答（贵且危险），
        # "规则假不通过"只是多一次重试（便宜，已用关键词修复与 facet 裁剪压掉）
        if score["passed"] and runtime.llm_grader is not None:
            grade_kwargs = {}
            if _accepts_parameter(runtime.llm_grader.grade, "obligations"):
                grade_kwargs["obligations"] = score.get("obligations") or []
            if _accepts_parameter(runtime.llm_grader.grade,
                                  "missing_obligations"):
                grade_kwargs["missing_obligations"] = (
                    score.get("missing_obligations") or [])
            if _accepts_parameter(runtime.llm_grader.grade, "route_coverage"):
                grade_kwargs["route_coverage"] = (
                    state.get("route_coverage") or [])
            review = runtime.llm_grader.grade(
                state["question"], hits, **grade_kwargs)
            score["llm_review"] = review
            if review.get("judgment") == "insufficient" and review.get("source") == "llm":
                score["passed"] = False
                score["rule_passed"] = True
                score["reason"] = "LLM 复核判定证据不足: {}".format(
                    review.get("reason") or review.get("missing") or "")
        return {
            "retrieval_score": score,
            "status": "graded",
            "trace": [{"node": "grade_retrieval", "passed": score["passed"],
                       "reason": score["reason"],
                       "coverage": score["coverage"],
                       "obligation_coverage": score.get("obligation_coverage"),
                       "missing_obligations": score.get("missing_obligations"),
                       "ambiguous_obligations": score.get("ambiguous_obligations"),
                       "llm_review": (score.get("llm_review") or {}).get("judgment"),
                       "documents": score["documents"]}],
        }

    def decide_next_step(state: Dict[str, Any]) -> Dict[str, Any]:
        score = state.get("retrieval_score") or {}
        hits = state.get("retrieved_chunks") or []
        retry_count = int(state.get("retry_count") or 0)
        low_confidence = not score.get("passed")

        if score.get("refusal_expected"):
            action, status = "finalize", "refused_no_evidence"
        elif score.get("passed"):
            action, status = "generate", "evidence_ok"
        elif retry_count < runtime.max_retries:
            action, status = "retrieve", "retry"
        elif not hits:
            action, status = "finalize", "refused_no_evidence"
        elif score.get("ambiguous_obligations"):
            action, status = "generate", "ambiguous_evidence"
            low_confidence = True
        else:
            # 证据不足但仍有候选：基于现有证据作答并声明不确定；
            # 是否交人工审核由 generate_answer 之后的条件边决定（先出草稿再人审）
            action, status = "generate", "answering_with_caveat"

        return {
            "next_action": action,
            "needs_human_review": bool(low_confidence),
            "status": status,
            "trace": [{"node": "decide_next_step", "action": action,
                       "retry_count": retry_count,
                       "missing_obligations": score.get("missing_obligations"),
                       "ambiguous_obligations": score.get("ambiguous_obligations"),
                       "reason": score.get("reason", "")}],
        }

    def rewrite_query(state: Dict[str, Any]) -> Dict[str, Any]:
        score = state.get("retrieval_score") or {}
        reason = score.get("reason", "")
        retry_count = int(state.get("retry_count") or 0) + 1
        previous = [state.get("rewritten_query") or state["question"]]
        previous.extend(state.get("sub_queries") or [])
        missing = score.get("missing_obligations") or []
        rewrite_kwargs = {}
        if _accepts_parameter(rewriter.rewrite, "obligations"):
            rewrite_kwargs["obligations"] = missing
        queries = rewriter.rewrite(
            state["question"], reason, previous, **rewrite_kwargs)
        if not queries:
            queries = [
                obligation_query(item) for item in missing
            ] or [state["question"]]
        retry_routes: List[Dict[str, Any]] = []
        if runtime.enable_pre_retrieval_routing:
            retry_routes = [
                route.as_dict() for route in route_for_missing_obligations(
                    state["question"],
                    missing,
                    state.get("routes") or [],
                    rewritten_queries=queries,
                )
            ]
            if retry_routes:
                queries = [route["query"] for route in retry_routes]
        return {
            "rewritten_query": queries[0],
            "sub_queries": queries,
            "routes": retry_routes if runtime.enable_pre_retrieval_routing
            else state.get("routes") or [],
            "retry_count": retry_count,
            "rewrite_history": [{
                "retry": retry_count,
                "reason": reason,
                "queries": queries,
                "obligations": [
                    item.get("obligation_id") for item in missing
                ],
                "routes": retry_routes,
                "source": getattr(rewriter, "last_source", "rule"),
            }],
            "route_history": ([{
                "retry": retry_count,
                "phase": "retry_planned",
                "missing_obligations": [
                    item.get("obligation_id") for item in missing
                ],
                "routes": retry_routes,
            }] if runtime.enable_pre_retrieval_routing else []),
            "status": "rewritten",
            "trace": [{"node": "rewrite_query", "retry": retry_count,
                       "queries": queries,
                       "routes": len(retry_routes),
                       "missing_obligations": [
                           item.get("obligation_id") for item in missing
                       ],
                       "source": getattr(rewriter, "last_source", "rule"),
                       "reason": reason}],
        }

    def generate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
        hits = state.get("retrieved_chunks") or []
        if not hits:
            return {
                "answer": REFUSAL_NO_EVIDENCE,
                "citations": [],
                "status": "refused_no_evidence",
                "trace": [{"node": "generate_answer", "skipped": "no_evidence"}],
            }
        if not runtime.generate_answers:
            return {
                "answer": None,
                "status": (
                    "ambiguous_evidence"
                    if state.get("status") == "ambiguous_evidence"
                    else "retrieval_only"
                ),
                "trace": [{"node": "generate_answer", "skipped": "retrieval_only"}],
            }

        question = state["question"]
        pipeline = runtime.pipeline
        history = _history(state.get("messages"))
        budget_result = build_context_with_budget(
            pipeline,
            question,
            hits,
            history=history,
            obligations=state.get("obligations") or [],
            max_context_tokens=runtime.max_context_tokens,
            output_reserved_tokens=runtime.output_reserved_tokens,
        )
        context = budget_result["context"]
        chunks = budget_result["chunks"]
        messages = pipeline.prompt_builder.build(
            question, context, history=budget_result["history"])
        memory_block = format_memories(state.get("memories") or [])
        if memory_block:
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + "\n" + memory_block +
                           "\n（以上只影响回答口径与风格；具体事实仍必须来自上下文。）",
            }
        extra_clauses = [NUMERIC_DISCIPLINE]
        if not (state.get("retrieval_score") or {}).get("passed", True):
            extra_clauses.append(UNCERTAIN_CLAUSE)
        # 上游 system prompt 已编号到 6，这里接着往下编
        numbered = "\n".join("{}. {}".format(index + 7, text)
                             for index, text in enumerate(extra_clauses))
        messages[0] = {"role": "system",
                       "content": messages[0]["content"] + "\n" + numbered}

        generator = pipeline.generator
        writer = _stream_writer(runtime)
        if writer is not None and hasattr(generator, "stream_generate"):
            # 逐片段推给 SSE，同时拼出完整答案供引用提取与状态落库
            parts: List[str] = []
            for chunk in generator.stream_generate(messages):
                if chunk:
                    parts.append(chunk)
                    writer({"type": "token", "text": chunk})
            answer = "".join(parts)
        else:
            answer = generator.generate(messages)
        if not (answer or "").strip() and hasattr(generator, "generate"):
            answer = generator.generate(messages)
        generation_status = "answered"
        if not (answer or "").strip():
            answer = REFUSAL_NO_EVIDENCE
            generation_status = "refused_no_evidence"
        citations = [enrich_citation(citation, runtime.page_offsets, chunks)
                     for citation in CitationExtractor().extract(answer, chunks)]
        grounding = verify_grounding(answer, citations, chunks, question)
        return {
            "answer": answer,
            "citations": citations,
            "context_chunks": chunks,
            "prompt_budget": budget_result["budget"],
            "grounding": grounding,
            "status": generation_status,
            "trace": [{"node": "generate_answer", "answer_len": len(answer or ""),
                       "context_chunks": len(chunks),
                       "prompt_tokens_estimate": budget_result["budget"][
                           "prompt_token_estimate"],
                       "history_tokens": budget_result["budget"][
                           "history_tokens"],
                       "skipped_history_turns": budget_result["budget"][
                           "skipped_history_turns"],
                       "budget_overflow": budget_result["budget"][
                           "budget_overflow"],
                       "citations": len(citations),
                       "ungrounded": len(grounding.get("ungrounded") or []),
                       "memories_used": len(state.get("memories") or [])}],
        }

    def remember_preference(state: Dict[str, Any]) -> Dict[str, Any]:
        """把"跨会话仍然成立"的输入写进 Store；兼作记忆写入请求的应答节点。"""
        update: Dict[str, Any] = {
            "trace": [{"node": "remember_preference", "written": 0}],
        }
        candidate = extract_memory(state.get("question") or "")
        already = state.get("memory_written") or []
        if candidate and not already:
            if runtime.memory_store is not None and runtime.memory_user_id:
                record = remember(runtime.memory_store, runtime.memory_user_id,
                                  candidate["content"], candidate["memory_type"])
                update["memory_written"] = [record]
                update["trace"] = [{"node": "remember_preference", "written": 1,
                                    "memory_id": record["memory_id"],
                                    "memory_type": record["memory_type"]}]
                if not state.get("answer"):
                    # 记忆写入请求本身没有知识库答案，给明确回执避免空响应
                    update["answer"] = MEMORY_ACK
                    update["status"] = "memory_saved"
            else:
                # 没配置 Store 就说没配置，不要假装写成功（静默失败最误导人）
                update["trace"] = [{"node": "remember_preference", "written": 0,
                                    "reason": "store_not_configured"}]
                if not state.get("answer"):
                    update["answer"] = MEMORY_UNAVAILABLE
                    update["status"] = "memory_unavailable"
        return update

    def verify_answer(state: Dict[str, Any]) -> Dict[str, Any]:
        """Audit claims after generation and repair only hard-violation spans."""
        answer = state.get("answer")
        citations = state.get("citations") or []
        chunks = state.get("context_chunks") or []
        question = state["question"]
        attempts = len(state.get("repair_history") or [])
        if not (isinstance(answer, str) and answer.strip()) or not chunks:
            grounding = state.get("grounding") or {}
            return {
                "status": state.get("status"),
                "claims": [],
                "answer_audit": {},
                "trace": [{"node": "verify_answer", "skipped": True,
                           "ungrounded": len(grounding.get("ungrounded") or [])}],
            }

        obligations = state.get("obligations") or []
        audit = audit_answer(
            answer, citations, chunks, question, obligations=obligations)
        hard = audit.get("hard_violations") or []
        soft = audit.get("soft_issues") or []
        effective_repair = (
            runtime.enable_claim_repair or runtime.enable_answer_repair)
        repair_result: Optional[Dict[str, Any]] = None

        if hard and effective_repair and attempts < runtime.max_repair_rounds:
            claims_by_id = {
                str(item.get("claim_id")): item
                for item in audit.get("claims") or []
            }
            target_claims = [
                claims_by_id[claim_id]
                for claim_id in audit.get("repairable_claim_ids") or []
                if claim_id in claims_by_id
            ]
            evidence_text = "\n".join(
                "[{}] {}".format(index + 1, (chunk.get("content") or "")[:700])
                for index, chunk in enumerate(chunks[:8]))
            if target_claims:
                repair_result = repair_claims(
                    runtime.pipeline.generator,
                    question,
                    evidence_text,
                    answer,
                    target_claims,
                    audit.get("missing_obligations") or [],
                )
            repaired = (repair_result or {}).get("answer")
            if repaired:
                citations_after = [
                    enrich_citation(citation, runtime.page_offsets, chunks)
                    for citation in CitationExtractor().extract(repaired, chunks)
                ]
                audit_after = audit_answer(
                    repaired, citations_after, chunks, question,
                    obligations=obligations)
                record = {
                    "round": attempts + 1,
                    "applied_claim_ids": (
                        repair_result or {}).get("applied_claim_ids") or [],
                    "hard_before": [item.get("code") for item in hard],
                    "hard_after": [
                        item.get("code")
                        for item in audit_after.get("hard_violations") or []
                    ],
                    "fixed": bool(audit_after.get("hard_gate_passed")),
                }
                if audit_after.get("hard_gate_passed"):
                    return {
                        "answer": repaired,
                        "citations": citations_after,
                        "claims": audit_after.get("claims") or [],
                        "answer_audit": audit_after,
                        "grounding": verify_grounding(
                            repaired, citations_after, chunks, question),
                        "needs_human_review": bool(
                            audit_after.get("soft_issues")),
                        "repair_history": [record],
                        "status": "answered_repaired",
                        "trace": [{
                            "node": "verify_answer",
                            "repairs": 1,
                            "fixed": True,
                            "applied_claims": record["applied_claim_ids"],
                            "soft_issues": len(
                                audit_after.get("soft_issues") or []),
                        }],
                    }
                repair_result = dict(repair_result or {})
                repair_result["rejected_answer"] = repaired

        hard_enforced = bool(runtime.enable_claim_hard_gate and hard)
        soft_review = bool(runtime.enable_claim_soft_warnings and soft)
        needs_review = hard_enforced or soft_review
        record = {
            "round": attempts + 1,
            "applied_claim_ids": (
                repair_result or {}).get("applied_claim_ids") or [],
            "hard_before": [item.get("code") for item in hard],
            "fixed": False,
        }
        update: Dict[str, Any] = {
            "claims": audit.get("claims") or [],
            "answer_audit": audit,
            "needs_human_review": needs_review,
            "status": (
                "answer_hard_gate_failed" if hard_enforced
                else "answered_soft_warnings" if soft_review
                else "answered"
            ),
            "trace": [{
                "node": "verify_answer",
                "hard_violations": len(hard),
                "soft_issues": len(soft),
                "repairs": 1 if repair_result else 0,
                "fixed": False,
            }],
        }
        if repair_result:
            repair_result = dict(repair_result)
            update["repair_history"] = [record]
        if hard_enforced:
            codes = "、".join(sorted({
                str(item.get("code")) for item in hard
            }))
            update["answer"] = (
                "{}\n\n（注：答案未通过 Claim 硬门禁：{}，已标记人工复核）"
            ).format(answer, codes)
        return update

    def human_review(state: Dict[str, Any]) -> Dict[str, Any]:
        """人工审核：interrupt 暂停，等 Command(resume=...) 带回审核结果。

        执行到这里时草稿答案已经生成（decide → generate → 人审），审核人拿到的
        是有上下文的完整草稿；恢复后按 approve / reject / supplement 收敛结果。
        """
        payload = build_review_payload(state, state.get("thread_id") or "")
        submission = interrupt(payload)
        submission = submission if isinstance(submission, dict) else {}
        decision = submission.get("decision", "approve")
        note = submission.get("note", "") or ""

        answer = state.get("answer")
        if decision == "reject":
            answer, status = REJECTED_ANSWER, "review_rejected"
        elif decision == "supplement":
            suffix = "\n\n人工补充：{}".format(note) if note else ""
            answer, status = (answer or PENDING_REVIEW) + suffix, "review_supplemented"
        else:
            status = "review_approved"

        return {
            "review_result": submission,
            "needs_human_review": False,
            "answer": answer,
            "citations": state.get("citations") or [],
            "status": status,
            "trace": [{"node": "human_review", "decision": decision,
                       "reviewer": submission.get("reviewer", ""),
                       "review_id": payload.get("review_id"),
                       "reason": (state.get("retrieval_score") or {}).get("reason", "")}],
        }

    def finalize_response(state: Dict[str, Any]) -> Dict[str, Any]:
        answer = state.get("answer")
        status = state.get("status") or "finished"

        if not answer and status == "out_of_scope":
            answer, status = OUT_OF_SCOPE, "out_of_scope"
        elif not answer and status == "refused_no_evidence":
            # 无证据时 decide_next_step 直接进 finalize（跳过生成），
            # 回答文本在这里补齐，避免"没有答案"变成空响应
            answer = REFUSAL_NO_EVIDENCE

        update: Dict[str, Any] = {
            "answer": answer,
            "status": status,
            "trace": [{"node": "finalize_response", "status": status,
                       "has_answer": bool(answer),
                       "citations": len(state.get("citations") or [])}],
        }
        if isinstance(answer, str) and answer.strip():
            update["messages"] = [{"role": "assistant", "content": answer}]
        return update

    return {
        "understand_query": understand_query,
        "retrieve_knowledge": retrieve_knowledge,
        "grade_retrieval": grade_retrieval,
        "decide_next_step": decide_next_step,
        "rewrite_query": rewrite_query,
        "generate_answer": generate_answer,
        "verify_answer": verify_answer,
        "remember_preference": remember_preference,
        "human_review": human_review,
        "finalize_response": finalize_response,
    }


def route_after_understand(state: Dict[str, Any]) -> str:
    if state.get("status") == "memory_only":
        return "remember_preference"
    if state.get("needs_retrieval", True):
        return "retrieve_knowledge"
    return "finalize_response"


def route_after_generate(state: Dict[str, Any]) -> str:
    """低置信度且开启审核时，先把草稿交人工；否则直接收尾。"""
    if state.get("needs_human_review") and state.get("status") != "reviewed":
        return "human_review"
    return "remember_preference"


def route_after_decide(state: Dict[str, Any]) -> str:
    action = state.get("next_action") or "generate"
    if action == "retrieve":
        return "rewrite_query"
    if action == "finalize":
        return "finalize_response"
    return "generate_answer"
