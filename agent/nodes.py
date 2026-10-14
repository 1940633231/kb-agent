"""主图节点实现。

职责边界（v2 §4.1）：判定（grade_retrieval）与重试（decide_next_step /
rewrite_query）全部在主图，检索子图只负责取回候选；所有重试痕迹
（retry_count、改写后的查询、每轮评分）都写进 State，随 checkpointer 落库，
便于回放与调试。
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.citation.citation import CitationExtractor
from langgraph.types import interrupt

from agent.grading import extract_facets, grade_evidence, needs_knowledge_base
from agent.memory import extract_memory, format_memories, recall, remember
from agent.review import build_review_payload
from agent.retrieval_subgraph import build_retrieval_subgraph
from agent.rewriter import RuleRewriter
from rag_adapter.rag import accumulate_evidence, enrich_citation

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
    "7. 本次检索证据未通过完整性校验。如果上下文不足以回答，请明确说明"
    "“根据现有信息无法回答”，不要给出未在上下文中出现的数字。"
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


def build_nodes(runtime: Any) -> Dict[str, Any]:
    """按运行时依赖构造节点函数集合。"""

    subgraph = build_retrieval_subgraph(runtime)
    rewriter = runtime.rewriter or RuleRewriter()

    def understand_query(state: Dict[str, Any]) -> Dict[str, Any]:
        question = (state.get("question") or "").strip()
        facets = extract_facets(question)
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
            "sub_queries": [question],
            "facets": facets,
            "needs_retrieval": needs_retrieval,
            "memories": memories,
            "status": status,
            "trace": [{"node": "understand_query", "facets": facets,
                       "needs_retrieval": needs_retrieval,
                       "memory_request": bool(memory_request),
                       "recalled_memories": len(memories)}],
        }

    def retrieve_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
        queries = state.get("sub_queries") or [state.get("rewritten_query")
                                               or state["question"]]
        retry = int(state.get("retry_count") or 0)
        budget = runtime.top_k if retry == 0 else runtime.widened_top_k
        result = subgraph.invoke({"queries": queries, "top_k": budget,
                                  "anchor": state["question"]})
        # 自纠错只做加法：重试把新候选追加到已有证据之后，不推翻上一轮（见 rag.py）
        candidates = accumulate_evidence(state.get("retrieved_chunks") or [],
                                        result.get("candidates") or [],
                                        runtime.evidence_limit)
        return {
            "retrieved_chunks": candidates,
            "retrieval_stats": dict(result.get("stats") or {},
                                    accumulated=len(candidates), budget=budget),
            "status": "retrieving",
            "trace": [{"node": "retrieve_knowledge", "queries": list(queries),
                       "candidate_count": len(candidates),
                       "budget": budget,
                       "retry": int(state.get("retry_count") or 0)}],
        }

    def grade_retrieval(state: Dict[str, Any]) -> Dict[str, Any]:
        hits = state.get("retrieved_chunks") or []
        score = grade_evidence(state["question"], hits,
                               min_coverage=runtime.min_coverage)
        return {
            "retrieval_score": score,
            "status": "graded",
            "trace": [{"node": "grade_retrieval", "passed": score["passed"],
                       "reason": score["reason"],
                       "coverage": score["coverage"],
                       "documents": score["documents"]}],
        }

    def decide_next_step(state: Dict[str, Any]) -> Dict[str, Any]:
        score = state.get("retrieval_score") or {}
        hits = state.get("retrieved_chunks") or []
        retry_count = int(state.get("retry_count") or 0)
        low_confidence = not score.get("passed")

        if score.get("passed"):
            action, status = "generate", "evidence_ok"
        elif retry_count < runtime.max_retries:
            action, status = "retrieve", "retry"
        elif not hits:
            action, status = "finalize", "refused_no_evidence"
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
                       "reason": score.get("reason", "")}],
        }

    def rewrite_query(state: Dict[str, Any]) -> Dict[str, Any]:
        score = state.get("retrieval_score") or {}
        reason = score.get("reason", "")
        retry_count = int(state.get("retry_count") or 0) + 1
        previous = [state.get("rewritten_query") or state["question"]]
        previous.extend(state.get("sub_queries") or [])
        queries = rewriter.rewrite(state["question"], reason, previous)
        if not queries:
            queries = [state["question"]]
        return {
            "rewritten_query": queries[0],
            "sub_queries": queries,
            "retry_count": retry_count,
            "rewrite_history": [{
                "retry": retry_count,
                "reason": reason,
                "queries": queries,
                "source": getattr(rewriter, "last_source", "rule"),
            }],
            "status": "rewritten",
            "trace": [{"node": "rewrite_query", "retry": retry_count,
                       "queries": queries,
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
                "status": "retrieval_only",
                "trace": [{"node": "generate_answer", "skipped": "retrieval_only"}],
            }

        question = state["question"]
        pipeline = runtime.pipeline
        context_result = pipeline.context_manager.build(question, hits)
        context = context_result["context"]
        chunks = context_result["chunks"]
        messages = pipeline.prompt_builder.build(question, context,
                                                history=_history(state.get("messages")))
        memory_block = format_memories(state.get("memories") or [])
        if memory_block:
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + "\n" + memory_block +
                           "\n（以上只影响回答口径与风格；具体事实仍必须来自上下文。）",
            }
        if not (state.get("retrieval_score") or {}).get("passed", True):
            messages[0] = {"role": "system",
                           "content": messages[0]["content"] + "\n" + UNCERTAIN_CLAUSE}

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
        citations = [enrich_citation(citation, runtime.page_offsets, chunks)
                     for citation in CitationExtractor().extract(answer, chunks)]
        return {
            "answer": answer,
            "citations": citations,
            "context_chunks": chunks,
            "status": "answered",
            "trace": [{"node": "generate_answer", "answer_len": len(answer or ""),
                       "context_chunks": len(chunks),
                       "citations": len(citations),
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
