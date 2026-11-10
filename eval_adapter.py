"""kb-agent adapters for the standalone rag-eval platform."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from rag_eval.contracts import (
    Citation,
    EvalSample,
    Evidence,
    Prediction,
    RetrievedChunk,
    RetrievedDocument,
)
from rag_eval.judges.base import JudgeResult
from rag_eval.registry import RunContext

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _document_id(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    return str(
        item.get("document_id")
        or metadata.get("document_id")
        or ""
    )


def _span(item: dict[str, Any]) -> dict[str, int] | None:
    try:
        start = int(item.get("start_offset") or 0)
        end = int(item.get("end_offset") or 0)
    except (TypeError, ValueError):
        return None
    return {"start": start, "end": end} if end > start else None


def _map_hits(
    hits: list[dict[str, Any]],
) -> tuple[list[RetrievedChunk], list[RetrievedDocument], list[Evidence]]:
    chunks: list[RetrievedChunk] = []
    documents: list[RetrievedDocument] = []
    evidence: list[Evidence] = []
    seen_documents: set[str] = set()
    for rank, hit in enumerate(hits, start=1):
        document_id = _document_id(hit)
        chunk_id = str(hit.get("chunk_id") or "")
        if not document_id or not chunk_id:
            continue
        score = float(
            hit.get("rerank_score", hit.get("score", 0.0)) or 0.0
        )
        span = _span(hit)
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                score=score,
                rank=rank,
                span=span,
                text=hit.get("content"),
                metadata=hit.get("metadata") or {},
            )
        )
        evidence.append(
            Evidence(
                document_id=document_id,
                chunk_id=chunk_id,
                span=span,
                text=hit.get("content"),
            )
        )
        if document_id not in seen_documents:
            seen_documents.add(document_id)
            documents.append(
                RetrievedDocument(
                    document_id=document_id,
                    score=score,
                    rank=len(documents) + 1,
                )
            )
    return chunks, documents, evidence


def _map_citations(
    citations: list[dict[str, Any]],
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    chunk_lookup = {chunk.chunk_id: chunk.document_id for chunk in chunks}
    mapped: list[Citation] = []
    for item in citations:
        chunk_id = str(item.get("chunk_id") or "") or None
        document_id = str(
            item.get("document_id")
            or chunk_lookup.get(chunk_id or "", "")
            or ""
        )
        if not document_id:
            continue
        mapped.append(
            Citation(
                document_id=document_id,
                chunk_id=chunk_id,
                quote=item.get("content_preview") or item.get("quote"),
            )
        )
    return mapped


def _index_errors(strategy: str, mode: str) -> list[str]:
    from rag_adapter.settings import CONFIG_PATH, index_dir

    errors: list[str] = []
    if not CONFIG_PATH.is_file():
        errors.append(f"config file not found: {CONFIG_PATH}")
    directory = index_dir(strategy)
    if not (directory / "metadata.json").is_file():
        errors.append(f"metadata not found: {directory / 'metadata.json'}")
    if mode in {"vector", "hybrid"} and not (directory / "faiss.index").is_file():
        errors.append(f"FAISS index not found: {directory / 'faiss.index'}")
    return errors


def _build_service(
    rag,
    *,
    strategy: str,
    mode: str,
    use_rerank: bool,
    retrieval_only: bool,
):
    """Build RAGService without requiring LLM config for retrieval-only runs."""
    from app.rag.service import RAGService

    config = rag.build_config()
    if retrieval_only:
        config.data.setdefault("generation", {})["backend"] = "stub"
    return RAGService(
        config=config,
        strategy=strategy,
        mode=mode,
        use_rerank=use_rerank,
    )


class KbAgentRetrievalAdapter:
    """Retrieval-only adapter using rag_adapter.rag.retrieve()."""

    capabilities = {"retrieval", "latency"}

    def __init__(
        self,
        strategy: str | None = None,
        mode: str | None = None,
        use_rerank: bool | None = None,
        top_k: int | None = None,
        service: Any | None = None,
    ):
        self.strategy = strategy or os.getenv(
            "KB_AGENT_EVAL_STRATEGY", "recursive"
        )
        self.mode = mode or os.getenv("KB_AGENT_EVAL_MODE", "hybrid")
        self.use_rerank = (
            use_rerank
            if use_rerank is not None
            else _env_bool("KB_AGENT_EVAL_RERANK", True)
        )
        self.top_k = int(
            top_k
            if top_k is not None
            else os.getenv("KB_AGENT_EVAL_TOP_K", "5")
        )
        self._service = service
        self._offsets: dict[str, Any] = {}
        self.name = "kb-agent-retrieval-{}-{}".format(
            self.mode, "rerank" if self.use_rerank else "norerank"
        )

    def validate(self) -> list[str]:
        if self._service is not None:
            return []
        errors = _index_errors(self.strategy, self.mode)
        if self.top_k <= 0:
            errors.append("top_k must be positive")
        return errors

    def _ensure_ready(self) -> None:
        if self._service is not None:
            return
        from rag_adapter import rag
        from rag_adapter.settings import load_env, use_project_root_as_cwd

        use_project_root_as_cwd()
        load_env()
        self._service = _build_service(
            rag,
            strategy=self.strategy,
            mode=self.mode,
            use_rerank=self.use_rerank,
            retrieval_only=True,
        )
        self._offsets = rag.load_page_offsets(self.strategy)

    def warmup(self, context: RunContext) -> None:
        self._ensure_ready()

    def run(self, sample: EvalSample, context: RunContext) -> Prediction:
        if sample.task != "retrieval":
            raise ValueError(
                f"retrieval adapter does not support task {sample.task!r}"
            )
        from rag_adapter import rag

        self._ensure_ready()
        started = time.perf_counter()
        hits = rag.retrieve(self._service, sample.query, top_k=self.top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        chunks, documents, evidence = _map_hits(hits)
        return Prediction(
            sample_id=sample.id,
            status="ok",
            documents=documents,
            chunks=chunks,
            evidence=evidence,
            latency_ms=elapsed_ms,
            usage={
                "backend": "kb_agent_retrieval",
                "mode": self.mode,
                "rerank": self.use_rerank,
            },
            metadata={
                "run_id": context.run_id,
                "retrieval_only": True,
                "behavior": "answer",
                "retry_count": 0,
                "rewrite_sources": 0,
            },
        )


class KbAgentVectorRetrievalAdapter(KbAgentRetrievalAdapter):
    """Deterministic local smoke: FAISS vector retrieval, no rerank."""

    def __init__(self, service: Any | None = None):
        super().__init__(
            mode="vector",
            use_rerank=False,
            service=service,
        )


class KbAgentL3Adapter:
    """Adapter for the kb-agent L3 self-correction graph."""

    capabilities = {"retrieval", "generation", "agent", "citation", "latency"}

    def __init__(
        self,
        strategy: str | None = None,
        top_k: int | None = None,
        max_retries: int | None = None,
        mode: str | None = None,
        use_rerank: bool | None = None,
        generate_answers: bool | None = None,
        use_llm_rewrite: bool | None = None,
        use_llm_grader: bool | None = None,
        enable_pre_retrieval_routing: bool | None = None,
        strict_route_filter: bool | None = None,
        route_candidate_limit: int | None = None,
        service: Any | None = None,
        graph: Any | None = None,
    ):
        self.strategy = strategy or os.getenv(
            "KB_AGENT_EVAL_STRATEGY", "recursive"
        )
        self.top_k = int(
            top_k
            if top_k is not None
            else os.getenv("KB_AGENT_EVAL_TOP_K", "5")
        )
        self.max_retries = int(
            max_retries
            if max_retries is not None
            else os.getenv("KB_AGENT_EVAL_MAX_RETRIES", "2")
        )
        self.mode = mode or os.getenv("KB_AGENT_EVAL_L3_MODE", "hybrid")
        self.use_rerank = (
            use_rerank
            if use_rerank is not None
            else _env_bool("KB_AGENT_EVAL_L3_RERANK", True)
        )
        self.generate_answers = (
            generate_answers
            if generate_answers is not None
            else _env_bool("KB_AGENT_EVAL_GENERATE", False)
        )
        self.use_llm_rewrite = (
            use_llm_rewrite
            if use_llm_rewrite is not None
            else _env_bool("KB_AGENT_EVAL_LLM_REWRITE", False)
        )
        self.use_llm_grader = (
            use_llm_grader
            if use_llm_grader is not None
            else _env_bool("KB_AGENT_EVAL_LLM_GRADER", False)
        )
        self.enable_pre_retrieval_routing = (
            enable_pre_retrieval_routing
            if enable_pre_retrieval_routing is not None
            else _env_bool("KB_AGENT_EVAL_PRE_ROUTING", True)
        )
        self.strict_route_filter = (
            strict_route_filter
            if strict_route_filter is not None
            else _env_bool("KB_AGENT_EVAL_STRICT_ROUTE_FILTER", True)
        )
        self.route_candidate_limit = int(
            route_candidate_limit
            if route_candidate_limit is not None
            else os.getenv("KB_AGENT_EVAL_ROUTE_CANDIDATE_LIMIT", "50")
        )
        self._service = service
        self._graph = graph
        self._offsets: dict[str, Any] = {}
        generation_mode = (
            "retrieval-only" if not self.generate_answers else "generated"
        )
        self.name = "kb-agent-l3-{}-{}-{}".format(
            generation_mode,
            self.mode,
            "rerank" if self.use_rerank else "norerank",
        )

    def validate(self) -> list[str]:
        if self._graph is not None:
            return []
        errors = _index_errors(self.strategy, self.mode)
        if self.top_k <= 0:
            errors.append("top_k must be positive")
        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")
        if self.route_candidate_limit <= 0:
            errors.append("route_candidate_limit must be positive")
        return errors

    def _ensure_ready(self) -> None:
        if self._graph is not None:
            return
        from agent.graph import build_graph, default_runtime
        from rag_adapter import rag
        from rag_adapter.settings import load_env, use_project_root_as_cwd

        use_project_root_as_cwd()
        load_env()
        if self._service is None:
            self._service = _build_service(
                rag,
                strategy=self.strategy,
                mode=self.mode,
                use_rerank=self.use_rerank,
                retrieval_only=not self.generate_answers,
            )
        runtime = default_runtime(
            strategy=self.strategy,
            use_llm_rewrite=self.use_llm_rewrite,
            generate_answers=self.generate_answers,
            max_retries=self.max_retries,
            service=self._service,
            use_llm_grader=self.use_llm_grader,
            top_k=self.top_k,
            enable_pre_retrieval_routing=self.enable_pre_retrieval_routing,
            strict_route_filter=self.strict_route_filter,
            route_candidate_limit=self.route_candidate_limit,
        )
        self._graph = build_graph(runtime)
        self._offsets = rag.load_page_offsets(self.strategy)

    def warmup(self, context: RunContext) -> None:
        self._ensure_ready()

    def run(self, sample: EvalSample, context: RunContext) -> Prediction:
        from agent.graph import ask

        self._ensure_ready()
        started = time.perf_counter()
        result = ask(
            self._graph,
            sample.query,
            thread_id=f"rag-eval-{context.run_id}-{sample.id}",
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        hits = list(result.get("retrieved_chunks") or [])
        chunks, documents, evidence = _map_hits(hits)
        citations = _map_citations(
            list(result.get("citations") or []), chunks
        )
        interrupted = bool(result.get("__interrupt__"))
        status = str(result.get("status") or "unknown")
        needs_review = bool(result.get("needs_human_review")) or interrupted
        if interrupted:
            status = "awaiting_human_review"
        if status in {"refused_no_evidence", "out_of_scope"}:
            behavior = "refuse"
        elif needs_review:
            behavior = "human_review"
        else:
            behavior = "answer"
        rewrite_history = list(result.get("rewrite_history") or [])
        answer = result.get("answer")
        text_for_tokens = " ".join(
            (hit.get("content") or "") for hit in hits[: self.top_k]
        ) + (answer or "")
        token_estimate = max(1, len(text_for_tokens) // 2)
        return Prediction(
            sample_id=sample.id,
            status="ok",
            documents=documents,
            chunks=chunks,
            answer=answer,
            citations=citations,
            evidence=evidence,
            latency_ms=elapsed_ms,
            usage={
                "backend": "kb_agent_l3",
                "total_tokens": token_estimate,
            },
            metadata={
                "run_id": context.run_id,
                "retrieval_mode": self.mode,
                "rerank": self.use_rerank,
                "behavior": behavior,
                "agent_status": status,
                "retry_count": int(result.get("retry_count") or 0),
                "rewrite_sources": len(rewrite_history),
                "needs_human_review": needs_review,
                "retrieval_score": result.get("retrieval_score") or {},
                "retrieval_stats": result.get("retrieval_stats") or {},
                "routes_planned": (
                    result.get("retrieval_stats") or {}
                ).get("routes_planned", 0),
                "repair_history": result.get("repair_history") or [],
                "generate_answers": self.generate_answers,
                "use_llm_rewrite": self.use_llm_rewrite,
                "use_llm_grader": self.use_llm_grader,
                "enable_pre_retrieval_routing":
                    self.enable_pre_retrieval_routing,
                "strict_route_filter": self.strict_route_filter,
                "route_candidate_limit": self.route_candidate_limit,
            },
        )


class KbAgentL3VectorAdapter(KbAgentL3Adapter):
    """Deterministic L3 orchestration with vector retrieval and no LLM."""

    def __init__(self, service: Any | None = None):
        super().__init__(
            mode="vector",
            use_rerank=False,
            generate_answers=False,
            use_llm_rewrite=False,
            use_llm_grader=False,
            service=service,
        )


class KbAgentAnswerJudge:
    """Bridge the existing kb-agent AnswerJudge into rag-eval's judge protocol."""

    name = "kb-agent-answer-judge"
    prompt_version = "kb-agent-answer-judge-v1"

    def __init__(self, service: Any | None = None, generator: Any | None = None):
        self._service = service
        self._generator = generator
        self._judge = None
        self.model = os.getenv("LLM_MODEL", "unknown")

    def warmup(self) -> None:
        if self._judge is not None:
            return
        if self._generator is None:
            if self._service is None:
                from rag_adapter import rag
                from rag_adapter.settings import load_env, use_project_root_as_cwd

                use_project_root_as_cwd()
                load_env()
                self._service = rag.build_service(
                    strategy="recursive", mode="hybrid", use_rerank=True
                )
            self._generator = self._service._pipeline.generator
        from evals.judge_answers import AnswerJudge

        self._judge = AnswerJudge(self._generator)
        self.model = os.getenv("LLM_MODEL", self.model)

    def judge(self, sample: EvalSample, prediction: Prediction) -> JudgeResult:
        self.warmup()
        metadata = sample.metadata
        item = {
            "question": sample.query,
            "expected_values": metadata.get("expected_values") or [],
            "expected_docs": metadata.get("expected_docs") or [],
        }
        outcome = self._judge.judge(item, prediction.answer or "")
        verdict = outcome.get("verdict")
        score = {"correct": 1.0, "partial": 0.5}.get(verdict, 0.0)
        parse_status = "ok"
        reason = str(outcome.get("reason") or "")
        if verdict == "unknown":
            score = None
            parse_status = (
                "call_failed" if "调用失败" in reason else "parse_failed"
            )
        return JudgeResult(
            metric="answer_correctness",
            score=score,
            rationale=reason,
            parse_status=parse_status,
            model=self.model,
            prompt_version=self.prompt_version,
        )
