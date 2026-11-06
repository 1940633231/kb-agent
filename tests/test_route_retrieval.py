"""Route-scoped retrieval and defensive document filtering."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.evidence_contract import (  # noqa: E402
    assess_route_coverage,
    build_obligations,
)
from agent.evidence import scope_evidence  # noqa: E402
from fakes import FakeService, make_hit  # noqa: E402
from rag_adapter.rag import retrieve_routes_with_evidence  # noqa: E402


def hit(year: int, metric: str = "营业收入"):
    return make_hit(
        "600519-{}".format(year),
        "贵州茅台 {}年年度报告 {} 100,000,000.00".format(year, metric),
    )


def route(query_id: str, year: int, obligation_id: str):
    return {
        "query_id": query_id,
        "query": "贵州茅台 {}年 营业收入 年度报告".format(year),
        "obligation_ids": [obligation_id],
        "document_ids": ["600519-{}".format(year)],
        "company": "贵州茅台",
        "year": year,
        "metrics": ["营业收入"],
        "unrouted": False,
    }


def test_each_route_passes_and_returns_only_its_document():
    service = FakeService(lambda query: [
        hit(2023 if "2023" in query else 2024)
    ])

    hits, routes = retrieve_routes_with_evidence(
        service,
        [route("route-1", 2023, "obligation-1"),
         route("route-2", 2024, "obligation-2")],
        top_k=5,
    )

    assert service._pipeline.retriever.document_filters == [
        {"600519-2023"}, {"600519-2024"}
    ]
    assert [item["document_id"] for item in hits] == [
        "600519-2023", "600519-2024"
    ]
    assert routes[0]["hits"][0]["retrieval_routes"] == ["route-1"]
    assert routes[1]["hits"][0]["retrieval_routes"] == ["route-2"]


def test_adapter_drops_cross_document_hits_when_backend_ignores_filter():
    class IgnoringPipeline:
        reranker = None

        def __init__(self):
            self.document_filters = []

        def _search_all(self, query, queries, document_ids=None):
            self.document_filters.append(document_ids)
            return [hit(2023), hit(2024)]

        def _filter_active_versions(self, candidates):
            return candidates

    pipeline = IgnoringPipeline()
    service = SimpleNamespace(_pipeline=pipeline)

    hits, routes = retrieve_routes_with_evidence(
        service,
        [route("route-1", 2023, "obligation-1")],
        top_k=5,
        strict_route_filter=True,
    )

    assert pipeline.document_filters == [{"600519-2023"}]
    assert [item["document_id"] for item in hits] == ["600519-2023"]
    assert routes[0]["post_filter_dropped"] == 1
    assert routes[0]["route_filter_degraded"] is True


def test_empty_route_does_not_fall_back_to_another_year():
    service = FakeService(lambda query: [hit(2023)])
    obligations = build_obligations(
        "贵州茅台2023年与2024年的营业收入分别是多少元？")

    hits, routes = retrieve_routes_with_evidence(
        service,
        [route("route-1", 2023, "obligation-1"),
         route("route-2", 2024, "obligation-2")],
        top_k=5,
    )
    coverage = assess_route_coverage(routes, obligations)

    assert [item["document_id"] for item in hits] == ["600519-2023"]
    assert routes[1]["returned"] == 0
    assert routes[1]["hits"] == []
    assert coverage[1]["missing"] == ["obligation-2"]


def test_missing_document_mapping_does_not_search_whole_corpus():
    class RecordingPipeline:
        reranker = None

        def __init__(self):
            self.calls = 0

        def _search_all(self, query, queries, document_ids=None):
            self.calls += 1
            return [hit(2023)]

        def _filter_active_versions(self, candidates):
            return candidates

    pipeline = RecordingPipeline()
    service = SimpleNamespace(_pipeline=pipeline)
    missing_doc_route = route("route-1", 2030, "obligation-1")
    missing_doc_route["document_ids"] = []

    hits, routes = retrieve_routes_with_evidence(
        service,
        [missing_doc_route],
        top_k=5,
        strict_route_filter=True,
    )

    assert hits == []
    assert pipeline.calls == 0
    assert routes[0]["empty_reason"] == "no_document_ids"


def test_rerank_does_not_drop_route_provenance():
    class CopyingReranker:
        def rerank(self, query, candidates, top_k=5):
            return [
                {
                    "chunk_id": item["chunk_id"],
                    "document_id": item["document_id"],
                    "score": 0.99,
                    "content": item["content"],
                    "metadata": item["metadata"],
                }
                for item in candidates[:top_k]
            ]

    service = FakeService(
        lambda query: [hit(2023 if "2023" in query else 2024)],
        use_rerank=True,
    )
    service._pipeline.reranker = CopyingReranker()

    hits, routes = retrieve_routes_with_evidence(
        service,
        [route("route-1", 2023, "obligation-1"),
         route("route-2", 2024, "obligation-2")],
        top_k=5,
    )

    assert {item["document_id"]: item["retrieval_routes"]
            for item in hits} == {
        "600519-2023": ["route-1"],
        "600519-2024": ["route-2"],
    }
    assert routes[0]["hits"][0]["retrieval_routes"] == ["route-1"]
    assert routes[1]["hits"][0]["retrieval_routes"] == ["route-2"]


def test_strict_scope_does_not_fall_back_when_filter_result_is_empty():
    kept, stats = scope_evidence(
        "贵州茅台2023年的营业收入是多少元？",
        [hit(2024)],
        allow_fallback=False,
    )

    assert kept == []
    assert stats["scope_degraded"] is True
    assert stats["dropped"] == 1
