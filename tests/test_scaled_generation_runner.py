"""Scaled generation runner configuration and service-reuse regressions."""
from __future__ import annotations

import sys
import inspect
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

import run_scaled_eval as scaled  # noqa: E402


def test_generation_runner_reuses_service_and_records_l3b_config(
        monkeypatch, tmp_path):
    service = object()
    build_calls = []
    evaluate_calls = []

    def fake_build_service(**kwargs):
        build_calls.append(kwargs)
        return service

    def fake_evaluate(questions, args, service=None):
        evaluate_calls.append((args, service))
        return {
            "metrics": {
                "answer_value_rate": 0.5,
                "citation_full_coverage_rate": 0.9,
                "refuse_correct_rate": 1.0,
            },
            "rows": [{
                "id": "q-1",
                "question_type": "fact",
                "expected_behavior": "answer",
                "status": "answered_unverified",
                "answer_value_hit": False,
                "claim_support_rate": 0.5,
                "unsupported_claim_rate": 0.5,
                "grounding_verified": False,
                "ungrounded_numbers": ["123"],
                "wanted_docs": ["600519-2023"],
                "retrieved_docs": ["600519-2023"],
                "answer": "wrong answer",
            }],
        }

    written = {}

    def fake_write(name, payload):
        written["name"] = name
        written["payload"] = payload
        return tmp_path / name

    monkeypatch.setattr(scaled.rag, "build_service", fake_build_service)
    monkeypatch.setattr(scaled.agent_eval, "evaluate", fake_evaluate)
    monkeypatch.setattr(scaled, "write_report", fake_write)
    monkeypatch.setattr(scaled, "dataset_meta",
                        lambda questions: {
                            "evaluated": len(questions),
                            "available": len(questions),
                        })
    monkeypatch.setattr(scaled, "summarize_repeats",
                        lambda runs: {
                            key: {"mean": value}
                            for key, value in runs[0]["metrics"].items()
                        })

    payload = scaled.run_generation(
        [{"id": "q-1"}],
        repeats=2,
        top_k=5,
        generation_mode="hybrid",
        max_retries=2,
        report_name="scaled_generation_hybrid_l3b.json",
    )

    assert build_calls == [{
        "strategy": "recursive",
        "mode": "hybrid",
        "use_rerank": True,
    }]
    assert len(evaluate_calls) == 2
    assert [call[1] for call in evaluate_calls] == [service, service]
    assert all(call[0].max_retries == 2 for call in evaluate_calls)
    assert all(call[0].no_pre_retrieval_routing is False
               for call in evaluate_calls)
    assert all(call[0].no_strict_route_filter is False
               for call in evaluate_calls)
    assert written["name"] == "scaled_generation_hybrid_l3b.json"
    assert payload["config"]["max_retries"] == 2
    assert payload["temporary"] is False
    assert payload["formal_report"] is True
    assert payload["config"]["generation_mode"] == "hybrid"
    assert payload["config"]["route_candidate_limit"] == 50
    assert payload["config"]["enable_pre_retrieval_routing"] is True
    assert "bge-reranker-base" in payload["model"]
    assert payload["runs"][0]["questions"][0]["id"] == "q-1"
    assert payload["answer_failures"]["1"][0]["id"] == "q-1"
    assert payload["answer_failures"]["1"][0]["answer"] == "wrong answer"


def test_generation_runner_defaults_to_single_pass_vector() -> None:
    parameters = inspect.signature(scaled.run_generation).parameters
    assert parameters["max_retries"].default == 0
    assert parameters["generation_mode"].default == "vector"
    assert parameters["report_name"].default == "scaled_generation.json"


def test_temporary_generation_report_is_marked_and_renamed(monkeypatch, tmp_path):
    service = object()
    written = {}

    monkeypatch.setattr(scaled.rag, "build_service", lambda **kwargs: service)
    monkeypatch.setattr(
        scaled.agent_eval,
        "evaluate",
        lambda questions, args, service=None: {
            "metrics": {},
            "rows": [],
        },
    )
    monkeypatch.setattr(
        scaled,
        "dataset_meta",
        lambda questions: {
            "path": "dataset.jsonl",
            "evaluated": len(questions),
            "available": 120,
            "manifest": {"sha256": "same-sha"},
        },
    )
    monkeypatch.setattr(scaled, "summarize_repeats", lambda runs: {})
    monkeypatch.setattr(
        scaled,
        "write_report",
        lambda name, payload: written.update(name=name, payload=payload),
    )

    payload = scaled.run_generation(
        [{"id": "q-1"}],
        repeats=1,
        generation_mode="hybrid",
        report_name="scaled_generation.json",
    )

    assert payload["temporary"] is True
    assert payload["formal_report"] is False
    assert written["name"] == "scaled_generation_debug.json"
