from __future__ import annotations

from pathlib import Path

from eval_adapter import (
    KbAgentAnswerJudge,
    KbAgentL3Adapter,
    KbAgentRetrievalAdapter,
    _build_service,
)
from rag_eval.contracts import EvalSample, Prediction, load_dataset
from rag_eval.registry import RunContext
from scripts.migrate_rag_eval_golden_set import migrate_golden_set
from scripts.run_rag_eval_judge import evaluate_judge
from tests.fakes import FakeGenerator, FakeService, make_hit, maotai_hit


class ExplodingGenerator(FakeGenerator):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        raise AssertionError("generation must stay disabled")

    def stream_generate(self, messages):
        self.calls += 1
        raise AssertionError("generation must stay disabled")


def _context() -> RunContext:
    return RunContext(run_id="kb-run", seed=42)


def _sample(task: str = "agent") -> EvalSample:
    return EvalSample.model_validate(
        {
            "id": "ar-fact-001",
            "task": task,
            "query": "贵州茅台2023年的营业收入是多少元？",
            "gold": {
                "documents": [
                    {
                        "document_id": "600519-2023",
                        "relevance": 1,
                        "spans": [],
                    }
                ],
                "answers": ["147,693,604,994.14"],
                "values": ["147,693,604,994.14"],
                "behavior": "answer",
            },
            "metadata": {
                "expected_values": [
                    {
                        "label": "营业收入",
                        "value": 147693604994.14,
                        "unit": "元",
                        "display": "147,693,604,994.14",
                    }
                ]
            },
        }
    )


def test_retrieval_adapter_does_not_call_generator() -> None:
    generator = ExplodingGenerator()
    service = FakeService(
        lambda query: [maotai_hit()],
        mode="vector",
        use_rerank=False,
        generator=generator,
    )
    adapter = KbAgentRetrievalAdapter(
        service=service,
        mode="vector",
        use_rerank=False,
        top_k=5,
    )

    prediction = adapter.run(_sample("retrieval"), _context())

    assert generator.calls == 0
    assert prediction.documents[0].document_id == "600519-2023"
    assert prediction.metadata["retrieval_only"] is True


def test_retrieval_only_service_uses_stub_generation(monkeypatch) -> None:
    captured = {}

    class FakeConfig:
        def __init__(self):
            self.data = {"generation": {"backend": "llm"}}

    class FakeRag:
        @staticmethod
        def build_config():
            return FakeConfig()

    class FakeRAGService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import app.rag.service

    monkeypatch.setattr(app.rag.service, "RAGService", FakeRAGService)
    _build_service(
        FakeRag,
        strategy="recursive",
        mode="vector",
        use_rerank=False,
        retrieval_only=True,
    )

    assert captured["config"].data["generation"]["backend"] == "stub"


def test_l3_adapter_defaults_to_retrieval_only() -> None:
    generator = ExplodingGenerator()
    service = FakeService(
        lambda query: [maotai_hit()],
        mode="hybrid",
        use_rerank=True,
        generator=generator,
    )
    adapter = KbAgentL3Adapter(
        service=service,
        generate_answers=False,
        use_llm_rewrite=False,
        use_llm_grader=False,
    )

    prediction = adapter.run(_sample("agent"), _context())

    assert generator.calls == 0
    assert prediction.answer is None
    assert prediction.documents[0].document_id == "600519-2023"
    assert prediction.metadata["agent_status"] == "retrieval_only"
    assert prediction.metadata["generate_answers"] is False


def test_l3_adapter_records_human_review_without_resuming() -> None:
    generator = ExplodingGenerator()
    service = FakeService(
        lambda query: [
            make_hit("600519-2022", "贵州茅台 2022 年年度报告 其他内容")
        ],
        mode="hybrid",
        use_rerank=True,
        generator=generator,
    )
    adapter = KbAgentL3Adapter(
        service=service,
        generate_answers=False,
        use_llm_rewrite=False,
        use_llm_grader=False,
        max_retries=0,
    )

    prediction = adapter.run(_sample("agent"), _context())

    assert generator.calls == 0
    assert prediction.metadata["behavior"] == "human_review"
    assert prediction.metadata["agent_status"] == "awaiting_human_review"
    assert prediction.metadata["needs_human_review"] is True


def test_answer_judge_bridges_to_rag_eval_result() -> None:
    generator = FakeGenerator(
        '{"verdict":"correct","reason":"value matches","category":"none"}'
    )
    judge = KbAgentAnswerJudge(generator=generator)

    result = judge.judge(
        _sample("agent"),
        Prediction(
            sample_id="ar-fact-001",
            answer="营业收入为 147,693,604,994.14 元。",
        ),
    )

    assert result.metric == "answer_correctness"
    assert result.score == 1.0
    assert result.parse_status == "ok"


def test_golden_set_migration_builds_two_50_question_datasets(
    tmp_path: Path,
) -> None:
    outputs = migrate_golden_set(tmp_path)

    retrieval = load_dataset(outputs["retrieval"])
    agent = load_dataset(outputs["agent"])

    assert len(retrieval.samples) == 50
    assert len(agent.samples) == 50
    assert retrieval.manifest.task == "retrieval"
    assert agent.manifest.task == "agent"
    assert all(sample.gold.documents for sample in agent.samples
               if sample.gold.behavior != "refuse")


def test_judge_summary_reports_unknown_rate() -> None:
    class Judge:
        model = "judge-model"
        prompt_version = "v1"

        def judge(self, sample, prediction):
            from rag_eval.judges.base import JudgeResult

            return JudgeResult(
                metric="answer_correctness",
                score=None,
                rationale="parse failed",
                parse_status="parse_failed",
                model=self.model,
                prompt_version=self.prompt_version,
            )

    payload = evaluate_judge(
        [_sample("agent")],
        [Prediction(sample_id="ar-fact-001", answer="answer")],
        Judge(),
    )

    assert payload["unknown_rate"] == 1.0
    assert payload["parse_statuses"]["parse_failed"] == 1
