"""Run the kb-agent answer judge over a rag-eval generation report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_adapter import KbAgentAnswerJudge  # noqa: E402
from rag_eval.contracts import EvalSample, Prediction, load_dataset  # noqa: E402


def evaluate_judge(
    samples: list[EvalSample],
    predictions: list[Prediction],
    judge: Any,
) -> dict[str, Any]:
    prediction_by_id = {prediction.sample_id: prediction for prediction in predictions}
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    verdicts: dict[str, int] = {}
    parse_statuses: dict[str, int] = {}

    for sample in samples:
        if sample.gold.behavior != "answer":
            continue
        prediction = prediction_by_id.get(sample.id)
        if prediction is None or prediction.status != "ok" or not prediction.answer:
            rows.append(
                {
                    "sample_id": sample.id,
                    "score": None,
                    "parse_status": "missing_answer",
                    "rationale": "prediction has no generated answer",
                    "model": getattr(judge, "model", ""),
                    "prompt_version": getattr(judge, "prompt_version", ""),
                }
            )
            parse_statuses["missing_answer"] = (
                parse_statuses.get("missing_answer", 0) + 1
            )
            continue
        result = judge.judge(sample, prediction)
        row = result.model_dump(mode="json")
        row["sample_id"] = sample.id
        rows.append(row)
        parse_statuses[result.parse_status] = (
            parse_statuses.get(result.parse_status, 0) + 1
        )
        if result.score is not None:
            scores.append(result.score)
            verdict = (
                "correct"
                if result.score == 1.0
                else "partial"
                if result.score > 0.0
                else "incorrect"
            )
            verdicts[verdict] = verdicts.get(verdict, 0) + 1

    answerable = sum(1 for sample in samples if sample.gold.behavior == "answer")
    return {
        "schema_version": "1",
        "answerable": answerable,
        "judged": len(rows),
        "scored": len(scores),
        "answer_correct_rate": (
            round(sum(scores) / len(scores), 6) if scores else None
        ),
        "unknown_rate": (
            round(sum(1 for row in rows if row.get("score") is None) / len(rows), 6)
            if rows
            else 0.0
        ),
        "verdicts": verdicts,
        "parse_statuses": parse_statuses,
        "judge_model": getattr(judge, "model", ""),
        "prompt_version": getattr(judge, "prompt_version", ""),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="rag-eval run directory containing predictions.jsonl",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--strict-unknown",
        action="store_true",
        help="exit 1 when any judge call or parse fails",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"predictions not found: {predictions_path}")
    predictions = [
        Prediction.model_validate(json.loads(line))
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = load_dataset(args.dataset).samples
    payload = evaluate_judge(samples, predictions, KbAgentAnswerJudge())
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    unknown = sum(
        1 for row in payload["rows"] if row.get("score") is None
    )
    return 1 if args.strict_unknown and unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
