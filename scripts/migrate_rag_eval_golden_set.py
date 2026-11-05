"""Convert the kb-agent golden set into rag-eval dataset directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_SET_PATH = PROJECT_ROOT / "evals/golden_set.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evals/rag_eval"


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: sample must be an object")
        items.append(payload)
    return items


def _dataset_sample(item: dict[str, Any], task: str) -> dict[str, Any]:
    expected_docs = list(item.get("expected_docs") or [])
    values = [
        str(entry.get("display") or entry.get("value"))
        for entry in item.get("expected_values") or []
    ]
    return {
        "id": str(item["id"]),
        "task": task,
        "query": str(item["question"]),
        "history": [],
        "gold": {
            "documents": [
                {
                    "document_id": "{}-{}".format(
                        document["ts_code"], document["year"]
                    ),
                    "relevance": 1,
                    "spans": [],
                }
                for document in expected_docs
            ],
            "chunks": [],
            "answers": values,
            "values": values,
            "behavior": item.get("expected_behavior") or "answer",
        },
        "metadata": {
            "question_type": item.get("question_type"),
            "expected_docs": expected_docs,
            "expected_values": item.get("expected_values") or [],
            "expected_keywords": item.get("expected_keywords") or [],
            "evidence_literals": item.get("evidence_literals") or [],
            "source_pages": item.get("source_pages") or [],
            "notes": item.get("notes") or "",
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_dataset(
    output_dir: Path,
    *,
    dataset_id: str,
    task: str,
    description: str,
    items: list[dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": dataset_id,
        "version": "1.0.0",
        "task": task,
        "samples_file": "samples.jsonl",
        "qrels_file": "qrels.json",
        "description": description,
        "source": "kb-agent",
        "created_at": "2026-11-05T00:00:00+08:00",
    }
    samples = [_dataset_sample(item, task) for item in items]
    qrels = {
        sample["id"]: {
            document["document_id"]: 1.0
            for document in sample["gold"]["documents"]
        }
        for sample in samples
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_jsonl(output_dir / "samples.jsonl", samples)
    _write_json(output_dir / "qrels.json", qrels)
    return output_dir


def migrate_golden_set(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    golden_set_path: Path = GOLDEN_SET_PATH,
) -> dict[str, Path]:
    items = load_golden_set(golden_set_path)
    if len(items) != 50:
        raise ValueError(
            f"expected exactly 50 golden-set questions, found {len(items)}"
        )
    ids = [str(item.get("id") or "") for item in items]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("golden-set ids must be non-empty and unique")
    return {
        "retrieval": _write_dataset(
            output_root / "kb_agent_retrieval_50",
            dataset_id="kb-agent-annual-reports-retrieval-50",
            task="retrieval",
            description=(
                "50-question kb-agent retrieval-only annual-report evaluation set."
            ),
            items=items,
        ),
        "agent": _write_dataset(
            output_root / "kb_agent_l3_50",
            dataset_id="kb-agent-annual-reports-agent-50",
            task="agent",
            description=(
                "50-question kb-agent L3 self-correction evaluation set."
            ),
            items=items,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", default=str(GOLDEN_SET_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    outputs = migrate_golden_set(
        Path(args.output_root),
        Path(args.golden_set),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
