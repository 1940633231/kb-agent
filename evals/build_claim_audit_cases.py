"""Generate the deterministic P1 claim-audit offline case set."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "evals" / "claim_audit_cases.jsonl"

COMPANIES = [
    ("贵州茅台", "600519", "营业收入", "147,693,604,994.14"),
    ("宁德时代", "300750", "营业收入", "400,917,044,900.00"),
    ("招商银行", "600036", "营业收入", "339,123,000,000.00"),
    ("比亚迪", "002594", "研发费用", "54,200,000,000.00"),
    ("美的集团", "000333", "净利润", "38,500,000,000.00"),
]


def make_chunk(company: str, code: str, year: int,
               metric: str, value: str) -> dict:
    return {
        "chunk_id": "{}-{}_chunk_1".format(code, year),
        "document_id": "{}-{}".format(code, year),
        "content": "{} {}年年度报告 {} {}".format(
            company, year, metric, value),
        "metadata": {
            "company": company,
            "year": year,
            "document_id": "{}-{}".format(code, year),
        },
    }


def make_citation(chunk: dict, number: int = 1) -> dict:
    return {
        "number": number,
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "page": 10 + number,
    }


def correct_cases() -> list[dict]:
    cases = []
    for company, code, metric, value in COMPANIES:
        for offset in range(10):
            year = 2023 + offset % 3
            chunk = make_chunk(company, code, year, metric, value)
            question = "{}{}年的{}是多少元？".format(company, year, metric)
            answer = "{}{}年{}为 {} 元 [1]。".format(
                company, year, metric, value)
            cases.append({
                "case_id": "correct-{:02d}".format(len(cases) + 1),
                "kind": "correct",
                "question": question,
                "answer": answer,
                "chunks": [chunk],
                "citations": [make_citation(chunk)],
                "expected_hard": [],
            })
    return cases


def polluted_cases() -> list[dict]:
    cases = []
    for index in range(50):
        company, code, metric, value = COMPANIES[index % len(COMPANIES)]
        year = 2023 + (index // len(COMPANIES)) % 3
        chunk = make_chunk(company, code, year, metric, value)
        error_type = [
            "unsupported_number",
            "entity_year_mismatch",
            "entity_year_mismatch",
            "metric_mismatch",
            "invalid_citation",
        ][index % 5]
        question = "{}{}年的{}是多少元？".format(company, year, metric)
        answer = "{}{}年{}为 {} 元 [1]。".format(
            company, year, metric, value)
        citations = [make_citation(chunk)]

        if error_type == "unsupported_number":
            answer = "{}{}年{}为 999,888,777,666.55 元 [1]。".format(
                company, year, metric)
        elif error_type == "entity_year_mismatch" and index % 2:
            wrong_company = next(
                item[0] for item in COMPANIES if item[0] != company)
            answer = "{}{}年{}为 {} 元 [1]。".format(
                wrong_company, year, metric, value)
        elif error_type == "entity_year_mismatch":
            answer = "{}{}年{}为 {} 元 [1]。".format(
                company, year + 1, metric, value)
        elif error_type == "metric_mismatch":
            chunk = make_chunk(company, code, year, "净利润", value)
            citations = [make_citation(chunk)]
            answer = "{}{}年{}为 {} 元 [1]。".format(
                company, year, metric, value)
        elif error_type == "invalid_citation":
            answer = "{}{}年{}为 {} 元 [2]。".format(
                company, year, metric, value)

        cases.append({
            "case_id": "polluted-{:02d}".format(len(cases) + 1),
            "kind": "polluted",
            "question": question,
            "answer": answer,
            "chunks": [chunk],
            "citations": citations,
            "expected_hard": [error_type],
        })
    return cases


def main() -> int:
    cases = correct_cases() + polluted_cases()
    OUTPUT.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False)
                  for item in cases) + "\n",
        encoding="utf-8",
    )
    print("wrote {} cases -> {}".format(len(cases), OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
