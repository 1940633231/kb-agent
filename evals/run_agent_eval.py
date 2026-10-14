"""L3 评估：在 L2 检索配置上叠加自纠错重试，量化编排层的净收益（v2 §6.4）。

指标与 L0~L2 对齐（doc_recall@5 / all_docs_hit@5 / doc_mrr / evidence_hit@5），
另外报告编排层特有的量：
    retry_rate / mean_retries       有多少题真的触发了重试、平均几次
    rewrite_sources                 改写由 LLM 还是规则兜底产生
    evidence_hit_final              按 Agent 实际证据预算（不只 top-5）的命中
    refuse_flagged_rate             应拒答题是否被标成低置信度/待审

--generate 打开完整生成链路（需要 DASHSCOPE_API_KEY），额外报告：
    answer_value_rate    答案里出现期望数值的比例
    citation_rate        有引用的答案比例
    citation_valid_rate  引用编号都能映射回证据的比例
    refuse_correct_rate  应拒答题是否给出明确拒答

用法：
    python evals/run_agent_eval.py
    python evals/run_agent_eval.py --limit 5
    python evals/run_agent_eval.py --generate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = PROJECT_ROOT / "evals"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVALS_DIR))

import run_eval as base  # noqa: E402  （复用 golden set 与指标口径）

from agent.graph import ask, build_graph, default_runtime  # noqa: E402
from rag_adapter.settings import load_env, use_project_root_as_cwd  # noqa: E402

REFUSAL_MARKERS = ("无法回答", "无法确认", "没有相关", "未提及", "不足以", "没有找到",
                   "无法提供", "不包含", "未包含")


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def citation_is_valid(citation: Dict, chunks: List[Dict]) -> bool:
    chunk_ids = {chunk.get("chunk_id") for chunk in chunks}
    return citation.get("chunk_id") in chunk_ids


def evaluate(questions: List[Dict], args) -> Dict:
    runtime = default_runtime(generate_answers=args.generate,
                              use_llm_rewrite=not args.rule_rewrite,
                              max_retries=args.max_retries)
    graph = build_graph(runtime)
    offsets = base.rag.load_page_offsets(args.strategy)

    rows: List[Dict] = []
    for item in questions:
        started = time.time()
        result = ask(graph, item["question"], thread_id="l3-{}".format(item["id"]))
        elapsed = time.time() - started

        evidence = result.get("retrieved_chunks") or []
        described = [base.rag.describe_hit(hit, offsets) for hit in evidence]
        wanted = {"{}-{}".format(doc["ts_code"], doc["year"])
                  for doc in item.get("expected_docs") or []}
        hit_docs = {doc.get("document_id") for doc in described}
        covered = wanted & hit_docs

        joined_full = base.normalize(" ".join(hit.get("content") or ""
                                              for hit in evidence))
        joined_top = base.normalize(" ".join(hit.get("content") or ""
                                             for hit in evidence[:args.top_k]))
        values = item.get("expected_values") or []
        value_flags = [any(variant and base.normalize(variant) in joined_full
                           for variant in base.text_variants(entry))
                       for entry in values]
        probes = base.evidence_probes(item)
        probe_flags_full = [probe in joined_full for probe in probes]
        probe_flags_top = [probe in joined_top for probe in probes]
        keywords = item.get("expected_keywords") or []
        keyword_flags = [base.normalize(k) in joined_full for k in keywords]

        if probes:
            evidence_hit_final = all(probe_flags_full)
            evidence_hit_top = all(probe_flags_top)
        elif keyword_flags:
            evidence_hit_final = all(keyword_flags)
            evidence_hit_top = all(base.normalize(k) in joined_top for k in keywords)
        else:
            evidence_hit_final = evidence_hit_top = None

        rank_first = None
        for position, doc in enumerate(described[:args.top_k], 1):
            if doc["document_id"] in wanted:
                rank_first = position
                break

        answer = result.get("answer") or ""
        row = {
            "id": item["id"],
            "question_type": item["question_type"],
            "expected_behavior": item.get("expected_behavior"),
            "wanted_docs": sorted(wanted),
            "retrieved_docs": sorted(hit_docs),
            "doc_recall": (len(covered) / len(wanted)) if wanted else None,
            "all_docs_hit": (len(covered) == len(wanted)) if wanted else None,
            "first_relevant_rank_top5": rank_first,
            "evidence_hit_top5": evidence_hit_top,
            "evidence_hit_final": evidence_hit_final,
            "evidence_count": len(evidence),
            "retry_count": result.get("retry_count", 0),
            "rewrite_sources": [entry.get("source")
                                for entry in result.get("rewrite_history") or []],
            "graded_passed": (result.get("retrieval_score") or {}).get("passed"),
            "graded_reason": (result.get("retrieval_score") or {}).get("reason"),
            "needs_human_review": result.get("needs_human_review"),
            "status": result.get("status"),
            "answer_value_hit": (any(base.normalize(variant) in base.normalize(answer)
                                     for variant in base.text_variants(entry))
                                 for entry in values) if answer else None,
            "answer": answer,
            "citations": result.get("citations") or [],
            "citation_count": len(result.get("citations") or []),
            "citation_valid": all(citation_is_valid(c, result.get("context_chunks") or [])
                                  for c in (result.get("citations") or [])) if
                (result.get("citations") or []) else None,
            "refused": any(marker in answer for marker in REFUSAL_MARKERS),
            "seconds": round(elapsed, 2),
        }
        if row["answer_value_hit"] is not None:
            row["answer_value_hit"] = all(row["answer_value_hit"])
        rows.append(row)
        print("  {} retry={} docs={} evidence={} hit_top5={} hit_final={}".format(
            item["id"], row["retry_count"], len(hit_docs), row["evidence_count"],
            row["evidence_hit_top5"], row["evidence_hit_final"]))

    answerable = [r for r in rows if r["doc_recall"] is not None]
    scored = [r for r in rows if r["evidence_hit_top5"] is not None]
    refuse_rows = [r for r in rows if r["expected_behavior"] == "refuse"]
    retried = [r for r in rows if r["retry_count"]]

    metrics = {
        "answerable_questions": len(answerable),
        "doc_recall@%d" % args.top_k: mean([r["doc_recall"] for r in answerable]),
        "all_docs_hit@%d" % args.top_k: mean(
            [1.0 if r["all_docs_hit"] else 0.0 for r in answerable]),
        "doc_mrr": mean([1.0 / r["first_relevant_rank_top5"]
                         if r["first_relevant_rank_top5"] else 0.0
                         for r in answerable]),
        "evidence_hit@%d" % args.top_k: mean(
            [1.0 if r["evidence_hit_top5"] else 0.0 for r in scored]),
        "evidence_hit_final": mean([1.0 if r["evidence_hit_final"] else 0.0
                                    for r in scored]),
        "mean_evidence_count": mean([r["evidence_count"] for r in rows]),
        "retry_rate": round(len(retried) / len(rows), 4) if rows else None,
        "mean_retries": mean([r["retry_count"] for r in rows]),
        "refuse_flagged_rate": mean([1.0 if r["needs_human_review"] else 0.0
                                     for r in refuse_rows]),
        "mean_seconds_per_query": mean([r["seconds"] for r in rows]),
    }
    if args.generate:
        answered = [r for r in rows if r["expected_behavior"] == "answer"]
        metrics.update({
            "answer_value_rate": mean([1.0 if r["answer_value_hit"] else 0.0
                                       for r in answered]),
            "citation_rate": mean([1.0 if r["citation_count"] else 0.0
                                   for r in answered]),
            "citation_valid_rate": mean([1.0 if r["citation_valid"] else 0.0
                                         for r in answered
                                         if r["citation_valid"] is not None]),
            "refuse_correct_rate": mean([1.0 if r["refused"] else 0.0
                                         for r in refuse_rows]),
        })
    return {"rows": rows, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description="L3 self-correction evaluation")
    parser.add_argument("--strategy", default="recursive")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--golden-set", default=str(base.GOLDEN_SET_PATH))
    parser.add_argument("--rule-rewrite", action="store_true",
                        help="改写只用规则兜底（不调 LLM）")
    parser.add_argument("--generate", action="store_true",
                        help="打开生成链路（消耗 API 额度）")
    args = parser.parse_args()

    use_project_root_as_cwd()
    load_env()

    questions = base.load_golden_set(Path(args.golden_set))
    if args.limit:
        questions = questions[:args.limit]
    print("L3 评估: {} 题 (generate={}, llm_rewrite={})".format(
        len(questions), args.generate, not args.rule_rewrite))

    started = time.time()
    outcome = evaluate(questions, args)
    print()
    print(json.dumps(outcome["metrics"], ensure_ascii=False, indent=2))

    report = {
        "layer": "L3",
        "label": "+自纠错重试（本项目）",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"mode": "hybrid", "use_rerank": True, "top_k": args.top_k,
                   "strategy": args.strategy, "max_retries": args.max_retries,
                   "generate": args.generate,
                   "llm_rewrite": not args.rule_rewrite},
        "metrics": outcome["metrics"],
        "questions": outcome["rows"],
        "elapsed_seconds": round(time.time() - started, 1),
    }

    reports_dir = EVALS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    target = reports_dir / "l3_report{}.json".format("_gen" if args.generate else "")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("\n报告 -> {}".format(target))

    matrix_path = reports_dir / "layer_matrix.json"
    if matrix_path.exists():
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        matrix.setdefault("layers", {})["L3"] = {
            "label": report["label"], "config": report["config"],
            "metrics": outcome["metrics"]}
        matrix_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print("已并入分层矩阵 -> {}".format(matrix_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
