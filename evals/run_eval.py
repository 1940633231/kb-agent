"""分层检索评估（v2 方案 §6.4 的对比矩阵）。

层次（每层只改一个变量）：
    L0  单路向量检索（基线，直接调 production-rag 向量检索）
    L1  L0 + 关键词检索 + RRF 融合（hybrid）
    L2  L1 + Cross-Encoder 重排
    L3  L2 + 自纠错重试（第一阶段的主图，本脚本暂不支持）

指标（全部只看检索，不调用生成）：
    doc_recall@5       期望文档是否进入 top-5（多文档题取覆盖比例）
    all_docs_hit@5     多文档题的期望文档是否全部命中
    doc_mrr            首个期望文档的排序倒数均值
    value_hit_rate     top-5 片段里出现期望数值的比例（证据级）
    evidence_hit@5     期望数值/关键词是否全部在 top-5 中出现
    refuse_top1_score  应拒答题的 top-1 相似度（为后续评分阈值提供依据）

产物：
    evals/baseline.json          L0 基线（§6.4 规则 2：灌库后立即测出）
    evals/reports/layer_matrix.json   本次跑过的所有层

用法：
    python evals/run_eval.py                 # 跑 L0/L1/L2
    python evals/run_eval.py --layers L0     # 只跑基线
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_adapter import rag  # noqa: E402
from rag_adapter.settings import (BASELINE_PATH, EVALS_DIR,  # noqa: E402
                                 GOLDEN_SET_PATH, load_env,
                                 use_project_root_as_cwd)

LAYERS = {
    "L0": {"mode": "vector", "use_rerank": False, "label": "单路向量（基线）"},
    "L1": {"mode": "hybrid", "use_rerank": False, "label": "+BM25+RRF"},
    "L2": {"mode": "hybrid", "use_rerank": True, "label": "+Rerank"},
}


def load_golden_set(path: Path) -> List[Dict]:
    items = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit("golden_set.jsonl 第 %d 行不是合法 JSON: %s" % (lineno, exc))
    return items


def _plain(value: float) -> str:
    """数值的可比较写法：去掉千分位与多余小数零。"""
    text = ("%.10f" % float(value)).rstrip("0").rstrip(".")
    return text


def _comma(value: float) -> str:
    return "{:,}".format(value) if float(value).is_integer() else "{:,}".format(value)


def text_variants(value_entry: Dict) -> List[str]:
    value = value_entry["value"]
    variants = {_plain(value), _comma(value)}
    # 百万元/千元口径在正文里也常以"亿元"表述，补一个换算版本
    if value_entry.get("unit") == "百万元":
        variants.add(_plain(round(float(value) / 100.0, 2)))
        variants.add(_comma(round(float(value) / 100.0, 2)))
    return [v for v in variants if v]


def normalize(text: str) -> str:
    return re.sub(r"[,\s\u3000]", "", text or "")


def evidence_probes(item: Dict) -> List[str]:
    """证据级判定用的字符串（已归一化）。

    优先用 item["evidence_literals"]：计算题与单位换算题的期望值（"616.89亿元"、
    "400,917,044,900元"）是算出来/换算出来的，原文里并不存在这个字面量，
    拿它去匹配会把"证据其实已经拿全"误判成未命中（第零阶段踩过这个坑）。
    没有 evidence_literals 时回退到期望值的千分位/无千分位变体。
    """
    literals = item.get("evidence_literals") or []
    if literals:
        return [normalize(literal) for literal in literals]
    return [normalize(variant)
            for entry in item.get("expected_values") or []
            for variant in text_variants(entry)]


def evaluate_layer(layer: str, questions: List[Dict], strategy: str,
                   top_k: int) -> Dict:
    config = LAYERS[layer]
    service = rag.build_service(strategy=strategy, mode=config["mode"],
                                use_rerank=config["use_rerank"])
    offsets = rag.load_page_offsets(strategy)

    per_question: List[Dict] = []
    for item in questions:
        started = time.time()
        hits = rag.retrieve(service, item["question"], top_k=top_k)
        elapsed = time.time() - started
        described = [rag.describe_hit(h, offsets) for h in hits]
        joined = normalize(" ".join(h.get("content") or "" for h in hits))

        expected_docs = item.get("expected_docs") or []
        hit_docs = {d.get("document_id") for d in described}
        wanted = {"{}-{}".format(d["ts_code"], d["year"]) for d in expected_docs}
        covered = wanted & hit_docs

        rank_first = None
        for position, doc in enumerate(described, 1):
            if doc["document_id"] in wanted:
                rank_first = position
                break

        values = item.get("expected_values") or []
        value_flags = [
            any(variant and normalize(variant) in joined
                for variant in text_variants(entry))
            for entry in values
        ]
        probes = evidence_probes(item)
        probe_flags = [probe in joined for probe in probes]
        keyword_flags = [normalize(k) in joined for k in item.get("expected_keywords") or []]

        if probes:
            evidence_hit = all(probe_flags)
        elif keyword_flags:
            evidence_hit = all(keyword_flags)
        else:
            evidence_hit = False  # 应拒答题不参与该指标

        per_question.append({
            "id": item["id"],
            "question_type": item["question_type"],
            "expected_behavior": item.get("expected_behavior"),
            "wanted_docs": sorted(wanted),
            "doc_recall": (len(covered) / len(wanted)) if wanted else None,
            "all_docs_hit": (len(covered) == len(wanted)) if wanted else None,
            "first_relevant_rank": rank_first,
            "value_hit_rate": (sum(value_flags) / len(value_flags)) if values else None,
            "probe_hit_rate": (sum(probe_flags) / len(probe_flags)) if probes else None,
            "evidence_hit": evidence_hit if (values or keyword_flags) else None,
            "top1_score": described[0]["score"] if described else None,
            "retrieved": described,
            "seconds": round(elapsed, 2),
        })

    answerable = [q for q in per_question if q["doc_recall"] is not None]
    scored = [q for q in per_question if q["evidence_hit"] is not None]
    refuse = [q for q in per_question if q["expected_behavior"] == "refuse"]

    def mean(values: Sequence[float]) -> Optional[float]:
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    metrics = {
        "answerable_questions": len(answerable),
        "doc_recall@%d" % top_k: mean([q["doc_recall"] for q in answerable]),
        "all_docs_hit@%d" % top_k: mean([1.0 if q["all_docs_hit"] else 0.0
                                        for q in answerable]),
        "doc_mrr": mean([1.0 / q["first_relevant_rank"] if q["first_relevant_rank"] else 0.0
                         for q in answerable]),
        "value_hit_rate": mean([q["value_hit_rate"] for q in scored]),
        "probe_hit_rate": mean([q["probe_hit_rate"] for q in scored]),
        "evidence_hit@%d" % top_k: mean([1.0 if q["evidence_hit"] else 0.0 for q in scored]),
        "refuse_top1_score": mean([q["top1_score"] for q in refuse]),
        "refuse_top1_max": max((q["top1_score"] or 0.0) for q in refuse) if refuse else None,
        "mean_seconds_per_query": mean([q["seconds"] for q in per_question]),
    }
    return {
        "layer": layer,
        "label": config["label"],
        "config": {"mode": config["mode"], "use_rerank": config["use_rerank"],
                   "top_k": top_k, "strategy": strategy},
        "metrics": metrics,
        "questions": per_question,
    }


def print_summary(reports: Dict[str, Dict], top_k: int) -> None:
    header = "{:<4} {:<16} {:>11} {:>11} {:>9} {:>10} {:>12}".format(
        "层", "说明", "Recall@%d" % top_k, "全命中率", "MRR", "证据命中", "拒答top1")
    print(header)
    print("-" * len(header))
    for layer in sorted(reports):
        report = reports[layer]
        m = report["metrics"]
        print("{:<4} {:<16} {:>11} {:>11} {:>9} {:>10} {:>12}".format(
            layer, report["label"],
            _fmt(m["doc_recall@%d" % top_k]), _fmt(m["all_docs_hit@%d" % top_k]),
            _fmt(m["doc_mrr"]), _fmt(m["evidence_hit@%d" % top_k]),
            _fmt(m["refuse_top1_score"])))


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else "{:.3f}".format(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Layered retrieval evaluation")
    parser.add_argument("--layers", nargs="+", default=["L0", "L1", "L2"],
                        choices=sorted(LAYERS))
    parser.add_argument("--strategy", default="recursive")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--golden-set", default=str(GOLDEN_SET_PATH))
    args = parser.parse_args()

    use_project_root_as_cwd()
    load_env()

    questions = load_golden_set(Path(args.golden_set))
    if not questions:
        print("golden set 为空，先写 evals/golden_set.jsonl")
        return 1
    by_type: Dict[str, int] = {}
    for item in questions:
        by_type[item["question_type"]] = by_type.get(item["question_type"], 0) + 1
    print("golden set: {} 题 {}".format(len(questions), by_type))

    reports: Dict[str, Dict] = {}
    for layer in args.layers:
        print("\n== {} {} ==".format(layer, LAYERS[layer]["label"]))
        reports[layer] = evaluate_layer(layer, questions, args.strategy, args.top_k)
        metrics = reports[layer]["metrics"]
        print("  Recall@{}={} 全命中={} MRR={} 证据命中={}".format(
            args.top_k, _fmt(metrics["doc_recall@%d" % args.top_k]),
            _fmt(metrics["all_docs_hit@%d" % args.top_k]),
            _fmt(metrics["doc_mrr"]),
            _fmt(metrics["evidence_hit@%d" % args.top_k])))

    print()
    print_summary(reports, args.top_k)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": args.strategy,
        "top_k": args.top_k,
        "golden_set": {"path": str(Path(args.golden_set)), "total": len(questions),
                       "by_type": by_type},
        "layers": {layer: {"label": report["label"], "config": report["config"],
                           "metrics": report["metrics"]}
                   for layer, report in reports.items()},
    }

    if "L0" in reports:
        baseline = dict(payload)
        baseline["baseline_layer"] = "L0"
        baseline["questions"] = reports["L0"]["questions"]
        BASELINE_PATH.write_text(json.dumps(baseline, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print("\nL0 基线 -> {}".format(BASELINE_PATH))

    reports_dir = EVALS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = reports_dir / "layer_matrix.json"
    matrix = dict(payload)
    matrix["questions"] = {layer: report["questions"] for layer, report in reports.items()}
    matrix_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print("分层矩阵 -> {}".format(matrix_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
