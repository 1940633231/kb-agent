"""证据 facet 裁剪：按问题点名的公司与年份收窄证据集合。

实测依据（17 道可答题，见 evals/reports/l3_report_gen.json）：
    证据混入同公司其他年份文档：11 道；混入其他公司文档：1 道（恰好就是答错那题）；
    答案引用到其他年份：5 道。
"重试只追加"（保证证据不丢）的代价是**污染也不丢**，所以需要一层显式裁剪。

裁剪规则（保守优先）：
    1. 只按问题**点名**的 facet 裁剪：公司名来自语料登记表、年份取问题里的 20xx；
       问题没点名公司就不按公司裁，没点名年份就不按年份裁；
    2. 多公司/多年份（跨文档题）取并集，绝不砍掉任何一个被点名的 facet；
    3. 裁剪后为空则回退原集合——宁可混入，也不能让证据为空导致必然拒答。

已知边界（不假装解决）：**"片段年份"不等于"数字年份"**。年报含上年对照列
（2024 年报的"主要会计数据"一行里同时有 2024 与 2023 的数），所以裁剪只解决
"混入"，解决不了"同一份文档里选错列 / 选错表 / 选错指标名"。
另外，按年份裁剪隐含"问题年份 = 该年份年报"的口径，用 year_window 可以放宽。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.grading import extract_facets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_REGISTRY = PROJECT_ROOT / "data" / "docs" / "annual_reports" / "metadata.json"


def _load_index() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """document_id → {ts_code, company, year}；公司名 → 代码。"""
    documents: Dict[str, Dict[str, Any]] = {}
    codes: Dict[str, str] = {}
    try:
        registry = json.loads(REPORTS_REGISTRY.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 语料缺失时不裁剪
        return documents, codes
    for entry in registry.get("documents", []):
        document_id = "{}-{}".format(entry["ts_code"], entry["year"])
        documents[document_id] = {"ts_code": entry["ts_code"],
                                  "company": entry["company"],
                                  "year": int(entry["year"])}
        codes[entry["company"]] = entry["ts_code"]
    return documents, codes


DOCUMENT_FACETS, COMPANY_CODES = _load_index()


def scope_evidence(question: str, hits: List[Dict], year_window: int = 0
                   ) -> Tuple[List[Dict], Dict[str, Any]]:
    """按问题点名的 facet 裁剪证据，返回 (保留的片段, 裁剪统计)。"""
    facets = extract_facets(question)
    codes = {COMPANY_CODES[name] for name in facets["companies"]
             if name in COMPANY_CODES}
    years = {int(year) for year in facets["years"]}
    total = len(hits)

    if not codes and not years:
        return hits, {"scoped": False, "reason": "问题未点名公司或年份",
                      "kept": total, "dropped": 0}

    kept: List[Dict] = []
    for hit in hits:
        document_id = hit.get("document_id") or \
            (hit.get("metadata") or {}).get("document_id") or ""
        entry = DOCUMENT_FACETS.get(document_id)
        if entry is None:
            kept.append(hit)          # 未知文档不裁（保守）
            continue
        if codes and entry["ts_code"] not in codes:
            continue
        if years and not any(abs(entry["year"] - year) <= year_window
                             for year in years):
            continue
        kept.append(hit)

    if not kept:
        return hits, {"scoped": False, "reason": "裁剪后为空，回退原集合",
                      "kept": total, "dropped": 0}
    return kept, {"scoped": True, "kept": len(kept), "dropped": total - len(kept),
                  "companies": sorted(codes), "years": sorted(years),
                  "year_window": year_window}
