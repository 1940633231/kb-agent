"""Build the 120-question scaled evaluation set from indexed corpus chunks."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pypdfium2 as pdfium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "data" / "docs" / "annual_reports" / "metadata.json"
INDEX_METADATA_PATH = PROJECT_ROOT / "data" / "index" / "recursive" / "metadata.json"
PAGE_OFFSETS_PATH = PROJECT_ROOT / "data" / "index" / "recursive" / "page_offsets.json"
OUTPUT_PATH = PROJECT_ROOT / "evals" / "golden_set_scaled_120.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "evals" / "golden_set_scaled_120.manifest.json"

PREFERRED_METRICS = (
    "营业收入",
    "归属于上市公司股东的净利润",
    "净利润",
    "研发费用",
    "研发投入合计",
    "销售费用",
    "经营活动产生的现金流量净额",
    "总资产",
)
NUMBER_AFTER_METRIC = re.compile(
    r"(?P<number>-?\d[\d,]{3,}(?:\.\d+)?)")
UNIT_PATTERN = re.compile(r"单位\s*[：:]\s*(亿元|百万元|万元|千元|元)")
REVENUE_ROW = re.compile(
    r"(?P<label>营业总收入|营业收入合计|一、营业总收入|一、营业收入|"
    r"其中：营业收入|营业收入)\s*(?:[（(][^）)]{0,20}[）)])?\s*"
    r"(?P<value>-?\d[\d,]{3,}(?:\.\d+)?)"
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def obligation(company: str, year: int, metric: str) -> dict:
    return {
        "obligation_id": None,
        "company": company,
        "year": year,
        "metric": metric,
        "expected_kind": "single_value",
        "expected_value": None,
        "required": True,
        "source_scope": "annual_report",
        "route_ids": [],
    }


def extract_fact(chunk: dict,
                 metric_candidate: str | None = None
                 ) -> tuple[str, str, str] | None:
    content = chunk.get("content") or ""
    metrics = (metric_candidate,) if metric_candidate else PREFERRED_METRICS
    for metric in metrics:
        search_start = 0
        metric_index = content.find(metric, search_start)
        while metric_index >= 0:
            prefix = content[max(0, metric_index - 12):metric_index]
            if "占" in prefix or "比例" in prefix or "比重" in prefix:
                metric_index = content.find(metric, metric_index + len(metric))
                continue
            window = content[
                metric_index + len(metric):metric_index + len(metric) + 160]
            literal = None
            for match in NUMBER_AFTER_METRIC.finditer(window):
                if match.start() > 100:
                    break
                between = window[:match.start()]
                if re.search(r"(占|比例|比重|同比|增长|下降|较)", between):
                    continue
                candidate = match.group("number").rstrip(".")
                if len(candidate.replace(",", "").replace(".", "").replace("-", "")) < 4:
                    continue
                if re.fullmatch(r"20\d{2}", candidate):
                    continue
                if any(
                        other != metric and other in between
                        for other in PREFERRED_METRICS):
                    continue
                literal = candidate
                absolute_number = (
                    metric_index + len(metric) + match.start())
                break
            if literal is None:
                metric_index = content.find(metric, metric_index + len(metric))
                continue
            prefix = content[max(0, absolute_number - 240):absolute_number]
            nearby_units = re.findall(
                r"(亿元|百万元|万元|千元|元)", prefix)
            unit_match = UNIT_PATTERN.search(content)
            unit = (
                nearby_units[-1] if nearby_units
                else unit_match.group(1) if unit_match
                else "%" if "率" in metric else "元"
            )
            return metric, literal, unit
    return None


def page_for_chunk(chunk: dict, page_offsets: dict) -> int:
    document_id = str(chunk.get("document_id") or "")
    offset = int(chunk.get("start_offset") or 0)
    entry = page_offsets.get(document_id) or {}
    for page, start, end in entry.get("pages") or []:
        if int(start) <= offset < int(end):
            return int(page)
    return 1


def page_text_entries(document_id: str, chunks: list[dict],
                      page_offsets: dict) -> list[dict]:
    by_page: dict[int, list[dict]] = {}
    for chunk in chunks:
        page = page_for_chunk(chunk, page_offsets)
        by_page.setdefault(page, []).append(chunk)
    entries = []
    for page, page_chunks in sorted(by_page.items()):
        page_chunks.sort(key=lambda item: int(item.get("start_offset") or 0))
        parts = []
        spans = []
        cursor = 0
        for chunk in page_chunks:
            content = chunk.get("content") or ""
            if parts:
                cursor += 1
            start = cursor
            parts.append(content)
            cursor += len(content)
            spans.append((start, cursor, chunk))
        entries.append({
            "page": page,
            "text": "\n".join(parts),
            "spans": spans,
        })
    return entries


def revenue_candidates_from_text(
        text: str, page_index: int) -> list[tuple[int, int, int, str, str, str]]:
    """Select exact revenue rows from main data or consolidated statements."""
    has_main = "主要会计数据" in text
    has_consolidated = any(marker in text for marker in (
        "合并利润表", "合并及银行利润表", "合并及公司利润表"))
    has_parent_only = "母公司利润表" in text and not has_consolidated
    segment_table = (
        any(marker in text for marker in (
            "分行业", "分产品", "分地区", "分季度"))
        and not has_main and not has_consolidated
    )
    if has_parent_only or segment_table or not (has_main or has_consolidated):
        return []

    source_priority = 0 if has_consolidated else 1
    label_priority = {
        "一、营业总收入": 0,
        "营业总收入": 1,
        "一、营业收入": 2,
        "营业收入合计": 3,
        "其中：营业收入": 4,
        "营业收入": 5,
    }
    candidates = []
    for match in REVENUE_ROW.finditer(text):
        label = match.group("label")
        prefix = text[max(0, match.start() - 16):match.start()]
        between = text[match.end("label"):match.start("value")]
        if re.search(r"(占|比例|比重)", prefix):
            continue
        if re.search(r"(占|比例|比重)", between):
            continue
        if label == "其中：营业收入" and not has_consolidated:
            continue
        value = match.group("value").rstrip(".")
        if re.fullmatch(r"20\d{2}", value):
            continue
        value_position = match.start("value")
        unit_matches = list(re.finditer(
            r"(?:单位\s*[：:]\s*(?:人民币)?|人民币)\s*"
            r"(亿元|百万元|万元|千元|元)",
            text[:value_position],
        ))
        if not unit_matches:
            unit_matches = list(re.finditer(
                r"(亿元|百万元|万元|千元|元)",
                text[max(0, value_position - 700):value_position],
            ))
        if not unit_matches:
            continue
        unit = unit_matches[-1].group(1)
        candidates.append((
            source_priority,
            label_priority.get(label, 9),
            page_index + 1,
            value,
            unit,
            label,
        ))
    return candidates


def revenue_fact_from_pdf(pdf_path: Path) -> tuple[str, str, str, int] | None:
    best = None
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for page_index in range(len(pdf)):
            text_page = pdf[page_index].get_textpage()
            try:
                text = text_page.get_text_range()
            finally:
                text_page.close()
            for candidate in revenue_candidates_from_text(text, page_index):
                if best is None or candidate < best:
                    best = candidate
    finally:
        pdf.close()
    if best is None:
        return None
    return best[5], best[3], best[4], best[2]


def fact_record(source: dict, chunk: dict | None, page_offsets: dict,
                sequence: int, metric_candidate: str,
                fact_candidate: tuple[str, str, str] | None = None,
                page_override: int | None = None) -> dict:
    fact = fact_candidate or extract_fact(chunk, metric_candidate)
    if fact is None:
        raise ValueError("no extractable fact in chunk {}".format(
            chunk.get("chunk_id")))
    metric, literal, unit = fact
    document_id = "{}-{}".format(source["ts_code"], source["year"])
    value = float(literal.replace(",", ""))
    source_page = page_override or (
        page_for_chunk(chunk, page_offsets) if chunk else 1)
    obligation_entry = obligation(
        source["company"], int(source["year"]), metric)
    obligation_entry["obligation_id"] = "obligation-1"
    obligation_entry["route_ids"] = ["route-1"]
    claim = {
        "claim_id": "claim-1",
        "claim_type": "numeric",
        "company": source["company"],
        "year": int(source["year"]),
        "metric": metric,
        "expected_value": value,
        "unit": unit,
    }
    return {
        "id": "scaled-fact-{:03d}".format(sequence),
        "dataset_source": "company",
        "question_type": "table" if sequence % 5 == 0 else "fact",
        "expected_behavior": "answer",
        "question": "{}{}年的{}是多少{}？".format(
            source["company"], source["year"], metric,
            "" if unit == "%" else unit),
        "expected_docs": [{
            "company": source["company"],
            "year": int(source["year"]),
            "ts_code": source["ts_code"],
        }],
        "expected_keywords": [metric],
        "expected_values": [{
            "label": metric,
            "value": value,
            "unit": unit,
            "display": "{}{}".format(literal, "" if unit == "%" else unit),
        }],
        "evidence_literals": [literal],
        "source_pages": [{
            "doc": document_id,
            "page": source_page,
            **({"chunk_id": chunk.get("chunk_id")} if chunk else {}),
        }],
        "required_evidence": [obligation_entry],
        "required_claims": [claim],
        "route_expectations": [{
            "query_id": "route-1",
            "required_obligations": ["obligation-1"],
        }],
        "notes": (
            "自动抽取自 PDF 第 {} 页{}".format(
                source_page,
                "（chunk {}）".format(chunk.get("chunk_id"))
                if chunk else "",
            )
        ),
    }


def combine_cross(records: list[dict], sequence: int) -> dict:
    left, right = records
    metrics = [left["expected_keywords"][0], right["expected_keywords"][0]]
    values = []
    for record in (left, right):
        source_value = dict(record["expected_values"][0])
        source_value["label"] = "{}{}年{}".format(
            record["expected_docs"][0]["company"],
            record["expected_docs"][0]["year"],
            record["expected_keywords"][0],
        )
        values.append(source_value)
    left_company = left["expected_docs"][0]["company"]
    right_company = right["expected_docs"][0]["company"]
    left_year = left["expected_docs"][0]["year"]
    right_year = right["expected_docs"][0]["year"]
    if left_company != right_company:
        question = "{}与{}{}年的{}分别是多少？".format(
            left_company, right_company, left_year, metrics[0])
    else:
        question = "{}{}年与{}年的{}分别是多少？".format(
            left_company, left_year, right_year, metrics[0])
    obligations = []
    for index, record in enumerate((left, right), start=1):
        item = record["required_evidence"][0]
        item = json.loads(json.dumps(item, ensure_ascii=False))
        item["obligation_id"] = "obligation-{}".format(index)
        item["route_ids"] = ["route-{}".format(index)]
        obligations.append(item)
    claims = []
    for index, record in enumerate((left, right), start=1):
        item = record["required_claims"][0]
        item = json.loads(json.dumps(item, ensure_ascii=False))
        item["claim_id"] = "claim-{}".format(index)
        claims.append(item)
    return {
        "id": "scaled-cross-{:03d}".format(sequence),
        "dataset_source": "cross",
        "question_type": "cross_document",
        "expected_behavior": "answer",
        "question": question,
        "expected_docs": left["expected_docs"] + right["expected_docs"],
        "expected_keywords": sorted(set(metrics)),
        "expected_values": values,
        "evidence_literals": (
            left["evidence_literals"] + right["evidence_literals"]),
        "source_pages": left["source_pages"] + right["source_pages"],
        "required_evidence": obligations,
        "required_claims": claims,
        "route_expectations": [
            {
                "query_id": "route-{}".format(index),
                "required_obligations": ["obligation-{}".format(index)],
            }
            for index in (1, 2)
        ],
        "notes": "跨文档组合题",
    }


def refusal_records(sequence: int) -> dict:
    companies = [
        "贵州茅台", "宁德时代", "招商银行", "比亚迪", "美的集团",
        "中国平安", "隆基绿能", "海康威视", "迈瑞医疗", "中国神华",
        "华为", "字节跳动", "苹果公司", "OpenAI", "未收录公司",
    ]
    company = companies[(sequence - 1) % len(companies)]
    unsupported_year = 2026 if sequence % 2 else 2022
    question = "{}{}年的营业收入是多少元？".format(company, unsupported_year)
    return {
        "id": "scaled-refuse-{:03d}".format(sequence),
        "dataset_source": "refusal",
        "question_type": "refuse",
        "expected_behavior": "refuse",
        "question": question,
        "expected_docs": [],
        "expected_keywords": [],
        "expected_values": [],
        "evidence_literals": [],
        "source_pages": [],
        "required_evidence": [],
        "required_claims": [],
        "route_expectations": [],
        "notes": "超出年报语料公司或年份范围",
    }


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    index_metadata = json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8"))
    page_offsets = json.loads(PAGE_OFFSETS_PATH.read_text(encoding="utf-8"))
    by_document: dict[str, list[dict]] = {}
    for chunk in index_metadata.values():
        document_id = str(
            chunk.get("document_id")
            or (chunk.get("metadata") or {}).get("document_id")
            or "")
        if document_id:
            by_document.setdefault(document_id, []).append(chunk)

    sources = {}
    for entry in registry.get("documents") or []:
        key = (str(entry["ts_code"]), int(entry["year"]))
        sources[key] = entry

    facts = []
    for source in registry.get("documents") or []:
        document_id = "{}-{}".format(source["ts_code"], source["year"])
        pdf_path = Path(source["file"])
        if not pdf_path.is_absolute():
            pdf_path = PROJECT_ROOT / pdf_path
        selected = revenue_fact_from_pdf(pdf_path)
        if selected is None:
            raise RuntimeError("no revenue row for {}".format(document_id))
        label, literal, unit, page = selected
        metric = label.removeprefix("一、").removeprefix("其中：").strip()
        chunk = next(
            (item for item in by_document.get(document_id, [])
             if literal in (item.get("content") or "")
             and page_for_chunk(item, page_offsets) == page),
            None,
        )
        facts.append(fact_record(
            source, chunk, page_offsets, len(facts) + 1, metric,
            (metric, literal, unit), page_override=page))

    def metric_of(record: dict) -> str:
        return str(record["expected_keywords"][0])

    def doc_of(record: dict) -> dict:
        return record["expected_docs"][0]

    cross_records = []
    same_year_groups: dict[tuple[int, str], list[dict]] = {}
    for record in facts:
        doc = doc_of(record)
        key = (int(doc["year"]), metric_of(record))
        same_year_groups.setdefault(key, []).append(record)
    for key in sorted(same_year_groups):
        group = same_year_groups[key]
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                left, right = group[left_index], group[right_index]
                if doc_of(left)["ts_code"] == doc_of(right)["ts_code"]:
                    continue
                cross_records.append(combine_cross(
                    [left, right], len(cross_records) + 1))
                if len(cross_records) == 8:
                    break
            if len(cross_records) == 8:
                break
        if len(cross_records) == 8:
            break

    same_company_groups: dict[tuple[str, str], list[dict]] = {}
    for record in facts:
        doc = doc_of(record)
        key = (str(doc["ts_code"]), metric_of(record))
        same_company_groups.setdefault(key, []).append(record)
    for key in sorted(same_company_groups):
        group = sorted(
            same_company_groups[key], key=lambda item: doc_of(item)["year"])
        if len(group) < 2:
            continue
        cross_records.append(combine_cross(
            [group[0], group[1]], len(cross_records) + 1))
        if len(cross_records) == 15:
            break
    if len(cross_records) != 15:
        raise RuntimeError(
            "unable to build 15 metric-consistent cross questions; got {}".format(
                len(cross_records)))
    refusals = [refusal_records(index + 1) for index in range(15)]
    records = facts + cross_records + refusals
    if len(records) != 120:
        raise RuntimeError("expected 120 records, got {}".format(len(records)))

    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset_id": "kb-agent-scaled-120",
        "version": "v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "path": str(OUTPUT_PATH),
        "sha256": sha256_of(OUTPUT_PATH),
        "corpus_metadata_sha256": sha256_of(REGISTRY_PATH),
        "companies": len({
            item["expected_docs"][0]["ts_code"]
            for item in facts
        }),
        "documents": len(facts),
        "counts": {
            "company": len(facts),
            "cross": len(cross_records),
            "refusal": len(refusals),
            "total": len(records),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("wrote {} records -> {}".format(len(records), OUTPUT_PATH))
    print("manifest -> {}".format(MANIFEST_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
