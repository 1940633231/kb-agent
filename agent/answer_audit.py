"""Claim-level answer audit with hard violations and soft review signals."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from agent.claims import AnswerClaim, extract_claims
from agent.evidence_contract import (
    DOCUMENT_FACETS,
    EvidenceObligation,
    assess_obligations,
)
from agent.grading import extract_facets, metric_mentioned, normalize
from agent.grounding import (
    detect_ambiguous_metric,
    has_statement_label,
    number_equivalents,
    verify_grounding,
)


def _chunk_facets(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = chunk.get("metadata") or {}
    document_id = str(
        chunk.get("document_id") or metadata.get("document_id") or "")
    try:
        year = int(metadata.get("year")) if metadata.get("year") is not None else None
    except (TypeError, ValueError):
        year = None
    if year is None and "-" in document_id:
        try:
            year = int(document_id.rsplit("-", 1)[1])
        except (TypeError, ValueError):
            year = None
    return {
        "document_id": document_id,
        "company": metadata.get("company"),
        "year": year,
    }


def _chunk_company(chunk: Dict[str, Any]) -> Optional[str]:
    """chunk 所属公司：优先 metadata，缺失时按 document_id 查文档登记表。"""
    facets = _chunk_facets(chunk)
    company = facets.get("company")
    if company:
        return company
    entry = DOCUMENT_FACETS.get(str(facets.get("document_id"))) or {}
    return entry.get("company")


def _citation_chunks(claim: AnswerClaim, citations: List[Dict],
                     chunks: List[Dict]) -> List[Dict]:
    by_number = {int(item.get("number")): item for item in citations or []
                 if item.get("number") is not None}
    by_chunk = {str(item.get("chunk_id")): item for item in chunks or []
                if item.get("chunk_id")}
    selected: List[Dict] = []
    for citation_id in claim.citation_ids:
        citation = by_number.get(citation_id)
        if citation is None:
            continue
        chunk = by_chunk.get(str(citation.get("chunk_id")))
        if chunk is not None:
            selected.append(chunk)
    return selected


def _number_present(value: Any, chunks: List[Dict]) -> bool:
    if value is None:
        return False
    text = normalize(str(value))
    variants = [normalize(item) for item in number_equivalents(text)]
    return any(
        any(variant and variant in normalize(chunk.get("content") or "")
            for variant in variants)
        for chunk in chunks
    )


def _has_metric(question: str, chunks: List[Dict]) -> bool:
    keywords = extract_facets(question)["keywords"]
    if not keywords:
        return True
    text = " ".join(chunk.get("content") or "" for chunk in chunks)
    return any(metric_mentioned(text, keyword) for keyword in keywords)


def _matched_metric(question: str, chunks: List[Dict]) -> Optional[str]:
    keywords = extract_facets(question)["keywords"]
    text = " ".join(chunk.get("content") or "" for chunk in chunks)
    for keyword in keywords:
        if metric_mentioned(text, keyword):
            return keyword
    return None


def _derived_supported(claim: AnswerClaim, citations: List[Dict],
                       chunks: List[Dict], question: str) -> bool:
    if claim.value is None:
        return False
    outcome = verify_grounding(
        claim.text_span, citations, chunks, question, accept_derived=True)
    target = normalize(str(claim.value))
    pool = {
        normalize(item)
        for item in (outcome.get("grounded") or [])
        + (outcome.get("derived") or [])
    }
    return target in pool


def _violation(code: str, claim: AnswerClaim, message: str,
               *, repairable: bool = True) -> Dict[str, Any]:
    return {
        "code": code,
        "claim_id": claim.claim_id,
        "claim_type": claim.claim_type,
        "message": message,
        "repairable": repairable,
    }


def _comparative_year_supported(claim: AnswerClaim,
                                cited: List[Dict]) -> bool:
    """上一年度数字常出现在下一年报的"上年同期"列，这是合法引用。

    只在同时满足以下条件时放行，避免弱化审计：
    1. 被引文档年份 = 声明年份 + 1（只有下一年报才可能带上年对比列）；
    2. 声明年份的文本确实出现在被引内容里；
    3. 数值型 Claim 还要求该年份出现在数值附近（±240 字窗口）。
    """
    if claim.year is None or not cited:
        return False
    try:
        target = int(claim.year)
    except (TypeError, ValueError):
        return False
    for chunk in cited:
        facets = _chunk_facets(chunk)
        doc_year = facets.get("year")
        if doc_year is None:
            continue
        try:
            if int(doc_year) != target + 1:
                continue
        except (TypeError, ValueError):
            continue
        content = normalize(chunk.get("content") or "")
        token = str(target)
        if token not in content:
            continue
        if claim.value is None:
            return True
        for variant in number_equivalents(normalize(str(claim.value))):
            needle = normalize(variant)
            if not needle:
                continue
            start = content.find(needle)
            while start >= 0:
                window = content[
                    max(0, start - 240): start + len(needle) + 240]
                if token in window:
                    return True
                start = content.find(needle, start + 1)
    return False


def audit_answer(answer: Optional[str],
                 citations: List[Dict],
                 chunks: List[Dict],
                 question: str,
                 obligations: Optional[
                     Iterable[EvidenceObligation | Dict[str, Any]]] = None
                 ) -> Dict[str, Any]:
    """Audit an answer and separate hard violations from soft warnings."""
    claims = extract_claims(answer, question)
    question_facets = extract_facets(question)
    expected_companies = set(question_facets.get("companies") or [])
    expected_years = {
        int(item) for item in question_facets.get("years") or []
    }
    hard: List[Dict[str, Any]] = []
    soft: List[Dict[str, Any]] = []
    repairable_claim_ids: List[str] = []
    citation_numbers = {
        int(item.get("number")) for item in citations or []
        if item.get("number") is not None
    }

    for claim in claims:
        if claim.claim_type == "citation":
            invalid = [item for item in claim.citation_ids
                       if item not in citation_numbers]
            if invalid:
                claim.support_status = "unsupported"
                hard.append(_violation(
                    "invalid_citation", claim,
                    "citation ids do not map to evidence: {}".format(invalid)))
            else:
                claim.support_status = "supported"
            continue

        if claim.claim_type == "entity_year":
            # 只有"公司+年份"这类可核对的事实断言才需要出处；纯公司名的
            # 引言句（如"……分别为："）既无年份也无数值，不应判硬违规。
            if not claim.year and not claim.citation_ids:
                claim.support_status = "soft"
                soft.append(_violation(
                    "uncited_entity_statement", claim,
                    "entity statement without year or citation"))
                continue
            cited = _citation_chunks(claim, citations, chunks)
            cited_facets = [_chunk_facets(item) for item in cited]
            company_ok = (
                not claim.company
                or any(item["company"] == claim.company for item in cited_facets)
            ) and (
                not claim.company
                or not expected_companies
                or claim.company in expected_companies
            )
            year_in_question_ok = (
                not claim.year
                or not expected_years
                or claim.year in expected_years
            )
            year_evidence_ok = (
                not claim.year
                or any(item["year"] == claim.year for item in cited_facets)
                or _comparative_year_supported(claim, cited)
            )
            year_ok = year_in_question_ok and year_evidence_ok
            if not (company_ok and year_ok):
                claim.support_status = "unsupported"
                hard.append(_violation(
                    "entity_year_mismatch", claim,
                    "company/year does not match the cited evidence"))
                repairable_claim_ids.append(claim.claim_id)
            else:
                claim.support_status = "supported"
            continue

        if claim.claim_type != "numeric":
            continue

        cited = _citation_chunks(claim, citations, chunks)
        if not claim.citation_ids:
            claim.support_status = "soft"
            soft.append(_violation(
                "uncited_numeric_claim", claim,
                "numeric claim has no citation"))
            continue
        if not cited:
            claim.support_status = "unsupported"
            hard.append(_violation(
                "invalid_citation", claim,
                "numeric claim citations do not map to chunks"))
            repairable_claim_ids.append(claim.claim_id)
            continue

        cited_facets = [_chunk_facets(item) for item in cited]
        company_ok = (
            not claim.company
            or any(item["company"] == claim.company for item in cited_facets)
        ) and (
            not claim.company
            or not expected_companies
            or claim.company in expected_companies
        )
        year_in_question_ok = (
            not claim.year
            or not expected_years
            or claim.year in expected_years
        )
        year_evidence_ok = (
            not claim.year
            or any(item["year"] == claim.year for item in cited_facets)
            or _comparative_year_supported(claim, cited)
        )
        year_ok = year_in_question_ok and year_evidence_ok
        if not company_ok or not year_ok:
            claim.support_status = "unsupported"
            hard.append(_violation(
                "entity_year_mismatch", claim,
                "company/year does not match the cited evidence"))
            repairable_claim_ids.append(claim.claim_id)
            continue

        number_ok = _number_present(claim.value, cited) or _derived_supported(
            claim, citations, chunks, question)
        if not number_ok:
            claim.support_status = "unsupported"
            hard.append(_violation(
                "unsupported_number", claim,
                "number does not appear in the cited evidence"))
            repairable_claim_ids.append(claim.claim_id)
            continue
        if not _has_metric(question, cited):
            claim.support_status = "unsupported"
            hard.append(_violation(
                "metric_mismatch", claim,
                "cited evidence does not contain the requested metric"))
            repairable_claim_ids.append(claim.claim_id)
            continue
        claim.support_status = "supported"

    parsed_obligations = [
        item if isinstance(item, EvidenceObligation)
        else EvidenceObligation.from_dict(item)
        for item in obligations or []
    ]
    missing_obligations: List[Dict[str, Any]] = []
    if parsed_obligations:
        obligation_result = assess_obligations(
            chunks, parsed_obligations, question)
        missing_obligations = [
            item for item in obligation_result["results"]
            if item["obligation_id"] in obligation_result["missing"]
        ]
        if missing_obligations:
            completeness = AnswerClaim(
                claim_id="completeness-1",
                claim_type="completeness",
                text_span=answer or "",
                support_status="unsupported",
            )
            claims.append(completeness)
            hard.append(_violation(
                "missing_obligation", completeness,
                "answer evidence misses required obligations: {}".format(
                    ", ".join(item["obligation_id"]
                              for item in missing_obligations)),
                repairable=False,
            ))

    ambiguity = detect_ambiguous_metric(question, chunks)
    if ambiguity.get("ambiguous") and not has_statement_label(answer or ""):
        soft.append({
            "code": "ambiguous_caliber",
            "claim_type": "completeness",
            "claim_id": None,
            "message": "same metric has multiple candidate calibers without a label",
            "repairable": False,
        })

    numeric_claims = [item for item in claims if item.claim_type == "numeric"]
    cited_numeric = [item for item in numeric_claims if item.citation_ids]
    citation_coverage = (
        round(len(cited_numeric) / len(numeric_claims), 4)
        if numeric_claims else None
    )
    if citation_coverage is not None and citation_coverage < 0.5:
        soft.append({
            "code": "low_citation_coverage",
            "claim_type": "citation",
            "claim_id": None,
            "message": "less than half of numeric claims carry citations",
            "repairable": False,
        })

    contract_items = []
    for claim in numeric_claims:
        cited = _citation_chunks(claim, citations, chunks)
        metric = _matched_metric(question, cited) if cited else None
        contract_items.append({
            "claim_id": claim.claim_id,
            "company": claim.company,
            "year": claim.year,
            "metric": metric,
            "raw_value": claim.value,
            "unit": claim.unit,
            "citation_ids": list(claim.citation_ids),
            "chunk_ids": [
                str(item.get("chunk_id")) for item in cited
                if item.get("chunk_id")
            ],
            "support_status": claim.support_status,
            "bound": bool(
                claim.company
                and claim.year
                and metric
                and claim.citation_ids
                and cited
                and claim.support_status == "supported"
            ),
        })

    return {
        "claims": [item.as_dict() for item in claims],
        "hard_violations": hard,
        "soft_issues": soft,
        "repairable_claim_ids": sorted(set(repairable_claim_ids)),
        "missing_obligations": missing_obligations,
        "citation_coverage": citation_coverage,
        "answer_contract": {
            "items": contract_items,
            "complete": bool(
                not hard
                and (
                    not contract_items
                    or all(item["bound"] for item in contract_items)
                )
            ),
        },
        "hard_gate_passed": not hard,
        "needs_human_review": bool(hard or soft),
    }


_SENTENCE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_CITATION_GROUP = re.compile(r"[\[【]\s*[\d\s,，、;；]+\s*[\]】]")
_REBINDABLE_CODES = (
    "entity_year_mismatch", "unsupported_number", "metric_mismatch")


def rebind_numeric_citations(answer: str, chunks: List[Dict],
                             question: str,
                             audit: Dict[str, Any]) -> Optional[str]:
    """把引错位置的数字 Claim 重绑到真正包含该数值的上下文片段。

    只处理已判定为硬违规的数值 Claim，且新片段必须同时满足：公司/年份与
    Claim 一致、数值逐字出现、片段含问题指标词。任一条件不满足就保留原
    违规——这里只修"引用绑错 chunk"，不放宽"数字对不上证据"的判定。

    只改该句里的引用编号，不动数字与措辞；同句里若有硬违规数值 Claim
    无法重绑，则整句跳过，避免连带破坏其它 Claim 的引用。无需重绑时返回
    None。
    """
    claim_ids = {
        str(item.get("claim_id"))
        for item in audit.get("hard_violations") or []
        if item.get("code") in _REBINDABLE_CODES
        and item.get("claim_type") == "numeric"
    }
    if not claim_ids:
        return None
    expected_companies = set(extract_facets(question)["companies"])
    chunk_facets = [_chunk_facets(item) for item in chunks]
    claims = [
        item for item in audit.get("claims") or []
        if item.get("claim_type") == "numeric"
    ]
    resolved: Dict[str, List[int]] = {}
    for claim in claims:
        if claim.get("value") is None:
            continue
        claim_id = str(claim.get("claim_id"))
        company = claim.get("company")
        year = claim.get("year")
        # 多公司问题里连公司都对不上的 Claim，不做重绑（宁可保留原违规）
        if not company and expected_companies:
            continue
        if company and expected_companies and company not in expected_companies:
            continue
        targets: List[int] = []
        for index, (chunk, facet) in enumerate(
                zip(chunks, chunk_facets), start=1):
            if company and _chunk_company(chunk) != company:
                continue
            if year is not None and facet.get("year") != int(year):
                continue
            if not _number_present(claim.get("value"), [chunk]):
                continue
            if not _has_metric(question, [chunk]):
                continue
            targets.append(index)
        if targets:
            resolved[claim_id] = targets[:2]
    if not resolved:
        return None

    edits: List[tuple] = []
    for match in _SENTENCE.finditer(answer or ""):
        start, end = match.span()
        sentence = (answer or "")[start:end]
        inside = [
            item for item in claims
            if start <= int(item.get("start") or 0) < end
        ]
        hard_inside = [
            item for item in inside
            if str(item.get("claim_id")) in claim_ids
        ]
        if not hard_inside or any(
                str(item.get("claim_id")) not in resolved
                for item in hard_inside):
            continue
        numbers: List[int] = []
        for item in inside:
            targets = resolved.get(str(item.get("claim_id")))
            numbers.extend(
                targets if targets
                else [int(n) for n in item.get("citation_ids") or []])
        markers = list(_CITATION_GROUP.finditer(sentence))
        if not markers:
            continue
        replacement = "".join(
            "[{}]".format(number) for number in sorted(set(numbers)))
        # 只替换标记本身：首个标记换成合并后的编号，其余标记删掉，
        # 标记之间的正文必须原样保留（同一句里常有多个数字）。
        pieces: List[str] = []
        cursor = 0
        for position, marker in enumerate(markers):
            pieces.append(sentence[cursor:marker.start()])
            if position == 0:
                pieces.append(replacement)
            cursor = marker.end()
        pieces.append(sentence[cursor:])
        rewritten = "".join(pieces)
        if rewritten != sentence:
            edits.append((start, end, rewritten))
    if not edits:
        return None
    updated = answer or ""
    for start, end, rewritten in reversed(edits):
        updated = updated[:start] + rewritten + updated[end:]
    return updated
