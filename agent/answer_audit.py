"""Claim-level answer audit with hard violations and soft review signals."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from agent.claims import AnswerClaim, extract_claims
from agent.evidence_contract import (
    EvidenceObligation,
    assess_obligations,
)
from agent.grading import extract_facets, normalize
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
    text = normalize(" ".join(chunk.get("content") or "" for chunk in chunks))
    return any(normalize(keyword) in text for keyword in keywords)


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
            year_ok = (
                not claim.year
                or any(item["year"] == claim.year for item in cited_facets)
            ) and (
                not claim.year
                or not expected_years
                or claim.year in expected_years
            )
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
        year_ok = (
            not claim.year
            or any(item["year"] == claim.year for item in cited_facets)
        ) and (
            not claim.year
            or not expected_years
            or claim.year in expected_years
        )
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

    return {
        "claims": [item.as_dict() for item in claims],
        "hard_violations": hard,
        "soft_issues": soft,
        "repairable_claim_ids": sorted(set(repairable_claim_ids)),
        "missing_obligations": missing_obligations,
        "citation_coverage": citation_coverage,
        "hard_gate_passed": not hard,
        "needs_human_review": bool(hard or soft),
    }
