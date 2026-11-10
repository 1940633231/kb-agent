"""Answer claims: small, independently auditable units of a generated answer."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from agent.grading import COMPANIES, extract_facets
from agent.grounding import extract_numbers

_NUMBER = re.compile(r"\d[\d,]{2,}\.?\d*")
_CITATION = re.compile(r"\[(\d+)\]")
_YEAR = re.compile(r"(20\d{2})年?")
_UNIT = re.compile(
    r"\s*(亿元|万元|千元|百万元|元|%|％|吨|户|人|个|倍|个百分点)"
)


@dataclass
class AnswerClaim:
    claim_id: str
    claim_type: str
    text_span: str
    value: Any = None
    unit: Optional[str] = None
    company: Optional[str] = None
    year: Optional[int] = None
    citation_ids: List[int] = None
    support_status: str = "unchecked"
    start: int = 0
    end: int = 0

    def __post_init__(self) -> None:
        if self.citation_ids is None:
            self.citation_ids = []

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AnswerClaim":
        return cls(
            claim_id=str(payload["claim_id"]),
            claim_type=str(payload["claim_type"]),
            text_span=str(payload.get("text_span") or ""),
            value=payload.get("value"),
            unit=payload.get("unit"),
            company=payload.get("company"),
            year=(int(payload["year"]) if payload.get("year") is not None
                  else None),
            citation_ids=[int(item) for item in payload.get("citation_ids") or []],
            support_status=str(payload.get("support_status") or "unchecked"),
            start=int(payload.get("start") or 0),
            end=int(payload.get("end") or 0),
        )


def _sentence_spans(text: str) -> List[tuple[int, int, str]]:
    spans: List[tuple[int, int, str]] = []
    for match in re.finditer(r"[^。！？!?；;\n]+[。！？!?；;]?", text):
        start, end = match.span()
        sentence = match.group(0).strip()
        if sentence:
            spans.append((start, end, sentence))
    return spans


def _nearest_company(sentence: str, position: int,
                     fallback: Optional[str]) -> Optional[str]:
    matches = [
        (sentence.rfind(company, 0, position), company)
        for company in COMPANIES if sentence.rfind(company, 0, position) >= 0
    ]
    return max(matches)[1] if matches else fallback


def _nearest_year(sentence: str, position: int,
                  fallback: Optional[int]) -> Optional[int]:
    matches = [
        (match.start(), int(match.group(1)))
        for match in _YEAR.finditer(sentence[:position])
    ]
    return max(matches)[1] if matches else fallback


def _sentence_citations(sentence: str) -> List[int]:
    return sorted({int(item) for item in _CITATION.findall(sentence)})


def extract_claims(answer: Optional[str],
                   question: Optional[str] = None) -> List[AnswerClaim]:
    """Extract numeric/entity-year/citation claims while preserving text spans."""
    text = answer or ""
    question_facets = extract_facets(question or "")
    fallback_company = (
        question_facets["companies"][0] if len(question_facets["companies"]) == 1
        else None
    )
    fallback_year = (
        int(question_facets["years"][0]) if len(question_facets["years"]) == 1
        else None
    )
    claims: List[AnswerClaim] = []

    for start, end, sentence in _sentence_spans(text):
        citation_ids = _sentence_citations(sentence)
        numeric_matches = list(_NUMBER.finditer(sentence))
        for match in numeric_matches:
            raw = match.group(0).rstrip(".")
            if raw not in extract_numbers(sentence):
                continue
            unit_match = _UNIT.match(sentence, match.end())
            company = _nearest_company(sentence, match.start(), fallback_company)
            year = _nearest_year(sentence, match.start(), fallback_year)
            try:
                value: Any = float(raw.replace(",", ""))
            except ValueError:
                value = raw
            claims.append(AnswerClaim(
                claim_id="claim-{}".format(len(claims) + 1),
                claim_type="numeric",
                text_span=sentence,
                value=value,
                unit=unit_match.group(1) if unit_match else None,
                company=company,
                year=year,
                citation_ids=list(citation_ids),
                start=start,
                end=end,
            ))

        if not numeric_matches:
            companies = [name for name in COMPANIES if name in sentence]
            years = [int(item) for item in _YEAR.findall(sentence)]
            if companies or years or citation_ids:
                claims.append(AnswerClaim(
                    claim_id="claim-{}".format(len(claims) + 1),
                    claim_type="entity_year",
                    text_span=sentence,
                    company=companies[0] if companies else fallback_company,
                    year=years[0] if years else fallback_year,
                    citation_ids=list(citation_ids),
                    start=start,
                    end=end,
                ))

        for citation_id in citation_ids:
            claims.append(AnswerClaim(
                claim_id="claim-{}".format(len(claims) + 1),
                claim_type="citation",
                text_span=sentence,
                citation_ids=[citation_id],
                start=start,
                end=end,
            ))
    return claims
