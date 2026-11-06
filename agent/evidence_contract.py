"""Evidence obligations: check every required company/year/metric explicitly.

The legacy grader answers "is coverage high enough?". That can still pass when a
multi-company or multi-year answer has the right total number of hits but is
missing one required slot. This module turns the question into a small contract
and grades each obligation independently.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agent.grading import extract_facets, normalize
from agent.grounding import extract_numbers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_REGISTRY = PROJECT_ROOT / "data" / "docs" / "annual_reports" / "metadata.json"

EXPECTED_KINDS = {
    "single_value",
    "multi_value",
    "calculation_input",
    "refusal",
    "comparison",
}
CALCULATION_MARKERS = (
    "增加", "减少", "差额", "相差", "比例", "占比", "较上年", "同比",
)
CALIBER_MARKERS = (
    "合并", "母公司", "主要会计数据", "利润表", "资产负债表",
    "分产品", "分地区", "分季度",
)


@dataclass(frozen=True)
class EvidenceObligation:
    obligation_id: str
    company: Optional[str]
    year: Optional[int]
    metric: Optional[str]
    expected_kind: str
    expected_value: Any = None
    required: bool = True
    source_scope: str = "annual_report"
    route_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.expected_kind not in EXPECTED_KINDS:
            raise ValueError("unsupported expected_kind: {}".format(self.expected_kind))

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["route_ids"] = list(self.route_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvidenceObligation":
        return cls(
            obligation_id=str(payload["obligation_id"]),
            company=payload.get("company"),
            year=int(payload["year"]) if payload.get("year") is not None else None,
            metric=payload.get("metric"),
            expected_kind=str(payload.get("expected_kind") or "single_value"),
            expected_value=payload.get("expected_value"),
            required=bool(payload.get("required", True)),
            source_scope=str(payload.get("source_scope") or "annual_report"),
            route_ids=tuple(payload.get("route_ids") or ()),
        )


@dataclass(frozen=True)
class RetrievalRoute:
    """A pre-retrieval route scoped to one company/year document set."""

    query_id: str
    query: str
    obligation_ids: tuple[str, ...] = field(default_factory=tuple)
    document_ids: tuple[str, ...] = field(default_factory=tuple)
    company: Optional[str] = None
    year: Optional[int] = None
    metrics: tuple[str, ...] = field(default_factory=tuple)
    #: True only when the question has no company/year facet and falls back to
    #: the legacy all-corpus query. A routed company/year with no matching
    #: document is intentionally different: it must not search the whole corpus.
    unrouted: bool = False

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["obligation_ids"] = list(self.obligation_ids)
        payload["document_ids"] = list(self.document_ids)
        payload["metrics"] = list(self.metrics)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RetrievalRoute":
        return cls(
            query_id=str(payload.get("query_id") or "route-1"),
            query=str(payload.get("query") or "").strip(),
            obligation_ids=tuple(
                str(item) for item in payload.get("obligation_ids") or []
            ),
            document_ids=tuple(
                str(item) for item in payload.get("document_ids") or []
            ),
            company=payload.get("company"),
            year=int(payload["year"]) if payload.get("year") is not None else None,
            metrics=tuple(str(item) for item in payload.get("metrics") or []),
            unrouted=bool(payload.get("unrouted", False)),
        )


def _load_document_facets() -> Dict[str, Dict[str, Any]]:
    try:
        registry = json.loads(REPORTS_REGISTRY.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - contract still works with hit metadata
        return {}
    facets: Dict[str, Dict[str, Any]] = {}
    for entry in registry.get("documents") or []:
        document_id = "{}-{}".format(entry.get("ts_code"), entry.get("year"))
        facets[document_id] = {
            "company": entry.get("company"),
            "year": int(entry.get("year")),
            "ts_code": entry.get("ts_code"),
        }
    return facets


DOCUMENT_FACETS = _load_document_facets()


def _document_ids_for_scope(company: Optional[str],
                            year: Optional[int]) -> tuple[str, ...]:
    if company and year is not None:
        return tuple(sorted(
            document_id
            for document_id, facets in DOCUMENT_FACETS.items()
            if facets.get("company") == company
            and facets.get("year") is not None
            and int(facets["year"]) == int(year)
        ))
    if company:
        return tuple(sorted(
            document_id
            for document_id, facets in DOCUMENT_FACETS.items()
            if facets.get("company") == company
        ))
    if year is not None:
        return tuple(sorted(
            document_id
            for document_id, facets in DOCUMENT_FACETS.items()
            if facets.get("year") is not None
            and int(facets["year"]) == int(year)
        ))
    return ()


def _route_query(question: str, company: Optional[str], year: Optional[int],
                 metrics: Sequence[str], *, single_route: bool) -> str:
    if single_route and question.strip():
        return question.strip()
    metric_text = "、".join(metrics)
    return " ".join(part for part in (
        company,
        "{}年".format(year) if year is not None else "",
        metric_text,
        "年度报告",
    ) if part).strip()


def build_initial_routes(
        question: str,
        obligations: Iterable[EvidenceObligation | Dict[str, Any]],
        facets: Optional[Dict[str, Any]] = None
        ) -> List[RetrievalRoute]:
    """Group non-refusal obligations by company/year and bind document IDs.

    The route order follows the obligation order (company-major, year-minor),
    which keeps traces deterministic and mirrors the evidence contract.
    """
    parsed = [
        item if isinstance(item, EvidenceObligation)
        else EvidenceObligation.from_dict(item)
        for item in obligations or []
    ]
    groups: Dict[tuple[Optional[str], Optional[int]],
                 List[EvidenceObligation]] = {}
    for obligation in parsed:
        if obligation.expected_kind == "refusal":
            continue
        key = (obligation.company, obligation.year)
        groups.setdefault(key, []).append(obligation)
    if not groups:
        return []

    def group_order(item):
        company, year = item[0]
        position = question.find(company) if company else len(question) + 1
        if position < 0:
            position = len(question) + 1
        return position, year if year is not None else 9999

    drafts: List[Dict[str, Any]] = []
    for (company, year), items in sorted(groups.items(), key=group_order):
        metrics: List[str] = []
        for item in items:
            if item.metric and item.metric not in metrics:
                metrics.append(item.metric)
        document_ids = _document_ids_for_scope(company, year)
        unrouted = company is None and year is None
        drafts.append({
            "obligation_ids": tuple(item.obligation_id for item in items),
            "document_ids": document_ids,
            "company": company,
            "year": year,
            "metrics": tuple(metrics),
            "unrouted": unrouted,
        })

    single_route = len(drafts) == 1
    routes: List[RetrievalRoute] = []
    for index, draft in enumerate(drafts, start=1):
        query = _route_query(
            question,
            draft["company"],
            draft["year"],
            draft["metrics"],
            single_route=single_route,
        )
        routes.append(RetrievalRoute(
            query_id="route-{}".format(index),
            query=query,
            **draft,
        ))
    return routes


def bind_routes_to_obligations(
        obligations: Iterable[EvidenceObligation | Dict[str, Any]],
        routes: Iterable[RetrievalRoute | Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
    """Return obligation dicts with their actual route IDs attached."""
    parsed_routes = [
        item if isinstance(item, RetrievalRoute)
        else RetrievalRoute.from_dict(item)
        for item in routes or []
    ]
    route_ids: Dict[str, List[str]] = {}
    for route in parsed_routes:
        for obligation_id in route.obligation_ids:
            route_ids.setdefault(obligation_id, []).append(route.query_id)

    bound: List[Dict[str, Any]] = []
    for item in obligations or []:
        obligation = (
            item if isinstance(item, EvidenceObligation)
            else EvidenceObligation.from_dict(item)
        )
        payload = obligation.as_dict()
        payload["route_ids"] = route_ids.get(obligation.obligation_id, [])
        bound.append(payload)
    return bound


def group_obligations_by_document(
        obligations: Iterable[EvidenceObligation | Dict[str, Any]]
        ) -> Dict[str, List[EvidenceObligation]]:
    """Map each mapped annual-report document to its required obligations."""
    groups: Dict[str, List[EvidenceObligation]] = {}
    for item in obligations or []:
        obligation = (
            item if isinstance(item, EvidenceObligation)
            else EvidenceObligation.from_dict(item)
        )
        if obligation.expected_kind == "refusal":
            continue
        for document_id in _document_ids_for_scope(
                obligation.company, obligation.year):
            groups.setdefault(document_id, []).append(obligation)
    return groups


def route_for_missing_obligations(
        question: str,
        missing_obligations: Iterable[EvidenceObligation | Dict[str, Any]],
        routes: Iterable[RetrievalRoute | Dict[str, Any]],
        rewritten_queries: Optional[Sequence[str]] = None,
        ) -> List[RetrievalRoute]:
    """Keep the original document scope and retry only missing slots."""
    missing = [
        item if isinstance(item, EvidenceObligation)
        else EvidenceObligation.from_dict(item)
        for item in missing_obligations or []
        if (item.get("expected_kind") if isinstance(item, dict)
            else item.expected_kind) != "refusal"
    ]
    if not missing:
        return []
    missing_by_id = {item.obligation_id: item for item in missing}
    missing_ids = set(missing_by_id)
    parsed_routes = [
        item if isinstance(item, RetrievalRoute)
        else RetrievalRoute.from_dict(item)
        for item in routes or []
    ]
    if not parsed_routes:
        parsed_routes = build_initial_routes(question, missing)

    query_candidates = [
        str(query).strip() for query in rewritten_queries or []
        if str(query).strip()
    ]
    retry_routes: List[RetrievalRoute] = []
    for route in parsed_routes:
        obligation_ids = tuple(
            obligation_id for obligation_id in route.obligation_ids
            if obligation_id in missing_ids
        )
        if not obligation_ids:
            continue
        metrics: List[str] = []
        for obligation_id in obligation_ids:
            metric = missing_by_id[obligation_id].metric
            if metric and metric not in metrics:
                metrics.append(metric)

        matched_queries = [
            query for query in query_candidates
            if (not route.company or route.company in query)
            and (route.year is None or str(route.year) in query)
            and (
                not metrics
                or any(metric in query for metric in metrics)
            )
        ]
        query = " ".join(dict.fromkeys(matched_queries)).strip()
        if not query:
            query = _route_query(
                question, route.company, route.year, metrics,
                single_route=False)
        retry_routes.append(RetrievalRoute(
            query_id=route.query_id,
            query=query,
            obligation_ids=obligation_ids,
            document_ids=route.document_ids,
            company=route.company,
            year=route.year,
            metrics=tuple(metrics),
            unrouted=route.unrouted,
        ))
    return retry_routes


def _kind_for_question(question: str, company_count: int,
                       year_count: int, metric_count: int) -> str:
    if not company_count:
        return "refusal"
    if any(marker in question for marker in CALCULATION_MARKERS):
        return "calculation_input"
    if company_count > 1:
        return "comparison"
    if year_count > 1 or metric_count > 1:
        return "multi_value"
    return "single_value"


def _metrics_by_company(question: str, companies: List[str],
                        metrics: List[str]) -> Dict[str, List[str]]:
    """Assign metrics to the company clause they appear in.

    A cross-company question can pair different metrics with different
    companies (e.g. "A 的研发费用与 B 的研发投入合计"). A full cartesian
    product would demand four slots instead of the two actually requested.
    """
    if len(companies) <= 1 or len(metrics) <= 1:
        return {company: list(metrics) for company in companies}
    positions = {
        company: question.find(company)
        for company in companies
    }
    ordered = sorted(companies, key=lambda item: positions[item])
    assigned: Dict[str, List[str]] = {}
    for index, company in enumerate(ordered):
        start = max(0, positions[company])
        end = (positions[ordered[index + 1]]
               if index + 1 < len(ordered) else len(question))
        clause = question[start:end]
        matched = [metric for metric in metrics if metric in clause]
        assigned[company] = matched
    if any(assigned.values()):
        for company in companies:
            if not assigned.get(company):
                assigned[company] = list(metrics)
    else:
        assigned = {company: list(metrics) for company in companies}
    return assigned


def _group_metric_slots(question: str, company: str,
                        metrics: List[str]) -> List[Optional[str]]:
    if len(metrics) <= 1:
        return list(metrics) or [None]
    start = max(0, question.find(company))
    next_positions = [
        position for position in (
            question.find(other) for other in _known_companies()
            if other != company
        ) if position > start
    ]
    end = min(next_positions) if next_positions else len(question)
    clause = question[start:end]
    separate = any(marker in clause for marker in ("与", "和", "、", "及", "分别"))
    separate = separate or any(marker in question for marker in CALCULATION_MARKERS)
    if separate:
        return list(metrics)
    return [" + ".join(metrics)]


def _known_companies() -> List[str]:
    return sorted({
        item.get("company") for item in DOCUMENT_FACETS.values()
        if item.get("company")
    }, key=len, reverse=True)


def scope_supported_by_corpus(
        question: str,
        facets: Optional[Dict[str, List[str]]] = None
        ) -> bool:
    """Whether the named company/year combination exists in the corpus."""
    facets = facets or extract_facets(question)
    companies = [str(item) for item in facets.get("companies") or []]
    years = [int(item) for item in facets.get("years") or []]
    if not companies:
        return False
    company_slots: Sequence[Optional[str]] = companies
    year_slots: Sequence[Optional[int]] = years or [None]
    return any(
        _document_ids_for_scope(company, year)
        for company in company_slots
        for year in year_slots
    )


def build_obligations(question: str,
                      facets: Optional[Dict[str, List[str]]] = None
                      ) -> List[EvidenceObligation]:
    """Derive one required evidence slot per company/year/metric combination."""
    facets = facets or extract_facets(question)
    companies = [str(item) for item in facets.get("companies") or []]
    years = [int(item) for item in facets.get("years") or []]
    metrics = [str(item) for item in facets.get("keywords") or []]

    if not companies and not years and not metrics:
        return []
    if not companies:
        return [EvidenceObligation(
            obligation_id="obligation-1",
            company=None,
            year=years[0] if len(years) == 1 else None,
            metric=metrics[0] if metrics else None,
            expected_kind="refusal",
            route_ids=("route-1",),
        )]

    company_slots: Sequence[Optional[str]] = companies or [None]
    year_slots: Sequence[Optional[int]] = years or [None]
    metrics_for_company = _metrics_by_company(question, companies, metrics)
    kind = _kind_for_question(question, len(companies), len(years), len(metrics))
    obligations: List[EvidenceObligation] = []
    for company in company_slots:
        metric_slots: Sequence[Optional[str]] = (
            _group_metric_slots(question, company, metrics_for_company.get(
                company, metrics))
            if company else (list(metrics) or [None])
        )
        for year in year_slots:
            for metric in metric_slots:
                index = len(obligations) + 1
                route_id = "route-{}".format(index)
                obligations.append(EvidenceObligation(
                    obligation_id="obligation-{}".format(index),
                    company=company,
                    year=year,
                    metric=metric,
                    expected_kind=kind,
                    route_ids=(route_id,),
                ))
    return obligations


def build_evidence_contract(question: str,
                            facets: Optional[Dict[str, List[str]]] = None
                            ) -> Dict[str, Any]:
    obligations = build_obligations(question, facets=facets)
    return {
        "obligations": [item.as_dict() for item in obligations],
        "required_count": sum(1 for item in obligations if item.required),
        "refusal_expected": bool(obligations) and all(
            item.expected_kind == "refusal" for item in obligations
        ),
    }


def obligation_query(obligation: EvidenceObligation | Dict[str, Any]) -> str:
    item = (obligation if isinstance(obligation, EvidenceObligation)
            else EvidenceObligation.from_dict(obligation))
    return " ".join(part for part in (
        item.company,
        "{}年".format(item.year) if item.year else "",
        item.metric,
        "年度报告",
    ) if part).strip()


def queries_for_obligations(
        obligations: Iterable[EvidenceObligation | Dict[str, Any]],
        max_queries: int = 8) -> List[str]:
    queries: List[str] = []
    for obligation in obligations or []:
        item = (obligation if isinstance(obligation, EvidenceObligation)
                else EvidenceObligation.from_dict(obligation))
        if item.expected_kind == "refusal":
            continue
        query = obligation_query(item)
        if query and query not in queries:
            queries.append(query)
    return queries[:max_queries]


def _hit_facets(hit: Dict[str, Any]) -> Dict[str, Any]:
    metadata = hit.get("metadata") or {}
    document_id = str(hit.get("document_id") or metadata.get("document_id") or "")
    derived = DOCUMENT_FACETS.get(document_id) or {}
    year_value = metadata.get("year", derived.get("year"))
    try:
        year = int(year_value) if year_value is not None else None
    except (TypeError, ValueError):
        year = None
    return {
        "document_id": document_id,
        "company": metadata.get("company") or derived.get("company"),
        "year": year,
    }


def hit_matches_obligation(hit: Dict[str, Any],
                           obligation: EvidenceObligation) -> bool:
    facets = _hit_facets(hit)
    if obligation.company and facets.get("company") != obligation.company:
        return False
    if obligation.year and facets.get("year") != obligation.year:
        return False
    if obligation.metric:
        content = normalize(hit.get("content") or "")
        metric_terms = [
            term.strip() for term in obligation.metric.split("+")
            if term.strip()
        ]
        if not metric_terms or not all(
                normalize(term) in content for term in metric_terms):
            return False
    if not obligation.company and not obligation.year and not obligation.metric:
        return False
    return True


def _route_ids(hit: Dict[str, Any]) -> List[str]:
    raw = hit.get("retrieval_routes") or hit.get("route_ids") or []
    if isinstance(raw, str):
        raw = [raw]
    return sorted({str(item) for item in raw if str(item).strip()})


def _ambiguous_calibers(hits: List[Dict[str, Any]],
                        metric: Optional[str]) -> Dict[str, Any]:
    """Flag only repeated metric mentions that carry different explicit calibers.

    A normal table row often contains current and prior-year columns. Those are
    not ambiguous by themselves. We require the same metric to appear at least
    twice in one chunk, with different statement/segment labels in the nearby
    text, before treating the obligation as ambiguous.
    """
    metric_text = normalize(metric or "")
    if not metric_text:
        return {"ambiguous": False}
    for hit in hits:
        flat = normalize(hit.get("content") or "")
        positions = [
            match.start()
            for match in re.finditer(re.escape(metric_text), flat)
        ]
        if len(positions) < 2:
            continue
        markers_by_position: List[List[str]] = []
        values: List[str] = []
        for position in positions:
            context = flat[max(0, position - 24):position + 80]
            markers_by_position.append([
                marker for marker in CALIBER_MARKERS if marker in context
            ])
            values.extend(extract_numbers(context))
        distinct_markers = {
            marker for markers in markers_by_position for marker in markers
        }
        if len(distinct_markers) >= 2 and len(set(values)) >= 2:
            document_id = _hit_facets(hit).get("document_id")
            return {
                "ambiguous": True,
                "document_id": document_id,
                "values": list(dict.fromkeys(values))[:4],
                "markers": sorted(distinct_markers),
            }
    return {"ambiguous": False}


def assess_obligations(
        hits: List[Dict[str, Any]],
        obligations: Iterable[EvidenceObligation | Dict[str, Any]],
        question: str = "") -> Dict[str, Any]:
    parsed = [
        item if isinstance(item, EvidenceObligation)
        else EvidenceObligation.from_dict(item)
        for item in obligations or []
    ]
    results: List[Dict[str, Any]] = []
    for obligation in parsed:
        if obligation.expected_kind == "refusal":
            results.append({
                **obligation.as_dict(),
                "status": "refusal",
                "matched_chunks": [],
                "matched_documents": [],
                "routes": [],
                "reason": "question names no supported company",
            })
            continue

        matched = [hit for hit in hits or []
                   if hit_matches_obligation(hit, obligation)]
        documents = sorted({
            _hit_facets(hit).get("document_id") for hit in matched
            if _hit_facets(hit).get("document_id")
        })
        routes = sorted({route for hit in matched for route in _route_ids(hit)})
        status = "covered" if matched else "missing"
        reason = "evidence slot covered" if matched else "missing evidence slot"

        if matched:
            ambiguity = _ambiguous_calibers(matched, obligation.metric)
            if ambiguity.get("ambiguous"):
                status = "ambiguous"
                reason = "same document has multiple candidate values for this metric"
                documents = sorted(set(documents) | {
                    str(ambiguity.get("document_id") or "")
                } - {""})

        results.append({
            **obligation.as_dict(),
            "status": status,
            "matched_chunks": [str(hit.get("chunk_id") or "") for hit in matched
                               if hit.get("chunk_id")],
            "matched_documents": documents,
            "routes": routes,
            "reason": reason,
        })

    covered = [item for item in results if item["status"] == "covered"]
    missing = [item for item in results if item["status"] == "missing"]
    ambiguous = [item for item in results if item["status"] == "ambiguous"]
    required = [item for item in results if item.get("required", True)]
    refusal_expected = bool(required) and all(
        item["status"] == "refusal" for item in required
    )
    return {
        "obligations": [item.as_dict() for item in parsed],
        "results": results,
        "covered": [item["obligation_id"] for item in covered],
        "missing": [item["obligation_id"] for item in missing],
        "ambiguous": [item["obligation_id"] for item in ambiguous],
        "coverage": round(len(covered) / len(required), 4) if required else 0.0,
        "passed": bool(required) and not missing and not ambiguous
        and not refusal_expected,
        "refusal_expected": refusal_expected,
    }


def assess_route_coverage(
        route_hits: Iterable[Dict[str, Any]],
        obligations: Iterable[EvidenceObligation | Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
    coverage: List[Dict[str, Any]] = []
    parsed_obligations = [
        item if isinstance(item, EvidenceObligation)
        else EvidenceObligation.from_dict(item)
        for item in obligations or []
    ]
    for route in route_hits or []:
        route_obligation_ids = {
            str(item) for item in route.get("obligation_ids") or []
        }
        scoped_obligations = (
            [
                item for item in parsed_obligations
                if item.obligation_id in route_obligation_ids
            ]
            if route_obligation_ids else parsed_obligations
        )
        result = assess_obligations(route.get("hits") or [], scoped_obligations)
        coverage.append({
            "query_id": route.get("query_id"),
            "query": route.get("query"),
            "obligation_ids": sorted(route_obligation_ids),
            "returned": len(route.get("hits") or []),
            "covered": result["covered"],
            "missing": result["missing"],
            "ambiguous": result["ambiguous"],
        })
    return coverage


def preserve_obligation_evidence(
        hits: List[Dict[str, Any]],
        route_hits: Iterable[Dict[str, Any]],
        obligations: Iterable[EvidenceObligation | Dict[str, Any]]
        ) -> tuple[List[Dict[str, Any]], List[str]]:
    """Append route candidates that satisfy obligations lost by global top-k.

    The reranker operates on a globally truncated pool. A route can therefore
    contain the one chunk needed for an obligation while that chunk falls just
    outside the merged top-k. We keep the route-ranked candidate for exactly
    those missing slots; this is additive and does not disturb the normal order.
    """
    parsed = [
        item if isinstance(item, EvidenceObligation)
        else EvidenceObligation.from_dict(item)
        for item in obligations or []
    ]
    merged = list(hits or [])
    assessment = assess_obligations(merged, parsed)
    missing_ids = set(assessment["missing"])
    positions = {
        str(hit.get("chunk_id") or hit.get("vector_id")): index
        for index, hit in enumerate(merged)
    }
    added: List[str] = []
    routes = list(route_hits or [])
    for obligation in parsed:
        if obligation.obligation_id not in missing_ids:
            continue
        for route in routes:
            for candidate in route.get("hits") or []:
                key = str(candidate.get("chunk_id") or candidate.get("vector_id"))
                if not hit_matches_obligation(candidate, obligation):
                    continue
                if key in positions:
                    current = merged[positions[key]]
                    if hit_matches_obligation(current, obligation):
                        continue
                    copied = dict(current)
                    copied.update(candidate)
                    copied["retrieval_routes"] = sorted(set(
                        current.get("retrieval_routes") or []
                    ) | set(candidate.get("retrieval_routes") or []))
                    copied["obligation_fill"] = obligation.obligation_id
                    merged[positions[key]] = copied
                    added.append(obligation.obligation_id)
                    break
                copied = dict(candidate)
                copied["obligation_fill"] = obligation.obligation_id
                merged.append(copied)
                positions[key] = len(merged) - 1
                added.append(obligation.obligation_id)
                break
            if obligation.obligation_id in added:
                break
    return merged, added
