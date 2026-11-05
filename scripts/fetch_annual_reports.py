"""Fetch A-share annual report PDFs from the CNINFO announcement API.

Data source (verified 2026-10-13):
    POST http://www.cninfo.com.cn/new/hisAnnouncement/query
    - A User-Agent header is required, otherwise the request is rejected.
    - Shanghai listings accept stock="600519,gssh0600519".
    - Shenzhen org ids are not reliably derivable (300750 rejected every
      prefix variant), so we query by searchkey="<company name>" instead.
    - Each item's adjunctUrl is turned into a direct PDF link:
      https://static.cninfo.com.cn/{adjunctUrl}

Outputs:
    data/docs/annual_reports/{ts_code}_{year}_annual_report.pdf
    data/docs/annual_reports/metadata.json   (title / announcement id / urls /
                                              bytes / sha256 / fetched_at)

Only the stdlib is used so this can run before the project venv exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
STATIC_BASE = "https://static.cninfo.com.cn/"
REFERER = "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_TARGETS_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "cninfo_targets.json"
)

# Excluded because they are summaries, translations, corrections or interim
# reports - we only want the full Chinese annual report of a fiscal year.
EXCLUDE_TITLE_PARTS = (
    "摘要", "英文", "更正", "已取消", "取消", "补充", "半年", "中期",
    "季度", "H股", "A股", "公告", "关于",
)

DEFAULT_YEARS = [2023, 2024, 2025]
MAX_QUERY_PAGES = 10
DOWNLOAD_DELAY_SECONDS = 1.0


def log(message: str) -> None:
    print(message, flush=True)


def _headers() -> dict:
    return {
        "User-Agent": USER_AGENT,
        "Referer": REFERER,
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }


def _as_bool(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def load_targets(path: Path, companies: list[str] | None = None,
                 include_disabled: bool = False) -> list[dict]:
    """Load, validate and normalize the external CNINFO target config."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_targets = payload.get("targets") if isinstance(payload, dict) else payload
    if not isinstance(raw_targets, list):
        raise ValueError("target config must be a list or contain a 'targets' list")

    targets: list[dict] = []
    seen_codes: set[str] = set()
    for index, raw in enumerate(raw_targets, start=1):
        if not isinstance(raw, dict):
            raise ValueError("target #{} must be an object".format(index))
        company = str(raw.get("company") or "").strip()
        ts_code = str(raw.get("ts_code") or "").strip()
        if not company or not re.fullmatch(r"\d{6}", ts_code):
            raise ValueError(
                "target #{} requires company and a six-digit ts_code".format(index)
            )
        if ts_code in seen_codes:
            raise ValueError("duplicate target ts_code: {}".format(ts_code))
        seen_codes.add(ts_code)

        enabled = _as_bool(raw.get("enabled"), default=True)
        stock = str(raw.get("stock") or "").strip()
        searchkey = str(raw.get("searchkey") or "").strip()
        if not stock and not searchkey:
            searchkey = company
        target = {
            "company": company,
            "ts_code": ts_code,
            "stock": stock,
            "searchkey": searchkey,
            "enabled": enabled,
        }
        if (enabled or include_disabled) and (
                not companies or company in companies):
            targets.append(target)

    if companies:
        requested = set(companies)
        available = {
            str(item.get("company"))
            for item in raw_targets
            if isinstance(item, dict)
        }
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError("unknown companies: {}".format(", ".join(unknown)))
    return targets


def dedupe_documents(documents: list[dict]) -> list[dict]:
    """Keep exactly one metadata record per (ts_code, year), last one wins."""
    by_key: dict[tuple[str, int], dict] = {}
    for document in documents or []:
        if not isinstance(document, dict):
            continue
        ts_code = str(document.get("ts_code") or "").strip()
        try:
            year = int(document.get("year"))
        except (TypeError, ValueError):
            continue
        if not ts_code:
            continue
        by_key[(ts_code, year)] = document
    return [by_key[key] for key in sorted(by_key)]


def registered_document_available(document: dict) -> bool:
    """A metadata record only counts as fetched when its PDF still exists."""
    raw_path = str(document.get("file") or "").strip()
    if not raw_path:
        return False
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.is_file()


def _request(url: str, data: bytes | None = None, retries: int = 4,
             timeout: int = 30) -> bytes:
    """GET/POST with exponential backoff. Raises on the final failure."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=_headers())
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retried below, surfaced at the end
            last_error = exc
            if attempt == retries:
                break
            backoff = 2 ** (attempt - 1)
            log(f"  ! {type(exc).__name__}: {exc} -> retry in {backoff}s "
                f"({attempt}/{retries - 1})")
            time.sleep(backoff)
    raise RuntimeError(f"request failed after {retries} attempts: {url}") from last_error


def query_announcements(target: dict, years: list[int],
                        max_pages: int = MAX_QUERY_PAGES) -> list[dict]:
    """Return every announcement row for one company, paging as needed."""
    del years  # title-year matching is done after the paged query
    rows: list[dict] = []
    page = 1
    stock = str(target.get("stock") or "").strip()
    searchkey = str(target.get("searchkey") or "").strip()
    if not stock and not searchkey:
        searchkey = str(target["company"])
    while page <= max_pages:
        form = {
            "pageNum": str(page),
            "pageSize": "30",
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": stock,
            "searchkey": searchkey,
            "secid": "",
            "category": "category_ndbg_szsh",
            "trade": "",
            "seDate": "",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        body = urllib.parse.urlencode(form).encode("utf-8")
        payload = json.loads(_request(QUERY_URL, data=body).decode("utf-8"))
        batch = payload.get("announcements") or []
        rows.extend(batch)
        total = int(payload.get("totalAnnouncement") or 0)
        if not batch or len(rows) >= total:
            break
        page += 1
        time.sleep(0.6)
    if page > max_pages and rows and len(rows) < total:
        log("  ! announcement pagination stopped at --max-pages={}".format(max_pages))
    return rows


def pick_annual_report(rows: list[dict], year: int) -> dict | None:
    """Pick the full Chinese annual report of `year` out of the announcement rows."""
    for row in rows:
        # searchkey queries return the title with <em> highlight tags
        title = re.sub(r"<[^>]+>", "", row.get("announcementTitle") or "").strip()
        if "年度报告" not in title:
            continue
        if any(part in title for part in EXCLUDE_TITLE_PARTS):
            continue
        match = re.search(r"(20\d{2})\s*年", title)
        if not match or int(match.group(1)) != year:
            continue
        return row
    return None


def download_pdf(adjunct_url: str, dest: Path) -> None:
    data = _request(STATIC_BASE + adjunct_url.lstrip("/"))
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"not a PDF payload: {adjunct_url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _failure(target: dict, year: int | None, stage: str,
             error: object) -> dict:
    return {
        "company": target.get("company"),
        "ts_code": target.get("ts_code"),
        "year": year,
        "stage": stage,
        "error": str(error),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch A-share annual reports")
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS_PATH),
                        help="external JSON target config")
    parser.add_argument("--out-dir", default="data/docs/annual_reports")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--companies", nargs="*", default=None,
                        help="company names to limit the run to")
    parser.add_argument("--include-candidates", action="store_true",
                        help="also fetch disabled replacement companies")
    parser.add_argument("--max-pages", type=int, default=MAX_QUERY_PAGES,
                        help="maximum announcement pages per company")
    parser.add_argument("--download-delay", type=float,
                        default=DOWNLOAD_DELAY_SECONDS,
                        help="seconds to wait before each PDF download")
    parser.add_argument("--report-path", default=None,
                        help="structured fetch report path "
                             "(default: <out-dir>/fetch_report.json)")
    args = parser.parse_args()

    targets_path = Path(args.targets)
    out_dir = Path(args.out_dir)
    metadata_path = out_dir / "metadata.json"
    report_path = (Path(args.report_path) if args.report_path
                   else out_dir / "fetch_report.json")
    targets = load_targets(targets_path, args.companies,
                           include_disabled=args.include_candidates)
    registry: dict = {"generated_at": datetime.now(timezone.utc).isoformat(),
                      "source": QUERY_URL,
                      "target_config": str(targets_path.as_posix()),
                      "documents": []}
    if metadata_path.exists():
        registry = json.loads(metadata_path.read_text(encoding="utf-8"))
        registry.setdefault("documents", [])
    registry["documents"] = dedupe_documents(registry["documents"])
    known = {
        (document["ts_code"], document["year"])
        for document in registry["documents"]
        if registered_document_available(document)
    }

    failures: list[dict] = []
    skipped = 0
    fetched = 0
    for target in targets:
        log(f"[{target['company']} {target['ts_code']}] querying announcements")
        try:
            rows = query_announcements(target, args.years, args.max_pages)
        except Exception as exc:  # noqa: BLE001 - reported as a failure at the end
            log(f"  ! query failed: {exc}")
            failures.append(_failure(target, None, "query", exc))
            continue
        log(f"  {len(rows)} announcement rows")

        for year in args.years:
            if (target["ts_code"], year) in known:
                log(f"  - {year}: already fetched, skipped")
                skipped += 1
                continue
            row = pick_annual_report(rows, year)
            if row is None:
                log(f"  ! {year}: no matching annual report found")
                failures.append(_failure(target, year, "select",
                                         "no matching annual report found"))
                continue

            title = re.sub(r"<[^>]+>", "", row["announcementTitle"] or "").strip()
            adjunct = row["adjunctUrl"]
            dest = out_dir / f"{target['ts_code']}_{year}_annual_report.pdf"
            try:
                if not dest.exists():
                    time.sleep(max(args.download_delay, 0.0))
                    download_pdf(adjunct, dest)
                record = {
                    "ts_code": target["ts_code"],
                    "company": target["company"],
                    "year": year,
                    "title": title,
                    "announcement_id": row.get("announcementId"),
                    "pdf_url": STATIC_BASE + adjunct.lstrip("/"),
                    "file": str(dest.as_posix()),
                    "bytes": dest.stat().st_size,
                    "sha256": sha256_of(dest),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc:  # noqa: BLE001 - reported as a failure at the end
                log(f"  ! {year}: download failed: {exc}")
                failures.append(_failure(target, year, "download", exc))
                continue

            registry["documents"] = [d for d in registry["documents"]
                                     if (d["ts_code"], d["year"]) != (target["ts_code"], year)]
            registry["documents"].append(record)
            known.add((target["ts_code"], year))
            fetched += 1
            log(f"  + {year}: {title} ({record['bytes']:,} bytes)")

    registry["generated_at"] = datetime.now(timezone.utc).isoformat()
    registry["documents"] = dedupe_documents(registry["documents"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(metadata_path, registry)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": QUERY_URL,
        "target_config": str(targets_path.as_posix()),
        "out_dir": str(out_dir.as_posix()),
        "years": args.years,
        "requested_targets": [
            {"company": target["company"], "ts_code": target["ts_code"]}
            for target in targets
        ],
        "fetched_count": fetched,
        "skipped_count": skipped,
        "failure_count": len(failures),
        "failures": failures,
        "registered_documents": len(registry["documents"]),
        "status": "failed" if failures else "ok",
    }
    _write_json(report_path, report)

    log(f"registry -> {metadata_path} ({len(registry['documents'])} documents)")
    log(f"fetch report -> {report_path} (status={report['status']})")
    if failures:
        log("failures:")
        for item in failures:
            log("  - {company} {year} [{stage}]: {error}".format(**item))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
