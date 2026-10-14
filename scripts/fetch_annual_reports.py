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

# Excluded because they are summaries, translations, corrections or interim
# reports - we only want the full Chinese annual report of a fiscal year.
EXCLUDE_TITLE_PARTS = (
    "摘要", "英文", "更正", "已取消", "取消", "补充", "半年", "中期",
    "季度", "H股", "A股", "公告", "关于",
)

TARGETS = [
    {
        "company": "贵州茅台",
        "ts_code": "600519",
        "stock": "600519,gssh0600519",
        "searchkey": "",
        "p0": True,
    },
    {
        "company": "宁德时代",
        "ts_code": "300750",
        "stock": "",
        "searchkey": "宁德时代",
        "p0": True,
    },
    {
        "company": "招商银行",
        "ts_code": "600036",
        "stock": "600036,gssh0600036",
        "searchkey": "",
        "p0": True,
    },
    # Candidates kept for the day a corpus document turns out to be unusable
    # (see README: 招商银行 PDFs extract as mojibake with pypdf).
    {
        "company": "中国平安",
        "ts_code": "601318",
        "stock": "",
        "searchkey": "中国平安",
        "p0": False,
    },
    {
        "company": "五粮液",
        "ts_code": "000858",
        "stock": "000858,gssz0000858",
        "searchkey": "",
        "p0": False,
    },
    {
        "company": "比亚迪",
        "ts_code": "002594",
        "stock": "",
        "searchkey": "比亚迪",
        "p0": False,
    },
    {
        "company": "美的集团",
        "ts_code": "000333",
        "stock": "",
        "searchkey": "美的集团",
        "p0": False,
    },
]

DEFAULT_YEARS = [2023, 2024, 2025]


def log(message: str) -> None:
    print(message, flush=True)


def _headers() -> dict:
    return {
        "User-Agent": USER_AGENT,
        "Referer": REFERER,
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }


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


def query_announcements(target: dict, years: list[int]) -> list[dict]:
    """Return every announcement row for one company, paging as needed."""
    rows: list[dict] = []
    page = 1
    while page <= 10:
        form = {
            "pageNum": str(page),
            "pageSize": "30",
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": target["stock"],
            "searchkey": target["searchkey"],
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch A-share annual reports")
    parser.add_argument("--out-dir", default="data/docs/annual_reports")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--companies", nargs="*", default=None,
                        help="company names to limit the run to")
    parser.add_argument("--include-candidates", action="store_true",
                        help="also fetch the non-P0 candidate companies")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    targets = [t for t in TARGETS
               if (t["p0"] or args.include_candidates)
               if not args.companies or t["company"] in args.companies]
    metadata_path = out_dir / "metadata.json"
    registry: dict = {"generated_at": datetime.now(timezone.utc).isoformat(),
                      "source": QUERY_URL, "documents": []}
    if metadata_path.exists():
        registry = json.loads(metadata_path.read_text(encoding="utf-8"))
        registry.setdefault("documents", [])
    known = {(d["ts_code"], d["year"]) for d in registry["documents"]}

    failures: list[str] = []
    for target in targets:
        log(f"[{target['company']} {target['ts_code']}] querying announcements")
        try:
            rows = query_announcements(target, args.years)
        except Exception as exc:  # noqa: BLE001 - reported as a failure at the end
            log(f"  ! query failed: {exc}")
            failures.append(f"{target['company']} query: {exc}")
            continue
        log(f"  {len(rows)} announcement rows")

        for year in args.years:
            if (target["ts_code"], year) in known:
                log(f"  - {year}: already fetched, skipped")
                continue
            row = pick_annual_report(rows, year)
            if row is None:
                log(f"  ! {year}: no matching annual report found")
                failures.append(f"{target['company']} {year}: not found")
                continue

            title = re.sub(r"<[^>]+>", "", row["announcementTitle"] or "").strip()
            adjunct = row["adjunctUrl"]
            dest = out_dir / f"{target['ts_code']}_{year}_annual_report.pdf"
            try:
                if not dest.exists():
                    time.sleep(1.0)
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
                failures.append(f"{target['company']} {year}: {exc}")
                continue

            registry["documents"] = [d for d in registry["documents"]
                                     if (d["ts_code"], d["year"]) != (target["ts_code"], year)]
            registry["documents"].append(record)
            known.add((target["ts_code"], year))
            log(f"  + {year}: {title} ({record['bytes']:,} bytes)")

    registry["generated_at"] = datetime.now(timezone.utc).isoformat()
    registry["documents"].sort(key=lambda d: (d["ts_code"], d["year"]))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"registry -> {metadata_path} ({len(registry['documents'])} documents)")
    if failures:
        log("failures:")
        for item in failures:
            log(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
