"""在年报原文里定位关键词/数字，带页码输出。

写 golden set 和复核 expected_value 时用：先在这里找到原文出处与页码，
再把题目与断言写进 evals/golden_set.jsonl，保证每题都能回原文核对。

用法：
    python scripts/report_lookup.py --list
    python scripts/report_lookup.py -c 600519 -y 2023 -k 研发投入
    python scripts/report_lookup.py -c 600036 -y 2023 -k 不良贷款率 -n 3
    python scripts/report_lookup.py -c 600519 -y 2023 --page 12
    python scripts/report_lookup.py -k 毛利率 --all
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pypdfium2 as pdfium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "docs" / "annual_reports"


def load_registry() -> List[Dict]:
    registry = json.loads((REPORTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return registry["documents"]


def page_texts(pdf_path: Path) -> Iterable[Tuple[int, str]]:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for index in range(len(pdf)):
            text_page = pdf[index].get_textpage()
            try:
                text = text_page.get_text_range()
            finally:
                text_page.close()
            yield index + 1, text.replace("\r\n", "\n").replace("\r", "\n")
    finally:
        pdf.close()


def squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Look up text in fetched annual reports")
    parser.add_argument("-c", "--company", help="ts_code, e.g. 600519")
    parser.add_argument("-y", "--year", type=int)
    parser.add_argument("-k", "--keyword", action="append", default=[])
    parser.add_argument("-n", "--max-hits", type=int, default=2)
    parser.add_argument("--page", type=int, help="dump one page")
    parser.add_argument("-w", "--window", type=int, default=220,
                        help="characters of context per hit")
    parser.add_argument("--all", action="store_true", help="search every report")
    parser.add_argument("--list", action="store_true", help="list available reports")
    args = parser.parse_args()

    documents = load_registry()
    if args.list:
        for doc in documents:
            print("{ts_code} {year} {company} | {title}".format(**doc))
        return 0

    if args.all:
        selected = documents
    else:
        if not args.company or not args.year:
            parser.error("需要 -c/--company 与 -y/--year，或用 --all")
        selected = [d for d in documents
                    if d["ts_code"] == args.company and d["year"] == args.year]
        if not selected:
            parser.error("登记表里没有该报告")

    for doc in selected:
        pdf_path = Path(doc["file"])
        if not pdf_path.is_absolute():
            pdf_path = PROJECT_ROOT / pdf_path
        header = "== {ts_code} {year} {company} ==".format(**doc)
        printed_header = False

        for page_no, text in page_texts(pdf_path):
            if args.page and page_no != args.page:
                continue
            if args.page:
                if not printed_header:
                    print(header)
                    printed_header = True
                print("--- page {} ({} chars) ---".format(page_no, len(text)))
                print(squeeze(text))
                continue

            flat = squeeze(text)
            for keyword in args.keyword:
                start = 0
                hits = 0
                while hits < args.max_hits:
                    found = flat.find(keyword, start)
                    if found < 0:
                        break
                    if not printed_header:
                        print(header)
                        printed_header = True
                    begin = max(0, found - args.window // 2)
                    end = found + args.window // 2
                    print("p{:>3} [{}] ...{}...".format(
                        page_no, keyword, flat[begin:end]))
                    start = found + len(keyword)
                    hits += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
