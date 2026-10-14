"""PDFium 抽取器：文本可用性与页区间映射。"""
from __future__ import annotations

import sys
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# kb-agent 同进程 import production-rag（app 包）；CI 里没有它时整模块跳过
pytest.importorskip("app.core.config", reason="需要 production-rag（app 包）")

from rag_adapter.pdfium_loader import PdfiumLoader  # noqa: E402
from rag_adapter.settings import ANNUAL_REPORTS_DIR  # noqa: E402

MAOTAI_2023 = ANNUAL_REPORTS_DIR / "600519_2023_annual_report.pdf"
CMB_2023 = ANNUAL_REPORTS_DIR / "600036_2023_annual_report.pdf"


def require(path: Path) -> Path:
    if not path.exists():
        pytest.skip("语料未抓取：{}".format(path.name))
    return path


def flat(text: str) -> str:
    """年报表格单元格会中途换行（如"归属于上市公 司股东的净利 润"），
    比对前先去掉全部空白。"""
    return re.sub(r"\s+", "", text)


def test_maotai_text_is_readable():
    extraction = PdfiumLoader().extract(require(MAOTAI_2023))
    assert len(extraction.content) > 100_000
    assert "研发投入合计" in extraction.content
    assert "归属于上市公司股东的净利润" in flat(extraction.content)


def test_cmb_text_is_not_mojibake():
    """招商银行是字体子集导致 pypdf 乱码的样本，PDFium 必须抽出可读中文。"""
    extraction = PdfiumLoader().extract(require(CMB_2023))
    assert "营业收入" in extraction.content
    assert "不良贷款率" in extraction.content
    assert "招商银行股份有限公司" in extraction.content


def test_page_offsets_are_monotonic_and_cover_content():
    extraction = PdfiumLoader().extract(require(MAOTAI_2023))
    pages = extraction.pages
    assert pages, "应至少抽出一页"
    for index, (page_no, start, end) in enumerate(pages):
        assert page_no >= 1
        assert start < end
        if index:
            assert start > pages[index - 1][2], "页区间应严格递增"
    assert pages[-1][2] == len(extraction.content)


def test_page_lookup_returns_page_number():
    extraction = PdfiumLoader().extract(require(MAOTAI_2023))
    marker = extraction.content.index("研发投入合计")
    page = extraction.page_for_offset(marker)
    assert page is not None
    assert 8 <= page <= 16, "研发投入情况表应在正文前段（实测第 11 页附近）"
