"""用 PDFium 抽取 PDF 文本，并保留每页在正文中的字符区间。

为什么不用 production-rag 默认的 pypdf：
实测（2026-10-13）招商银行年报（600036 的 2023/2024/2025）正文使用字体子集且
缺少 ToUnicode 映射，pypdf 抽出来是乱码——600036_2023 只能抽出 206,412 字符，
关键词"营业收入"命中 0 次；同一份 PDF 用 PDFium 抽出 392,667 字符，
"营业收入" 20 次、"净利润" 29 次、"不良贷款" 20 次，中文与数字完整。

结论：kb-agent 的 PDF 文本抽取统一走 PDFium（单条管线，不做双管线并行），
切片 / embedding / 索引写入仍完全复用 production-rag。

页区间：逐页按 production-rag 的 DocumentCleaner 规则清洗后再用 "\\n\\n" 拼接，
因此每页的 [start, end) 与写库后 chunk 的 start_offset/end_offset 处在同一坐标系。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from app.ingestion.cleaner.cleaner import DocumentCleaner
from app.ingestion.document import Document
from app.ingestion.loader.base import BaseLoader


@dataclass
class PdfExtraction:
    """抽取结果：正文 + 每页字符区间 [[page_no, start, end], ...]。"""

    content: str
    pages: List[List[int]] = field(default_factory=list)

    def page_for_offset(self, offset: int) -> Optional[int]:
        """把正文坐标映射到页码；落在页面之外时返回 None。"""
        for page_no, start, end in self.pages:
            if start <= offset < end:
                return page_no
        return None


class PdfiumLoader(BaseLoader):
    """PDFium 文本抽取器（与 production-rag 的 BaseLoader 接口兼容）。"""

    def __init__(self, metadata: Optional[Dict] = None) -> None:
        self.metadata: Dict = dict(metadata or {})

    def extract(self, file_path: Union[str, Path]) -> PdfExtraction:
        import pypdfium2 as pdfium

        cleaner = DocumentCleaner()
        cursor = 0
        page_texts: List[str] = []
        pages: List[List[int]] = []

        pdf = pdfium.PdfDocument(str(file_path))
        try:
            for index in range(len(pdf)):
                text_page = pdf[index].get_textpage()
                try:
                    raw = text_page.get_text_range()
                finally:
                    text_page.close()

                clean = cleaner.clean(
                    Document(document_id="page", content=raw, metadata={})
                ).content
                if not clean:
                    continue
                if page_texts:
                    cursor += 2  # 分页拼接插入的 "\n\n"
                pages.append([index + 1, cursor, cursor + len(clean)])
                cursor += len(clean)
                page_texts.append(clean)
        finally:
            pdf.close()

        return PdfExtraction(content="\n\n".join(page_texts), pages=pages)

    def load(self, source: Union[str, Path]) -> Document:
        extraction = self.extract(source)
        metadata = dict(self.metadata)
        metadata.setdefault("document_id",
                            metadata.get("document_id") or Path(str(source)).stem)
        metadata.setdefault("page_count", len(extraction.pages))
        return Document(
            document_id=metadata["document_id"],
            content=extraction.content,
            metadata=metadata,
        )
