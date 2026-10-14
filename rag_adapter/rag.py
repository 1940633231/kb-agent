"""对 production-rag 的薄封装：本地轻量配置、年报装载、检索入口。

接入方式与 v2 方案 §2.2 一致：同进程 import production-rag（可编辑安装），
不引入跨服务调用。这一层只负责"检索 + 返回候选与分数"，
相关性判定与重试属于主图（第一阶段），不在这里实现。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.config import Config
from app.ingestion.document import Document
from app.rag.service import RAGService

from rag_adapter.pdfium_loader import PdfiumLoader
from rag_adapter.settings import (CONFIG_PATH, PROJECT_ROOT, REPORTS_REGISTRY,
                                 index_dir, page_offsets_path)


def build_config() -> Config:
    """本地轻量配置（见 configs/kb_agent.yaml）。"""
    return Config(config_path=str(CONFIG_PATH))


def build_service(strategy: str = "recursive", mode: str = "vector",
                  use_rerank: bool = False,
                  tenant_id: str = "default") -> RAGService:
    """构建 RAGService；mode 取 vector / bm25 / hybrid。"""
    return RAGService(config=build_config(), strategy=strategy, mode=mode,
                      use_rerank=use_rerank, tenant_id=tenant_id)


def retrieve(service: RAGService, query: str, top_k: int = 5) -> List[Dict]:
    """只做检索、不做生成。

    复刻 RAGPipeline._retrieve_and_context 的检索段：
        _search_all（rerank 启用时候选池放宽到 rerank_candidate_pool）
        → 活跃版本过滤 → Cross-Encoder 重排（或按分数排序）→ 截断 top_k

    为什么不能只调 pipeline.retriever.search()：那样会跳过重排，L2 就退化成
    "RRF 排序 + 更宽候选池"，层与层之间的增量归因不成立（第零阶段踩过这个坑）。
    也不能调 pipeline.run()：它会顺便调用 LLM 生成，检索指标不该依赖生成。
    上游只暴露 query()，这里以不改动 production-rag 代码为约束。
    """
    pipeline = service._pipeline  # noqa: SLF001
    candidates = pipeline._search_all(query, [query])  # noqa: SLF001
    candidates = pipeline._filter_active_versions(candidates)  # noqa: SLF001
    if pipeline.reranker is not None:
        return pipeline.reranker.rerank(query, candidates, top_k=top_k)
    return sorted(candidates, key=lambda r: r.get("score", 0.0),
                  reverse=True)[:top_k]


def retrieve_multi(service: RAGService, queries: List[str], top_k: int = 5,
                   anchor: Optional[str] = None) -> List[Dict]:
    """多路查询一次检索：各路取候选池 → 合并去重 → **只重排一次** → 截断 top_k。

    为什么不逐路各自重排再合并：逐路重排要跑 N 次 Cross-Encoder（CPU 上每次
    50 个候选约 25s，见下），而且各路的分数不在同一尺度上、合并顺序没有意义。
    这里对齐 pipeline 的多路召回语义（_search_all 按路均分候选池后合并），
    再用 anchor（用户原始问题）统一重排，排序口径一致。
    """
    pipeline = service._pipeline  # noqa: SLF001
    ordered = [q for q in (queries or []) if q and q.strip()]
    if not ordered:
        return []
    primary = anchor or ordered[0]
    candidates = pipeline._search_all(primary, ordered)  # noqa: SLF001
    candidates = pipeline._filter_active_versions(candidates)  # noqa: SLF001
    candidates = _dedupe_keep_best(candidates)
    if pipeline.reranker is not None:
        return pipeline.reranker.rerank(primary, candidates, top_k=top_k)
    return sorted(candidates, key=lambda r: r.get("score", 0.0),
                  reverse=True)[:top_k]


def _dedupe_keep_best(candidates: List[Dict]) -> List[Dict]:
    """按 chunk_id 去重并保留最高分（多路合并后的防御性去重）。"""
    best: Dict[str, Dict] = {}
    for hit in candidates or []:
        key = str(hit.get("chunk_id") or hit.get("vector_id"))
        current = best.get(key)
        if current is None or float(hit.get("score", 0.0)) > float(
                current.get("score", 0.0)):
            best[key] = hit
    return list(best.values())


def accumulate_evidence(previous: List[Dict], incoming: List[Dict],
                        limit: int) -> List[Dict]:
    """把本轮新检索到的候选**追加**到已有证据之后（按 chunk_id 去重）。

    自纠错只能做加法：第一轮的证据并不因为"不够完整"就作废，把它丢掉会让
    重试后的证据反而不如不重试（L3 第一版实测到的负向结果就是这么来的）。

    为什么不按分数全局重排：各轮查询不同，RRF/重排 logits 跨轮不可比；
    保留"第一轮在前、后续轮追加"的顺序，可以保证 L3 的 top-5 与 L2 完全一致，
    L3 的增量只体现在更靠后的追加证据上，便于归因。
    """
    merged = list(previous or [])
    seen = {str(hit.get("chunk_id") or hit.get("vector_id")) for hit in merged}
    for hit in incoming or []:
        key = str(hit.get("chunk_id") or hit.get("vector_id"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged[:limit]


def load_annual_reports(registry_path: Optional[Path] = None
                        ) -> Tuple[List[Document], Dict[str, Dict]]:
    """按抓取登记表读取年报，返回 (documents, page_offsets)。"""
    registry_file = registry_path or REPORTS_REGISTRY
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    documents: List[Document] = []
    offsets: Dict[str, Dict] = {}

    for entry in registry["documents"]:
        pdf_path = Path(entry["file"])
        if not pdf_path.is_absolute():
            pdf_path = PROJECT_ROOT / pdf_path
        document_id = "{}-{}".format(entry["ts_code"], entry["year"])

        loader = PdfiumLoader({
            "document_id": document_id,
            "document_title": entry["title"],
            "source": pdf_path.name,
            "url": entry["pdf_url"],
            "company": entry["company"],
            "ts_code": entry["ts_code"],
            "year": entry["year"],
            "doc_type": "annual_report",
            "content_hash": entry["sha256"],
        })
        extraction = loader.extract(pdf_path)
        documents.append(Document(
            document_id=document_id,
            content=extraction.content,
            metadata=dict(loader.metadata, page_count=len(extraction.pages)),
        ))
        offsets[document_id] = {
            "title": entry["title"],
            "company": entry["company"],
            "year": entry["year"],
            "pages": extraction.pages,
        }
    return documents, offsets


def save_page_offsets(offsets: Dict[str, Dict], strategy: str = "recursive") -> Path:
    target = page_offsets_path(strategy)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(offsets, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


def load_page_offsets(strategy: str = "recursive") -> Dict[str, Dict]:
    target = page_offsets_path(strategy)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def page_for_offset(offsets: Dict[str, Dict], document_id: str,
                    offset: int) -> Optional[int]:
    entry = offsets.get(document_id)
    if not entry:
        return None
    for page_no, start, end in entry["pages"]:
        if start <= offset < end:
            return page_no
    return None


def describe_hit(hit: Dict, offsets: Optional[Dict[str, Dict]] = None) -> Dict:
    """把检索命中整理成评估/引用需要的字段（公司、年份、页码）。"""
    metadata = hit.get("metadata") or {}
    document_id = hit.get("document_id") or metadata.get("document_id") or ""
    page = None
    if offsets and document_id:
        page = page_for_offset(offsets, document_id,
                              int(hit.get("start_offset") or 0))
    return {
        "document_id": document_id,
        "company": metadata.get("company"),
        "year": metadata.get("year"),
        "title": metadata.get("document_title"),
        "page": page,
        "score": hit.get("score"),
        "chunk_id": hit.get("chunk_id"),
        "snippet": (hit.get("content") or "")[:160].replace("\n", " "),
    }


def enrich_citation(citation: Dict, offsets: Dict[str, Dict],
                    chunks: Optional[List[Dict]] = None) -> Dict:
    """给引用补上可追溯信息：文档标题、公司、年份与原文页码。

    上游 CitationExtractor 从 chunk metadata 取 file_name，而我们的语料元数据用的是
    source/document_title 字段，于是它只能给出 "unknown"；页码在 chunk 上也不存在。
    不改上游代码，在适配层用 page_offsets（document_id → 每页字符区间）把
    chunk 的 start_offset 映射回页码，引用才真正可回溯（对应验收项 M6）。
    """
    document_id = citation.get("document_id") or ""
    # 上游只从 chunk.metadata 取 document_id；真实语料里 chunk 顶层也有该字段，
    # 元数据缺失时会退化成 "unknown"，这里按 chunk_id 回查一次兜底
    if (not document_id or document_id == "unknown") and chunks:
        for chunk in chunks:
            if chunk.get("chunk_id") == citation.get("chunk_id"):
                document_id = (chunk.get("document_id")
                               or (chunk.get("metadata") or {}).get("document_id")
                               or "")
                break
    entry = offsets.get(document_id) or {}
    page = page_for_offset(offsets, document_id, int(citation.get("start_offset") or 0))
    return dict(
        citation,
        document_id=document_id,
        document_title=entry.get("title") or citation.get("document_title"),
        company=entry.get("company"),
        year=entry.get("year"),
        file_name=entry.get("title") or citation.get("file_name"),
        page=page,
    )


def index_files(strategy: str = "recursive") -> List[Path]:
    """当前索引产物文件（重建时用于清理）。"""
    directory = index_dir(strategy)
    return [directory / "faiss.index", directory / "metadata.json",
            directory / "page_offsets.json", directory / "write.lock"]
