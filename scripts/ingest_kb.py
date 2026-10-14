"""把年报 PDF 灌进本地 FAISS + metadata.json 索引。

复用 production-rag 的切片 / embedding / 索引写入链路，只替换 PDF 文本抽取
（见 rag_adapter/pdfium_loader.py）。第零阶段为一次性全量灌库：
默认先清理 data/index/{strategy}/ 下的旧索引产物，再整批写入。

用法：
    python scripts/ingest_kb.py                     # 重建 recursive 索引
    python scripts/ingest_kb.py --strategy fixed
    python scripts/ingest_kb.py --keep-existing     # 不清理，直接追加写入
    python scripts/ingest_kb.py --verify-only       # 只核对已有索引并写页码映射
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.writer import IndexWriter  # noqa: E402

from rag_adapter import rag  # noqa: E402
from rag_adapter.settings import (PROJECT_ROOT, load_env,  # noqa: E402
                                 use_project_root_as_cwd)


def reset_index(strategy: str) -> list[str]:
    """清理索引产物；只允许删除 data/index/{strategy}/ 下的已知文件。"""
    allowed_root = (PROJECT_ROOT / "data" / "index").resolve()
    removed: list[str] = []
    for path in rag.index_files(strategy):
        resolved = path.resolve()
        if not str(resolved).startswith(str(allowed_root)):
            raise RuntimeError("拒绝清理工作区之外的路径: {}".format(resolved))
        if resolved.exists():
            resolved.unlink()
            removed.append(resolved.name)
    return removed


def verify_index(strategy: str) -> dict:
    """等效于 production-rag 的发布前校验：metadata 与 FAISS 的 vector_id 必须一致。

    上游实现（app/ingestion/validation.py → FaissVectorStore.ids）用了
    index.id_map.tolist()，在 faiss>=1.15 上会抛 AttributeError，因此这里用
    faiss.vector_to_array 自行核对同样的事情。
    """
    import faiss

    from app.storage.metadata_store import MetadataStore

    directory = rag.index_dir(strategy)
    metadata_path = directory / "metadata.json"
    index_path = directory / "faiss.index"
    if not metadata_path.exists() or not index_path.exists():
        raise RuntimeError("索引不完整：缺少 metadata.json 或 faiss.index")

    entries = MetadataStore().load(str(metadata_path)) or {}
    meta_ids = set()
    for key, entry in entries.items():
        meta_ids.add(int(entry.get("vector_id", key)))

    index = faiss.read_index(str(index_path))
    faiss_ids = {int(i) for i in faiss.vector_to_array(index.id_map)}

    report = {
        "metadata_chunks": len(entries),
        "faiss_vectors": len(faiss_ids),
        "missing_in_faiss": sorted(meta_ids - faiss_ids)[:5],
        "missing_in_metadata": sorted(faiss_ids - meta_ids)[:5],
        "consistent": meta_ids == faiss_ids,
    }
    if not report["consistent"]:
        raise RuntimeError("索引校验失败: {}".format(json.dumps(report, ensure_ascii=False)))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest annual reports into the local index")
    parser.add_argument("--strategy", choices=["fixed", "recursive"],
                        default="recursive")
    parser.add_argument("--keep-existing", action="store_true",
                        help="不清理旧索引，直接追加写入")
    parser.add_argument("--verify-only", action="store_true",
                        help="跳过写入，只核对已有索引并写页码映射")
    args = parser.parse_args()

    use_project_root_as_cwd()
    load_env()

    if args.verify_only:
        report = verify_index(args.strategy)
        print("索引校验通过: chunks={metadata_chunks}, vectors={faiss_vectors}".format(**report))
        _, offsets = rag.load_annual_reports()
        print("页码映射: {}".format(rag.save_page_offsets(offsets, args.strategy)))
        return 0

    if not args.keep_existing:
        removed = reset_index(args.strategy)
        print("已清理旧索引产物: {}".format(removed or "无"))

    started = time.time()
    documents, offsets = rag.load_annual_reports()
    print("载入年报 {} 份，正文合计 {:,} 字符".format(
        len(documents), sum(len(d.content) for d in documents)))

    writer = IndexWriter(rag.build_config())
    result = writer.write(
        documents=documents,
        strategy=args.strategy,
        overwrite_existing=False,
    )
    print("写入完成: documents={}, chunks={}, dimension={}, backend={}".format(
        result.get("document_count"), result.get("chunk_count"),
        result.get("dimension"), result.get("vector_backend")))
    print("索引: {} | 元数据: {}".format(result.get("index_path"),
                                        result.get("metadata_path")))

    target = rag.save_page_offsets(offsets, args.strategy)
    print("页码映射: {}".format(target))
    report = verify_index(args.strategy)
    print("索引校验通过: chunks={metadata_chunks}, vectors={faiss_vectors}".format(**report))
    print("耗时 {:.1f}s".format(time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
