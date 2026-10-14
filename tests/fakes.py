"""测试替身：不加载模型、不调用 LLM。

接口层的用例要覆盖"生成 + 引用 + SSE"，所以这里的假服务比
test_agent_graph.py 里的版本多了 context_manager / prompt_builder / generator
（含 stream_generate）。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List

COMPANY_BY_CODE = {"600519": "贵州茅台", "300750": "宁德时代", "600036": "招商银行"}


def make_hit(document_id: str, content: str, score: float = 0.7) -> Dict[str, Any]:
    code, year = document_id.split("-")
    return {
        "chunk_id": "{}#1".format(document_id),
        "document_id": document_id,
        "score": score,
        "content": content,
        "start_offset": 0,
        "end_offset": len(content),
        "metadata": {"company": COMPANY_BY_CODE.get(code, code), "year": int(year),
                     "document_id": document_id,
                     "document_title": "{}年年度报告".format(year)},
    }


def maotai_hit(year: int = 2023, score: float = 0.7) -> Dict[str, Any]:
    content = ("贵州茅台酒股份有限公司 {} 年年度报告 主要会计数据 营业收入 "
               "147,693,604,994.14").format(year)
    return make_hit("600519-{}".format(year), content, score)


class FakeRetriever:
    def __init__(self, resolver) -> None:
        self.resolver = resolver
        self.queries: List[str] = []

    def search(self, query, top_k=5, document_ids=None):
        self.queries.append(query)
        return self.resolver(query)[:top_k]


class FakeGenerator:
    """固定文本生成器：答案里带一个 [1] 引用，便于验证引用解析。"""

    def __init__(self, answer: str = "根据上下文[1]，营业收入为 147,693,604,994.14 元。"):
        self.answer = answer

    def generate(self, messages: List[Dict[str, str]]) -> str:
        return self.answer

    def stream_generate(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        for index in range(0, len(self.answer), 6):
            yield self.answer[index:index + 6]


class FakeContextManager:
    def build(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        context = "\n\n".join(
            "[{}] {}".format(i + 1, hit.get("content", ""))
            for i, hit in enumerate(results))
        return {"context": context, "chunks": list(results), "stats": {}}


class FakePromptBuilder:
    def build(self, query: str, context: str, history: Iterable | None = None):
        return [{"role": "system", "content": "你是严谨的问答助手"},
                {"role": "user", "content": "上下文：{}\n\n问题：{}".format(context, query)}]


class FakePipeline:
    def __init__(self, resolver, generator: FakeGenerator | None = None) -> None:
        self.retriever = FakeRetriever(resolver)
        self.generator = generator or FakeGenerator()
        self.context_manager = FakeContextManager()
        self.prompt_builder = FakePromptBuilder()
        self.reranker = None
        self.top_k = 5
        self.rerank_candidate_pool = 50

    def _search_all(self, query, queries, document_ids=None):
        collected: List[Dict] = []
        for text in queries:
            collected.extend(self.retriever.search(text, top_k=self.top_k))
        return collected

    def _filter_active_versions(self, candidates):
        return candidates


class FakeService:
    def __init__(self, resolver, mode: str = "vector", use_rerank: bool = False,
                 generator: FakeGenerator | None = None) -> None:
        self.mode = mode
        self.use_rerank = use_rerank
        self.strategy = "recursive"
        self._pipeline = FakePipeline(resolver, generator)
        self.generator = self._pipeline.generator


class DictStore:
    """Store 的最小替身（幂等表、记忆读写都能用）。"""

    def __init__(self) -> None:
        self.data: Dict[Any, Dict[str, Any]] = {}

    def put(self, namespace, key, value):
        self.data[(tuple(namespace), key)] = dict(value)

    def get(self, namespace, key):
        item = self.data.get((tuple(namespace), key))
        if item is None:
            return None
        return type("Item", (), {"key": key, "value": item,
                                 "namespace": tuple(namespace)})()

    def search(self, namespace_prefix, filter=None, limit=10, query=None, offset=0):
        prefix = tuple(namespace_prefix)
        results = []
        for (namespace, key), value in self.data.items():
            if namespace[:len(prefix)] != prefix:
                continue
            if filter and any(value.get(k) != v for k, v in filter.items()):
                continue
            results.append(type("Item", (), {
                "key": key, "value": value, "namespace": namespace,
                "created_at": value.get("created_at"), "score": None})())
        return results[:limit]

    def delete(self, namespace, key):
        self.data.pop((tuple(namespace), key), None)
