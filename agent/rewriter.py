"""查询改写：LLM 改写优先，规则改写兜底（v2 §4.1 的重试动作）。

两类改写动作都对应实测到的失败模式：
    1. 年份/实体补全：问题问 2023 年，检索却召回同公司其他年报，
       于是把"XX公司2023年 指标 年度报告"写清楚；
    2. 子问题拆分：跨文档多值题（多个年份/多家公司）在 top-5 内凑不齐证据，
       按年份拆成多条子查询分别检索后再合并。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from agent.grading import extract_facets

REWRITE_SYSTEM_PROMPT = (
    "你是检索查询改写器。用户问题在知识库里没有检索到足够证据，请改写成更适合"
    "检索的查询。要求：\n"
    "1. 补全实体与年份，例如把“宁德时代2023年营业收入”写成"
    "“宁德时代 2023年年度报告 营业收入 主要会计数据”；\n"
    "2. 问题涉及多个年份或多家公司时，拆成多条子查询（每年一条）；\n"
    "3. 去掉口语化与疑问语气，保留指标名、年份、公司名等检索关键词；\n"
    "4. 只输出 JSON，格式：{\"queries\": [\"...\", \"...\"]}，不要输出别的文字。\n"
)


class RuleRewriter:
    """不调用 LLM 的兜底改写：按缺失要素补关键词，或按年份/公司拆子问题。"""

    def rewrite(self, question: str, reason: str,
                previous_queries: Sequence[str] = ()) -> List[str]:
        facets = extract_facets(question)
        companies = facets["companies"]
        years = facets["years"]
        metric = " ".join(facets["keywords"][:3])
        self.last_source = "rule"

        if len(years) > 1 or len(companies) > 1:
            queries: List[str] = []
            for company in (companies or [""]):
                for year in (years or [""]):
                    queries.append(" ".join(part for part in [
                        company, "{}年".format(year) if year else "", metric,
                        "年度报告",
                    ] if part).strip())
            unique = [q for q in queries if q and q not in previous_queries]
            if unique:
                return unique[:6]

        company = companies[0] if companies else ""
        year = years[0] if years else ""
        query = " ".join(part for part in [
            company, "{}年".format(year) if year else "", metric, "年度报告",
        ] if part).strip()
        if not query or query in previous_queries:
            query = "{} {}".format(question, metric).strip()
        return [query]


class LLMRewriter:
    """用一次轻量 LLM 调用生成改写查询；异常或解析失败时退回规则改写。"""

    def __init__(self, generator: Any, max_queries: int = 4):
        self.generator = generator
        self.max_queries = max_queries
        self.fallback = RuleRewriter()
        self.last_source = ""

    def rewrite(self, question: str, reason: str,
                previous_queries: Sequence[str] = ()) -> List[str]:
        user_prompt = (
            "原始问题：{question}\n"
            "已尝试的查询：{previous}\n"
            "未通过的原因：{reason}\n"
            "请输出改写后的查询 JSON。"
        ).format(question=question,
                 previous=json.dumps(list(previous_queries), ensure_ascii=False),
                 reason=reason or "检索证据不足")
        queries: List[str] = []
        try:
            answer = self.generator.generate([
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ])
            queries = self._parse(answer)
        except Exception:  # noqa: BLE001 - 网络/额度异常不应阻塞主流程
            queries = []

        fresh = [q for q in queries if q not in previous_queries]
        if fresh:
            self.last_source = "llm"
            return fresh[:self.max_queries]
        self.last_source = "rule"
        return self.fallback.rewrite(question, reason, previous_queries)

    def _parse(self, answer: Optional[str]) -> List[str]:
        if not answer:
            return []
        match = re.search(r"\{.*\}", answer, re.S)
        if not match:
            return []
        try:
            payload: Dict = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        queries = payload.get("queries") or []
        if isinstance(queries, str):
            queries = [queries]
        return [str(q).strip() for q in queries if str(q).strip()]
