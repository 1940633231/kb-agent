"""轻量 LLM 语义评分：规则评分通过后复核一次"证据够不够唯一确定答案"。

为什么需要（实测）：规则评分只看年份/公司/关键词覆盖，判不出"缺的是那个具体数字"。
茅台两道元级数字题（`ar-fact-001` 营业收入、`ar-fact-004` 归母净利润）的规则评分
都是"通过"，于是根本不触发重试，而正确数值其实排在 rerank 第 6 / 第 16 位。
方案 §4.1 明确允许"必要时再用一次轻量 LLM 评分"，这就是那一次。

为什么只在规则**通过**时调用：
    "规则说够、其实不够"会造成沉默的错答（贵且危险）；
    "规则说不够、其实够"只是多一次重试（便宜），已经用关键词修复与 facet 裁剪压掉了。
    所以把 LLM 预算花在前者上。

失败即回退：调用异常、解析失败一律保留规则结论，不让外部依赖决定主流程。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

GRADE_SYSTEM_PROMPT = (
    "你是检索证据审核员。给定一个问题与若干候选片段，判断这些片段是否"
    "**足以唯一确定**问题的答案。\n"
    "判定标准：\n"
    "1. 片段要同时覆盖问题要求的实体（公司、年份）与指标名；\n"
    "2. 问题要具体数值时，片段里必须真的出现该数值——只有同类指标、"
    "只有其它年份的数、只有明细项而不见合计，都算不足；\n"
    "3. 存在歧义（多个候选数值、年份或指标名相近）即算不足。\n"
    "只输出 JSON，不要输出别的文字："
    "{\"judgment\": \"answerable\" 或 \"insufficient\", "
    "\"reason\": \"一句话理由\", \"missing\": \"缺什么（没有就写空字符串）\"}"
)


class LLMGrader:
    """一次调用完成语义评分；结果只用于把"假通过"翻成不通过。"""

    def __init__(self, generator: Any, max_chunks: int = 5,
                 max_chars_per_chunk: int = 500) -> None:
        self.generator = generator
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk
        self.last_error: Optional[str] = None

    def _evidence_block(self, hits: List[Dict]) -> str:
        lines = []
        for index, hit in enumerate(hits[:self.max_chunks], 1):
            document_id = hit.get("document_id") or ""
            content = (hit.get("content") or "")[:self.max_chars_per_chunk]
            lines.append("[{}] {} {}".format(index, document_id, content))
        return "\n".join(lines)

    def grade(self, question: str, hits: List[Dict]) -> Dict[str, Any]:
        """返回 {judgment, reason, missing, source}；异常时 source=fallback。"""
        if not hits:
            return {"judgment": "insufficient", "reason": "没有候选片段",
                    "missing": "全部证据", "source": "rule"}
        prompt = "问题：{}\n\n候选片段：\n{}".format(question, self._evidence_block(hits))
        try:
            answer = self.generator.generate([
                {"role": "system", "content": GRADE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            parsed = self._parse(answer)
        except Exception as exc:  # noqa: BLE001 - 外部依赖异常不能阻塞主流程
            self.last_error = "{}: {}".format(type(exc).__name__, exc)
            return {"judgment": "answerable", "reason": "LLM 评分不可用，沿用规则结论",
                    "missing": "", "source": "fallback", "error": self.last_error}

        if parsed is None:
            return {"judgment": "answerable", "reason": "LLM 评分解析失败，沿用规则结论",
                    "missing": "", "source": "fallback"}
        parsed["source"] = "llm"
        return parsed

    @staticmethod
    def _parse(answer: Optional[str]) -> Optional[Dict[str, Any]]:
        if not answer:
            return None
        match = re.search(r"\{.*\}", answer, re.S)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        judgment = str(payload.get("judgment", "")).strip().lower()
        if judgment not in ("answerable", "insufficient"):
            return None
        return {"judgment": judgment,
                "reason": str(payload.get("reason", ""))[:200],
                "missing": str(payload.get("missing", ""))[:120]}
