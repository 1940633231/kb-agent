"""跨会话长期记忆（LangGraph Store）。

写什么：只写"跨会话仍然成立"的内容——用户明确表达的偏好、业务约定、稳定事实
（例如"我们对外口径用万元"）。临时信息（"这次先看 2023 年"）不写。
判定必须出现**明确的长期化标记**（记住/以后/一直/我的偏好/我是……），
并限制长度：宁可少写，也不要把临时信息写成长期事实。

namespace：("kb-agent", "users", {user_id}, "memories")。隔离由 namespace 前缀
保证，而不是"取回来再按字段过滤"——不同用户的记忆在存储层就互不可见。

与知识库的边界：知识库内容来自 data/docs 语料索引，属于"世界知识"；
记忆只存"关于用户与约定的元信息"。两者混在一起会污染检索结果，
也会让"删掉文档"无法真正清除信息。
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

APP_NS = "kb-agent"
MEMORY_TYPES = ("preference", "fact", "task")

# 长期化标记：出现这些词才认为用户是在表达"跨会话仍然成立"的信息
DURABLE_MARKERS = (
    "记住", "以后", "今后", "一直", "每次都", "以后都", "我的偏好", "我偏好",
    "我们的口径", "我们公司", "我是", "请默认", "默认用", "养成习惯",
)
MAX_MEMORY_CHARS = 200


def memory_namespace(user_id: str) -> Tuple[str, ...]:
    return (APP_NS, "users", user_id, "memories")


def extract_memory(question: str) -> Optional[Dict[str, Any]]:
    """从用户输入里判定是否包含值得长期记住的信息。

    规则优先：只有出现长期化标记且长度合理才写，不调用 LLM，
    避免把"临时要求"误判成"长期事实"（方案 §4.5 的核心问题之一）。
    """
    text = (question or "").strip()
    if not text or len(text) > MAX_MEMORY_CHARS:
        return None
    if not any(marker in text for marker in DURABLE_MARKERS):
        return None
    memory_type = "preference"
    if re.search(r"我是|我们的口径|我们公司", text):
        memory_type = "fact"
    if re.search(r"任务|待办|提醒我", text):
        memory_type = "task"
    return {"content": text, "memory_type": memory_type}


def remember(store: Any, user_id: str, content: str,
             memory_type: str = "preference", source: str = "user") -> Dict[str, Any]:
    """写入一条记忆，返回落库记录。"""
    if store is None:
        raise RuntimeError("未配置 Store，无法写入长期记忆")
    if memory_type not in MEMORY_TYPES:
        raise ValueError("未知记忆类型: {}".format(memory_type))
    now = datetime.now(timezone.utc).isoformat()
    key = uuid.uuid4().hex[:12]
    value = {
        "content": content.strip(),
        "memory_type": memory_type,
        "source": source,
        "created_at": now,
        "updated_at": now,
    }
    store.put(memory_namespace(user_id), key, value)
    return dict(value, memory_id=key, namespace=list(memory_namespace(user_id)))


def recall(store: Any, user_id: str, limit: int = 5,
           memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """按 namespace 取回记忆（可选按类型过滤）。"""
    if store is None:
        return []
    filt = {"memory_type": memory_type} if memory_type else None
    try:
        items = store.search(memory_namespace(user_id), filter=filt, limit=limit)
    except Exception:  # noqa: BLE001 - 记忆不可用时不应阻塞问答
        return []
    memories: List[Dict[str, Any]] = []
    for item in items:
        value = dict(getattr(item, "value", {}) or {})
        value["memory_id"] = getattr(item, "key", None)
        memories.append(value)
    return memories


def forget(store: Any, user_id: str, memory_id: str) -> None:
    """删除一条记忆（冲突/过期时的最小操作；复杂遗忘策略不在首期范围）。"""
    if store is not None:
        store.delete(memory_namespace(user_id), memory_id)


def format_memories(memories: List[Dict[str, Any]]) -> str:
    """把记忆拼成给 LLM 的提示块（没有记忆时返回空串）。"""
    if not memories:
        return ""
    lines = ["用户长期偏好/约定（跨会话有效）："]
    for index, memory in enumerate(memories, 1):
        lines.append("{}. [{}] {}".format(index, memory.get("memory_type", "preference"),
                                          memory.get("content", "")))
    return "\n".join(lines)
