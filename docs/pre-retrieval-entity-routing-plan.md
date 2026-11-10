# 检索前实体路由改造计划

> 目标：在检索前按公司、年份和证据义务做硬路由，避免多公司、多年份问题在
> FAISS/BM25 候选阶段发生交叉污染。

## 1. 问题定义

当前首轮检索仍把原始问题作为单一查询：

```text
问题
→ 识别公司、年份、指标
→ 生成证据义务
→ 但首轮仍检索整个语料
→ 检索后才做公司/年份 scope
```

这会导致：

- 多公司题只召回其中一家公司。
- 同公司不同年份高度相似，召回错误年份。
- 多值题缺少某个义务，但合并后数量看起来足够。
- 后置 scope 可能因过滤后为空而回退原集合，重新带回污染。
- 缺失实体只能进入重试后才按义务拆分。

## 2. 改造边界

### 2.1 本计划内

- `kb-agent` 编排层。
- `kb-agent` 的 production-rag 适配层。
- 初始 route 规划。
- 每条 route 绑定 document_ids。
- 路由级检索统计与义务覆盖。
- 路由级重试。
- 单元测试、图测试和消融评测。

### 2.2 本计划外

- 不修改 production-rag 的检索主体。
- 不修改 FAISS 索引结构。
- 不重新 embedding。
- 不重建 metadata。
- 不接 Milvus。
- 不接 Elasticsearch。
- 不修改 Postgres。
- 不处理同一文档内部的年份对照列错误。

### 2.3 底座能力确认

production-rag 已具备文档级过滤：

```text
RAGPipeline._search_all(..., document_ids=None)
Retriever.search(..., document_ids=None)
BM25Search.search(..., document_ids=None)
```

Vector Retriever 会把 document_ids 转换为允许的 vector_id 集合，FAISS 只在这些
vector 中检索。本地 BM25 也会按 document_ids 限制候选。

因此当前 FAISS + 本地 BM25 路径**不需要改 production-rag**。

## 3. 目标流程

```text
问题
→ 实体、年份、指标识别
→ 生成 EvidenceObligation
→ 按 company + year 聚合 obligation
→ 生成初始 RetrievalRoute
→ 每条 route 绑定 document_ids
→ 逐 route 做 FAISS/BM25 检索
→ route 内再次校验 company/year
→ 合并候选并保留 retrieval_routes
→ 统一 Cross-Encoder Rerank
→ 逐义务检查覆盖
→ 只重试缺失 obligation 对应的 route
```

## 4. 核心数据结构

### 4.1 RetrievalRoute

建议放在 `agent/evidence_contract.py`：

```python
@dataclass(frozen=True)
class RetrievalRoute:
    query_id: str
    query: str
    obligation_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    company: str | None
    year: int | None
    metrics: tuple[str, ...]
```

序列化字段：

```json
{
  "query_id": "route-1",
  "query": "贵州茅台 2023年 营业收入 年度报告",
  "obligation_ids": ["obligation-1"],
  "document_ids": ["600519-2023"],
  "company": "贵州茅台",
  "year": 2023,
  "metrics": ["营业收入"]
}
```

### 4.2 路由聚合规则

- 同一公司、同一年份、多个指标：合并为一条 document route。
- 同一公司、不同年份：每个年份一条 route。
- 不同公司、同一年份：每家公司一条 route。
- 多公司多年份：生成完整公司×年份 route 矩阵。
- refusal obligation：不生成检索 route。
- 无公司、无年份：退化到现有单路检索，并标记 `unrouted=True`。

示例：

```text
贵州茅台2023年、2024年营业收入
```

生成：

```text
route-1: document_ids={600519-2023}
route-2: document_ids={600519-2024}
```

```text
贵州茅台与招商银行2023年营业收入
```

生成：

```text
route-1: document_ids={600519-2023}
route-2: document_ids={600036-2023}
```

## 5. 具体改造

### 5.1 `agent/evidence_contract.py`

新增：

```python
build_initial_routes(question, obligations, facets) -> list[RetrievalRoute]
group_obligations_by_document(obligations) -> dict[str, list[...]]
route_for_missing_obligations(...) -> list[RetrievalRoute]
```

职责：

- 把 obligation 聚合成公司/年份 route。
- 从 metadata 映射 `company + year -> document_id`。
- 为每条 route 生成定向 query。
- 保留 obligation_ids，供评分和定向重试。

### 5.2 `agent/nodes.py`

`understand_query`：

- 保留 facets 和 obligations。
- 首轮生成 `routes` 和 `sub_queries`。
- 单 route 时 query 尽量保持接近原始问题。
- 多 route 时每条 query 带公司、年份、指标、年度报告。
- 把 routes 写入 State 和 trace。

`retrieve_knowledge`：

- 首轮直接传 routes，而不是只传字符串。
- 每条 route 传自己的 document_ids。
- route hits 统计写入 retrieval_stats。
- 重试轮只传缺失 obligation 对应 routes。

### 5.3 `rag_adapter/rag.py`

新增兼容接口：

```python
retrieve_routes_with_evidence(
    service,
    routes: list[dict],
    top_k: int,
    anchor: str,
) -> tuple[list[dict], list[dict]]
```

行为：

1. 对每个 route 调用：

   ```python
   pipeline._search_all(
       route["query"],
       [route["query"]],
       document_ids=set(route["document_ids"]),
   )
   ```

2. 检索后再做防御性 doc_id 校验：

   ```python
   if document_ids and hit.document_id not in document_ids:
       drop
   ```

3. 给候选附加：

   ```json
   {
     "retrieval_routes": ["route-1"],
     "route_obligations": ["obligation-1"]
   }
   ```

4. 合并去重。
5. 只统一 rerank 一次。

保留现有 `retrieve_multi_with_routes()`，内部可适配为无 document_ids 的 route，
避免破坏旧测试。

### 5.4 `agent/retrieval_subgraph.py`

RetrievalState 增加：

```python
routes: List[Dict]
route_document_ids: Dict[str, List[str]]
route_obligations: Dict[str, List[str]]
```

stats 增加：

```text
routes_planned
routes_with_hits
routes_empty
per_route_candidates
per_route_documents
per_route_obligations_covered
per_route_obligations_missing
route_filter_degraded
```

### 5.5 `agent/state.py`

建议新增：

```python
routes: List[Dict[str, Any]]
route_history: List[Dict[str, Any]]
```

### 5.6 `agent/rewriter.py`

重试接口改为可接收 missing obligations 和 route：

- 只对 missing obligation 生成查询。
- 不重新查询已满足义务。
- route document_ids 保持不变。
- 重试 query 可以换表述，但不能跨 document_ids。

### 5.7 `agent/evidence.py`

`scope_evidence` 保留，但降级为兜底。

新增统计：

```json
{
  "scope_degraded": false,
  "route_filter_missing": []
}
```

如果 route 过滤结果为空，不得回退到全库；只允许回退到同 route 的宽候选池。

## 6. Runtime 开关

在 `AgentRuntime` 增加：

```python
enable_pre_retrieval_routing: bool = True
strict_route_filter: bool = True
route_candidate_limit: int = 50
```

语义：

- `enable_pre_retrieval_routing=False`：完全回退当前行为，用于回滚和消融。
- `strict_route_filter=True`：route 不允许任何跨 document_id 候选。
- `route_candidate_limit`：每条 route 的 rerank 前候选上限。

## 7. 测试计划

### 7.1 `tests/test_entity_routing.py`

- 单公司单年份只生成一条 route。
- 单公司多年份生成多年份 route。
- 多公司单年份生成多公司 route。
- 多公司多年份生成完整路由矩阵。
- 同一公司年份多指标聚合为一条 document route。
- refusal obligation 不生成检索 route。
- route 必须携带 document_ids 和 obligation_ids。

### 7.2 `tests/test_route_retrieval.py`

- route-1 只返回 `600519-2023`。
- route-2 只返回 `600519-2024`。
- 后端忽略 document_ids 时，适配层防御性过滤仍生效。
- 某 route 为空时其他 route 不被污染。
- 合并后总数足够但缺少一个义务时不能通过。
- rerank 后 route provenance 不丢失。

### 7.3 Graph 回归

- 多公司题首轮至少发生两次 route 检索。
- 多公司多年份首轮生成完整 route 矩阵。
- 已满足义务不会在重试时重复检索。
- 缺失义务只重试对应 route。
- 无实体问题仍使用原单路检索。
- `enable_pre_retrieval_routing=False` 时行为与当前实现一致。

### 7.4 FakeService 改造

FakeRetriever 必须执行 document_ids：

```python
if document_ids is not None and hit.document_id not in document_ids:
    drop
```

防止测试替身忽略过滤后产生假通过。

## 8. 消融评测

在相同 120 题、相同 FAISS、相同本地 BM25、相同 embedding 和 reranker 下比较：

| 层 | 配置 |
|---|---|
| L2 | hybrid + rerank，无预路由 |
| L3a | L2 + 检索前公司/年份路由 |
| L3b | L3a + 缺失义务重试 |

记录：

- doc_recall@5
- all_docs_hit@5
- evidence_obligation_coverage
- route_obligation_coverage
- missing_obligation_rate
- retry_rate
- mean_seconds_per_query
- cross_document_contamination_rate

建议验收门槛：

- cross_document_contamination_rate = 0
- L3a doc_recall@5 >= 0.95
- L3b route_obligation_coverage >= 0.95
- missing_obligation_rate <= 0.05
- retry_rate 不高于当前 L3 基线
- 单公司单年份延迟增加不超过 10%

## 9. 不修改 production-rag 的条件

当前不修改底座，因为：

- document_ids 已支持检索前过滤。
- 年报 document_id 已包含公司和年份。
- 当前 route 最大数量很小。
- FAISS 和本地 BM25 都能按 document_ids 过滤。

只有在以下情况才另开 production-rag 改造：

- 需要按任意 metadata 字段过滤，而不是 document_id。
- 需要 Milvus/ES 端复合字段预过滤。
- route 数量大到需要批量多过滤集检索接口。

这些均不属于本计划。

## 10. 提交顺序

1. 文档与测试骨架  
   `补充检索前实体路由计划`

2. 路由模型与初始 planner  
   `实现检索前实体路由规划`

3. route-level document_ids 检索  
   `实现按公司和年份检索前过滤`

4. 定向重试与统计  
   `实现路由级缺失义务重试`

5. 消融评测报告  
   `增加检索前路由消融评测`

提交说明全部使用中文。

## 11. 回滚方案

- 设置 `enable_pre_retrieval_routing=False`。
- 保留原 `retrieve_multi_with_routes()` 接口。
- 新 route 检索和旧多查询检索并存。
- 旧 scaled 报告已标记 stale，不作为回滚基线。

## 12. 完成标准

1. 多公司题首轮不再检索全库。
2. 多公司多年份题首轮生成完整 route 矩阵。
3. 每条 route 只允许自己的 document_ids。
4. 某 route 为空不会回退到其他公司或年份。
5. 缺失义务只重试对应 route。
6. rerank 前后 route provenance 一致。
7. 120 题结构不变。
8. L3a/L3b 指标可与 L2 归因比较。
9. 不接入 Milvus/ES。
10. 全量测试通过并形成中文提交。
