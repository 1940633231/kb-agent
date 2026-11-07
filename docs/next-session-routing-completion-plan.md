# 检索前实体路由收尾执行计划

> 用途：下一个 Codex 会话直接读取并执行。
> 创建日期：2026-11-07
> 当前分支：`codex/stage0-corpus-expansion`
> 当前 HEAD：`2dc40c4 优化生成证据与 Claim 审计`

## 0. 2026-11-07 实施进度

已完成：

- 无公司且无年份的通用问题回退为单条 `unrouted=True` 原始查询。
- 未知实体带年份仍生成拒答义务，且不执行检索。
- 纯寒暄仍为 `out_of_scope`，且不执行检索。
- layer matrix 使用报告自身 `layer` 作为键，L2 / L3a / L3b 互不覆盖。
- 新增 `evals/run_routing_ablation.py`，支持一次 service 初始化、三档复用、
  每档完成即落盘、`--resume`、正式/调试报告隔离。
- 正式 120 题 L2 / L3a / L3b 消融已完成，三档各 120 条逐题明细。
- 正式 summary 的 `acceptance.all_passed = true`。
- layer matrix 已按当前 120 题 manifest 重建，仅保留 L2 / L3a / L3b。
- 最终结果文档：`docs/routing-ablation-results.md`。
- 全量测试：`190 passed, 1 skipped`。

正式执行命令（已完成）：

```powershell
.\.venv\Scripts\python.exe evals\run_routing_ablation.py `
  --limit 120 `
  --top-k 5 `
  --route-candidate-limit 50 `
  --resume
```

正式运行总耗时约 5 小时 45 分钟。`--resume` 只会复用配置、manifest SHA、
题数和逐题明细均匹配的已完成层。

## 1. 当前结论

检索前实体路由的主体已经实现，以下能力已验证通过：

- 多公司题按公司拆分 route。
- 同一公司多年份按年份拆分 route。
- 多公司多年份生成完整公司 × 年份矩阵。
- 每条 route 绑定并强制校验 `document_ids`。
- route 为空时不会回退到其他公司或年份。
- 缺失义务只重试对应 route。
- rerank 前后保留 route provenance。
- `enable_pre_retrieval_routing=False` 可回退旧单查询路径。
- 120 题结构未变化。

当前检查结果：

```text
计划相关测试：58 passed
全量测试：177 passed, 1 skipped
跳过项：未配置 Postgres

120 题审计：
records = 120
normal = 120
review = 0
high_risk = 0
fail = 0

当前 scaled L3：
doc_recall@5 = 1.0
route_obligation_coverage = 0.9905
missing_obligation_rate = 0
retry_rate = 0.0917
cross_document_contamination_rate = 0
mean_seconds_per_query = 43.8359
```

但不能将计划标记为全部完成，仍有两个必修缺口和一个报告问题。

## 2. 必修缺口

### 2.1 无实体问题没有回退到单路检索

计划要求：

```text
无公司、无年份 → 退化到现有单路检索，并标记 unrouted=True
```

当前行为：

```text
问题：经营活动现金流如何理解？
queries = []
routes = []
status = refused_no_evidence
```

原因位置：

- `agent/evidence_contract.py`
  - `build_obligations()` 在 `not companies` 时直接生成 refusal obligation。
  - `build_initial_routes()` 没有 route 时返回空列表。
- `agent/retrieval_subgraph.py`
  - routing 开启但没有 route 时直接返回空候选。
- `agent/nodes.py`
  - routing 开启且 routes 为空时 `sub_queries` 为空。

实施要求：

1. 只在“无公司且无年份”的通用知识问题上回退单路检索。
2. 未知实体带年份时仍保持拒答，例如“华为2024年的研发投入是多少？”。
3. 回退 route 必须：

```json
{
  "query_id": "route-1",
  "query": "原始问题",
  "obligation_ids": [],
  "document_ids": [],
  "company": null,
  "year": null,
  "metrics": [],
  "unrouted": true
}
```

建议修改：

- `agent/evidence_contract.py`
  - 区分“无实体通用问题”和“未知公司问题”。
  - 无公司且无年份时不要生成 refusal obligation。
- `agent/nodes.py`
  - `needs_retrieval=True` 且 routing 开启但没有 planned routes 时，构造一条
    `unrouted=True` route。
- `agent/retrieval_subgraph.py`
  - 无 route 但有 `unrouted` route 时仍执行原始问题检索。

新增测试：

- `tests/test_pre_retrieval_graph.py`
  - 通用无实体问题只发起一次原始 query。
  - `routes` 长度为 1，且 `unrouted=True`。
  - `document_filters` 为 `[None]`。
  - 结果不是 `refused_no_evidence`。
  - 未知实体带年份仍为 refusal，且检索次数为 0。
  - 纯寒暄仍为 out_of_scope，且不检索。

提交说明建议：

```text
修复无实体问题的单路检索回退
```

### 2.2 L3a/L3b 会覆盖 layer matrix 中的 L3

当前 `evals/run_agent_eval.py` 无论实际运行 L3a 还是 L3b，都固定写入：

```python
matrix["layers"]["L3"]
```

因此：

- 运行 L3a 会覆盖 L3。
- 运行 L3b 会覆盖 L3 或 L3a。
- 无法形成 L2 / L3a / L3b 三档比较。

实施要求：

1. 将矩阵更新逻辑抽成独立函数，例如：

```python
update_layer_matrix(path, layer, label, config, metrics)
```

2. 写入键必须使用 `report["layer"]`，即：

```python
matrix.setdefault("layers", {})[layer] = {
    "label": label,
    "config": config,
    "metrics": metrics,
}
```

3. 新增测试：

- 顺序写入 L2、L3a、L3b 后三个键同时存在。
- 重跑 L3a 只覆盖 L3a，不删除 L3b。
- 重跑 L3b 只覆盖 L3b。

提交说明建议：

```text
修复 L3a/L3b 消融矩阵覆盖
```

## 3. 当前报告问题

### 3.1 layer_matrix.json 当前不可使用

当前工作区 `evals/reports/layer_matrix.json` 已被一次单题运行覆盖：

```text
answerable_questions = 1
max_retries = 0
mean_seconds_per_query = 65.8
```

该文件不能用于最终报告。应用提交版本核对后，在完整消融跑完后重新生成。

### 3.2 旧 L3 报告不能与新的 120 题报告混用

当前报告口径不同：

```text
evals/reports/scaled_l3.json：120 题，新一轮 120 题基线
evals/reports/l3_report.json：20 题，旧报告
evals/reports/layer_matrix.json：旧 45 题矩阵
```

最终报告只能使用同一数据集、同一 manifest SHA、同一检索配置的结果。

### 3.3 当前 scaled L3 的路由覆盖是事后重算

`scaled_l3.json` 的 `route_coverage_rate` 已按最终选中证据重算，代码也有回归测试。
但为了正式验收，建议在新的 L2/L3a/L3b 消融中直接由当前代码生成完整逐题数据，
不要再依赖手工后处理报告。

## 4. 新增 120 题消融 runner

建议新增：

```text
evals/run_routing_ablation.py
```

不要直接覆盖现有 `scaled_l3.json`。

### 4.1 三档配置

| 档位 | 预路由 | max_retries | 说明 |
|---|---:|---:|---|
| L2 | false | 0 | hybrid + rerank，无路由 |
| L3a | true | 0 | L2 + 检索前公司/年份路由 |
| L3b | true | 2 | L3a + 缺失义务定向重试 |

三档必须保持一致：

```text
dataset = evals/golden_set_scaled_120.jsonl
strategy = recursive
mode = hybrid
use_rerank = True
top_k = 5
route_candidate_limit = 50
generate = False
rule_rewrite = True
llm_grader = False
```

### 4.2 性能要求

hybrid service 只初始化一次，三档复用同一个 service，避免重复加载：

- FAISS
- BM25
- embedding
- reranker

不要每档重新创建 RAGService。

### 4.3 输出文件

```text
evals/reports/routing_ablation_l2.json
evals/reports/routing_ablation_l3a.json
evals/reports/routing_ablation_l3b.json
evals/reports/routing_ablation_summary.json
```

每档报告必须包含：

- `dataset.manifest.sha256`
- `dataset.manifest.corpus_metadata_sha256`
- `dataset.evaluated = 120`
- 完整 `questions` 逐题明细
- `generated_at`
- `elapsed_seconds`
- `config`
- `metrics`

`routing_ablation_summary.json` 至少包含：

- L2 / L3a / L3b 指标表
- `L3a - L2` 增量
- `L3b - L3a` 增量
- 各题型的延迟和召回拆分

### 4.4 建议命令

```powershell
.\.venv\Scripts\python.exe evals\run_routing_ablation.py `
  --limit 120 `
  --top-k 5 `
  --route-candidate-limit 50
```

如果支持分批执行，建议只提供显式调试参数，默认必须跑完整 120 题：

```powershell
--limit 10
```

调试结果不得写入正式消融报告，或必须在报告中标记：

```json
{
  "temporary": true,
  "formal_report": false
}
```

## 5. 指标与验收门槛

### 5.1 必查指标

每档都必须输出：

- `doc_recall@5`
- `all_docs_hit@5`
- `doc_mrr`
- `evidence_obligation_coverage`
- `route_obligation_coverage`
- `missing_obligation_rate`
- `retry_rate`
- `mean_retries`
- `refuse_flagged_rate`
- `mean_seconds_per_query`
- `cross_document_contamination_rate`
- `foreign_evidence_rate`

### 5.2 延迟拆分

总平均延迟不足以验证计划中的 10% 门槛。额外计算：

- `p50_seconds`
- `p95_seconds`
- `single_company_single_year_mean_seconds`
- `fact_mean_seconds`
- `table_mean_seconds`
- `cross_document_mean_seconds`
- `refusal_mean_seconds`

“单公司单年份”定义：

```text
expected_behavior = answer
question_type in {fact, table}
wanted_docs 数量 = 1
```

延迟门槛比较：

```text
L3a single_company_single_year_mean_seconds
<=
L2 single_company_single_year_mean_seconds * 1.10
```

### 5.3 验收门槛

```text
cross_document_contamination_rate = 0
L3a doc_recall@5 >= 0.95
L3b route_obligation_coverage >= 0.95
L3b missing_obligation_rate <= 0.05
L3b retry_rate <= 0.0917
L3a single-company-single-year latency <= L2 * 1.10
所有报告 manifest SHA 与当前数据集一致
所有报告包含 120 条逐题明细
```

说明：

- `0.0917` 是当前 `scaled_l3.json` 的 120 题 retry rate。
- 不要把 old 45 题 L3 的 `retry_rate=0.46` 当成本次门槛。
- 不要报告单次最好成绩，三档必须使用完整同一题集。

## 6. 测试计划

### 6.1 第一轮定向测试

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_entity_routing.py `
  tests\test_route_retrieval.py `
  tests\test_pre_retrieval_graph.py `
  tests\test_multiroute_coverage.py `
  tests\test_evidence_contract.py `
  -q -rs
```

### 6.2 报告与数据集测试

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_scaled_golden_set.py `
  tests\test_scaled_golden_set_audit.py `
  tests\test_scaled_eval_reports.py `
  -q -rs
```

### 6.3 全量测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -rs
```

全量测试通过标准：

```text
0 failed
允许 1 skipped：Postgres 未配置
```

## 7. 报告更新顺序

1. 修复无实体回退。
2. 修复 L3a/L3b 矩阵键。
3. 新增消融 runner 和测试。
4. 跑完整 120 题 L2。
5. 跑完整 120 题 L3a。
6. 跑完整 120 题 L3b。
7. 生成 summary。
8. 重新生成 layer matrix。
9. 检查报告 SHA、题数、配置和延迟门槛。
10. 更新最终报告文档。

不要先跑 hybrid 生成三轮。路由消融未完成前，生成指标无法归因。

## 8. 提交顺序

建议三个独立中文提交：

```text
修复无实体问题的单路检索回退
修复 L3a/L3b 消融矩阵覆盖
增加检索前路由 120 题消融报告
```

提交前检查：

```powershell
git status --short --branch
git diff --check
git diff --ignore-space-at-eol --stat
```

只暂存本计划涉及的文件，不要执行：

```powershell
git add -A
```

## 9. 回滚方案

运行时可立即回退：

```python
enable_pre_retrieval_routing = False
```

预期回退行为：

- routes 为空。
- route_history 为空。
- `retrieval_stats.routing_enabled = False`。
- 使用原始问题执行旧单路检索。

不要通过删除 route 代码回滚。保留旧 `retrieve_multi_with_routes()` 兼容路径。

## 10. 明确不在本轮范围

- 不修改 production-rag 的 FAISS/BM25 主体。
- 不重新 embedding。
- 不重建 metadata。
- 不接 Milvus。
- 不接 Elasticsearch。
- 不修改 Postgres。
- 不处理同一文档内部年份对照列错误。
- 不继续 hybrid 三轮生成，直到路由消融完成。

## 11. 下一会话启动命令

```powershell
Get-Content docs\next-session-routing-completion-plan.md -Encoding UTF8
git status --short --branch
git log -6 --oneline --decorate
```

启动后先检查：

```powershell
git diff --ignore-space-at-eol --stat
```

注意：

- 工作区原本已有用户改动和 CRLF 差异。
- `evals/reports/layer_matrix.json` 当前已被单题运行覆盖。
- 不要回滚未参与本计划的文件。
- 不要使用当前 layer_matrix 作为最终报告。

## 12. 完成标准

本计划完成后必须能回答：

1. 无实体通用问题是否仍能走原始单路检索？
2. 未知实体带年份是否仍然拒答且不检索？
3. L2、L3a、L3b 是否同时存在于 layer matrix？
4. L3a 的 `doc_recall@5` 是否达到 0.95？
5. L3b 的 `route_obligation_coverage` 是否达到 0.95？
6. L3b 的 `missing_obligation_rate` 是否不超过 0.05？
7. L3b 的 retry rate 是否不超过 0.0917？
8. 跨文档污染率是否为 0？
9. 单公司单年份延迟是否不超过 L2 的 10%？
10. 所有报告是否使用同一 120 题 manifest SHA？
11. 全量测试是否通过？
