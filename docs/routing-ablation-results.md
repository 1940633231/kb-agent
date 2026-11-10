# 检索前实体路由 120 题消融结果

> 执行日期：2026-11-07 至 2026-11-08
> 数据集：`evals/golden_set_scaled_120.jsonl`
> manifest SHA256：`9b91a2457f47afab5de9968f9dfebef72e72598abdf4e8a58ef95c4e1417d29f`
> corpus metadata SHA256：`38bbabd11e54afbe1c3a4c5f67112bfb4640c503ad750fc824ebbe06e982deba`

## 配置

三档使用同一份 120 题数据、同一个 hybrid/rerank service：

| 档位 | 预路由 | max_retries |
|---|---:|---:|
| L2 | false | 0 |
| L3a | true | 0 |
| L3b | true | 2 |

固定配置：

```text
strategy = recursive
mode = hybrid
use_rerank = true
top_k = 5
route_candidate_limit = 50
generate = false
rule_rewrite = true
llm_grader = false
```

正式运行总耗时 `16697.32` 秒，约 4 小时 38 分钟。

## 核心指标

| 档位 | Recall@5 | All docs@5 | MRR | 证据义务覆盖 | Route 覆盖 | 缺失义务率 | 重试率 | 平均秒/题 | 跨文档污染 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 | 0.9762 | 0.9619 | 0.9905 | 0.9476 | 0.9476 | 0.0762 | 0.0000 | 44.6214 | 0.0095 |
| L3a | 1.0000 | 1.0000 | 1.0000 | 0.9905 | 0.9905 | 0.0000 | 0.0000 | 37.2332 | 0.0000 |
| L3b | 1.0000 | 1.0000 | 1.0000 | 0.9905 | 0.9905 | 0.0000 | 0.0917 | 55.4650 | 0.0000 |

关键增量：

- `L3a - L2`：Recall@5 `+0.0238`，All docs@5 `+0.0381`，
  证据义务覆盖 `+0.0429`，缺失义务率 `-0.0762`，
  跨文档污染率 `-0.0095`，平均延迟 `-7.3882` 秒。
- `L3b - L3a`：召回和最终证据覆盖不变，平均重试次数 `+0.1833`，
  平均延迟 `+18.2318` 秒。

## 延迟拆分

| 指标 | L2 | L3a | L3b |
|---|---:|---:|---:|
| p50 秒 | 36.7150 | 34.0550 | 40.3450 |
| p95 秒 | 70.9920 | 72.7405 | 173.7720 |
| 单公司单年份均值 | 47.0920 | 36.2287 | 44.5896 |
| fact 均值 | 46.8468 | 36.0153 | 44.3056 |
| table 均值 | 48.0728 | 37.0822 | 45.7256 |
| 跨文档均值 | 40.5320 | 80.4533 | 174.8393 |
| 拒答均值 | 33.8873 | 0.0407 | 1.3433 |

单公司单年份延迟门槛：

```text
L3a / L2 = 36.228667 / 47.092 = 0.769317 <= 1.10
```

跨文档题需要并行执行多个 route，因此 L3a/L3b 的跨文档延迟明显上升；
单公司单年份题没有出现计划中的 10% 延迟回退。

## 验收结果

正式 summary：`evals/reports/routing_ablation_summary.json`

```text
acceptance.all_passed = true
L3a doc_recall@5 = 1.0
L3b route_obligation_coverage = 0.9905
L3b missing_obligation_rate = 0.0
L3b retry_rate = 0.0917
L3a single-company single-year latency ratio = 0.769317
L3a/L3b cross_document_contamination_rate = 0.0
all reports evaluated = 120
all reports manifest SHA = 9b91a245...
```

L2 基线唯一的跨文档污染来自 `scaled-fact-082`：无路由检索混入
`603259-2024`、`603259-2025`，并漏掉目标文档 `603259-2023`。
L3a/L3b 的路由过滤消除了该污染，并把 Recall@5 提升到 1.0。

## 产物

- `evals/reports/routing_ablation_l2.json`
- `evals/reports/routing_ablation_l3a.json`
- `evals/reports/routing_ablation_l3b.json`
- `evals/reports/routing_ablation_summary.json`
- `evals/reports/layer_matrix.json`
