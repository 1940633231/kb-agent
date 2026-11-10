# kb-agent 阶段最终交付说明

> 收尾日期：2026-11-10
> 分支：`codex/stage0-corpus-expansion`
> 评测结果提交：`a9684f1 完成 v2 路由与 L3b 三轮生成评测`
> 数据集：`evals/golden_set_scaled_120.jsonl` v2
> manifest SHA256：`9b91a2457f47afab5de9968f9dfebef72e72598abdf4e8a58ef95c4e1417d29f`

## 1. 阶段状态

本阶段可以收尾。

已完成范围：

- 30 家公司、90 份年报、37,958 chunk 的语料扩展与审计。
- P0 证据义务契约、P1 Claim 门禁、P2 上下文预算。
- P3 120 题扩展评测集和 v2 营业总收入口径修正。
- 多公司、多年份和完整公司 × 年份矩阵的检索前实体路由。
- L2 / L3a / L3b 三档正式消融。
- 三轮 hybrid L3b 生成评测。
- 评测适配器、runner、报告契约测试和逐题失败归因。

## 2. 数据与评测

120 题源文审计：

```text
records = 120
normal = 120
review = 0
high_risk = 0
fail = 0
```

v2 修正内容：

- 美的集团 2023-2025 年三道题由错误的“营业收入”口径改为“营业总收入”。
- 数值改为利润表精确千元值，而不是正文四舍五入的亿元概数。
- 数据构建器同步修正，后续重建不会再次把营业总收入误标为营业收入。

## 3. 路由消融

| 档位 | Recall@5 | Route 覆盖 | 缺失义务率 | 重试率 | 平均秒/题 | 污染率 |
|---|---:|---:|---:|---:|---:|---:|
| L2 | 0.9762 | 0.9476 | 0.0762 | 0.0000 | 44.6214 | 0.0095 |
| L3a | 1.0000 | 0.9905 | 0.0000 | 0.0000 | 37.2332 | 0.0000 |
| L3b | 1.0000 | 0.9905 | 0.0000 | 0.0917 | 55.4650 | 0.0000 |

正式 summary：

```text
acceptance.all_passed = true
L3a single-company single-year latency ratio = 0.769317
all reports evaluated = 120
all reports use the same v2 manifest SHA
```

## 4. 三轮 hybrid L3b 生成

| 指标 | 均值 | 标准差 |
|---|---:|---:|
| 答案正确率 | 0.6603 | 0.0119 |
| Claim 支持率 | 0.7988 | 0.0528 |
| 不支持 Claim 率 | 0.2012 | 0.0528 |
| Grounding 通过率 | 0.9403 | 0.0131 |
| 引用完整覆盖率 | 0.9930 | 0.0049 |
| 拒答正确率 | 1.0000 | 0.0000 |
| 上下文溢出率 | 0.0000 | 0.0000 |
| 重试率 | 0.0917 | 0.0000 |
| 平均秒/题 | 92.8829 | 9.8357 |

分题型答案正确率：

```text
fact = 0.6713
table = 0.7593
cross_document = 0.4889
```

## 5. 验证结果

```text
全量测试：198 passed, 1 skipped
跳过项：未配置 Postgres
```

所有正式报告都包含 120 条逐题明细，三轮生成合计 360 条逐题记录。

## 6. 主要产物

- `evals/reports/routing_ablation_summary.json`
- `evals/reports/scaled_generation.json`
- `evals/reports/context_budget.json`
- `evals/reports/scaled_golden_set_audit.json`
- `evals/reports/layer_matrix.json`
- `docs/routing-ablation-results.md`
- `docs/generation-eval-v2.md`

## 7. 已知短板

1. 跨文档题答案正确率只有 `0.4889`，是生成阶段的首要问题。
2. Claim 支持率 `0.798833`，低于计划中的 `0.8` 优化门槛约 `0.001167`。
3. 表格抽取后仍是扁平文本，行列语义需要模型重新推断。
4. hybrid L3b 平均约 92.88 秒/题，主要耗时仍在 CPU rerank。
5. 工作区可能存在历史 CRLF 统计差异，但
   `git diff --ignore-space-at-eol` 为零语义改动。

## 8. 收尾结论

检索、路由、评测基础设施和完整报告已经闭环，可以结束当前阶段。
生成质量优化应作为独立下一阶段，不应继续阻塞本阶段交付。
