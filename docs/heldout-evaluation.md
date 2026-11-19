# Held-out 40 题证据评测

这份报告只回答一个问题：拿到检索证据后，答案是否真的把公司、年份、指标和
数值绑定在同一份被引证据里。

## 数据集

- 文件：`evals/heldout_40.jsonl`
- 版本：`v1-frozen`
- 规模：40 题，8 个场景各 5 题
- dataset SHA256：`1d2c98a0b49f6a7c23299166c7a36bc2d700048e633708ebc6d706ead4001e39`
- 人工确认：`40/40`
- 开发集：`evals/golden_set_scaled_120.jsonl`，SHA256 `9b91a245...`

40 题覆盖已知/未知实体混问、同公司多年份、多口径、正文概数与表格值、
公司别名、应拒答和跨页表格。它是小样本诊断集，不用来宣称泛化正确率。

## 评分口径

每题按五类记录：

- `fully_correct`
- `partially_correct`
- `incorrect`
- `correctly_refused`
- `wrongly_refused`

数值出现在答案里只算诊断信息，不能单独算正确。`evidence_binding_rate` 要求
公司、年份、指标和数值能从被引 chunk 中得到支持；引用页匹配使用
`chunk_start_page`，chunk 绑定仍使用 `chunk_id`。

对于 mixed/partial 题，另一半未知义务必须明确拒答。静默遗漏算
`not_addressed`，编造数字算 `leaked_value`，不能因为一半答对就判为完整正确。

## 前后对比

基线和修复后报告使用同一套最终评分器重评：

| 指标 | 基线 | 修复后 |
|---|---:|---:|
| `fully_correct` | 3/40 | 6/40 |
| `partially_correct` | 16/40 | 15/40 |
| `incorrect` | 6/40 | 5/40 |
| `correctly_refused` | 5/40 | 5/40 |
| `wrongly_refused` | 10/40 | 9/40 |
| `wrong_refusal_rate` | 0.2857 | 0.2571 |
| `evidence_binding_rate` | 0.3714 | 0.4571 |
| `expected_value_presence_rate` | 0.6200 | 0.6400 |
| `citation_chunk_coverage_rate` | 0.1429 | 0.1714 |

分组结果中，别名题从 `0 full / 5 wrongly_refused` 变为
`3 full / 2 wrongly_refused`；跨页题从 `1 full + 4 incorrect` 变为
`1 full + 1 partial + 3 incorrect`。这些是场景级诊断，不是稳定的泛化估计。

## 发现

1. 原先把“期望数值字符串出现”当作答案正确率会把错绑掩盖掉。加入
   公司/年份/指标/数值联合校验后，`0.6603` 只能解释为开发集上的自动代理
   命中率，不能解释为真实业务正确率。
2. 别名归一化是有效的边界改造：`茅台/美的/顺丰/海康/平安` 被路由到正式
   公司名后，别名组的绑定率和完整率都上升，但“美的研发费用”等仍在
   口径/年份邻近值上发生错误拒答。
3. mixed/partial 题的关键不是“答得更多”，而是把未知部分明确标为语料边界。
   当前纯未知题 5/5 正确拒答，但不支持部分仍可能在运行时 Claim 门禁中留下
   `missing_obligation` 提示；评测已经按“明确拒答”而不是按“答满”计分。
4. 跨页剩余失败主要是精确表格 chunk 未召回、上下文压缩后证据丢失，以及
   同一指标的正文概数/表格精确值共存。不能把这三篇论文级问题包装成已解决。

## 面试表述

> 120 题用于开发和路由消融，`0.6603` 是期望数值字符串命中的自动代理指标。
> 为避免把错绑算成答对，我另建了 40 题人工确认的 held-out，要求公司、年份、
> 指标、数值和被引 chunk 联合成立；同口径重评后，完整正确从 3 题提升到
> 6 题，绑定率从 0.3714 提升到 0.4571。剩余问题集中在跨页表格、口径选择和
> 不支持义务的显式拒答，不再继续扩题或堆功能。

## 复现

```powershell
.\.venv\Scripts\python.exe evals\run_heldout_eval.py
.\.venv\Scripts\python.exe evals\run_heldout_eval.py --rescore-existing `
  --output evals\reports\heldout_40_baseline.json
```
