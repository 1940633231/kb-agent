# 下一阶段执行计划

> 用途：在新 Codex 会话中直接读取并执行。  
> 当前日期：2026-11-06  
> 当前分支：`codex/stage0-corpus-expansion`

## 1. 当前起点

### 1.1 已完成

- 30 家公司、90 份年报、37,958 chunk。
- P0 证据义务契约与多路覆盖。
- P1 Claim 硬门禁与文本 span 定向修复。
- P2 统一上下文预算。
- P3 120 题评测集与评测 runner。
- 120 题目动源文审计：
  - `normal=120`
  - `review=0`
  - `high_risk=0`
  - `fail=0`

最新关键提交：

```text
f32c527 修正扩展评测集源文绑定与审计
e7cbeeb 实现 P3 扩展评测与三轮报告
688d90b 实现 P2 统一上下文预算
17a7284 feat: add claim audit hard gate for P1
a4c821a feat: add evidence obligation coverage for P0
831ac6e feat: expand annual report corpus to 30 companies
```

### 1.2 当前不可引用的数据

以下报告基于修正前的 120 题，当前均标记为：

```text
stale_dataset_manifest_mismatch
```

- `evals/reports/scaled_retrieval.json`
- `evals/reports/scaled_l3.json`
- `evals/reports/scaled_generation.json`
- `evals/reports/context_budget.json`

不得继续把这些报告中的指标当作当前数据集结果。

### 1.3 当前存储与检索实现

当前 L0-L3 使用：

- 本地 FAISS flat index。
- 本地 BM25。
- 本地 `BAAI/bge-small-zh-v1.5`。
- 本地 `BAAI/bge-reranker-base`。
- LangGraph 编排。

当前不使用：

- Milvus。
- Elasticsearch。
- MySQL。
- Redis。
- Postgres：只有持久化测试和线程持久化场景需要。

因此本轮不要连接 WSL Docker 中的 Milvus/ES。

## 2. 新会话启动步骤

第一条命令：

```powershell
Get-Content docs\next-session-execution-plan.md -Encoding UTF8
git status --short --branch
git log -6 --oneline --decorate
```

要求：

- 工作区可能已有用户改动和 CRLF 变化。
- 不要执行 `git add -A`。
- 不要回滚未参与本计划的文件。
- 只暂存本计划明确涉及的文件。
- 提交说明使用中文。

## 3. 阶段 A：冻结并复核评测集

### A1. 检查 manifest

```powershell
Get-Content -Raw evals\golden_set_scaled_120.manifest.json |
  ConvertFrom-Json |
  Format-List
```

必须满足：

```text
companies = 30
documents = 90
counts.company = 90
counts.cross = 15
counts.refusal = 15
counts.total = 120
```

### A2. 重跑数据审计

```powershell
.\.venv\Scripts\python.exe evals\audit_scaled_golden_set.py
```

通过标准：

```text
records = 120
normal = 120
review = 0
high_risk = 0
fail = 0
```

报告：

- `evals/reports/scaled_golden_set_audit.json`
- `evals/reports/scaled_golden_set_review_shortlist.json`

如果不是 120/120，先修数据集，不得开始模型评测。

### A3. 跑结构测试

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_scaled_golden_set.py `
  tests\test_scaled_golden_set_audit.py `
  tests\test_scaled_eval_reports.py `
  -q -rs
```

## 4. 阶段 B：重跑修正后 L0

### B1. 执行 L0

```powershell
.\.venv\Scripts\python.exe evals\run_scaled_eval.py `
  --mode retrieval `
  --top-k 5
```

L0 定义：

```text
问题向量化 → FAISS → Top-5
```

报告：

```text
evals/reports/scaled_retrieval.json
```

### B2. 检查指标

重点字段：

- `doc_recall@5`
- `all_docs_hit@5`
- `evidence_obligation_coverage`
- `missing_obligation_rate`
- `refuse_top1_score`
- `mean_seconds_per_query`

建议基线门槛：

- `doc_recall@5 >= 0.85`
- manifest SHA 与当前数据集一致
- 报告中不得再出现 `stale_dataset_manifest_mismatch`

### B3. 查失败题

```powershell
$r = Get-Content -Raw evals\reports\scaled_retrieval.json |
  ConvertFrom-Json

$r.questions |
  Where-Object { $_.doc_recall -lt 1 } |
  Select-Object id,question_type,doc_recall,wanted_docs,retrieved_docs |
  Format-Table -AutoSize
```

分流规则：

- 标准值或页码错误：回到数据集修复。
- 正确文档完全未召回：修检索或题目表述。
- 文档召回但证据义务缺失：进入 L3 对比。

### B4. 提交 L0 基线

提交说明建议：

```text
重跑修正评测集 L0 基线
```

## 5. 阶段 C：重跑修正后 L3 检索

### C1. 前置条件

只有 L0 已重跑且失败题完成归因后，才执行 L3。

### C2. 执行 L3

```powershell
.\.venv\Scripts\python.exe evals\run_scaled_eval.py `
  --mode l3 `
  --top-k 5
```

L3 检索定义：

```text
FAISS + 本地 BM25 + RRF + Cross-Encoder Rerank
+ 证据义务 + 路由覆盖 + 缺失义务重试
```

本命令只做检索，不生成答案。

报告：

```text
evals/reports/scaled_l3.json
```

### C3. 验收门槛

- `doc_recall@5 >= 0.95`
- `route_obligation_coverage >= 0.95`
- `missing_obligation_rate <= 0.05`
- `context_overflow_rate` 不适用，生成阶段再检查
- 报告 SHA 与当前 manifest 一致

如果 L3 明显高于 L0：

- 记录编排带来的增量。
- 检查增量是否来自证据义务重试。

如果 L3 没有提升：

- 检查多路路由是否实际触发。
- 检查 rewrite 是否只针对缺失义务。
- 检查 rerank 候选池是否把关键证据截掉。

### C4. 提交 L3 基线

提交说明建议：

```text
重跑修正评测集 L3 基线
```

## 6. 阶段 D：vector 生成三轮

### D1. 执行

```powershell
.\.venv\Scripts\python.exe evals\run_scaled_eval.py `
  --mode generation `
  --repeats 3 `
  --generation-mode vector `
  --top-k 5
```

报告：

```text
evals/reports/scaled_generation.json
```

### D2. 必查指标

- `answer_correctness.mean`
- `answer_correctness.stddev`
- `claim_support_rate.mean`
- `unsupported_claim_rate.mean`
- `citation_full_coverage.mean`
- `refusal_correctness.mean`
- `context_overflow_rate.mean`
- `mean_seconds_per_query.mean`

验收：

- `context_overflow_rate.mean == 0.0`
- 三轮都完整跑完
- 失败题按轮次记录
- 不报告最好单轮成绩

### D3. 生成失败题归因

按以下类别分组：

- 标准值正确，但检索没拿到
- 检索拿到，但答案取错行/列
- 公司或年份串绑
- 多值漏答
- 单位错误
- 引用不覆盖
- Claim 不支持
- 应拒答却作答

### D4. 提交 vector 生成基线

提交说明建议：

```text
重跑修正评测集三轮生成基线
```

## 7. 阶段 E：质量优化与 L3 生成

### E1. 优化触发条件

出现以下任一情况时先优化，不直接跑 L3 生成：

- `claim_support_rate < 0.8`
- 单位错误或跨公司串绑反复出现
- 多值题漏答集中
- 引用完整覆盖率低于 0.95
- 应拒答正确率低于 0.8

### E2. 优化顺序

1. 修评测集标准值和指标绑定。
2. 修检索证据义务覆盖。
3. 修答案生成 prompt 的逐值输出要求。
4. 修 Claim 修复只改硬违规 span。
5. 修引用选择。

### E3. L3 生成三轮

只有前三轮 vector 生成稳定后执行：

```powershell
.\.venv\Scripts\python.exe evals\run_scaled_eval.py `
  --mode generation `
  --repeats 3 `
  --generation-mode hybrid `
  --top-k 5
```

注意：

- 这是最重的一步。
- 不要与抓取、索引重建或其他 CPU 重任务并行。
- 预计数小时。

## 8. 阶段 F：最终报告与提交

最终必须满足：

- 120 题结构通过。
- 数据集 manifest SHA 固定。
- L0/L3 报告不再 stale。
- 三轮生成报告可比较。
- Claim 审计通过。
- 上下文预算溢出为 0。
- 全量测试通过。

执行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -rs
```

提交说明建议：

```text
完成修正评测集三层基线
```

## 9. Milvus / Elasticsearch 边界

本轮不要接 Milvus/ES。

原因：

- 当前 L0-L3 基线统一使用 FAISS + 本地 BM25。
- 中途换后端会引入不可归因变量。
- Milvus 和 ES 需要重新灌入同一批 37,958 chunk。
- 不能拿 Milvus/ES 指标与 FAISS 基线直接混比。

后续如确需生产化实验，单独开分支：

```text
codex/milvus-es-backend-experiment
```

实验要求：

1. 使用同一 embedding。
2. 使用同一 chunk 集合和 metadata。
3. 先做 FAISS vs Milvus 的纯向量对比。
4. 再做本地 BM25 vs ES 的纯稀疏对比。
5. 最后才做 hybrid 对比。
6. 单独输出报告，不覆盖当前 L0-L3 基线和 stale 标记。

## 10. 完成标准

新会话完成时应能回答：

1. 修正后的 120 题是否 120/120 可信？
2. L0 和 L3 的检索差距是多少？
3. L3 提升来自哪一路、哪个义务？
4. 三轮生成答案的均值和标准差是多少？
5. Claim 支持率、引用覆盖率和拒答正确率是多少？
6. 上下文是否零溢出？
7. 哪些失败是数据集问题，哪些是检索问题，哪些是生成问题？
8. 是否所有报告都与当前 manifest SHA 一致？
