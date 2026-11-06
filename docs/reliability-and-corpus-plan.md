# kb-agent 数据扩展与可靠性补强计划

> 状态：阶段 0、P0、P1、P2 已完成；P3 数据集已定稿，模型评测待重跑
> 适用范围：`kb-agent` 为主，阶段 2 可能涉及 `production-rag` 的小幅扩展  
> 不在本计划内：`sql-mcp-agent` 的 MCP 链路改造；它只作为后续展示材料

阶段 0 执行结果：30 家公司、90 份年报审计通过；37,958 chunk；原 50 题检索与
生成回归可完整运行。报告见 `evals/reports/corpus_audit.json`、
`evals/reports/stage0_expanded_retrieval.json` 与
`evals/reports/stage0_expanded_generation.json`。

P0 执行结果：证据义务契约、逐路覆盖与缺失义务定向重试已接入；50 题文档
Recall@5 为 `1.0`，证据义务覆盖率 `0.8333`，达到 `0.82+` 目标。报告见
`evals/reports/p0_l3_evidence.json`。

P1 执行结果：Claim 解析、硬门禁、软提示和文本 span 定向修复已接入；
100 条离线污染集硬违规召回 `1.0`，正常答案误伤率 `0.0`。报告见
`evals/reports/claim_audit.json`。

P2 执行结果：system prompt、history、query、evidence 与输出预留已纳入统一预算；
105 个 scaled answerable case 预算溢出率为 `0.0`，P95 预算开销约 `1.78ms`，
端到端估算开销 `0.0027%`。报告见 `evals/reports/context_budget.json`。

P3 执行结果：新增 120 题扩展集与 manifest；确定性检索、L3 编排检索和三轮
vector 生成评测均完成。报告见 `scaled_retrieval.json`、`scaled_l3.json`、
`scaled_generation.json`、`claim_audit.json` 与 `context_budget.json`。

## 1. 目标与基线

### 1.1 目标

先把语料从 3 家公司扩展到 30 家公司，再按顺序补齐 Agent 的证据完整性、
答案校验、上下文预算和评测体系。

本计划按“30 家公司总数”定义：

- 现有：3 家公司 x 3 年 = 9 份年报
- 目标：30 家公司 x 3 年 = 90 份年报
- 新增：27 家公司

如果后续明确要求“在现有基础上再新增 30 家”，则总数改为 33 家，其他流程不变。

### 1.2 当前基线

- 语料：贵州茅台、宁德时代、招商银行，2023-2025 年
- 年报：9 份
- chunk：3,576 个
- 评测集：50 题，其中 45 道可答题、5 道应拒答题
- 本地测试：`75 passed, 1 skipped`
- 当前 L3 证据完整度：`0.7333`
- 当前 L3 答案正确率：`0.778`（LLM 裁判口径）
- 当前主要短板：
  1. 多路检索缺少逐子查询、逐证据义务的完整性判断
  2. 输出校验以数字正则为主，且修复默认关闭
  3. 上下文预算没有统一覆盖 system、history、evidence 和输出预留
  4. 评测数据规模小，生成结果和 LLM 裁判存在波动

### 1.3 执行原则

- 数据扩展必须先完成并通过语料审计，之后才进入 P0-P3。
- 每个阶段都必须留下代码、测试、评测报告和可复述的结论。
- 不通过扩大模型、增加 Agent 数量或更换向量库来掩盖证据模型问题。
- 现有 50 题保留为回归集，不能因为语料扩展而失去历史可比性。
- PDF 和本地索引不提交 Git；`metadata.json`、目标配置和审计报告提交。
- 每个阶段单独形成可回滚的提交，避免把数据、算法和评测口径混在一个变更里。

---

## 阶段 0：扩展到 30 家公司并完成语料审计

### 0.1 为什么排第一

当前 3 家公司不足以稳定观察多公司、跨文档、同指标跨年份和表格口径问题。
数据扩展应先于证据契约和 Claim 校验，否则后续规则会在过小的语料上过拟合。

### 0.2 目标公司清单

现有 3 家保留：

| 公司 | 代码 | 行业 |
|---|---|---|
| 贵州茅台 | 600519 | 消费 |
| 宁德时代 | 300750 | 新能源制造 |
| 招商银行 | 600036 | 银行 |

新增 27 家：

| 公司 | 代码 | 行业 |
|---|---|---|
| 五粮液 | 000858 | 消费 |
| 美的集团 | 000333 | 家电制造 |
| 海尔智家 | 600690 | 家电制造 |
| 伊利股份 | 600887 | 食品饮料 |
| 海天味业 | 603288 | 食品饮料 |
| 泸州老窖 | 000568 | 消费 |
| 比亚迪 | 002594 | 新能源汽车 |
| 隆基绿能 | 601012 | 光伏 |
| 阳光电源 | 300274 | 新能源设备 |
| 汇川技术 | 300124 | 工业自动化 |
| 三一重工 | 600031 | 工程机械 |
| 中联重科 | 000157 | 工程机械 |
| 中国平安 | 601318 | 保险 |
| 兴业银行 | 601166 | 银行 |
| 中信证券 | 600030 | 证券 |
| 东方财富 | 300059 | 金融信息 |
| 海康威视 | 002415 | 电子 |
| 科大讯飞 | 002230 | 人工智能 |
| 中芯国际 | 688981 | 半导体 |
| 立讯精密 | 002475 | 电子制造 |
| 京东方A | 000725 | 面板 |
| 恒瑞医药 | 600276 | 医药 |
| 迈瑞医疗 | 300760 | 医疗器械 |
| 药明康德 | 603259 | 医药服务 |
| 中国神华 | 601088 | 能源 |
| 长江电力 | 600900 | 公用事业 |
| 顺丰控股 | 002352 | 物流 |

备选替换公司：

- 中国中免（601888）
- 万华化学（600309）
- 紫金矿业（601899）
- 长城汽车（601633）
- 山西汾酒（600809）

替换规则：连续三年缺报、PDF 非法、PDFium 文本不可读或明显乱码时替换。
不能为了凑数保留无法检索的文档。

### 0.3 抓取目标配置化

当前 `scripts/fetch_annual_reports.py` 把目标写死在 `TARGETS` 中。先改成外部配置：

- 新增 `configs/cninfo_targets.json`
- 为 `fetch_annual_reports.py` 增加 `--targets` 参数
- 配置字段：`company`、`ts_code`、`stock`、`searchkey`、`enabled`
- 保留现有命令行兼容，避免破坏已记录命令
- 如果 `stock` 查询为空，自动回退到 `searchkey=company`

建议命令：

```powershell
python scripts/fetch_annual_reports.py `
  --targets configs/cninfo_targets.json `
  --years 2023 2024 2025 `
  --out-dir data/docs/annual_reports
```

抓取要求：

- 带 User-Agent 和 Referer
- 每个公司查询分页，限制最大页数
- 每次 PDF 下载前限速
- 已有 `(ts_code, year)` 可跳过
- 失败写入审计结果，不是只在终端打印
- 同一 `(ts_code, year)` 只保留一条 metadata

### 0.4 新增语料审计脚本

新增 `scripts/audit_annual_reports.py`，输出：

```text
evals/reports/corpus_audit.json
```

审计内容：

- 公司数、年报数、年份覆盖矩阵
- `(ts_code, year)` 唯一性
- PDF 文件存在、文件头为 `%PDF`、大小为合理正数
- SHA-256 唯一性
- 每个文件能否被 PDFium 打开
- 抽取字符数下限
- 是否出现“年度报告”、公司名、至少一个财务指标词
- 每条 metadata 的标题年份与记录年份一致
- 缺失公司或缺失年份清单
- 索引重建前的通过/失败状态

建议初始质量门槛：

- 30/30 家公司存在
- 每家公司 3/3 年存在
- 90/90 份 PDF 可打开
- 90/90 份文本抽取非空
- 单份文档抽取字符数不少于 20,000
- 中文关键指标命中率 100%
- SHA-256 无重复

门槛不通过时不得进入 P0。

### 0.5 重建索引与回归

语料审计通过后执行：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_kb.py
.\.venv\Scripts\python.exe scripts\ingest_kb.py --verify-only
.\.venv\Scripts\python.exe -m pytest tests -q -rs
```

索引验收：

- metadata 条目数等于 FAISS 向量 ID 集合
- 每份文档都有 chunk
- 页码映射覆盖所有文档
- 原 50 题仍能完成检索和生成
- 不出现因公司字段、年份字段或文档 ID 变化导致的评测中断

### 0.6 阶段 0 交付物

- `configs/cninfo_targets.json`
- 扩展后的 `data/docs/annual_reports/metadata.json`
- `scripts/audit_annual_reports.py`
- `evals/reports/corpus_audit.json`
- 索引重建记录与测试结果
- README 中的语料规模和抓取说明更新

### 0.7 阶段 0 验收

- 30 家公司、90 份年报全部通过审计
- 索引校验一致
- 原 50 题无评测中断
- 形成“新增前/新增后”chunk 数、语料字符数和检索耗时记录

---

## P0：证据契约与多路完整性

### P0.1 目标

把“证据够不够”从年份/公司/关键词的词面覆盖，升级为可逐项核对的证据义务。

### P0.2 核心模型

新增 `agent/evidence_contract.py`，定义：

```text
EvidenceObligation
- obligation_id
- company
- year
- metric
- expected_kind
- expected_value
- required
- source_scope
- route_ids
```

`expected_kind` 至少包括：

- `single_value`
- `multi_value`
- `calculation_input`
- `refusal`
- `comparison`

### P0.3 检索链路改动

- `RetrievalState` 增加 `obligations`、`route_hits`、`route_coverage`
- 多路检索保留每个 `query_id` 到命中的映射
- `merge_candidates` 去重后仍能追溯命中来自哪些查询
- `retrieve_multi` 增加每个子查询的候选统计
- 统计不只含 `returned`，还要含：
  - `per_query_returned`
  - `per_query_documents`
  - `obligations_covered`
  - `obligations_missing`

### P0.4 评分与路由改动

修改 `agent/grading.py`：

- 保留现有年份/公司/关键词检查作为低成本预筛
- 新增逐义务状态：`covered / missing / ambiguous`
- 返回缺失义务，而不是只返回一个覆盖率
- 同一文档多口径冲突标记为 `ambiguous`
- 计算题必须检查计算输入是否都在证据中
- 多公司和多年份题必须逐个检查，不能因为总数足够就通过

修改 `agent/nodes.py`：

- `decide_next_step` 根据缺失义务决定是否重试
- `rewrite_query` 针对缺失义务改写，不再泛化重写
- 达到重试上限后区分：
  - 完全无证据：拒答
  - 部分义务缺失：带保留作答并人工审核
  - 存在歧义：不能静默作答

### P0.5 LLM 复核边界

`agent/llm_grade.py` 只作为二级复核：

- 不再只看前 5 个 chunk
- 优先传入每个义务对应的候选证据
- 输入必须包括缺失义务和路由覆盖信息
- LLM 复核失败时回退规则结论，不能让外部依赖阻塞主流程

### P0.6 测试

新增 `tests/test_evidence_contract.py` 和 `tests/test_multiroute_coverage.py`：

- 单公司单年份单值
- 单公司多年份多值
- 多公司单值
- 多公司多年份
- 某一路为空但其他路有结果
- 合并后数量足够但缺少一个义务
- 同文档多口径冲突
- 计算题缺少一个输入
- 应拒答题没有有效义务

### P0.7 验收

- 现有 50 题文档召回不下降
- L3 证据完整度从 `0.7333` 提升到 `0.82+`
- 构造的 30 个“合并后假通过”用例，缺失义务检出率不低于 90%
- 重试次数不因误判明显增加
- 所有测试通过

---

## P1：Claim 校验与硬门禁

### P1.1 目标

把答案拆成可验证 Claim，区分“必须拦截”和“只提示”的问题，避免当前修复规则
因为误伤而被整体关闭。

### P1.2 新增模块

新增 `agent/claims.py`：

```text
AnswerClaim
- claim_id
- claim_type: numeric / entity_year / citation / completeness
- text_span
- value
- unit
- company
- year
- citation_ids
- support_status
```

新增 `agent/answer_audit.py`：

- 解析自然语言答案中的 Claim
- 映射引用片段
- 产出硬违规、软问题和可修复项

### P1.3 硬违规与软问题

硬违规：

- 数字在被引片段中完全不存在
- 公司、年份与引用片段绑定错误
- 计算结果的输入不在证据中
- 多值题缺少被点名的公司或年份

软问题：

- 引用覆盖不足但答案数字可能正确
- 同文档多口径未显式说明
- 答案表述不完整但无法自动修复
- 引用页码只覆盖 chunk 首页

只有硬违规进入自动修复；软问题优先进人工审核。

### P1.4 定向修复

修改 `agent/repair.py`：

- 只接收硬违规 Claim
- 修复提示必须包含 Claim、对应证据、缺失义务
- 不允许重新自由生成整段答案
- 修复后重新执行 Claim 审计
- 修复无效时保留原答案并标记人工审核

修改 `agent/runtime.py`：

- 不再用单一 `enable_answer_repair` 开关覆盖全部情况
- 增加硬门禁和软提示的独立开关
- 默认启用硬门禁，默认关闭自由修复

### P1.5 测试

新增 `tests/test_claims.py`、`tests/test_answer_audit.py`：

- 正确数字和引用
- 数字不存在
- 串公司
- 串年份
- 选错指标名
- 单位换算
- 多值漏答
- 单文档多口径
- 正确引用但页码偏移

新增离线污染集：

```text
evals/claim_audit_cases.jsonl
```

至少覆盖 50 条正确样本和 50 条污染样本。

### P1.6 验收

- 硬违规召回率不低于 95%
- 正常答案误伤率不高于 5%
- 修复只改硬违规涉及的 Claim
- `before/after` 审计结果写入评估报告
- 现有测试不退化

---

## P2：统一上下文预算

### P2.1 目标

让 system prompt、历史对话、当前问题、检索证据和输出预留共享一个总预算。

### P2.2 项目边界

主改动在 `kb-agent`，新增：

```text
agent/context_budget.py
```

如果要做干净的统一预算，需要给 `production-rag` 增加可选参数：

- `extra_reserved_tokens`
- 或在 `ContextManager.build()` 中接收 `history_tokens`

`kb-agent` 计算历史预算，`production-rag` 只负责把外部预留纳入检索上下文预算。
如果不修改上游，则只能在 `kb-agent` 外层估算，存在口径重复，作为降级方案。

### P2.3 分配策略

优先级：

1. system prompt
2. 当前问题
3. 输出预留
4. 必要的证据义务
5. 历史对话

历史超预算时：

- 丢弃最早回合
- 保留最近指代所需历史
- 不裁剪已满足证据义务的片段
- 证据不足时优先保留能补齐义务的片段

### P2.4 可观测字段

写入 `retrieval_stats` 或新增 `prompt_budget`：

- `max_context_tokens`
- `system_tokens`
- `query_tokens`
- `history_tokens`
- `evidence_tokens`
- `reserved_output_tokens`
- `skipped_history_turns`
- `skipped_evidence_chunks`
- `budget_overflow`

### P2.5 测试

新增 `tests/test_context_budget.py`：

- 正常短问题
- 5 轮长历史
- 历史加长证据
- 多文档证据
- 输出预留不足
- tokenizer 与字符计数差异
- 预算裁剪后仍保留关键证据

### P2.6 验收

- 长历史加长证据不超模型上下文
- 预算溢出率在测试集为 0
- 证据优先于历史裁剪
- 评测 P50/P95 延迟不因预算计算增加超过 5%

---

## P3：评测体系扩展

### P3.1 目标

数据扩展后，评测要从“50 题年报问答”升级为能验证 30 家公司、跨文档、
多证据义务和答案 Claim 的体系。

### P3.2 数据集

保留现有 50 题：

```text
evals/golden_set.jsonl
```

新增扩展集：

```text
evals/golden_set_scaled_120.jsonl
```

建议构成：

- 30 家公司 x 3 题 = 90 题
- 跨公司/跨年份多值题 = 15 题
- 应拒答题 = 15 题

每题新增字段：

- `required_evidence`
- `required_claims`
- `route_expectations`
- `source_pages`
- `evidence_literals`

### P3.3 新指标

- `Evidence Obligation Coverage`
- `Route Coverage`
- `Missing Obligation Rate`
- `Claim Support Rate`
- `Unsupported Claim Rate`
- `Hard Gate Precision / Recall`
- `Context Overflow Rate`
- `Answer Correctness`
- `Citation Full Coverage`
- `Refusal Correctness`

### P3.4 运行策略

CI 主门禁：

- 单元测试
- golden set 结构校验
- 确定性检索评测
- 证据契约与 Claim 审计

定时或手动运行：

- 生成答案评测
- LLM 裁判
- 同一数据集重复 3 次
- 输出均值、标准差、失败题清单

不能只报告最好一轮。

### P3.5 评测报告

新增或扩展：

```text
evals/reports/scaled_retrieval.json
evals/reports/scaled_l3.json
evals/reports/scaled_generation.json
evals/reports/claim_audit.json
evals/reports/context_budget.json
```

每个报告必须包含：

- 数据集版本
- 模型版本
- 语料版本或 metadata SHA
- 运行时间和次数
- 指标均值和波动
- 失败样本及归因

### P3.6 验收

- 120 题扩展集结构校验通过
- 检索和生成评测都能稳定跑完
- 同模型 3 次运行的结果可比较
- 报告能区分检索、编排、校验和口径变化带来的影响
- 答案正确率以同模型基线为准提升至少 5 个百分点

---

## 推荐执行顺序与工作量

| 顺序 | 阶段 | 预计工作量 | 主要交付 |
|---:|---|---:|---|
| 0 | 30 家公司数据扩展与审计 | 3-5 天 | 90 份年报、审计报告、重建索引 |
| 1 | P0 证据契约与多路完整性 | 3 天 | 义务模型、路由覆盖、回归测试 |
| 2 | P1 Claim 校验与硬门禁 | 3 天 | Claim 模型、硬违规审计、定向修复 |
| 3 | P2 统一上下文预算 | 1-2 天 | 预算器、上游预留参数、超限测试 |
| 4 | P3 评测扩展 | 3-4 天 | 120 题扩展集、新指标、三次运行报告 |

数据抓取和 PDFium 审计可能受巨潮接口、PDF 大小和本地磁盘影响，阶段 0 的时间
应预留余量。索引重建和 CPU embedding 可能耗时较长，建议单独执行，不要和测试混跑。

---

## 新会话执行入口

新会话开始时按以下顺序执行：

1. 阅读本文件。
2. 执行 `git status --short --branch`，确认当前未提交改动。
3. 先完成阶段 0，不得提前改 P0-P3。
4. 阶段 0 审计通过后，提交一次数据和抓取配置变更。
5. 再按 P0、P1、P2、P3 顺序实施，每个阶段单独提交。
6. 每阶段结束执行 `pytest tests -q -rs` 和对应评测。

新会话的第一条执行命令：

```powershell
Get-Content docs\reliability-and-corpus-plan.md -Encoding UTF8
git status --short --branch
```

阶段 0 的第一批代码改动：

```text
configs/cninfo_targets.json
scripts/fetch_annual_reports.py
scripts/audit_annual_reports.py
tests/test_fetch_annual_reports.py
tests/test_corpus_audit.py
```

---

## 完成标准

项目最终应能回答以下问题，并有测试和报告支撑：

1. 30 家公司、90 份年报是否全部可读、可检索、无重复？
2. 一个问题需要哪些证据义务，当前每一路检索分别满足了什么？
3. 答案里的每一个关键 Claim 是否都能映射到具体引用？
4. 哪些错误必须拦截，哪些错误只能提示人工审核？
5. 长历史加长证据时是否仍能保证上下文不溢出？
6. 扩展语料后，检索、生成和校验的收益是否可归因、可复现？
