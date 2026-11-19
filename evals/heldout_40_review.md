# Held-out 40 Review

Status: frozen after human confirmation.

Revision: v3 makes page semantics explicit and reclassifies the former `body_vs_table_conflict` rows, which are unit/rounding or caliber variants rather than contradictions.

Source audit: `evidence_page` is the page that actually carries the literal, located by the literal's offset inside its chunk; the jsonl keeps the chunk start page separately in `chunk_start_page`. Conflict evidence is page-located the same way.

Review focus: rows marked `high` should be checked before freezing. Normal rows already passed chunk/page/literal audit and only need a scan.

Scoring rule: `expected_values` are the parts that must be answered. `unsupported_parts` must be explicitly refused or marked outside the corpus. Supplying a number for an unsupported part makes the row incorrect rather than partially correct. Restating the same value in another unit (元/千元/万元/亿元) or with reasonable rounding is equivalent and must be accepted; only a different numeral, or the `different caliber` rows, count as wrong.

Runner note: `run_heldout_eval.py` consumes `unsupported_obligations` and scores mixed/partial answers separately. Retrieval page matching uses `chunk_start_page`, while answer citations use expected chunk IDs.

| ID | Priority | Scenario | Question | Behavior | Expected | Source |
|---|---|---|---|---|---|---|
| heldout-mixed-001 | high | mixed_known_unknown | 贵州茅台2024年销售费用是多少元？同时请给出华为2024年研发费用。 | partial_answer | 销售费用=5,639,300,059.49元; explicitly unsupported: 华为 2024 研发费用 | 600519-2024 p8 600519-2024_chunk_12 |
| heldout-mixed-002 | high | mixed_known_unknown | 美的集团2024年研发费用是多少千元？另外给出苹果公司2024年研发费用。 | partial_answer | 研发费用=16,232,771千元; explicitly unsupported: 苹果公司 2024 研发费用 | 000333-2024 p52 000333-2024_chunk_90 |
| heldout-mixed-003 | high | mixed_known_unknown | 宁德时代2024年经营活动产生的现金流量净额是多少千元？并给出小米集团2024年同一指标。 | partial_answer | 经营活动产生的现金流量净额=96,990,345千元; explicitly unsupported: 小米集团 2024 经营活动产生的现金流量净额 | 300750-2024 p9 300750-2024_chunk_8 |
| heldout-mixed-004 | high | mixed_known_unknown | 海康威视2024年研发费用是多少元？同时说明字节跳动2024年研发费用。 | partial_answer | 研发费用=11,864,013,392.27元; explicitly unsupported: 字节跳动 2024 研发费用 | 002415-2024 p94 002415-2024_chunk_120 |
| heldout-mixed-005 | high | mixed_known_unknown | 恒瑞医药2024年总资产是多少元？另给出辉瑞2024年总资产。 | partial_answer | 总资产=50,135,645,304.10元; explicitly unsupported: 辉瑞 2024 总资产 | 600276-2024 p7 600276-2024_chunk_7 |
| heldout-multiyear-001 | normal | multi_year_same_company | 美的集团2023年和2024年研发费用分别是多少千元？ | answer | 研发费用（2023年）=14,583,311千元; 研发费用（2024年）=16,232,771千元 | 000333-2023 p54 000333-2023_chunk_98, 000333-2024 p52 000333-2024_chunk_90 |
| heldout-multiyear-002 | normal | multi_year_same_company | 五粮液2023年和2024年销售费用分别是多少元？ | answer | 销售费用（2023年）=7,796,298,418.13元; 销售费用（2024年）=10,692,376,117.41元 | 000858-2023 p16 000858-2023_chunk_23, 000858-2024 p16 000858-2024_chunk_23 |
| heldout-multiyear-003 | normal | multi_year_same_company | 海康威视2023年和2024年研发费用分别是多少元？ | answer | 研发费用（2023年）=11,392,948,404.47元; 研发费用（2024年）=11,864,013,392.27元 | 002415-2023 p96 002415-2023_chunk_119, 002415-2024 p94 002415-2024_chunk_120 |
| heldout-multiyear-004 | normal | multi_year_same_company | 恒瑞医药2023年和2024年研发费用分别是多少元？ | answer | 研发费用（2023年）=4,953,887,105.16元; 研发费用（2024年）=6,582,915,544.87元 | 600276-2023 p69 600276-2023_chunk_112, 600276-2024 p53 600276-2024_chunk_70 |
| heldout-multiyear-005 | normal | multi_year_same_company | 长江电力2023年和2024年销售费用分别是多少元？ | answer | 销售费用（2023年）=192,385,627.70元; 销售费用（2024年）=188,142,529.94元 | 600900-2023 p13 600900-2023_chunk_21, 600900-2024 p12 600900-2024_chunk_21 |
| heldout-caliber-001 | normal | multi_caliber | 美的集团2024年归属于上市公司股东的净利润和扣除非经常性损益后的净利润分别是多少千元？ | answer | 归属于上市公司股东的净利润=38,537,237千元; 扣除非经常性损益后的净利润=35,741,418千元 | 000333-2024 p9 000333-2024_chunk_8, 000333-2024 p9 000333-2024_chunk_8 |
| heldout-caliber-002 | normal | multi_caliber | 贵州茅台2024年合并利润表中的净利润与主要会计数据中的归母净利润分别是多少元？ | answer | 合并利润表净利润=89,334,728,025.90元; 归属于上市公司股东的净利润=86,228,146,421.62元 | 600519-2024 p64 600519-2024_chunk_109, 600519-2024 p5 600519-2024_chunk_5 |
| heldout-caliber-003 | normal | multi_caliber | 宁德时代2024年归母净利润和扣非归母净利润分别是多少千元？ | answer | 归属于上市公司股东的净利润=50,744,682千元; 扣除非经常性损益后的归母净利润=44,992,919千元 | 300750-2024 p9 300750-2024_chunk_8, 300750-2024 p9 300750-2024_chunk_8 |
| heldout-caliber-004 | normal | multi_caliber | 中国平安2024年归属于母公司股东的营运利润和归母净利润分别是多少百万元？ | answer | 归属于母公司股东的营运利润=121,862百万元; 归属于母公司股东的净利润=126,607百万元 | 601318-2024 p71 601318-2024_chunk_111, 601318-2024 p71 601318-2024_chunk_111 |
| heldout-caliber-005 | normal | multi_caliber | 中国神华2024年归属于本公司股东的净利润和扣非归母净利润分别是多少百万元？ | answer | 归属于本公司股东的净利润=58,671百万元; 扣除非经常性损益后的归母净利润=60,125百万元 | 601088-2024 p7 601088-2024_chunk_9, 601088-2024 p7 601088-2024_chunk_9 |
| heldout-conflict-001 | high | body_vs_table_crosscheck | 美的集团2024年归母净利润是多少千元？ | answer | 归属于上市公司股东的净利润=38,537,237千元; equivalent rounded restatement, must be accepted: 385亿元 ≈ 385.37237亿元 (000333-2024 p20 000333-2024_chunk_28) | 000333-2024 p9 000333-2024_chunk_8 |
| heldout-conflict-002 | high | body_vs_table_crosscheck | 宁德时代2024年归母净利润是多少千元？ | answer | 归属于上市公司股东的净利润=50,744,682千元; equivalent unit restatement, must be accepted: 507.45亿元 = 507.44682亿元 (300750-2024 p17 300750-2024_chunk_22) | 300750-2024 p9 300750-2024_chunk_8 |
| heldout-conflict-003 | high | body_vs_table_crosscheck | 海康威视2024年归母净利润是多少元？ | answer | 归属于上市公司股东的净利润=11,977,327,023.54元; equivalent unit restatement, must be accepted: 119.77亿元 = 119.77327亿元 (002415-2024 p88 002415-2024_chunk_111) | 002415-2024 p9 002415-2024_chunk_11 |
| heldout-conflict-004 | high | body_vs_table_crosscheck | 伊利股份2024年归母净利润是多少元？ | answer | 归属于上市公司股东的净利润=8,452,859,993.18元; different caliber, not a conflict: 84.64亿元 = 合并净利润 8,463,710,355.65元（含少数股东损益），非归母净利润 (600887-2024 p14 600887-2024_chunk_17) | 600887-2024 p7 600887-2024_chunk_5 |
| heldout-conflict-005 | high | body_vs_table_crosscheck | 海尔智家2024年归母净利润是多少元？ | answer | 归属于上市公司股东的净利润=18,741,120,122.93元; equivalent unit restatement, must be accepted: 187.41亿元 = 187.41120亿元 (600690-2024 p14 600690-2024_chunk_21) | 600690-2024 p10 600690-2024_chunk_16 |
| heldout-partial-001 | high | partial_answer | 美的集团2024年销售费用是多少千元？截至2025年6月30日收盘价是多少元？ | partial_answer | 销售费用=38,753,649千元; explicitly unsupported: 美的集团 2025-06-30 股票收盘价 | 000333-2024 p52 000333-2024_chunk_90 |
| heldout-partial-002 | high | partial_answer | 宁德时代2024年研发费用是多少千元？2025年6月30日收盘价是多少元？ | partial_answer | 研发费用=18,606,756千元; explicitly unsupported: 宁德时代 2025-06-30 股票收盘价 | 300750-2024 p24 300750-2024_chunk_34 |
| heldout-partial-003 | high | partial_answer | 贵州茅台2024年归属于上市公司股东的净利润是多少元？2025年6月30日总市值是多少亿元？ | partial_answer | 归属于上市公司股东的净利润=86,228,146,421.62元; explicitly unsupported: 贵州茅台 2025-06-30 总市值 | 600519-2024 p5 600519-2024_chunk_5 |
| heldout-partial-004 | normal | partial_answer | 长江电力2024年研发费用是多少元？2025年6月30日收盘价是多少元？ | partial_answer | 研发费用=890,719,278.34元; explicitly unsupported: 长江电力 2025-06-30 股票收盘价 | 600900-2024 p12 600900-2024_chunk_21 |
| heldout-partial-005 | normal | partial_answer | 中国神华2024年研发费用是多少百万元？2025年7月秦皇岛动力煤现货价是多少元/吨？ | partial_answer | 研发费用=2,727百万元; explicitly unsupported: 中国神华 2025-07 秦皇岛动力煤现货价 | 601088-2024 p18 601088-2024_chunk_26 |
| heldout-alias-001 | high | alias_or_abbreviation | 茅台2024年的销售费用是多少元？ | answer | 销售费用=5,639,300,059.49元 | 600519-2024 p8 600519-2024_chunk_12 |
| heldout-alias-002 | normal | alias_or_abbreviation | 美的2024年的研发费用是多少千元？ | answer | 研发费用=16,232,771千元 | 000333-2024 p52 000333-2024_chunk_90 |
| heldout-alias-003 | normal | alias_or_abbreviation | 顺丰2024年的研发费用是多少千元？ | answer | 研发费用=2,533,607千元 | 002352-2024 p45 002352-2024_chunk_64 |
| heldout-alias-004 | normal | alias_or_abbreviation | 海康2024年的经营现金流净额是多少元？ | answer | 经营活动产生的现金流量净额=13,264,092,022.73元 | 002415-2024 p9 002415-2024_chunk_11 |
| heldout-alias-005 | high | alias_or_abbreviation | 平安2024年的归母净利润是多少百万元？ | answer | 归属于母公司股东的净利润=126,607百万元 | 601318-2024 p71 601318-2024_chunk_111 |
| heldout-refuse-001 | high | unsupported_scope | 华为2024年的研发费用是多少元？ | refuse | correct refusal: 华为 2024 研发费用 | corpus scope boundary |
| heldout-refuse-002 | high | unsupported_scope | 贵州茅台2026年的销售费用是多少元？ | refuse | correct refusal: 贵州茅台 2026 销售费用 | corpus scope boundary |
| heldout-refuse-003 | high | unsupported_scope | 小米集团2024年的营业收入是多少元？ | refuse | correct refusal: 小米集团 2024 营业收入 | corpus scope boundary |
| heldout-refuse-004 | normal | unsupported_scope | 宁德时代2022年的研发费用是多少元？ | refuse | correct refusal: 宁德时代 2022 研发费用 | corpus scope boundary |
| heldout-refuse-005 | normal | unsupported_scope | 招商银行2026年的净利润是多少元？ | refuse | correct refusal: 招商银行 2026 净利润 | corpus scope boundary |
| heldout-crosspage-001 | normal | cross_page_table | 美的集团2024年归母净利润和研发费用分别是多少千元？ | answer | 归属于上市公司股东的净利润=38,537,237千元; 研发费用=16,232,771千元 | 000333-2024 p9 000333-2024_chunk_8, 000333-2024 p52 000333-2024_chunk_90 |
| heldout-crosspage-002 | normal | cross_page_table | 宁德时代2024年归母净利润和研发费用分别是多少千元？ | answer | 归属于上市公司股东的净利润=50,744,682千元; 研发费用=18,606,756千元 | 300750-2024 p9 300750-2024_chunk_8, 300750-2024 p24 300750-2024_chunk_34 |
| heldout-crosspage-003 | normal | cross_page_table | 海康威视2024年营业收入和研发费用分别是多少元？ | answer | 营业收入=92,495,525,118.30元; 研发费用=11,864,013,392.27元 | 002415-2024 p9 002415-2024_chunk_11, 002415-2024 p94 002415-2024_chunk_120 |
| heldout-crosspage-004 | normal | cross_page_table | 恒瑞医药2024年总资产和研发费用分别是多少元？ | answer | 总资产=50,135,645,304.10元; 研发费用=6,582,915,544.87元 | 600276-2024 p7 600276-2024_chunk_7, 600276-2024 p53 600276-2024_chunk_70 |
| heldout-crosspage-005 | normal | cross_page_table | 海尔智家2024年归母净利润和销售费用分别是多少元？ | answer | 归属于上市公司股东的净利润=18,741,120,122.93元; 销售费用=33,585,903,561.60元 | 600690-2024 p10 600690-2024_chunk_16, 600690-2024 p41 600690-2024_chunk_67 |

## Review Decisions

- [x] Approve all high-priority rows
- [x] Confirm the cross-check rows are unit/rounding or caliber variants, not contradictions
- [x] Replace or edit rows that encode the wrong expected behavior
- [x] Confirm 40/40 after changes
