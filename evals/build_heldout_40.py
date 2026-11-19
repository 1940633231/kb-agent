"""Build and source-audit the 40-question held-out candidate set.

The generated dataset remains a candidate until a human reviewer confirms it.
This script deliberately loads raw chunks rather than model outputs so question
selection is not driven by known agent failures.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = PROJECT_ROOT / "evals"
INDEX_METADATA = PROJECT_ROOT / "data" / "index" / "recursive" / "metadata.json"
PAGE_OFFSETS = PROJECT_ROOT / "data" / "index" / "recursive" / "page_offsets.json"
CORPUS_METADATA = PROJECT_ROOT / "data" / "docs" / "annual_reports" / "metadata.json"
DEV_SET = EVALS_DIR / "golden_set_scaled_120.jsonl"
OUTPUT = EVALS_DIR / "heldout_40.jsonl"
MANIFEST = EVALS_DIR / "heldout_40.manifest.json"
REVIEW = EVALS_DIR / "heldout_40_review.md"
BASELINE = EVALS_DIR / "reports" / "heldout_40_baseline.json"

CATEGORIES = (
    "mixed_known_unknown",
    "multi_year_same_company",
    "multi_caliber",
    "body_vs_table_crosscheck",
    "partial_answer",
    "alias_or_abbreviation",
    "unsupported_scope",
    "cross_page_table",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def F(
    company: str,
    ts_code: str,
    year: int,
    metric: str,
    value: str,
    unit: str,
    display: str,
    doc_id: str,
    chunk_start_page: int,
    chunk_id: str,
) -> dict[str, Any]:
    return {
        "company": company,
        "ts_code": ts_code,
        "year": year,
        "metric": metric,
        "value": value,
        "unit": unit,
        "display": display,
        "doc_id": doc_id,
        "chunk_start_page": chunk_start_page,
        "chunk_id": chunk_id,
    }


def S(
    id_: str,
    category: str,
    question: str,
    behavior: str,
    answerability: str,
    facts: list[dict[str, Any]] | None = None,
    *,
    unsupported: list[dict[str, Any]] | None = None,
    conflicts: list[dict[str, Any]] | None = None,
    review_priority: str = "normal",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": id_,
        "category": category,
        "question": question,
        "expected_behavior": behavior,
        "answerability_status": answerability,
        "facts": facts or [],
        "unsupported": unsupported or [],
        "conflicts": conflicts or [],
        "review_priority": review_priority,
        "notes": notes,
    }


# Reused source-audited facts.
MOUTAI_SALES_2024 = F(
    "贵州茅台", "600519", 2024, "销售费用", "5639300059.49", "元",
    "5,639,300,059.49元", "600519-2024", 8, "600519-2024_chunk_12")
MOUTAI_NET_PROFIT_ATTR_2024 = F(
    "贵州茅台", "600519", 2024, "归属于上市公司股东的净利润",
    "86228146421.62", "元", "86,228,146,421.62元",
    "600519-2024", 4, "600519-2024_chunk_5")
MOUTAI_CONSOLIDATED_NET_PROFIT_2024 = F(
    "贵州茅台", "600519", 2024, "合并利润表净利润",
    "89334728025.90", "元", "89,334,728,025.90元",
    "600519-2024", 64, "600519-2024_chunk_109")

MIDEA_RND_2023 = F(
    "美的集团", "000333", 2023, "研发费用", "14583311", "千元",
    "14,583,311千元", "000333-2023", 53, "000333-2023_chunk_98")
MIDEA_RND_2024 = F(
    "美的集团", "000333", 2024, "研发费用", "16232771", "千元",
    "16,232,771千元", "000333-2024", 51, "000333-2024_chunk_90")
MIDEA_SALES_2024 = F(
    "美的集团", "000333", 2024, "销售费用", "38753649", "千元",
    "38,753,649千元", "000333-2024", 51, "000333-2024_chunk_90")
MIDEA_NET_PROFIT_ATTR_2024 = F(
    "美的集团", "000333", 2024, "归属于上市公司股东的净利润",
    "38537237", "千元", "38,537,237千元",
    "000333-2024", 8, "000333-2024_chunk_8")
MIDEA_DEDUCTED_NET_PROFIT_2024 = F(
    "美的集团", "000333", 2024, "扣除非经常性损益后的净利润",
    "35741418", "千元", "35,741,418千元",
    "000333-2024", 8, "000333-2024_chunk_8")

CATL_RND_2024 = F(
    "宁德时代", "300750", 2024, "研发费用", "18606756", "千元",
    "18,606,756千元", "300750-2024", 23, "300750-2024_chunk_34")
CATL_CASHFLOW_2024 = F(
    "宁德时代", "300750", 2024, "经营活动产生的现金流量净额",
    "96990345", "千元", "96,990,345千元",
    "300750-2024", 8, "300750-2024_chunk_8")
CATL_NET_PROFIT_ATTR_2024 = F(
    "宁德时代", "300750", 2024, "归属于上市公司股东的净利润",
    "50744682", "千元", "50,744,682千元",
    "300750-2024", 8, "300750-2024_chunk_8")
CATL_DEDUCTED_NET_PROFIT_2024 = F(
    "宁德时代", "300750", 2024, "扣除非经常性损益后的归母净利润",
    "44992919", "千元", "44,992,919千元",
    "300750-2024", 8, "300750-2024_chunk_8")

HIKVISION_RND_2023 = F(
    "海康威视", "002415", 2023, "研发费用", "11392948404.47", "元",
    "11,392,948,404.47元", "002415-2023", 95, "002415-2023_chunk_119")
HIKVISION_RND_2024 = F(
    "海康威视", "002415", 2024, "研发费用", "11864013392.27", "元",
    "11,864,013,392.27元", "002415-2024", 93, "002415-2024_chunk_120")
HIKVISION_CASHFLOW_2024 = F(
    "海康威视", "002415", 2024, "经营活动产生的现金流量净额",
    "13264092022.73", "元", "13,264,092,022.73元",
    "002415-2024", 9, "002415-2024_chunk_11")
HIKVISION_NET_PROFIT_ATTR_2024 = F(
    "海康威视", "002415", 2024, "归属于上市公司股东的净利润",
    "11977327023.54", "元", "11,977,327,023.54元",
    "002415-2024", 9, "002415-2024_chunk_11")
HIKVISION_REVENUE_2024 = F(
    "海康威视", "002415", 2024, "营业收入", "92495525118.30", "元",
    "92,495,525,118.30元", "002415-2024", 9, "002415-2024_chunk_11")

HENGRUI_TOTAL_ASSETS_2024 = F(
    "恒瑞医药", "600276", 2024, "总资产", "50135645304.10", "元",
    "50,135,645,304.10元", "600276-2024", 6, "600276-2024_chunk_7")
HENGRUI_RND_2023 = F(
    "恒瑞医药", "600276", 2023, "研发费用", "4953887105.16", "元",
    "4,953,887,105.16元", "600276-2023", 68, "600276-2023_chunk_112")
HENGRUI_RND_2024 = F(
    "恒瑞医药", "600276", 2024, "研发费用", "6582915544.87", "元",
    "6,582,915,544.87元", "600276-2024", 52, "600276-2024_chunk_70")

WULIANGYE_SALES_2023 = F(
    "五粮液", "000858", 2023, "销售费用", "7796298418.13", "元",
    "7,796,298,418.13元", "000858-2023", 16, "000858-2023_chunk_23")
WULIANGYE_SALES_2024 = F(
    "五粮液", "000858", 2024, "销售费用", "10692376117.41", "元",
    "10,692,376,117.41元", "000858-2024", 15, "000858-2024_chunk_23")

CHANGJIANG_SALES_2023 = F(
    "长江电力", "600900", 2023, "销售费用", "192385627.70", "元",
    "192,385,627.70元", "600900-2023", 13, "600900-2023_chunk_21")
CHANGJIANG_SALES_2024 = F(
    "长江电力", "600900", 2024, "销售费用", "188142529.94", "元",
    "188,142,529.94元", "600900-2024", 11, "600900-2024_chunk_21")
CHANGJIANG_RND_2024 = F(
    "长江电力", "600900", 2024, "研发费用", "890719278.34", "元",
    "890,719,278.34元", "600900-2024", 11, "600900-2024_chunk_21")

PINGAN_OPERATING_PROFIT_2024 = F(
    "中国平安", "601318", 2024, "归属于母公司股东的营运利润",
    "121862", "百万元", "121,862百万元",
    "601318-2024", 70, "601318-2024_chunk_111")
PINGAN_NET_PROFIT_ATTR_2024 = F(
    "中国平安", "601318", 2024, "归属于母公司股东的净利润",
    "126607", "百万元", "126,607百万元",
    "601318-2024", 70, "601318-2024_chunk_111")

SHENHUA_NET_PROFIT_ATTR_2024 = F(
    "中国神华", "601088", 2024, "归属于本公司股东的净利润",
    "58671", "百万元", "58,671百万元",
    "601088-2024", 7, "601088-2024_chunk_9")
SHENHUA_DEDUCTED_NET_PROFIT_2024 = F(
    "中国神华", "601088", 2024, "扣除非经常性损益后的归母净利润",
    "60125", "百万元", "60,125百万元",
    "601088-2024", 7, "601088-2024_chunk_9")
SHENHUA_RND_2024 = F(
    "中国神华", "601088", 2024, "研发费用", "2727", "百万元",
    "2,727百万元", "601088-2024", 17, "601088-2024_chunk_26")

YILI_NET_PROFIT_ATTR_2024 = F(
    "伊利股份", "600887", 2024, "归属于上市公司股东的净利润",
    "8452859993.18", "元", "8,452,859,993.18元",
    "600887-2024", 7, "600887-2024_chunk_5")
HAIER_NET_PROFIT_ATTR_2024 = F(
    "海尔智家", "600690", 2024, "归属于上市公司股东的净利润",
    "18741120122.93", "元", "18,741,120,122.93元",
    "600690-2024", 10, "600690-2024_chunk_16")
HAIER_SALES_2024 = F(
    "海尔智家", "600690", 2024, "销售费用", "33585903561.60", "元",
    "33,585,903,561.60元", "600690-2024", 41, "600690-2024_chunk_67")

SF_RND_2024 = F(
    "顺丰控股", "002352", 2024, "研发费用", "2533607", "千元",
    "2,533,607千元", "002352-2024", 44, "002352-2024_chunk_64")


SPECS = [
    S("heldout-mixed-001", "mixed_known_unknown",
      "贵州茅台2024年销售费用是多少元？同时请给出华为2024年研发费用。",
      "partial_answer", "partially_supported", [MOUTAI_SALES_2024],
      unsupported=[{"entity": "华为", "year": 2024, "metric": "研发费用"}],
      review_priority="high"),
    S("heldout-mixed-002", "mixed_known_unknown",
      "美的集团2024年研发费用是多少千元？另外给出苹果公司2024年研发费用。",
      "partial_answer", "partially_supported", [MIDEA_RND_2024],
      unsupported=[{"entity": "苹果公司", "year": 2024, "metric": "研发费用"}],
      review_priority="high"),
    S("heldout-mixed-003", "mixed_known_unknown",
      "宁德时代2024年经营活动产生的现金流量净额是多少千元？并给出小米集团2024年同一指标。",
      "partial_answer", "partially_supported", [CATL_CASHFLOW_2024],
      unsupported=[{"entity": "小米集团", "year": 2024,
                    "metric": "经营活动产生的现金流量净额"}],
      review_priority="high"),
    S("heldout-mixed-004", "mixed_known_unknown",
      "海康威视2024年研发费用是多少元？同时说明字节跳动2024年研发费用。",
      "partial_answer", "partially_supported", [HIKVISION_RND_2024],
      unsupported=[{"entity": "字节跳动", "year": 2024, "metric": "研发费用"}],
      review_priority="high"),
    S("heldout-mixed-005", "mixed_known_unknown",
      "恒瑞医药2024年总资产是多少元？另给出辉瑞2024年总资产。",
      "partial_answer", "partially_supported", [HENGRUI_TOTAL_ASSETS_2024],
      unsupported=[{"entity": "辉瑞", "year": 2024, "metric": "总资产"}],
      review_priority="high"),

    S("heldout-multiyear-001", "multi_year_same_company",
      "美的集团2023年和2024年研发费用分别是多少千元？",
      "answer", "fully_supported", [MIDEA_RND_2023, MIDEA_RND_2024]),
    S("heldout-multiyear-002", "multi_year_same_company",
      "五粮液2023年和2024年销售费用分别是多少元？",
      "answer", "fully_supported", [WULIANGYE_SALES_2023, WULIANGYE_SALES_2024]),
    S("heldout-multiyear-003", "multi_year_same_company",
      "海康威视2023年和2024年研发费用分别是多少元？",
      "answer", "fully_supported", [HIKVISION_RND_2023, HIKVISION_RND_2024]),
    S("heldout-multiyear-004", "multi_year_same_company",
      "恒瑞医药2023年和2024年研发费用分别是多少元？",
      "answer", "fully_supported", [HENGRUI_RND_2023, HENGRUI_RND_2024]),
    S("heldout-multiyear-005", "multi_year_same_company",
      "长江电力2023年和2024年销售费用分别是多少元？",
      "answer", "fully_supported",
      [CHANGJIANG_SALES_2023, CHANGJIANG_SALES_2024]),

    S("heldout-caliber-001", "multi_caliber",
      "美的集团2024年归属于上市公司股东的净利润和扣除非经常性损益后的净利润分别是多少千元？",
      "answer", "fully_supported",
      [MIDEA_NET_PROFIT_ATTR_2024, MIDEA_DEDUCTED_NET_PROFIT_2024]),
    S("heldout-caliber-002", "multi_caliber",
      "贵州茅台2024年合并利润表中的净利润与主要会计数据中的归母净利润分别是多少元？",
      "answer", "fully_supported",
      [MOUTAI_CONSOLIDATED_NET_PROFIT_2024, MOUTAI_NET_PROFIT_ATTR_2024]),
    S("heldout-caliber-003", "multi_caliber",
      "宁德时代2024年归母净利润和扣非归母净利润分别是多少千元？",
      "answer", "fully_supported",
      [CATL_NET_PROFIT_ATTR_2024, CATL_DEDUCTED_NET_PROFIT_2024]),
    S("heldout-caliber-004", "multi_caliber",
      "中国平安2024年归属于母公司股东的营运利润和归母净利润分别是多少百万元？",
      "answer", "fully_supported",
      [PINGAN_OPERATING_PROFIT_2024, PINGAN_NET_PROFIT_ATTR_2024]),
    S("heldout-caliber-005", "multi_caliber",
      "中国神华2024年归属于本公司股东的净利润和扣非归母净利润分别是多少百万元？",
      "answer", "fully_supported",
      [SHENHUA_NET_PROFIT_ATTR_2024, SHENHUA_DEDUCTED_NET_PROFIT_2024]),

    S("heldout-conflict-001", "body_vs_table_crosscheck",
      "美的集团2024年归母净利润是多少千元？",
      "answer", "fully_supported", [MIDEA_NET_PROFIT_ATTR_2024],
      conflicts=[{"literal": "385亿元", "doc_id": "000333-2024",
                  "chunk_start_page": 20, "chunk_id": "000333-2024_chunk_28",
                  "kind": "rounded_restatement",
                  "relation": "≈ 385.37237亿元"}],
      review_priority="high"),
    S("heldout-conflict-002", "body_vs_table_crosscheck",
      "宁德时代2024年归母净利润是多少千元？",
      "answer", "fully_supported", [CATL_NET_PROFIT_ATTR_2024],
      conflicts=[{"literal": "507.45亿元", "doc_id": "300750-2024",
                  "chunk_start_page": 16, "chunk_id": "300750-2024_chunk_22",
                  "kind": "equivalent_unit",
                  "relation": "= 507.44682亿元"}],
      review_priority="high"),
    S("heldout-conflict-003", "body_vs_table_crosscheck",
      "海康威视2024年归母净利润是多少元？",
      "answer", "fully_supported", [HIKVISION_NET_PROFIT_ATTR_2024],
      conflicts=[{"literal": "119.77亿元", "doc_id": "002415-2024",
                  "chunk_start_page": 88, "chunk_id": "002415-2024_chunk_111",
                  "kind": "equivalent_unit",
                  "relation": "= 119.77327亿元"}],
      review_priority="high"),
    S("heldout-conflict-004", "body_vs_table_crosscheck",
      "伊利股份2024年归母净利润是多少元？",
      "answer", "fully_supported", [YILI_NET_PROFIT_ATTR_2024],
      conflicts=[{"literal": "84.64亿元", "doc_id": "600887-2024",
                  "chunk_start_page": 13, "chunk_id": "600887-2024_chunk_17",
                  "kind": "different_caliber",
                  "relation": "= 合并净利润 8,463,710,355.65元（含少数股东损益），非归母净利润"}],
      review_priority="high"),
    S("heldout-conflict-005", "body_vs_table_crosscheck",
      "海尔智家2024年归母净利润是多少元？",
      "answer", "fully_supported", [HAIER_NET_PROFIT_ATTR_2024],
      conflicts=[{"literal": "187.41亿元", "doc_id": "600690-2024",
                  "chunk_start_page": 12, "chunk_id": "600690-2024_chunk_21",
                  "kind": "equivalent_unit",
                  "relation": "= 187.41120亿元"}],
      review_priority="high"),

    S("heldout-partial-001", "partial_answer",
      "美的集团2024年销售费用是多少千元？截至2025年6月30日收盘价是多少元？",
      "partial_answer", "partially_supported", [MIDEA_SALES_2024],
      unsupported=[{"entity": "美的集团", "date": "2025-06-30",
                    "metric": "股票收盘价"}],
      review_priority="high"),
    S("heldout-partial-002", "partial_answer",
      "宁德时代2024年研发费用是多少千元？2025年6月30日收盘价是多少元？",
      "partial_answer", "partially_supported", [CATL_RND_2024],
      unsupported=[{"entity": "宁德时代", "date": "2025-06-30",
                    "metric": "股票收盘价"}],
      review_priority="high"),
    S("heldout-partial-003", "partial_answer",
      "贵州茅台2024年归属于上市公司股东的净利润是多少元？2025年6月30日总市值是多少亿元？",
      "partial_answer", "partially_supported", [MOUTAI_NET_PROFIT_ATTR_2024],
      unsupported=[{"entity": "贵州茅台", "date": "2025-06-30",
                    "metric": "总市值"}],
      review_priority="high"),
    S("heldout-partial-004", "partial_answer",
      "长江电力2024年研发费用是多少元？2025年6月30日收盘价是多少元？",
      "partial_answer", "partially_supported", [CHANGJIANG_RND_2024],
      unsupported=[{"entity": "长江电力", "date": "2025-06-30",
                    "metric": "股票收盘价"}]),
    S("heldout-partial-005", "partial_answer",
      "中国神华2024年研发费用是多少百万元？2025年7月秦皇岛动力煤现货价是多少元/吨？",
      "partial_answer", "partially_supported", [SHENHUA_RND_2024],
      unsupported=[{"entity": "中国神华", "date": "2025-07",
                    "metric": "秦皇岛动力煤现货价"}]),

    S("heldout-alias-001", "alias_or_abbreviation",
      "茅台2024年的销售费用是多少元？",
      "answer", "fully_supported", [MOUTAI_SALES_2024],
      review_priority="high"),
    S("heldout-alias-002", "alias_or_abbreviation",
      "美的2024年的研发费用是多少千元？",
      "answer", "fully_supported", [MIDEA_RND_2024]),
    S("heldout-alias-003", "alias_or_abbreviation",
      "顺丰2024年的研发费用是多少千元？",
      "answer", "fully_supported", [SF_RND_2024]),
    S("heldout-alias-004", "alias_or_abbreviation",
      "海康2024年的经营现金流净额是多少元？",
      "answer", "fully_supported", [HIKVISION_CASHFLOW_2024]),
    S("heldout-alias-005", "alias_or_abbreviation",
      "平安2024年的归母净利润是多少百万元？",
      "answer", "fully_supported", [PINGAN_NET_PROFIT_ATTR_2024],
      review_priority="high"),

    S("heldout-refuse-001", "unsupported_scope",
      "华为2024年的研发费用是多少元？", "refuse", "unsupported_entity",
      unsupported=[{"entity": "华为", "year": 2024, "metric": "研发费用"}],
      review_priority="high"),
    S("heldout-refuse-002", "unsupported_scope",
      "贵州茅台2026年的销售费用是多少元？", "refuse", "unsupported_year",
      unsupported=[{"entity": "贵州茅台", "year": 2026, "metric": "销售费用"}],
      review_priority="high"),
    S("heldout-refuse-003", "unsupported_scope",
      "小米集团2024年的营业收入是多少元？", "refuse", "unsupported_entity",
      unsupported=[{"entity": "小米集团", "year": 2024, "metric": "营业收入"}],
      review_priority="high"),
    S("heldout-refuse-004", "unsupported_scope",
      "宁德时代2022年的研发费用是多少元？", "refuse", "unsupported_year",
      unsupported=[{"entity": "宁德时代", "year": 2022, "metric": "研发费用"}]),
    S("heldout-refuse-005", "unsupported_scope",
      "招商银行2026年的净利润是多少元？", "refuse", "unsupported_year",
      unsupported=[{"entity": "招商银行", "year": 2026, "metric": "净利润"}]),

    S("heldout-crosspage-001", "cross_page_table",
      "美的集团2024年归母净利润和研发费用分别是多少千元？",
      "answer", "fully_supported",
      [MIDEA_NET_PROFIT_ATTR_2024, MIDEA_RND_2024]),
    S("heldout-crosspage-002", "cross_page_table",
      "宁德时代2024年归母净利润和研发费用分别是多少千元？",
      "answer", "fully_supported",
      [CATL_NET_PROFIT_ATTR_2024, CATL_RND_2024]),
    S("heldout-crosspage-003", "cross_page_table",
      "海康威视2024年营业收入和研发费用分别是多少元？",
      "answer", "fully_supported",
      [HIKVISION_REVENUE_2024, HIKVISION_RND_2024]),
    S("heldout-crosspage-004", "cross_page_table",
      "恒瑞医药2024年总资产和研发费用分别是多少元？",
      "answer", "fully_supported",
      [HENGRUI_TOTAL_ASSETS_2024, HENGRUI_RND_2024]),
    S("heldout-crosspage-005", "cross_page_table",
      "海尔智家2024年归母净利润和销售费用分别是多少元？",
      "answer", "fully_supported", [HAIER_NET_PROFIT_ATTR_2024, HAIER_SALES_2024]),
]


def numeric_value(raw: str) -> float | int:
    value = float(raw.replace(",", ""))
    return int(value) if value.is_integer() else value


def expected_docs(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    docs = []
    for fact in facts:
        key = (fact["company"], fact["year"], fact["ts_code"])
        if key in seen:
            continue
        seen.add(key)
        docs.append({"company": fact["company"], "year": fact["year"],
                     "ts_code": fact["ts_code"]})
    return docs


def literal_index(content: str, literal: str) -> int | None:
    """Raw character index of a literal inside a chunk.

    Tolerates whitespace inside the literal: PDF extraction splits long
    figures ("1,422, 602") and body prose separates a number from its unit
    ("507.45 亿元").
    """
    if literal and literal in content:
        return content.index(literal)
    compact = re.sub(r"\s+", "", literal or "")
    if not compact:
        return None
    positions = [i for i, ch in enumerate(content) if not ch.isspace()]
    stripped = "".join(ch for ch in content if not ch.isspace())
    at = stripped.find(compact)
    return positions[at] if at >= 0 else None


def locate_fact_literal(content: str,
                        fact: dict[str, Any]) -> tuple[int, str] | None:
    """Locate the literal form of a fact that actually occurs in its chunk."""
    candidates = [str(fact["value"]), fact["display"]]
    try:
        number = float(fact["value"])
    except ValueError:
        pass
    else:
        candidates.extend([
            "{:,.2f}".format(number),
            "{:,.0f}".format(number),
            "{:.2f}".format(number),
            "{:.0f}".format(number),
        ])
    for candidate in candidates:
        at = literal_index(content, candidate)
        if at is not None:
            return at, candidate
    return None


def build_record(spec: dict[str, Any], chunks: dict[str, dict[str, Any]],
                 offsets: dict[str, Any]) -> dict[str, Any]:
    facts = spec["facts"]
    evidence = []
    claims = []
    expected_values = []
    source_pages = []
    for index, fact in enumerate(facts, start=1):
        chunk = chunks[fact["chunk_id"]]
        content = chunk.get("content") or ""
        located = locate_fact_literal(content, fact)
        if located is None:
            raise ValueError("literal missing in {} for {}".format(
                fact["chunk_id"], fact["metric"]))
        literal_at, literal = located
        page = page_for_offset(
            fact["doc_id"], int(chunk.get("start_offset") or 0) + literal_at,
            offsets)
        if page is None:
            raise ValueError("no page for {} in {}".format(
                fact["chunk_id"], fact["doc_id"]))
        evidence.append({
            "obligation_id": "obligation-{}".format(index),
            "company": fact["company"],
            "year": fact["year"],
            "metric": fact["metric"],
            "expected_kind": "multi_value" if len(facts) > 1 else "single_value",
            "expected_value": numeric_value(fact["value"]),
            "required": True,
            "source_scope": "annual_report",
            "route_ids": ["route-{}".format(index)],
        })
        claims.append({
            "claim_id": "claim-{}".format(index),
            "claim_type": "numeric",
            "company": fact["company"],
            "year": fact["year"],
            "metric": fact["metric"],
            "expected_value": numeric_value(fact["value"]),
            "unit": fact["unit"],
        })
        expected_values.append({
            "label": fact["metric"],
            "company": fact["company"],
            "year": fact["year"],
            "value": numeric_value(fact["value"]),
            "unit": fact["unit"],
            "display": fact["display"],
        })
        source_pages.append({
            "doc": fact["doc_id"],
            "evidence_page": page,
            "chunk_start_page": fact["chunk_start_page"],
            "chunk_id": fact["chunk_id"],
            "literal": literal,
        })
    conflicts = []
    for item in spec["conflicts"]:
        chunk = chunks[item["chunk_id"]]
        content = chunk.get("content") or ""
        literal_at = literal_index(content, item["literal"])
        if literal_at is None:
            raise ValueError("conflict literal missing in {}".format(
                item["chunk_id"]))
        page = page_for_offset(
            item["doc_id"], int(chunk.get("start_offset") or 0) + literal_at,
            offsets)
        if page is None:
            raise ValueError("no page for conflict in {}".format(
                item["chunk_id"]))
        conflicts.append(dict(
            item,
            evidence_page=page,
        ))

    unsupported_obligations = [
        {
            "obligation_id": "unsupported-{}".format(index),
            "entity": item.get("entity"),
            "year": item.get("year"),
            "date": item.get("date"),
            "metric": item.get("metric"),
            "expected_kind": "must_refuse",
            "required": True,
            "reason": "outside_corpus",
        }
        for index, item in enumerate(spec["unsupported"], start=1)
    ]
    return {
        "id": spec["id"],
        "dataset_source": "heldout_candidate_source_audited",
        "scenario": spec["category"],
        "question_type": spec["category"],
        "expected_behavior": spec["expected_behavior"],
        "answerability_status": spec["answerability_status"],
        "question": spec["question"],
        "expected_docs": expected_docs(facts),
        "expected_keywords": list(dict.fromkeys(
            fact["metric"] for fact in facts)),
        "expected_values": expected_values,
        "evidence_literals": [fact["value"] for fact in facts],
        "source_pages": source_pages,
        "required_evidence": evidence,
        "required_claims": claims,
        "unsupported_parts": spec["unsupported"],
        "unsupported_obligations": unsupported_obligations,
        "conflicting_evidence": conflicts,
        "review": {
            "source_audit": "pending",
            "human_confirmation": "pending",
            "priority": spec["review_priority"],
            "notes": spec["notes"],
        },
    }


def page_for_offset(document_id: str, offset: int,
                    offsets: dict[str, Any]) -> int | None:
    for page, start, end in (offsets.get(document_id) or {}).get("pages", []):
        if int(start) <= int(offset) < int(end):
            return int(page)
    return None


def validate(records: list[dict[str, Any]],
             chunks: dict[str, dict[str, Any]],
             offsets: dict[str, Any]) -> None:
    if len(records) != 40:
        raise ValueError("expected 40 records, got {}".format(len(records)))
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids")
    questions = [re.sub(r"\s+", "", record["question"]) for record in records]
    if len(questions) != len(set(questions)):
        raise ValueError("duplicate held-out questions")
    dev_questions = {
        re.sub(r"\s+", "", json.loads(line)["question"])
        for line in DEV_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    overlap = sorted(set(questions) & dev_questions)
    if overlap:
        raise ValueError("held-out overlaps development set: {}".format(overlap[:3]))

    facts_by_doc: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["expected_behavior"] == "refuse":
            if not record["unsupported_parts"]:
                raise ValueError("refusal without unsupported parts: {}".format(
                    record["id"]))
            continue
        if not record["expected_values"]:
            raise ValueError("answer without expected values: {}".format(
                record["id"]))
        for value, source in zip(record["expected_values"],
                                 record["source_pages"]):
            chunk = chunks.get(source["chunk_id"])
            if chunk is None:
                raise ValueError("unknown chunk {}".format(source["chunk_id"]))
            if chunk["document_id"] != source["doc"]:
                raise ValueError("chunk/doc mismatch {}".format(record["id"]))
            content = chunk.get("content") or ""
            chunk_page = page_for_offset(
                source["doc"], int(chunk.get("start_offset") or 0), offsets)
            if chunk_page != source["chunk_start_page"]:
                raise ValueError("chunk page mismatch {}: {} != {}".format(
                    source["chunk_id"], chunk_page,
                    source["chunk_start_page"]))
            located = literal_index(
                content, source.get("literal") or str(value["value"]))
            if located is None:
                raise ValueError("literal missing in {} for {}".format(
                    source["chunk_id"], value["label"]))
            evidence_page = page_for_offset(
                source["doc"], int(chunk.get("start_offset") or 0) + located,
                offsets)
            if evidence_page != source["evidence_page"]:
                raise ValueError("evidence page mismatch {}: {} != {}".format(
                    source["chunk_id"], evidence_page,
                    source["evidence_page"]))
            facts_by_doc.setdefault(source["doc"], []).append(value)
        for conflict in record["conflicting_evidence"]:
            chunk = chunks.get(conflict["chunk_id"])
            if chunk is None:
                raise ValueError("unknown conflict chunk {}".format(
                    conflict["chunk_id"]))
            content = chunk.get("content") or ""
            located = literal_index(content, conflict["literal"])
            if located is None:
                raise ValueError("conflict literal missing in {}".format(
                    conflict["chunk_id"]))
            chunk_page = page_for_offset(
                conflict["doc_id"], int(chunk.get("start_offset") or 0), offsets)
            if chunk_page != conflict["chunk_start_page"]:
                raise ValueError("conflict chunk page mismatch {}".format(
                    conflict["chunk_id"]))
            evidence_page = page_for_offset(
                conflict["doc_id"], int(chunk.get("start_offset") or 0) + located,
                offsets)
            if evidence_page != conflict["evidence_page"]:
                raise ValueError("conflict evidence page mismatch {}".format(
                    conflict["chunk_id"]))
    for record in records:
        if len(record["unsupported_obligations"]) != len(
                record["unsupported_parts"]):
            raise ValueError("unsupported obligation mismatch {}".format(
                record["id"]))
        for obligation in record["unsupported_obligations"]:
            if obligation["expected_kind"] != "must_refuse":
                raise ValueError("bad unsupported obligation {}".format(
                    record["id"]))
    counts = Counter(record["scenario"] for record in records)
    if any(counts[category] != 5 for category in CATEGORIES):
        raise ValueError("category counts are not 5 each: {}".format(dict(counts)))


def expected_label(item: dict[str, Any], record: dict[str, Any]) -> str:
    label = str(item["label"])
    if record["scenario"] == "multi_year_same_company" and item.get("year"):
        return "{}（{}年）".format(label, item["year"])
    return label


def conflict_phrase(item: dict[str, Any]) -> str:
    kind = item.get("kind") or "conflict"
    relation = item.get("relation") or ""
    where = "{} p{} {}".format(
        item["doc_id"], item["evidence_page"], item["chunk_id"])
    if kind == "different_caliber":
        return "different caliber, not a conflict: {} {} ({})".format(
            item["literal"], relation, where)
    if kind == "rounded_restatement":
        return ("equivalent rounded restatement, must be accepted: "
                "{} {} ({})").format(item["literal"], relation, where)
    return "equivalent unit restatement, must be accepted: {} {} ({})".format(
        item["literal"], relation, where)


def review_markdown(records: list[dict[str, Any]]) -> str:
    frozen = all(
        record["review"]["human_confirmation"] == "confirmed"
        for record in records
    )
    lines = [
        "# Held-out 40 Review",
        "",
        "Status: {}.".format(
            "frozen after human confirmation" if frozen
            else "source-audited candidates; human confirmation pending"),
        "",
        "Revision: v3 makes page semantics explicit and reclassifies "
        "the former `body_vs_table_conflict` rows, which are unit/rounding or "
        "caliber variants rather than contradictions.",
        "",
        "Source audit: `evidence_page` is the page that actually carries the literal, "
        "located by the literal's offset inside its chunk; the jsonl keeps the "
        "chunk start page separately in `chunk_start_page`. Conflict evidence is "
        "page-located the same way.",
        "",
        "Review focus: rows marked `high` should be checked before freezing. "
        "Normal rows already passed chunk/page/literal audit and only need a scan.",
        "",
        "Scoring rule: `expected_values` are the parts that must be answered. "
        "`unsupported_parts` must be explicitly refused or marked outside the "
        "corpus. Supplying a number for an unsupported part makes the row "
        "incorrect rather than partially correct. Restating the same value in "
        "another unit (元/千元/万元/亿元) or with reasonable rounding is "
        "equivalent and must be accepted; only a different numeral, or the "
        "`different caliber` rows, count as wrong.",
        "",
        "Runner note: `run_heldout_eval.py` consumes `unsupported_obligations` "
        "and scores mixed/partial answers separately. Retrieval page matching "
        "uses `chunk_start_page`, while answer citations use expected chunk IDs.",
        "",
        "| ID | Priority | Scenario | Question | Behavior | Expected | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        if record["expected_values"]:
            expected = "; ".join(
                "{}={}".format(expected_label(item, record), item["display"])
                for item in record["expected_values"])
        else:
            expected = "correct refusal: " + "; ".join(
                "{} {} {}".format(item.get("entity", ""), item.get("year", ""),
                                  item.get("metric", ""))
                for item in record["unsupported_parts"])
        if record["unsupported_parts"] and record["expected_values"]:
            expected += "; explicitly unsupported: " + "; ".join(
                "{} {} {}".format(item.get("entity", ""),
                                  item.get("year") or item.get("date", ""),
                                  item.get("metric", ""))
                for item in record["unsupported_parts"])
        if record["conflicting_evidence"]:
            expected += "; " + "; ".join(
                conflict_phrase(item) for item in record["conflicting_evidence"])
        source = ", ".join(
            "{} p{} {}".format(
                item["doc"], item["evidence_page"], item["chunk_id"])
            for item in record["source_pages"]) or "corpus scope boundary"
        question = record["question"].replace("|", "\\|")
        expected = expected.replace("|", "\\|")
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            record["id"], record["review"]["priority"], record["scenario"],
            question, record["expected_behavior"], expected, source))
    lines.extend([
        "",
        "## Review Decisions",
        "",
        "- [{}] Approve all high-priority rows".format("x" if frozen else " "),
        "- [{}] Confirm the cross-check rows are unit/rounding or caliber "
        "variants, not contradictions".format("x" if frozen else " "),
        "- [{}] Replace or edit rows that encode the wrong expected behavior"
        .format("x" if frozen else " "),
        "- [{}] Confirm 40/40 after changes".format("x" if frozen else " "),
    ])
    return "\n".join(lines) + "\n"


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        + "\n",
        encoding="utf-8",
    )


def substantive_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in record.items() if key != "review"}
        for record in records
    ]


def update_baseline_for_freeze(candidate_sha: str, frozen_sha: str,
                               frozen_at: str) -> dict[str, Any] | None:
    if not BASELINE.exists():
        return None
    report = json.loads(BASELINE.read_text(encoding="utf-8"))
    report_sha = report.get("dataset_sha256")
    if report_sha not in (candidate_sha, frozen_sha):
        raise ValueError(
            "baseline dataset hash does not match candidate/frozen dataset")
    report["dataset_candidate_sha256"] = candidate_sha
    report["dataset_sha256"] = frozen_sha
    report["dataset_frozen_at"] = frozen_at
    BASELINE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(BASELINE),
        "sha256": sha256_of(BASELINE),
        "dataset_candidate_sha256": candidate_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build/source-audit the held-out candidate dataset")
    parser.add_argument("--freeze", action="store_true",
                        help="mark the source-audited 40 rows as human-confirmed")
    args = parser.parse_args()

    raw = json.loads(INDEX_METADATA.read_text(encoding="utf-8"))
    chunks = {item["chunk_id"]: item for item in raw.values()}
    offsets = json.loads(PAGE_OFFSETS.read_text(encoding="utf-8"))
    records = [build_record(spec, chunks, offsets) for spec in SPECS]
    validate(records, chunks, offsets)

    existing_manifest = (
        json.loads(MANIFEST.read_text(encoding="utf-8"))
        if MANIFEST.exists() else {}
    )
    already_frozen = existing_manifest.get("status") == "frozen"
    if already_frozen and not args.freeze:
        raise SystemExit(
            "held-out dataset is frozen; pass --freeze to verify/rebuild it")

    candidate_sha = (
        sha256_of(OUTPUT) if OUTPUT.exists()
        else hashlib.sha256(b"").hexdigest()
    )
    if args.freeze and OUTPUT.exists():
        previous_records = [
            json.loads(line)
            for line in OUTPUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if substantive_records(previous_records) != substantive_records(records):
            raise SystemExit(
                "frozen dataset content changed; create a new dataset version "
                "instead of mutating held-out labels")
        if already_frozen:
            frozen_at = existing_manifest["frozen_at"]
            candidate_sha = existing_manifest["candidate_dataset_sha256"]
            confirmed_at = previous_records[0]["review"].get(
                "confirmed_at") or frozen_at
        else:
            frozen_at = datetime.now(timezone.utc).isoformat()
            confirmed_at = frozen_at
    else:
        frozen_at = None
        confirmed_at = None

    for record in records:
        record["review"]["source_audit"] = "passed"
        if args.freeze:
            record["review"]["human_confirmation"] = "confirmed"
            record["review"]["confirmed_at"] = confirmed_at
            record["review"]["confirmation_source"] = "project_owner"

    write_jsonl(records, OUTPUT)
    frozen_sha = sha256_of(OUTPUT)
    REVIEW.write_text(review_markdown(records), encoding="utf-8")
    counts = Counter(record["scenario"] for record in records)
    baseline_info = (
        update_baseline_for_freeze(candidate_sha, frozen_sha, frozen_at)
        if args.freeze and frozen_at else None
    )
    manifest = {
        "dataset_id": (
            "kb-agent-heldout-40" if args.freeze
            else "kb-agent-heldout-40-candidate"),
        "version": "v1-frozen" if args.freeze else "v3-candidate",
        "generated_at": (
            existing_manifest.get("generated_at")
            if already_frozen else datetime.now(timezone.utc).isoformat()
        ),
        "frozen_at": frozen_at,
        "status": "frozen" if args.freeze else "candidate_pending_user_review",
        "path": str(OUTPUT),
        "sha256": frozen_sha,
        "candidate_dataset_sha256": candidate_sha,
        "review_path": str(REVIEW),
        "dev_set": {
            "path": str(DEV_SET),
            "sha256": sha256_of(DEV_SET),
        },
        "corpus_metadata": {
            "path": str(CORPUS_METADATA),
            "sha256": sha256_of(CORPUS_METADATA),
        },
        "field_semantics": {
            "evidence_page": (
                "PDF page containing the expected value or conflict literal"
            ),
            "chunk_start_page": (
                "PDF page returned by application retrieval for the chunk start; "
                "use this for retrieval page matching"
            ),
            "unsupported_obligations": (
                "Obligations that must be explicitly refused or marked outside "
                "the corpus"
            ),
        },
        "runner": {
            "path": str(EVALS_DIR / "run_heldout_eval.py"),
            "consumes_unsupported_obligations": True,
            "retrieval_page_field": "chunk_start_page",
            "answer_binding_field": "chunk_id",
        },
        "counts": {
            "total": len(records),
            "by_scenario": dict(sorted(counts.items())),
            "high_priority_review": sum(
                record["review"]["priority"] == "high" for record in records),
            "source_audit_passed": sum(
                record["review"]["source_audit"] == "passed"
                for record in records),
            "human_confirmation_pending": (
                0 if args.freeze else len(records)),
            "human_confirmation_confirmed": (
                len(records) if args.freeze else 0),
        },
        "baseline": baseline_info,
        "freeze_policy": (
            "Frozen after human confirmation. Do not change questions, labels, "
            "or evidence in place; use a new dataset version for changes."
            if args.freeze else
            "Do not tune retrieval, prompts, or routing rules against this set "
            "before human confirmation and hash freeze."
        ),
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT),
        "manifest": str(MANIFEST),
        "review": str(REVIEW),
        "sha256": manifest["sha256"],
        "counts": manifest["counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
