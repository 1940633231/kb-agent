"""Answer prompt tuned for auditable annual-report responses."""
from __future__ import annotations

from app.generation.prompt import PromptBuilder


ANSWER_RULES = """
数值回答补充规则（与上文同等约束）：
- 问题要求具体指标时，答案必须逐字写出题目中的指标名称，不能只写“收入”“实现收入”等泛称。
- 如果原文使用等价口径（例如题目问“营业收入”、原文写“营业总收入”），答案写成
  “营业收入（原文口径：营业总收入）”，并保留原文单位。
- 每个数值必须保留原文数值和原文单位；如需换算，同时给出原文值与换算值，不能只给
  四舍五入后的概数。
- 同一指标同时出现精确表格值和正文概数时，必须优先引用精确表格值。正文中的
  “约”“XX亿元”等概数不能替代精确值，也不能拿概数换算成题目要求的精确单位。
- 多公司、多年份或多指标题必须逐项回答，每项都包含公司、年份、指标、数值、单位和
  引用编号，不能合并、省略或只回答其中一部分。
- 引用必须指向包含该指标和数值的原表或原句，不要引用封面、目录、审计意见或无关
  财务项目。
- 如果上下文只有利润、费用、资产等相关分项，没有题目要求的指标原表或原句，应明确
  说明无法回答，不能自行推导或把相关分项当作题目指标。
"""


class GroundedAnswerPromptBuilder(PromptBuilder):
    """Extend the upstream prompt while keeping its history sanitization."""

    SYSTEM_PROMPT = PromptBuilder.SYSTEM_PROMPT + ANSWER_RULES
