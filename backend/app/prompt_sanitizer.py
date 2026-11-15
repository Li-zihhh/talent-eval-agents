"""Prompt 脱敏模块 - 在将 Prompt 发送给外部模型前替换其中的个人身份信息 (PII)。

支持检测和替换三类敏感信息：
    - 标记姓名（如"候选人姓名：张三"中的姓名部分）
    - 邮箱地址
    - 身份证号码
    - 手机号码

替换后的占位符格式：__PII_{TYPE}_{序号}__，保证同一值替换为相同占位符。
"""

from __future__ import annotations

from collections import Counter
import re

# 匹配标注的候选人姓名 / 员工姓名模式
LABELED_NAME = re.compile(
    r"(?P<prefix>(?:候选人姓名|员工姓名|姓名)[：:\s]+)"
    r"(?P<value>[\u4e00-\u9fff·]{2,20})(?=[，,。；;\s]|$)"
)

# 敏感信息检测器列表：类型名 -> 匹配正则
DETECTORS = (
    ("email", re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")),
    ("id_card", re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")),
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
)


def sanitize_prompt(original_prompt: str) -> str:
    """返回替换了常见 PII 后的 Prompt 副本。

    同一原始值在整个 Prompt 中会被替换为相同的占位符，
    便于后续在模型输出中还原。

    Args:
        original_prompt: 原始 Prompt 文本。

    Returns:
        脱敏后的 Prompt 文本。
    """
    indexes: Counter[str] = Counter()
    placeholders: dict[tuple[str, str], str] = {}  # (类型, 原始值) -> 占位符

    def replace(pii_type: str, value: str) -> str:
        """为给定的 PII 类型和值生成或返回已有的占位符。"""
        key = (pii_type, value)
        if key not in placeholders:
            indexes[pii_type] += 1
            placeholders[key] = f"__PII_{pii_type.upper()}_{indexes[pii_type]:03d}__"
        return placeholders[key]

    result = original_prompt
    # 先替换标注的姓名
    names = [match.group("value") for match in LABELED_NAME.finditer(result)]
    for name in dict.fromkeys(names):
        result = result.replace(name, replace("person", name))
    # 再替换邮箱、身份证、手机号
    for pii_type, pattern in DETECTORS:
        result = pattern.sub(
            lambda match, kind=pii_type: replace(kind, match.group(0)),
            result,
        )
    return result