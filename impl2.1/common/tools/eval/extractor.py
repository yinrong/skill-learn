"""从模型文本输出中提取 violations 列表和 CPK 数值。"""
from __future__ import annotations
import re
from typing import Optional

# 匹配 rule1~rule8（大小写均可，允许 rule1 或 rule 1 两种写法）
_RULE_PATTERN = re.compile(r'\brule\s*([1-8])\b', re.IGNORECASE)

# 中文映射（汉字/圆圈数字 → 阿拉伯数字）
_CHINESE_NUM = {'一': '1', '二': '2', '三': '3', '四': '4',
                '五': '5', '六': '6', '七': '7', '八': '8',
                # Unicode 圆圈数字 ①~⑧ (U+2460~U+2467)
                '①': '1', '②': '2', '③': '3', '④': '4',
                '⑤': '5', '⑥': '6', '⑦': '7', '⑧': '8'}

# "第X条" 格式（第1条 / 第一条）
_CHINESE_RULE_PATTERN = re.compile(r'第\s*([1-8一二三四五六七八①②③④⑤⑥⑦⑧])\s*条')
# "规则X" 格式（规则1 / 规则② / 规则三）—— 圆圈数字/汉字均支持
_GUIZE_PATTERN = re.compile(r'规则\s*([1-8一二三四五六七八①②③④⑤⑥⑦⑧])')

# CPK 提取：匹配 CPK=1.234 或 CPK：1.234 或 cpk 为 1.234 等
_CPK_PATTERNS = [
    re.compile(r'CPK\s*[=：:]\s*([+-]?\d+\.?\d*)', re.IGNORECASE),
    re.compile(r'cpk\s+为\s*([+-]?\d+\.?\d*)', re.IGNORECASE),
    re.compile(r'min\([^)]+\)\s*[=：:]\s*([+-]?\d+\.?\d*)'),
    # 匹配 "CPK=min(1.049,1.672)=1.049" 中的最后一个数
    re.compile(r'CPK=min\([^)]+\)=([+-]?\d+\.?\d*)', re.IGNORECASE),
]

# 处置建议废话模式（用于质量关卡过滤）
_BAD_DISPOSAL_PATTERNS = [
    re.compile(r'建议(进行)?过程能力改善', re.IGNORECASE),
    re.compile(r'escalate to engineer', re.IGNORECASE),
    re.compile(r'参考.{0,10}SPC.{0,10}(控制)?程序'),
    re.compile(r'^(检查|核查)设备$'),          # 过于笼统，无具体对象
    re.compile(r'建议评估'),
]


def extract_violations(text: str) -> list[str]:
    """从模型输出文本中提取触发的 rule 列表。

    只提取 think 块之外（正文部分）的结论性违规——若无 think 块则扫描全文。
    返回去重后按 rule1~rule8 排序的列表。
    """
    # 优先扫描 </think> 之后的正文
    think_end = text.find("</think>")
    scan_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    triggered = set()
    # Scan line-by-line: skip lines that indicate "not triggered" (未触发/正常/❌)
    for line in scan_text.splitlines():
        # Skip lines that clearly say a rule was NOT triggered
        # "未触发 Rule 4", "❌ Rule 4", "rule4: 正常", "rule4 — 未触发" etc.
        not_triggered = bool(re.search(r'(未触发|✅\s*正常|❌\s*(?!.*触发))', line))
        if not_triggered:
            continue
        for m in _RULE_PATTERN.finditer(line):
            triggered.add(f"rule{m.group(1)}")
        # Also capture Chinese "第X条" and "规则X" formats
        for pat in (_CHINESE_RULE_PATTERN, _GUIZE_PATTERN):
            for m in pat.finditer(line):
                n = m.group(1)
                n = _CHINESE_NUM.get(n, n)  # convert to digit
                if n in '12345678':
                    triggered.add(f"rule{n}")
    return sorted(triggered, key=lambda r: int(r[4:]))


def extract_violations_from_think(text: str) -> list[str]:
    """从 think 块内提取触发的 rule（用于调试对比）。"""
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    if think_start < 0 or think_end < 0:
        return []
    think_text = text[think_start + 7:think_end]

    # 在 think 块内查找"触发"关键词附近的 rule
    triggered = []
    for line in think_text.splitlines():
        if "触发" in line:
            m = _RULE_PATTERN.search(line)
            if m:
                triggered.append(f"rule{m.group(1)}")
    return sorted(set(triggered), key=lambda r: int(r[4:]))


def extract_cpk(text: str) -> Optional[float]:
    """从文本中提取 CPK 数值，返回 None 表示未找到。"""
    # 优先扫描 </think> 之后的正文
    think_end = text.find("</think>")
    scan_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    for pattern in _CPK_PATTERNS:
        m = pattern.search(scan_text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue

    # 若正文找不到，尝试全文
    for pattern in _CPK_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue

    return None


def has_reasoning_chain(text: str) -> bool:
    """判断输出是否包含 <think> 推理链。"""
    return "<think>" in text and "</think>" in text


def check_disposal_quality(text: str) -> tuple[bool, list[str]]:
    """检查处置建议是否含有废话，返回 (is_clean, bad_matches)。"""
    # 提取 </think> 之后的正文
    think_end = text.find("</think>")
    scan_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    bad = []
    for pat in _BAD_DISPOSAL_PATTERNS:
        m = pat.search(scan_text)
        if m:
            bad.append(m.group(0))
    return len(bad) == 0, bad
