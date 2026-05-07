"""GRPO 奖励函数：用规则引擎评判模型输出的正确性。

供 LLaMA-Factory GRPO 训练时作为自定义 reward_model 使用。
实现标准 reward_func(completions, **kwargs) -> list[float] 接口。
"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from tools.eval.extractor import extract_violations, extract_cpk


def compute_f1(pred: list[str], gt: list[str]) -> float:
    pred_set = set(pred)
    gt_set = set(gt)
    if not pred_set and not gt_set:
        return 1.0
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def reward_func(completions: list[str], ground_truths: list[dict], **kwargs) -> list[float]:
    """标准 GRPO 奖励函数接口。

    Args:
        completions:   模型生成的文本列表（每个 prompt 对应 G 个生成）
        ground_truths: 对应的 ground truth（来自数据集的 ground_truth 字段）

    Returns:
        rewards: 每个生成的奖励分数（0.0 ~ 1.2）

    奖励设计：
        - rule_f1 (0~1.0):  规则检测 F1
        - cpk_bonus (0~0.1): |pred_cpk - gt_cpk| < 0.1 则加 0.1
        - format_bonus (0~0.1): 输出中含 "rule" 英文标识符则加 0.05
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        gt_violations = gt.get("violations", [])
        gt_cpk = gt.get("cpk")

        pred_violations = extract_violations(completion)
        pred_cpk = extract_cpk(completion)

        # 主奖励：规则 F1
        rule_f1 = compute_f1(pred_violations, gt_violations)

        # CPK 精度奖励
        cpk_bonus = 0.0
        if pred_cpk is not None and gt_cpk is not None:
            cpk_bonus = 0.1 if abs(pred_cpk - gt_cpk) < 0.1 else 0.05

        # 格式奖励：使用英文 rule 标识符
        import re
        format_bonus = 0.05 if re.search(r'\brule[1-8]\b', completion, re.IGNORECASE) else 0.0

        total = rule_f1 + cpk_bonus + format_bonus
        rewards.append(round(total, 4))

    return rewards


if __name__ == "__main__":
    # 快速测试
    completions = [
        "rule1 触发，点位超出控制限。CPK=1.23",
        "第1条触发，没有CPK",
        "过程受控，无异常。CPK=0.95",
    ]
    gts = [
        {"violations": ["rule1"], "cpk": 1.25},
        {"violations": ["rule1"], "cpk": 0.90},
        {"violations": [], "cpk": 0.92},
    ]
    rewards = reward_func(completions, gts)
    for i, (c, r) in enumerate(zip(completions, rewards)):
        print(f"[{i}] reward={r:.4f}  '{c[:40]}...'")
