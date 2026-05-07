"""Phase 2 质量关卡：验证训练集 ground_truth 正确性 + 处置建议质量。

用法：
    python tools/eval/quality_gate.py --file data/demo/train_N4.jsonl --n 50
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from tools.spc.rules import check_nelson_rules, compute_cpk
from tools.eval.extractor import check_disposal_quality, has_reasoning_chain


def _parse_input(input_text: str) -> tuple[list[float], float, float, float]:
    """从 input 字段解析 data、usl、lsl、cl。"""
    import re
    # 解析数据列表
    data_match = re.search(r'\[([^\]]+)\]', input_text)
    data = [float(x.strip()) for x in data_match.group(1).split(',')]

    usl = float(re.search(r'USL=([+-]?\d+\.?\d*)', input_text).group(1))
    lsl = float(re.search(r'LSL=([+-]?\d+\.?\d*)', input_text).group(1))
    cl  = float(re.search(r'CL=([+-]?\d+\.?\d*)', input_text).group(1))
    return data, usl, lsl, cl


def run_quality_gate(file_path: str, n: int = 50, seed: int = 0) -> float:
    """随机抽取 n 条样本进行验证，返回通过率。"""
    with open(file_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    rng = random.Random(seed)
    subset = rng.sample(samples, min(n, len(samples)))

    passed = 0
    failed_items: list[dict] = []

    for idx, s in enumerate(subset):
        issues = []

        # 1. 解析 input
        try:
            data, usl, lsl, cl = _parse_input(s["input"])
        except Exception as e:
            issues.append(f"input解析失败：{e}")
            failed_items.append({"idx": idx, "issues": issues})
            continue

        gt = s.get("ground_truth", {})

        # 2. 规则引擎重算，与 ground_truth 对比
        sigma = 0.0
        if data:
            import statistics
            sigma = statistics.stdev(data)
        engine_violations = check_nelson_rules(data, cl, sigma)
        gt_violations = set(gt.get("violations", []))
        eng_set = set(engine_violations)
        if gt_violations != eng_set:
            issues.append(
                f"violations 不一致：gt={sorted(gt_violations)} vs engine={sorted(eng_set)}"
            )

        # 3. CPK 对比（允许 ±0.05 误差）
        gt_cpk = gt.get("cpk")
        eng_cpk = compute_cpk(data, usl, lsl)
        if gt_cpk is not None and eng_cpk is not None:
            if abs(gt_cpk - eng_cpk) > 0.05:
                issues.append(f"CPK 偏差：gt={gt_cpk} vs engine={eng_cpk:.3f}")

        # 4. output 字段有推理链
        output = s.get("output", "")
        if not has_reasoning_chain(output):
            issues.append("output 缺少 <think> 推理链")

        # 5. 处置建议质量
        disposal_ok, bad_items = check_disposal_quality(output)
        if not disposal_ok:
            issues.append(f"处置建议含废话：{bad_items[:2]}")

        if not issues:
            passed += 1
        else:
            failed_items.append({"idx": idx, "issues": issues})

    pass_rate = passed / len(subset)

    print(f"\n=== Phase 2 质量关卡 ===")
    print(f"抽检文件：{file_path}")
    print(f"抽检数量：{len(subset)}")
    print(f"通过数量：{passed}")
    print(f"通过率：  {pass_rate * 100:.1f}%")

    if failed_items:
        print(f"\n失败样本（前5条）：")
        for item in failed_items[:5]:
            print(f"  样本#{item['idx']}：{'; '.join(item['issues'])}")

    threshold = 0.80
    if pass_rate >= threshold:
        print(f"\n✓ 通过质量关卡（{pass_rate * 100:.1f}% ≥ {threshold * 100:.0f}%），可继续 Phase 3")
    else:
        print(f"\n✗ 未通过质量关卡（{pass_rate * 100:.1f}% < {threshold * 100:.0f}%）")
        print(f"  请修复 generator.py 并重新执行 Phase 1")
        sys.exit(1)

    return pass_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 质量关卡验证")
    parser.add_argument("--file", required=True, help="训练集 JSONL 路径")
    parser.add_argument("--n", type=int, default=50, help="抽检样本数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_quality_gate(args.file, args.n, args.seed)


if __name__ == "__main__":
    main()
