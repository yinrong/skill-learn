"""混合多个 JSONL 数据集，支持比例控制和随机打乱。

用法：
    python tools/data/mix_dataset.py \\
        --inputs data/demo/train_claude_teacher.jsonl:1.0 \\
                 data/demo/train_N2_v1.jsonl:0.4 \\
        --output data/demo/train_mix_v1.jsonl \\
        --seed 42
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path


def mix_datasets(
    inputs: list[tuple[str, float]],   # [(path, weight), ...]
    output_path: str,
    seed: int = 42,
    max_total: int | None = None,
) -> None:
    """按权重混合多个 JSONL 文件并打乱。

    weights 是相对比例（不需要归一），最终样本数 = 各文件按权重缩放后的数量之和。
    若某文件样本不足，则全部使用（不过采样）。
    """
    rng = random.Random(seed)

    # 读取所有文件
    all_samples_by_source = []
    for path, weight in inputs:
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠ 解析失败（{path}）：{e}")
        all_samples_by_source.append((path, weight, samples))
        print(f"  读取 {len(samples)} 条 from {path}（权重={weight}）")

    if not all_samples_by_source:
        raise ValueError("没有可用的输入文件")

    # 确定目标数量：以最小 weight=1.0 的文件数量为基准
    max_weight = max(w for _, w, _ in all_samples_by_source)
    base_count = None
    for path, weight, samples in all_samples_by_source:
        if abs(weight - max_weight) < 1e-9:
            base_count = len(samples)
            break
    assert base_count is not None

    # 按权重采样
    mixed = []
    for path, weight, samples in all_samples_by_source:
        target = int(base_count * weight / max_weight)
        target = min(target, len(samples))
        chosen = rng.sample(samples, target)
        mixed.extend(chosen)
        print(f"  选取 {target}/{len(samples)} 条 from {Path(path).name}")

    if max_total and len(mixed) > max_total:
        mixed = rng.sample(mixed, max_total)
        print(f"  截断到 {max_total} 条")

    rng.shuffle(mixed)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in mixed:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✓ 混合完成：{len(mixed)} 条样本 → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="混合多个 JSONL 数据集")
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        metavar="FILE:WEIGHT",
        help="输入文件和权重，格式 file.jsonl:1.0，可多个",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_total", type=int, default=None,
                        help="混合后最大样本数（可选截断）")
    args = parser.parse_args()

    inputs = []
    for item in args.inputs:
        if ":" in item:
            parts = item.rsplit(":", 1)
            inputs.append((parts[0], float(parts[1])))
        else:
            inputs.append((item, 1.0))

    mix_datasets(inputs, args.output, seed=args.seed, max_total=args.max_total)


if __name__ == "__main__":
    main()
