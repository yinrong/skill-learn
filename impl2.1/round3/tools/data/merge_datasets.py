"""合并多个 JSONL 数据集。

用法：
    python round3/tools/data/merge_datasets.py \\
        --inputs data1.jsonl data2.jsonl data3.jsonl \\
        --output merged.jsonl \\
        --shuffle --seed 42
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_samples: list[dict] = []
    for path in args.inputs:
        count = 0
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))
                count += 1
        print(f"  {path}: {count} 条")

    if args.shuffle:
        random.Random(args.seed).shuffle(all_samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✓ 合并完成：{args.output}（{len(all_samples)} 条，shuffle={args.shuffle}）")
    for path in args.inputs:
        print(f"  来源：{path}")


if __name__ == "__main__":
    main()
