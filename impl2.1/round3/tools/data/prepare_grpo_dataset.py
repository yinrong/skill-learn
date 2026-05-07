"""将 SFT 格式数据集转换为 GRPO 格式。

GRPO 训练需要的字段：
  - system, instruction, input（prompt 部分）
  - ground_truth（奖励函数读取）

与 SFT 数据集的区别：不需要 output 字段（由模型生成），
但需要 ground_truth 字段供 reward_func 使用。

用法：
    python round3/tools/data/prepare_grpo_dataset.py \\
        --input round3/data/train_R3-AB.jsonl \\
        --output round3/data/train_R3-grpo.jsonl
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
# (no common imports needed in this file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    count = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for line in open(args.input, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # GRPO 格式：保留 prompt 字段，去掉 output，保留 ground_truth
            grpo_sample = {
                "system":      d.get("system", ""),
                "instruction": d.get("instruction", ""),
                "input":       d.get("input", ""),
                # output 字段留空或删除（GRPO 训练时不用）
                "output":      "",
                "ground_truth": d.get("ground_truth", {"violations": [], "cpk": None}),
            }
            fout.write(json.dumps(grpo_sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"✓ GRPO 数据集：{args.output}（{count} 条）")


if __name__ == "__main__":
    main()
