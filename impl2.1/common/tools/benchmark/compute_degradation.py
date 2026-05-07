#!/usr/bin/env python3
"""
计算 SFT 模型相对基座的通用能力退化率，并输出汇总表格。

用法：
    python compute_degradation.py --results_dir <path>

输出：
    <results_dir>/degradation_summary.json
    <results_dir>/degradation_summary.md
"""

import argparse
import json
import os
from pathlib import Path


# 每个任务的主指标键名（lm-eval 输出格式），列表按优先级排列
TASK_METRICS: dict[str, list[str]] = {
    "mmlu":           ["acc,none"],
    "gsm8k":          ["exact_match,flexible-extract", "exact_match,strict-match"],
    "arc_challenge":  ["acc_norm,none", "acc,none"],
    "hellaswag":      ["acc_norm,none", "acc,none"],
    "winogrande":     ["acc,none"],
    "truthfulqa_mc1": ["acc,none"],
    "cmmlu":          ["acc,none"],
}

# 模型对：(SFT tag, base tag)
MODEL_PAIRS = [
    ("sft-8B-expLLL",   "base-8B"),
    ("sft-14B-expYYY",  "base-14B"),
    ("sft-32B-expHHH",  "base-32B"),
]


def load_results(results_dir: Path, tag: str) -> dict[str, float]:
    """从 lm-eval 输出目录读取所有任务的分数，返回 {task: score}。"""
    scores: dict[str, list[float]] = {}
    for subfolder in ["standard", "cn"]:
        folder = results_dir / tag / subfolder
        if not folder.exists():
            continue
        # lm-eval 在 output_path 下再嵌套一层以模型路径命名的目录
        for fname in sorted(folder.rglob("results*.json")):
            try:
                data = json.loads(fname.read_text())
                for task, task_results in data.get("results", {}).items():
                    task_base = task.split(",")[0]  # strip subtask suffix
                    wanted = TASK_METRICS.get(task_base)
                    if wanted is None:
                        continue
                    # 按优先级取第一个有值的指标
                    for metric_key in wanted:
                        score = task_results.get(metric_key)
                        if isinstance(score, (int, float)):
                            scores.setdefault(task_base, []).append(score)
                            break
            except Exception as e:
                print(f"  [warn] could not parse {fname}: {e}")
    # 对多子任务取均值（mmlu 有57个子任务，取整体 acc,none 即可）
    # 但 mmlu 的顶层 "mmlu" key 已包含整体分，不需要子任务均值
    # 只保留 task_base 在 TASK_METRICS 中的顶层任务
    top_level = {t: vals for t, vals in scores.items() if t in TASK_METRICS}
    return {task: sum(vals) / len(vals) for task, vals in top_level.items()}


def compute_degradation(base_score: float, sft_score: float) -> float:
    """退化率 = (base - sft) / base，若 sft > base 则为负（提升）。"""
    if base_score == 0:
        return 0.0
    return (base_score - sft_score) / base_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir
    summary = {}
    tasks_seen = set()

    for sft_tag, base_tag in MODEL_PAIRS:
        base_scores = load_results(results_dir, base_tag)
        sft_scores  = load_results(results_dir, sft_tag)

        if not base_scores or not sft_scores:
            print(f"[skip] missing results for {sft_tag} or {base_tag}")
            continue

        tasks = sorted(set(base_scores) & set(sft_scores))
        tasks_seen.update(tasks)

        pair_summary = {}
        degradations = []
        for task in tasks:
            b = base_scores[task]
            s = sft_scores[task]
            d = compute_degradation(b, s)
            pair_summary[task] = {"base": b, "sft": s, "degradation": d}
            degradations.append(d)

        avg_deg = sum(degradations) / len(degradations) if degradations else 0.0
        pair_summary["_avg_degradation"] = avg_deg

        summary[sft_tag] = {
            "base_tag": base_tag,
            "scores": pair_summary,
        }

        print(f"\n{'='*60}")
        print(f"  {sft_tag}  vs  {base_tag}")
        print(f"{'='*60}")
        print(f"  {'Task':<20} {'Base':>8} {'SFT':>8} {'Degr%':>8}")
        print(f"  {'-'*44}")
        for task in tasks:
            b = pair_summary[task]["base"]
            s = pair_summary[task]["sft"]
            d = pair_summary[task]["degradation"]
            print(f"  {task:<20} {b:>8.3f} {s:>8.3f} {d*100:>+8.2f}%")
        print(f"  {'Average degradation':<20} {'':<8} {'':<8} {avg_deg*100:>+8.2f}%")

    # 保存 JSON
    out_json = results_dir / "degradation_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {out_json}")

    # 生成 Markdown 表格
    _write_markdown(results_dir, summary, sorted(tasks_seen))


def _write_markdown(results_dir: Path, summary: dict, tasks: list[str]):
    lines = ["# 通用能力退化率汇总\n"]

    for sft_tag, data in summary.items():
        base_tag = data["base_tag"]
        scores = data["scores"]
        avg = scores.get("_avg_degradation", 0.0)

        lines.append(f"## {sft_tag} vs {base_tag}\n")
        lines.append(f"平均退化率：**{avg*100:+.2f}%**\n")
        lines.append("| 任务 | 基座 | SFT | 退化率 |")
        lines.append("|------|------|-----|--------|")
        for task in tasks:
            if task not in scores:
                continue
            b = scores[task]["base"]
            s = scores[task]["sft"]
            d = scores[task]["degradation"]
            lines.append(f"| {task} | {b:.3f} | {s:.3f} | {d*100:+.2f}% |")
        lines.append("")

    # 综合得分补丁提示
    lines.append("---\n")
    lines.append("> 用平均退化率更新 `docs/2.1_round2_report.md §5.4` 综合得分公式中的退化率参数。\n")

    out_md = results_dir / "degradation_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_md}")


if __name__ == "__main__":
    main()
