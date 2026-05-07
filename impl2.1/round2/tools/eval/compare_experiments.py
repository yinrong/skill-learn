"""实验结果对比分析脚本。

用法：
    python tools/eval/compare_experiments.py \
        --results_dir history-route2.1.1/results/ \
        --output history-route2.1.1/reports/experiment_comparison.md
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def load_result(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("summary", data)
    except Exception as e:
        print(f"  ⚠ 加载失败 {path}: {e}")
        return None


def format_rule_bar(recall_dict: dict) -> str:
    rules = [f"rule{i}" for i in range(1, 9)]
    bars = []
    for rule in rules:
        val = recall_dict.get(rule)
        if val is None:
            bars.append(f"{rule}:N/A")
        else:
            mark = "✓" if val > 0.2 else ("△" if val > 0 else "✗")
            bars.append(f"{rule}:{val:.2f}{mark}")
    return "  ".join(bars)


def analyze_results(results_dir: str, output_path: str | None = None) -> None:
    results_dir = Path(results_dir)
    result_files = sorted(results_dir.glob("*.json"))

    if not result_files:
        print(f"没有找到结果文件：{results_dir}")
        return

    # 读取所有结果
    rows = []
    for path in result_files:
        s = load_result(path)
        if s is None:
            continue
        rows.append({
            "name": path.stem,
            "f1": s.get("rule_detection_f1", 0),
            "cpk_found": s.get("cpk_found_rate", 0),
            "n_train": s.get("n_train_samples", "?"),
            "per_rule": s.get("per_rule_recall", {}),
            "inf_ms": s.get("inference_time_ms_mean", 0),
        })

    # 排序
    rows.sort(key=lambda r: r["f1"], reverse=True)

    # 计算规则覆盖（recall>0 的规则数）
    for r in rows:
        covered = sum(1 for v in r["per_rule"].values() if v is not None and v > 0)
        min_recall = min((v for v in r["per_rule"].values() if v is not None), default=0)
        r["rules_covered"] = covered
        r["min_rule_recall"] = min_recall

    lines = []
    lines.append("# 实验结果对比报告\n")
    lines.append(f"生成时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    # Summary table
    lines.append("## 汇总表（按 rule_f1 降序）\n")
    lines.append("| 实验 | rule_f1 | CPK_found | rules_covered | min_recall | inf_ms | n_train |")
    lines.append("|------|---------|-----------|---------------|------------|--------|---------|")
    for r in rows:
        lines.append(
            f"| {r['name']} | **{r['f1']:.3f}** | {r['cpk_found']:.2f} | "
            f"{r['rules_covered']}/8 | {r['min_rule_recall']:.2f} | "
            f"{r['inf_ms']:.0f}ms | {r['n_train']} |"
        )

    # Per-rule detail
    lines.append("\n## Per-Rule Recall 详情\n")
    lines.append("| 实验 | rule1 | rule2 | rule3 | rule4 | rule5 | rule6 | rule7 | rule8 |")
    lines.append("|------|-------|-------|-------|-------|-------|-------|-------|-------|")
    for r in rows:
        pr = r["per_rule"]
        cells = []
        for i in range(1, 9):
            v = pr.get(f"rule{i}")
            if v is None:
                cells.append("N/A")
            elif v >= 0.5:
                cells.append(f"**{v:.2f}**")
            elif v > 0:
                cells.append(f"{v:.2f}")
            else:
                cells.append("0.00")
        lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")

    # Analysis
    lines.append("\n## 关键发现\n")
    if len(rows) >= 2:
        best = rows[0]
        baseline = next((r for r in rows if "expA" in r["name"]), rows[-1])
        lines.append(f"- **最佳实验**：{best['name']}（F1={best['f1']:.3f}）")
        lines.append(f"- **基线实验**：{baseline['name']}（F1={baseline['f1']:.3f}）")
        delta = best['f1'] - baseline['f1']
        lines.append(f"- **最大提升**：+{delta:.3f}（相对基线）")

        # Find if balanced weights helped
        expA = next((r for r in rows if r["name"] == "expA"), None)
        expB = next((r for r in rows if r["name"] == "expB"), None)
        if expA and expB:
            delta_ab = expB["f1"] - expA["f1"]
            lines.append(
                f"- **均衡权重效果（Exp-B vs A）**：F1 {'+' if delta_ab >= 0 else ''}{delta_ab:.3f}，"
                f"规则覆盖 {expB['rules_covered']} vs {expA['rules_covered']} /8"
            )

        # Find 8B vs 14B
        expE = next((r for r in rows if r["name"] == "expE"), None)
        if expB and expE:
            delta_e = expE["f1"] - expB["f1"]
            lines.append(
                f"- **8B vs 14B（Exp-E vs B）**：F1 {'+' if delta_e >= 0 else ''}{delta_e:.3f}"
            )

    lines.append("\n## 后续方向建议\n")
    # Suggest next experiments based on results
    best_f1 = rows[0]["f1"] if rows else 0
    best_covered = rows[0]["rules_covered"] if rows else 0
    if best_f1 < 0.15:
        lines.append("- ⚠ 最佳 F1 < 0.15，建议：")
        lines.append("  - 增加训练数据量（N=400+）")
        lines.append("  - 尝试 Claude 教师数据（E2）")
        lines.append("  - 检查训练是否正常完成")
    elif best_f1 < 0.20:
        lines.append("- F1 在 0.15~0.20 区间，建议：")
        lines.append("  - 执行 E2（Claude 教师数据 500条）")
        lines.append("  - 混合最优实验数据 + Claude 教师数据（E3）")
        lines.append("  - 尝试 32B 基座（若显存允许）")
    else:
        lines.append("- F1 >= 0.20，建议：")
        lines.append("  - 执行 GRPO 强化训练（E4）")
        lines.append("  - 扩大数据量至 1000-2000 条")
        lines.append("  - 测试 Qwen3-32B 基座的上界")

    if best_covered < 8:
        lines.append(f"- ⚠ 最佳模型只覆盖 {best_covered}/8 条规则，建议增加规则均衡性")

    report = "\n".join(lines)
    print(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ 报告已写入：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="实验结果对比分析")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default=None, help="输出 Markdown 文件路径")
    args = parser.parse_args()
    analyze_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()
