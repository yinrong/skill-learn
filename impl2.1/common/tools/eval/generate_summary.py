"""生成最终评测摘要报告 demo_summary.md（3条件：base-no-skill / base-with-skill / sft-no-skill）。

用法：
    python tools/eval/generate_summary.py \\
        --results_dir results/demo/ \\
        --output reports/demo_summary.md
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from tools.eval.scaling_curve import (
    _load_results,
    _power_law_fit,
    _extract_sft_points,
    _find_baseline,
    _saturating_pred,
)


def generate_summary(results_dir: str, output_path: str) -> str:
    results = _load_results(results_dir)
    points = _extract_sft_points(results)
    base_no  = _find_baseline(results, skill=False)
    base_with = _find_baseline(results, skill=True)

    ns   = [p[0] for p in points]
    f1s  = [p[1] for p in points]

    # 幂律拟合
    a, b, r2 = _power_law_fit(ns, f1s)
    use_sat = a > 0 and ns and (a * (100_000 ** b)) > 0.95

    def pred(n: int) -> str:
        if a <= 0 or not ns:
            return "N/A"
        if use_sat:
            val = _saturating_pred(ns, f1s, n)
            return f"{val:.3f}" if val else "N/A"
        return f"{min(a * (n ** b), 0.99):.3f}"

    base_no_f1   = base_no.get("rule_detection_f1")  if base_no   else None
    base_with_f1 = base_with.get("rule_detection_f1") if base_with else None
    base_no_ms   = base_no.get("inference_time_ms_mean")  if base_no   else None
    base_with_ms = base_with.get("inference_time_ms_mean") if base_with else None
    skill_tokens = base_with.get("skill_tokens", 0) if base_with else 0

    # 3-condition table rows (base rows + sft rows)
    cond_rows: list[dict] = []
    if base_no:
        cond_rows.append({
            "label": "base-no-skill",
            "n_train": 0,
            "skill": "No",
            "f1": base_no_f1,
            "cpk_mae": base_no.get("cpk_mae"),
            "inf_ms": base_no_ms,
            "train_s": 0,
        })
    if base_with:
        cond_rows.append({
            "label": "base-with-skill",
            "n_train": 0,
            "skill": "Yes",
            "f1": base_with_f1,
            "cpk_mae": base_with.get("cpk_mae"),
            "inf_ms": base_with_ms,
            "train_s": 0,
        })
    _node_labels = {100: "sft-N1", 200: "sft-N2", 300: "sft-N3", 500: "sft-N4"}
    for p in points:
        n_train = p[0]
        f1_val = p[1]
        # find matching result for this node
        node_summary = None
        for name, s in results.items():
            from tools.eval.scaling_curve import _NODE_SAMPLES
            node_id = next((k for k in _NODE_SAMPLES if name.upper().endswith(k)), None)
            if node_id and _NODE_SAMPLES[node_id] == n_train and "sft" in name:
                node_summary = s
                break
        cond_rows.append({
            "label": _node_labels.get(n_train, f"sft-N-{n_train}"),
            "n_train": n_train,
            "skill": "No",
            "f1": f1_val,
            "cpk_mae": node_summary.get("cpk_mae") if node_summary else None,
            "inf_ms": node_summary.get("inference_time_ms_mean") if node_summary else None,
            "train_s": node_summary.get("train_time_s") if node_summary else None,
        })

    def _fmt(v, fmt=".3f"):
        return f"{v:{fmt}}" if v is not None else "—"

    # Skill internalization check
    n4_f1 = f1s[-1] if f1s else None
    internalization_ratio = (n4_f1 / base_with_f1) if (n4_f1 and base_with_f1 and base_with_f1 > 0) else None
    if internalization_ratio is not None:
        if internalization_ratio >= 0.80:
            intern_verdict = f"**Excellent** ({internalization_ratio:.2f} ≥ 0.80) → 进入 POC 扩数据"
        elif internalization_ratio >= 0.60:
            intern_verdict = f"**Good** ({internalization_ratio:.2f} ≥ 0.60) → POC 可继续"
        else:
            intern_verdict = f"**Insufficient** ({internalization_ratio:.2f} < 0.60) → 检查数据格式/训练配置"
    else:
        intern_verdict = "N/A（缺少 base-with-skill 对照数据）"

    # Go/No-Go checks
    checks = [
        ("链路打通",  "4个节点均完成训练+评测",              len(ns) >= 4),
        ("内化效果",  f"N4 sft-no-skill ≥ base-with-skill×0.70",
         (internalization_ratio >= 0.70 if internalization_ratio else False)),
        ("正向信号",  f"N4 f1 > base-no-skill + 0.10",
         (n4_f1 is not None and base_no_f1 is not None and n4_f1 > base_no_f1 + 0.10)),
        ("曲线可用",  f"R²={r2:.3f} (>0.60)",               r2 > 0.60),
        ("成本节省",  f"Skill tokens > 500",                 skill_tokens > 500),
    ]
    all_pass = all(c[2] for c in checks)
    go_nogo = "**Go ✓**" if all_pass else "**No-Go ✗**"

    # 最弱规则（N4 SFT）
    n4_result = next(
        (s for name, s in results.items()
         if name.upper().endswith("N4") and "sft" in name and "baseline" not in name),
        None,
    )
    weak_rules_text = "N/A"
    if n4_result:
        pr = n4_result.get("per_rule_recall", {})
        sorted_rules = sorted([(k, v) for k, v in pr.items() if v is not None], key=lambda x: x[1])
        if sorted_rules:
            weak_rules_text = ", ".join(f"{r}({v:.2f})" for r, v in sorted_rules[:3])

    # Edge saturation
    ratio_val = None
    if len(ns) >= 3:
        g_first = (f1s[1] - f1s[0]) / (ns[1] - ns[0]) if (ns[1] - ns[0]) > 0 else 0
        g_last  = (f1s[-1] - f1s[-2]) / (ns[-1] - ns[-2]) if (ns[-1] - ns[-2]) > 0 else 0
        ratio_val = g_last / g_first if g_first > 0 else None
    if ratio_val is not None:
        if ratio_val > 0.60:
            curve_advice = f"Still rising fast (ratio={ratio_val:.2f}) — raise POC data target"
        elif ratio_val >= 0.30:
            curve_advice = f"Slowing but positive (ratio={ratio_val:.2f}) — proceed to POC as planned"
        else:
            curve_advice = f"Near saturation (ratio={ratio_val:.2f}) — shift POC focus to DPO/RL"
    else:
        curve_advice = "Insufficient data to assess saturation"

    # ── Build Markdown ────────────────────────────────────────────────────────
    lines = [
        "# Demo 阶段评测摘要（Skill 内化实验）",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> 实验模型：Qwen3-8B  场景：SPC Nelson 规则检测 Skill 内化",
        "",
        "## 三条件指标对比",
        "",
        "| 条件 | 训练集 | Skill | rule_f1 | cpk_mae | 推理时间(ms) | 训练时长 |",
        "|------|--------|-------|---------|---------|------------|--------|",
    ]
    for r in cond_rows:
        bold_start = "**" if r["label"].startswith("sft-N4") else ""
        bold_end   = "**" if r["label"].startswith("sft-N4") else ""
        f1_str = f"{bold_start}{_fmt(r['f1'])}{bold_end}" if r['f1'] is not None else "—"
        train_str = f"{r['train_s']:.0f}s" if r.get("train_s") else "—"
        ms_str = f"{r['inf_ms']:.0f}" if r.get("inf_ms") else "—"
        lines.append(
            f"| {r['label']} | {r['n_train']} | {r['skill']} "
            f"| {f1_str} | {_fmt(r['cpk_mae'])} | {ms_str} | {train_str} |"
        )

    lines += [
        "",
        "## Skill 内化效果",
        "",
    ]
    if base_no_f1 is not None and base_with_f1 is not None and n4_f1 is not None:
        token_saving_pct = int((1 - 250 / (250 + skill_tokens)) * 100) if skill_tokens > 0 else 0
        lines += [
            f"- **sft-no-skill N4** rule_f1 = {n4_f1:.3f}  ",
            f"- **base-with-skill** rule_f1 = {base_with_f1:.3f} (上界参考)  ",
            f"- **base-no-skill**   rule_f1 = {base_no_f1:.3f} (当前不带 Skill 的效果)",
            "",
            f"内化成功率：{intern_verdict}  ",
            f"Token 节省：{skill_tokens} tokens/req → 约 {token_saving_pct}% 节省  ",
        ]
    else:
        lines.append(f"内化效果：{intern_verdict}")

    lines += [
        "",
        "## 效果预测（幂律拟合，基于 sft-no-skill 节点）",
        "",
        f"拟合模型：rule_f1 = {a:.4f} × N^{b:.4f}  （R²={r2:.3f}）",
        "",
        "| 目标数据量 | 预测 rule_f1 |",
        "|-----------|------------|",
    ]
    for target in [1_000, 5_000, 10_000, 100_000]:
        lines.append(f"| {target:,} | {pred(target)} |")

    lines += [
        "",
        f"**曲线趋势**：{curve_advice}",
        "",
        "## 最弱规则（N4 sft-no-skill 快照）",
        "",
        f"Recall 最低 Top3：{weak_rules_text}",
        "",
        "> POC 阶段需针对这些规则重点增补黄金样本。",
        "",
        "## 验收检验",
        "",
        "| 验收项 | 要求 | 结果 |",
        "|--------|------|------|",
    ]
    for name, detail, passed in checks:
        icon = "✓ 通过" if passed else "✗ 未通过"
        lines.append(f"| {name} | {detail} | {icon} |")

    lines += [
        "",
        f"## 结论：{go_nogo}",
        "",
    ]
    if all_pass:
        lines.append("全部验收项通过，建议进入 **route2.2 POC 阶段**。")
    else:
        failed = [c[0] for c in checks if not c[2]]
        lines.append(f"以下验收项未通过，需修复后重新评估：{', '.join(failed)}。")

    md = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✓ 摘要已写入：{output_path}")
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Demo 阶段评测摘要")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default="reports/demo_summary.md")
    args = parser.parse_args()
    generate_summary(args.results_dir, args.output)


if __name__ == "__main__":
    main()
