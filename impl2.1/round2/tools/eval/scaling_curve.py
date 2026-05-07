"""幂律拟合 + 6子图输出（3条件：base-no-skill / base-with-skill / sft-no-skill）。

用法：
    python tools/eval/scaling_curve.py \\
        --results_dir results/demo/ \\
        --target_n 100000 \\
        --output reports/scaling_curve.png
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

# 文件名节点 → 训练集大小
_NODE_SAMPLES = {"N1": 100, "N2": 200, "N3": 300, "N4": 500}


def _load_results(results_dir: str) -> dict[str, dict]:
    """加载 results_dir 下所有 JSON 文件，返回 {filename_stem: summary_dict}。"""
    results = {}
    for p in sorted(Path(results_dir).glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            summary = data.get("summary", data)
            results[p.stem] = summary
        except Exception as e:
            print(f"⚠ 跳过 {p.name}：{e}")
    return results


def _find_baseline(results: dict[str, dict], skill: bool) -> Optional[dict]:
    """找到 base-no-skill 或 base-with-skill 的评测结果。"""
    condition = "with_skill" if skill else "no_skill"
    # 优先匹配 eval_condition 字段
    for name, s in results.items():
        if "baseline" in name and s.get("eval_condition") == condition:
            return s
    # 回退：文件名匹配
    for name, s in results.items():
        if "baseline" in name:
            if skill and "with_skill" in name:
                return s
            if not skill and ("no_skill" in name or "with_skill" not in name):
                return s
    return None


def _extract_sft_points(results: dict[str, dict]) -> list[tuple[int, float, float, float]]:
    """提取 SFT no-skill 节点 (n_train, rule_f1, cpk_found_rate, inf_ms)。"""
    points = []
    for name, summary in results.items():
        if "sft" not in name and "sft" not in summary.get("model", ""):
            continue
        if "baseline" in name:
            continue
        # 从文件名提取节点 N1..N4
        node_id = next((k for k in _NODE_SAMPLES if name.upper().endswith(k)), None)
        if node_id:
            n = _NODE_SAMPLES[node_id]
        else:
            n = summary.get("n_train_samples")
        if n is None:
            continue
        f1 = summary.get("rule_detection_f1")
        if f1 is None:
            continue
        cpk_rate = summary.get("cpk_found_rate") or 0.0
        inf_ms = summary.get("inference_time_ms_mean") or 0.0
        points.append((int(n), float(f1), float(cpk_rate), float(inf_ms)))

    points.sort(key=lambda x: x[0])
    return points


def _power_law_fit(ns: list[int], ys: list[float]) -> tuple[float, float, float]:
    """幂律拟合 y = a * x^b，返回 (a, b, r2)。跳过 y<=0 的点。"""
    import math
    valid = [(n, y) for n, y in zip(ns, ys) if y > 0]
    if len(valid) < 2:
        return 0.0, 0.0, 0.0
    lx = [math.log(v[0]) for v in valid]
    ly = [math.log(v[1]) for v in valid]
    n = len(lx)
    xm, ym = sum(lx) / n, sum(ly) / n
    ssxy = sum((lx[i] - xm) * (ly[i] - ym) for i in range(n))
    ssxx = sum((lx[i] - xm) ** 2 for i in range(n))
    if ssxx == 0:
        return 0.0, 0.0, 0.0
    b = ssxy / ssxx
    a = math.exp(ym - b * xm)
    ss_res = sum((ly[i] - (math.log(a) + b * lx[i])) ** 2 for i in range(n))
    ss_tot = sum((ly[i] - ym) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, r2


def _saturating_pred(ns: list[int], f1s: list[float], target_n: int, alpha: float = 0.65) -> Optional[float]:
    """饱和修正预测，当幂律外推超 0.95 时使用。"""
    import math
    if not ns or not f1s:
        return None
    n_last, f1_last = ns[-1], f1s[-1]
    if target_n <= n_last:
        return round(f1_last, 3)
    log_inc = math.log10(target_n / n_last)
    gain = (1.0 - f1_last) * (1.0 - alpha ** log_inc)
    return round(min(f1_last + gain, 0.99), 3)


def _predict_series(a: float, b: float, ns_obs: list[int], f1s_obs: list[float],
                    target_ns: list[int], use_saturation: bool) -> list[float]:
    """生成外推预测序列，保证单调。"""
    preds = []
    for tn in target_ns:
        if use_saturation:
            v = _saturating_pred(ns_obs, f1s_obs, tn)
        else:
            v = round(min(a * (tn ** b), 0.99), 3)
        preds.append(v or 0.0)
    return preds


def generate_curves(
    results_dir: str,
    output_path: str,
    target_n: int = 100_000,
    show: bool = False,
) -> dict:
    """生成 6 子图并保存为 PNG，返回拟合结果字典。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import math

    results = _load_results(results_dir)
    points = _extract_sft_points(results)
    base_no_skill = _find_baseline(results, skill=False)
    base_with_skill = _find_baseline(results, skill=True)

    ns   = [p[0] for p in points]
    f1s  = [p[1] for p in points]
    cpks = [p[2] for p in points]
    imss = [p[3] for p in points]

    base_no_f1   = base_no_skill.get("rule_detection_f1", 0.0) if base_no_skill else None
    base_with_f1 = base_with_skill.get("rule_detection_f1", 0.0) if base_with_skill else None
    base_no_ms   = base_no_skill.get("inference_time_ms_mean", 0.0) if base_no_skill else None
    base_with_ms = base_with_skill.get("inference_time_ms_mean", 0.0) if base_with_skill else None
    skill_tokens = base_with_skill.get("skill_tokens", 0) if base_with_skill else 0

    # 幂律拟合（仅用 SFT no-skill 节点）
    a, b, r2 = _power_law_fit(ns, f1s)
    use_sat = a > 0 and ns and (a * (target_n ** b)) > 0.95
    pred_at_target = None
    if a > 0 and ns:
        pred_at_target = (_saturating_pred(ns, f1s, target_n) if use_sat
                          else round(min(a * (target_n ** b), 0.99), 3))

    pred_ns = [1_000, 5_000, 10_000, 100_000]
    pred_vals = _predict_series(a, b, ns, f1s, pred_ns, use_sat) if a > 0 and ns else []

    ratio = None
    if len(ns) >= 3 and ns[-1] != ns[-2] and ns[1] != ns[0]:
        g_first = (f1s[1] - f1s[0]) / (ns[1] - ns[0]) if (ns[1] - ns[0]) > 0 else 0
        g_last  = (f1s[-1] - f1s[-2]) / (ns[-1] - ns[-2]) if (ns[-1] - ns[-2]) > 0 else 0
        ratio = g_last / g_first if g_first > 0 else None

    # ── 绘图 ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Fig1: 3-condition rule_f1 curve ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Fig1: rule_f1 (3 conditions)", fontsize=10)
    if ns and f1s:
        ax1.plot(ns, f1s, "o-b", label="sft-no-skill", zorder=3)
    if base_no_f1 is not None:
        ax1.axhline(base_no_f1, color="gray", ls=":", lw=1.5, label=f"base-no-skill ({base_no_f1:.3f})")
    if base_with_f1 is not None:
        ax1.axhline(base_with_f1, color="orange", ls="--", lw=1.5,
                    label=f"base-with-skill ({base_with_f1:.3f})")
    if pred_at_target and ns:
        ax1.annotate(f"*@{target_n//1000}K={pred_at_target}",
                     xy=(max(ns), min(pred_at_target, 0.99)),
                     xytext=(max(ns) * 0.45, min(pred_at_target, 0.99) + 0.06),
                     fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax1.set_xlabel("Training samples (N)")
    ax1.set_ylabel("rule_f1")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=7)

    # ── Fig2: CPK found rate ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Fig2: CPK Found Rate vs. N", fontsize=10)
    if ns and cpks:
        ax2.plot(ns, cpks, "o-g", label="sft-no-skill")
    if base_no_skill:
        ax2.axhline(base_no_skill.get("cpk_found_rate", 0), color="gray", ls=":", label="base-no-skill")
    if base_with_skill:
        ax2.axhline(base_with_skill.get("cpk_found_rate", 0), color="orange", ls="--", label="base-with-skill")
    ax2.set_xlabel("Training samples (N)")
    ax2.set_ylabel("CPK found rate")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7)

    # ── Fig3: per-rule recall at N4 ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Fig3: Per-Rule Recall at N4 (sft-no-skill)", fontsize=10)
    n4_result = next(
        (s for name, s in results.items() if name.upper().endswith("N4") and "sft" in name),
        None
    )
    if n4_result:
        pr = n4_result.get("per_rule_recall", {})
        rules = [f"rule{i}" for i in range(1, 9)]
        vals = [pr.get(r) or 0 for r in rules]
        min_val = min(vals) if vals else 0
        colors = ["red" if v == min_val else "steelblue" for v in vals]
        ax3.bar(rules, vals, color=colors)
        ax3.set_ylim(0, 1.05)
        ax3.tick_params(axis="x", rotation=45, labelsize=8)
        ax3.set_ylabel("Recall")

    # ── Fig4: marginal gain ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title("Fig4: Marginal Gain (dF1 per 100 samples)", fontsize=10)
    if len(ns) >= 2:
        pairs = [(ns[i], ns[i-1], f1s[i], f1s[i-1]) for i in range(1, len(ns)) if ns[i] != ns[i-1]]
        delta_f1 = [(f1 - fp) / (n - np) * 100 for n, np, f1, fp in pairs]
        mid_ns = [(n + np) // 2 for n, np, _, _ in pairs]
        ax4.bar([str(x) for x in mid_ns], delta_f1, color="orange")
        ax4.set_xlabel("Midpoint N")
        ax4.set_ylabel("dF1 per 100 samples (%)")

    # ── Fig5: inference time comparison ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title("Fig5: Inference Time vs. N", fontsize=10)
    if ns and imss and any(v > 0 for v in imss):
        ax5.plot(ns, imss, "o-b", label="sft-no-skill")
    if base_no_ms:
        ax5.axhline(base_no_ms, color="gray", ls=":", label=f"base-no-skill ({base_no_ms:.0f}ms)")
    if base_with_ms:
        ax5.axhline(base_with_ms, color="orange", ls="--", label=f"base-with-skill ({base_with_ms:.0f}ms)")
    ax5.set_xlabel("Training samples (N)")
    ax5.set_ylabel("Inference time (ms)")
    ax5.legend(fontsize=7)

    # ── Fig6: Skill Internalization summary ───────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title("Fig6: Skill Internalization Summary", fontsize=10)
    ax6.axis("off")
    lines = [
        f"Power law: f1 = {a:.4f} x N^{b:.4f}" if a > 0 else "Power law: insufficient data",
        f"R2 = {r2:.3f}",
        "",
    ]
    for tn, pv in zip(pred_ns, pred_vals):
        lines.append(f"@{tn:,}: rule_f1 ~ {pv:.3f}")
    lines.append("")
    if base_no_f1 is not None and base_with_f1 is not None:
        lines.append(f"base-no-skill:   {base_no_f1:.3f}")
        lines.append(f"base-with-skill: {base_with_f1:.3f}")
    if f1s and base_with_f1 and base_with_f1 > 0:
        ratio_sft = f1s[-1] / base_with_f1
        lines.append(f"N4 sft / with-skill = {ratio_sft:.2f}")
    if skill_tokens > 0:
        lines.append(f"Skill tokens saved: {skill_tokens}/req")
    lines.append("")
    if ratio is not None:
        if ratio > 0.60:
            lines.append(f"Still rising fast (ratio={ratio:.2f})")
        elif ratio >= 0.30:
            lines.append(f"Slowing but positive (ratio={ratio:.2f})")
        else:
            lines.append(f"Near saturation (ratio={ratio:.2f})")
    else:
        lines.append("Insufficient data for saturation check")
    ax6.text(0.05, 0.95, "\n".join(lines), transform=ax6.transAxes,
             fontsize=9, va="top", family="monospace")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ 曲线图已保存：{output_path}")
    if show:
        plt.show()
    plt.close(fig)

    return {
        "a": a, "b": b, "r2": round(r2, 3),
        "pred_at_target": pred_at_target,
        "target_n": target_n,
        "n_sft_points": len(ns),
        "base_no_skill_f1": base_no_f1,
        "base_with_skill_f1": base_with_f1,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="生成效果-数据量曲线图（3条件）")
    parser.add_argument("--results_dir", required=True, help="评测结果 JSON 所在目录")
    parser.add_argument("--output", default="reports/scaling_curve.png")
    parser.add_argument("--target_n", type=int, default=100_000, help="外推目标数据量")
    parser.add_argument("--show", action="store_true", help="弹出交互式窗口")
    args = parser.parse_args()

    result = generate_curves(args.results_dir, args.output, args.target_n, args.show)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
