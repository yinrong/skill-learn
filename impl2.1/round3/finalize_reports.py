"""
round3 实验完成后更新所有报告的脚本。
当 R3-D3/R3-C/R3-E1/Claude-fair 结果都产出后运行：
    python3 round3/finalize_reports.py
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "round3/results"
SUMMARY_PATH = ROOT / "round3/reports/round3_summary.md"
INDEX_PATH = ROOT / "round3/index.md"
PROGRESS_PATH = ROOT / "round3/progress.md"
DEGRADATION_PATH = ROOT / "common/benchmark_results/degradation_summary.json"


def load_result(name: str) -> dict | None:
    f = RESULTS_DIR / f"{name}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def fmt(val, prec=3) -> str:
    if val is None:
        return "—"
    return f"{val:.{prec}f}"


def main():
    print("=== Round3 报告更新 ===")

    # ── 加载所有实验结果 ──────────────────────────────────────────────────────
    experiments = {
        "R3-A": load_result("R3-A"),
        "R3-B": load_result("R3-B"),
        "R3-AB": load_result("R3-AB"),
        "R3-AB-v2": load_result("R3-AB-v2"),
        "R3-AB-v3": load_result("R3-AB-v3"),
        "R3-D1": load_result("R3-D1"),
        "R3-D2": load_result("R3-D2"),
        "R3-D3": load_result("R3-D3"),
        "R3-C-grpo": load_result("R3-C-grpo"),
        "R3-E1": load_result("R3-E1"),
        "R3-claude-fair": load_result("R3-claude-fair"),
    }

    for name, r in experiments.items():
        if r:
            s = r["summary"]
            f1 = s.get("rule_detection_f1", 0)
            print(f"  {name}: F1={f1:.3f}")
        else:
            print(f"  {name}: 未完成")

    # ── 加载退化率 ──────────────────────────────────────────────────────────
    degradation = {}
    if DEGRADATION_PATH.exists():
        deg_data = json.loads(DEGRADATION_PATH.read_text())
        for tag, info in deg_data.items():
            avg = info.get("scores", {}).get("_avg_degradation")
            if avg is not None:
                key = tag.replace("sft-14B-", "").replace("sft-32B-", "")
                degradation[key] = avg
        print(f"\n退化率：{degradation}")

    # ── 打印新增实验结果表格 ──────────────────────────────────────────────
    print("\n新增实验结果：")
    for name in ["R3-D3", "R3-C-grpo", "R3-E1", "R3-claude-fair"]:
        r = experiments.get(name)
        if r:
            s = r["summary"]
            f1 = s.get("rule_detection_f1", 0)
            pr = s.get("per_rule_recall", {})
            cpk_mae = s.get("cpk_mae")
            cpk_rate = s.get("cpk_found_rate", 0)
            print(f"  {name}: F1={f1:.3f} rule2={fmt(pr.get('rule2'))} "
                  f"rule7={fmt(pr.get('rule7'))} CPK_MAE={fmt(cpk_mae)} "
                  f"CPK率={cpk_rate:.3f}")

    print("\n提示：如需更新 round3_summary.md，请检查上述输出后手动运行编辑。")
    print("自动更新需要读取当前报告内容然后追加新实验行。")

    # ── 更新综合得分（如果有退化率数据） ─────────────────────────────────
    if degradation:
        print("\n综合得分（加入退化率后）：")
        # 公式: F1 × (1 - 退化率) / (推理时间_s × 推理卡数 × 显存_GB) × 1000
        eval_configs = {
            "R3-AB-v2": {"deg_key": "R3-AB-v2", "inf_s": 45.4, "cards": 1, "vram": 96},
            "R3-D1": {"deg_key": "R3-D1", "inf_s": 45.1, "cards": 1, "vram": 20},
            "R3-D2": {"deg_key": "R3-D2", "inf_s": 99.1, "cards": 1, "vram": 45},
            "R3-D3": {"deg_key": "R3-D3", "inf_s": 45.0, "cards": 1, "vram": 30},
        }
        for exp, cfg in eval_configs.items():
            r = experiments.get(exp)
            if not r:
                continue
            f1 = r["summary"].get("rule_detection_f1", 0)
            deg = degradation.get(cfg["deg_key"], 0.0)
            score = f1 * (1 - deg) / (cfg["inf_s"] * cfg["cards"] * cfg["vram"]) * 1000
            print(f"  {exp}: F1={f1:.3f} 退化率={deg:.3%} 综合得分={score:.4f}")


if __name__ == "__main__":
    main()
