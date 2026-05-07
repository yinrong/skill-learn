"""
计算 SFT 模型相对基座模型的退化率，更新 degradation_summary.json。

用法：
    python common/tools/eval/compute_degradation.py \
        --benchmark_dir common/benchmark_results \
        --sft_tags sft-14B-R3-AB-v2 sft-14B-R3-D1 sft-32B-R3-D2 \
        --base_map 14B=base-14B 32B=base-32B \
        --output common/benchmark_results/degradation_summary.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

TASK_KEYS = {
    "arc_challenge": "acc_norm,none",
    "gsm8k": "exact_match,flexible-extract",
    "hellaswag": "acc_norm,none",
    "mmlu": "acc,none",
    "truthfulqa_mc1": "acc,none",
    "winogrande": "acc,none",
}


def find_results_file(benchmark_dir: Path, tag: str) -> Path | None:
    tag_dir = benchmark_dir / tag / "standard"
    if not tag_dir.exists():
        return None
    # Find results_*.json recursively
    files = sorted(tag_dir.rglob("results_*.json"))
    return files[-1] if files else None


def load_task_scores(results_path: Path) -> dict[str, float]:
    data = json.loads(results_path.read_text())
    results = data.get("results", {})
    scores = {}
    for task, key in TASK_KEYS.items():
        if task in results and key in results[task]:
            scores[task] = results[task][key]
    return scores


def compute_degradation(base_scores: dict, sft_scores: dict) -> dict:
    out = {}
    degradations = []
    for task in TASK_KEYS:
        if task in base_scores and task in sft_scores:
            base = base_scores[task]
            sft = sft_scores[task]
            # degradation = (base - sft) / base (positive = degraded, negative = improved)
            deg = (base - sft) / base if base > 0 else 0.0
            out[task] = {"base": base, "sft": sft, "degradation": deg}
            degradations.append(deg)
    if degradations:
        out["_avg_degradation"] = sum(degradations) / len(degradations)
    return out


def infer_base_tag(sft_tag: str, base_map: dict[str, str]) -> str | None:
    """推断 SFT tag 对应的 base tag。

    base_map 示例: {"14B": "base-14B", "32B": "base-32B"}
    """
    for size_key, base_tag in base_map.items():
        if size_key in sft_tag:
            return base_tag
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", default="common/benchmark_results")
    parser.add_argument("--sft_tags", nargs="+", required=True,
                        help="SFT model tags to evaluate (subdirs of benchmark_dir)")
    parser.add_argument("--base_map", nargs="+", default=["14B=base-14B", "32B=base-32B"],
                        help="Mapping from model size to base tag, e.g. '14B=base-14B'")
    parser.add_argument("--output", default="common/benchmark_results/degradation_summary.json")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    base_map = {k: v for k, v in [x.split("=") for x in args.base_map]}

    # Load existing summary
    output_path = Path(args.output)
    summary = {}
    if output_path.exists():
        summary = json.loads(output_path.read_text())

    # Preload base scores
    base_scores_cache = {}
    for base_tag in set(base_map.values()):
        base_file = find_results_file(benchmark_dir, base_tag)
        if base_file:
            base_scores_cache[base_tag] = load_task_scores(base_file)
            print(f"Loaded base scores for {base_tag}: {list(base_scores_cache[base_tag].keys())}")
        else:
            print(f"WARNING: No results file found for {base_tag}")

    # Process each SFT tag
    for sft_tag in args.sft_tags:
        sft_file = find_results_file(benchmark_dir, sft_tag)
        if not sft_file:
            print(f"WARNING: No results file found for {sft_tag}, skipping")
            continue

        base_tag = infer_base_tag(sft_tag, base_map)
        if not base_tag or base_tag not in base_scores_cache:
            print(f"WARNING: Cannot find base for {sft_tag}, skipping")
            continue

        sft_scores = load_task_scores(sft_file)
        base_scores = base_scores_cache[base_tag]
        deg = compute_degradation(base_scores, sft_scores)

        summary[sft_tag] = {
            "base_tag": base_tag,
            "scores": deg,
        }

        avg_deg = deg.get("_avg_degradation", float("nan"))
        print(f"\n{sft_tag} (base: {base_tag}):")
        for task, d in deg.items():
            if task != "_avg_degradation":
                print(f"  {task}: base={d['base']:.4f} sft={d['sft']:.4f} deg={d['degradation']:.4f}")
        print(f"  → avg_degradation: {avg_deg:.4f} ({avg_deg*100:.2f}%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDegradation summary written to {output_path}")


if __name__ == "__main__":
    main()
