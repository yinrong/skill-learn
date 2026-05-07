"""用 Claude API 对 SPC 测试集进行评测（无 Skill 文档），作为上限参考。

用法：
    python tools/eval/spc_eval_claude.py \
        --test data/demo/test.jsonl \
        --output history-xxx/results/base_claude_no_skill.json \
        --max_samples 50
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from tools.eval.extractor import (
    extract_violations,
    extract_cpk,
    has_reasoning_chain,
    check_disposal_quality,
)

# Claude API 通过环境变量配置
_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "http://model.mify.ai.srv/anthropic")
_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
_MODEL = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "ppio/pa/claude-sonnet-4-6")


def call_claude(
    system: str,
    prompt: str,
    max_tokens: int = 3000,
    timeout: int = 180,
    enable_thinking: bool = True,
) -> tuple[str, float]:
    """调用 Claude Messages API，返回 (输出文本, 推理时间ms)。"""
    import urllib.request

    headers = {
        "Content-Type": "application/json",
        "x-api-key": _AUTH_TOKEN,
        "anthropic-version": "2023-06-01",
    }
    body: dict = {
        "model": _MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    if enable_thinking:
        # Claude extended thinking（budget 较高用于 SPC 分析）
        body["thinking"] = {"type": "enabled", "budget_tokens": 2000}

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{_BASE_URL.rstrip('/')}/v1/messages",
        data=data,
        headers=headers,
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        elapsed_ms = (time.time() - t0) * 1000

        parts = []
        thinking_text = ""
        answer_text = ""
        for block in result.get("content", []):
            if block.get("type") == "thinking":
                thinking_text = block.get("thinking", "")
            elif block.get("type") == "text":
                answer_text = block.get("text", "")

        if thinking_text and answer_text:
            return f"<think>\n{thinking_text}\n</think>\n\n{answer_text}", elapsed_ms
        elif answer_text:
            return answer_text, elapsed_ms
        elif thinking_text:
            return thinking_text, elapsed_ms
        return "[ERROR] empty response", elapsed_ms

    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        return f"[ERROR] {e}", elapsed_ms


def compute_sample_metrics(
    pred_text: str,
    gt_violations: list[str],
    gt_cpk: Optional[float],
) -> dict:
    pred_violations = extract_violations(pred_text)
    pred_cpk = extract_cpk(pred_text)
    has_reason = has_reasoning_chain(pred_text)
    disposal_ok, bad_items = check_disposal_quality(pred_text)

    gt_set = set(gt_violations)
    pred_set = set(pred_violations)
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    cpk_error = abs(pred_cpk - gt_cpk) if (pred_cpk is not None and gt_cpk is not None) else None

    per_rule_correct = {}
    for rule in [f"rule{i}" for i in range(1, 9)]:
        if rule in gt_set:
            per_rule_correct[rule] = rule in pred_set
        else:
            per_rule_correct[rule] = None

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "cpk_error": cpk_error,
        "cpk_found": pred_cpk is not None,
        "has_reasoning": has_reason,
        "disposal_clean": disposal_ok,
        "per_rule_correct": per_rule_correct,
        "pred_violations": pred_violations,
        "pred_cpk": pred_cpk,
    }


def run_eval(
    test_path: str,
    output_path: str,
    max_samples: Optional[int] = None,
    enable_thinking: bool = True,
    delay: float = 1.0,
) -> dict:
    samples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    n = len(samples)
    print(f"评测模型: {_MODEL}  样本数: {n}  thinking: {enable_thinking}")

    all_f1 = []
    all_cpk_error = []
    all_inference_ms = []
    cpk_found_count = 0
    reasoning_count = 0
    per_rule_tp: dict[str, int] = {f"rule{i}": 0 for i in range(1, 9)}
    per_rule_total: dict[str, int] = {f"rule{i}": 0 for i in range(1, 9)}
    per_sample_results = []

    for i, sample in enumerate(samples):
        system_text = sample.get("system", "你是一名 SPC 工程师。")
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        gt = sample.get("ground_truth", {})
        gt_violations = gt.get("violations", [])
        gt_cpk = gt.get("cpk")

        prompt = f"{instruction}\n\n{input_text}" if input_text else instruction
        pred_text, elapsed_ms = call_claude(
            system=system_text,
            prompt=prompt,
            enable_thinking=enable_thinking,
        )

        metrics = compute_sample_metrics(pred_text, gt_violations, gt_cpk)
        metrics["inference_ms"] = elapsed_ms

        per_sample_results.append({
            "idx": i,
            "gt_violations": gt_violations,
            "gt_cpk": gt_cpk,
            **metrics,
        })

        all_f1.append(metrics["f1"])
        all_inference_ms.append(elapsed_ms)
        if metrics["cpk_error"] is not None:
            all_cpk_error.append(metrics["cpk_error"])
        if metrics["cpk_found"]:
            cpk_found_count += 1
        if metrics["has_reasoning"]:
            reasoning_count += 1
        for rule in per_rule_tp:
            correct = metrics["per_rule_correct"].get(rule)
            if correct is not None:
                per_rule_total[rule] += 1
                if correct:
                    per_rule_tp[rule] += 1

        if (i + 1) % 10 == 0:
            running_f1 = sum(all_f1) / len(all_f1)
            print(f"  [{i+1}/{n}] running rule_f1={running_f1:.3f}", flush=True)

        if delay > 0:
            time.sleep(delay)

    rule_detection_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    cpk_mae = sum(all_cpk_error) / len(all_cpk_error) if all_cpk_error else None
    sorted_ms = sorted(all_inference_ms)
    inf_mean = sum(sorted_ms) / len(sorted_ms) if sorted_ms else 0.0
    p95_idx = int(len(sorted_ms) * 0.95)
    inf_p95 = sorted_ms[min(p95_idx, len(sorted_ms) - 1)] if sorted_ms else 0.0

    per_rule_recall = {}
    for rule in per_rule_tp:
        total = per_rule_total[rule]
        per_rule_recall[rule] = round(per_rule_tp[rule] / total, 3) if total > 0 else None

    summary = {
        "model": _MODEL,
        "eval_condition": "no_skill_claude",
        "n_train_samples": 0,
        "n_test_samples": n,
        "rule_detection_f1": round(rule_detection_f1, 3),
        "cpk_mae": round(cpk_mae, 3) if cpk_mae is not None else None,
        "cpk_found_rate": round(cpk_found_count / n, 3),
        "has_reasoning_rate": round(reasoning_count / n, 3),
        "inference_time_ms_mean": round(inf_mean, 1),
        "inference_time_ms_p95": round(inf_p95, 1),
        "per_rule_recall": per_rule_recall,
    }

    result = {"summary": summary, "per_sample": per_sample_results}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== Claude 评测结果 ===")
    print(f"  rule_detection_f1:  {summary['rule_detection_f1']}")
    print(f"  cpk_mae:            {summary['cpk_mae']}")
    print(f"  cpk_found_rate:     {summary['cpk_found_rate']}")
    print(f"  has_reasoning_rate: {summary['has_reasoning_rate']}")
    print(f"\n各规则 recall：")
    for rule, val in summary["per_rule_recall"].items():
        print(f"  {rule}: {val}")
    print(f"\n结果已写入：{output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Claude API 评测 SPC 任务（上限参考）")
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_thinking", action="store_true", help="禁用 extended thinking")
    parser.add_argument("--delay", type=float, default=1.0, help="请求间隔秒数（避免限速）")
    args = parser.parse_args()

    run_eval(
        test_path=args.test,
        output_path=args.output,
        max_samples=args.max_samples,
        enable_thinking=not args.no_thinking,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
