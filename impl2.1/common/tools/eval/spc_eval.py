"""SPC 评测脚本：调用模型 API，计算 rule_f1 / cpk_mae 等指标，输出 JSON。

用法：
    # 无 Skill（基座 base-no-skill 或 SFT 内化评测）
    python tools/eval/spc_eval.py \\
        --model_url http://localhost:8020 \\
        --model_name qwen3-8b-base \\
        --test data/demo/test.jsonl \\
        --n_train_samples 0 \\
        --output results/demo/baseline_8b_no_skill.json

    # 有 Skill（基座 base-with-skill 对照评测）
    python tools/eval/spc_eval.py \\
        --model_url http://localhost:8020 \\
        --model_name qwen3-8b-base \\
        --test data/demo/test_with_skill.jsonl \\
        --n_train_samples 0 \\
        --output results/demo/baseline_8b_with_skill.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from tools.spc.rules import check_nelson_rules


# ── API 调用 ──────────────────────────────────────────────────────────────────
def call_model(
    url: str,
    system: str,
    instruction: str,
    input_text: str,
    max_tokens: int = 2048,
    timeout: int = 180,
    enable_thinking: bool = False,
) -> tuple[str, float]:
    """调用 vLLM OpenAI 兼容接口，返回 (输出文本, 推理时间ms)。"""
    prompt = f"{instruction}\n\n{input_text}" if input_text else instruction
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # enable_thinking=True allows the model to use <think> reasoning chains
        # which improves accuracy significantly (trained models were trained with thinking)
        # enable_thinking=False is faster but causes models to skip reasoning → lower quality
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }).encode()

    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        elapsed_ms = (time.time() - t0) * 1000
        msg = result["choices"][0]["message"]
        # vllm 0.19.1 with reasoning_parser: thinking in "reasoning", answer in "content"
        reasoning = msg.get("reasoning") or ""
        content = msg.get("content") or ""
        if content:
            # Model completed thinking + answer: wrap reasoning so extractor looks at content
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n\n{content}", elapsed_ms
            return content, elapsed_ms
        elif reasoning:
            # Model put all output into reasoning (e.g. enable_thinking=False with reasoning_parser).
            # Return as plain text so extractor scans full text for violations/CPK.
            return reasoning, elapsed_ms
        return "[ERROR] empty response", elapsed_ms
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        return f"[ERROR] {e}", elapsed_ms


# ── 单样本指标计算 ─────────────────────────────────────────────────────────────
def compute_sample_metrics(
    pred_text: str,
    gt_violations: list[str],
    gt_cpk: Optional[float],
) -> dict:
    pred_violations = extract_violations(pred_text)
    pred_cpk = extract_cpk(pred_text)
    has_reason = has_reasoning_chain(pred_text)
    disposal_ok, bad_items = check_disposal_quality(pred_text)

    # Rule detection F1（基于 set 比较）
    gt_set = set(gt_violations)
    pred_set = set(pred_violations)
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # CPK
    cpk_error = abs(pred_cpk - gt_cpk) if (pred_cpk is not None and gt_cpk is not None) else None
    cpk_found = pred_cpk is not None

    # Per-rule recall（gt 中有该规则时，pred 是否也有）
    per_rule_correct = {}
    for rule in [f"rule{i}" for i in range(1, 9)]:
        if rule in gt_set:
            per_rule_correct[rule] = rule in pred_set
        else:
            per_rule_correct[rule] = None  # gt 无此规则，不计入该规则 recall

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "cpk_error": cpk_error,
        "cpk_found": cpk_found,
        "has_reasoning": has_reason,
        "disposal_clean": disposal_ok,
        "per_rule_correct": per_rule_correct,
        "pred_violations": pred_violations,
        "pred_cpk": pred_cpk,
    }


# ── 主评测函数 ────────────────────────────────────────────────────────────────
def run_eval(
    model_url: str,
    model_name: str,
    test_path: str,
    output_path: str,
    n_train_samples: int = 0,
    max_samples: Optional[int] = None,
    delay: float = 0.0,
    concurrency: int = 8,
    enable_thinking: bool = False,
    max_tokens: int = 2048,
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
    # Detect whether this is a with-skill or no-skill eval from first sample
    has_skill_in_data = "Nelson" in samples[0].get("system", "") if samples else False
    eval_condition = "with_skill" if has_skill_in_data else "no_skill"
    # Count skill tokens in first sample
    skill_tokens = len(samples[0].get("system", "").split()) if has_skill_in_data else 0

    print(f"评测模型: {model_name}  样本数: {n}  并发: {concurrency}  条件: {eval_condition}")

    per_sample_results_map: dict[int, dict] = {}
    all_f1 = []
    all_cpk_error = []
    all_inference_ms: list[float] = []
    cpk_found_count = 0
    reasoning_count = 0
    per_rule_tp: dict[str, int] = {f"rule{i}": 0 for i in range(1, 9)}
    per_rule_total: dict[str, int] = {f"rule{i}": 0 for i in range(1, 9)}
    completed = 0

    def _eval_one(idx_sample):
        idx, sample = idx_sample
        system_text = sample.get("system", "")
        instruction = sample.get("instruction", "")
        input_text  = sample.get("input", "")
        gt = sample.get("ground_truth", {})
        gt_violations = gt.get("violations", [])
        gt_cpk = gt.get("cpk")
        pred_text, elapsed_ms = call_model(model_url, system_text, instruction, input_text,
                                           max_tokens=max_tokens, enable_thinking=enable_thinking)
        metrics = compute_sample_metrics(pred_text, gt_violations, gt_cpk)
        metrics["inference_ms"] = elapsed_ms
        return idx, metrics, gt_violations, gt_cpk

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_eval_one, (i, s)): i for i, s in enumerate(samples)}
        for future in as_completed(futures):
            idx, metrics, gt_violations, gt_cpk = future.result()
            per_sample_results_map[idx] = {
                "idx": idx,
                "gt_violations": gt_violations,
                "gt_cpk": gt_cpk,
                **metrics,
            }
            all_f1.append(metrics["f1"])
            all_inference_ms.append(metrics["inference_ms"])
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
            completed += 1
            if completed % 20 == 0:
                running_f1 = sum(all_f1) / len(all_f1)
                print(f"  [{completed}/{n}] running rule_f1={running_f1:.3f}", flush=True)
            if delay > 0:
                time.sleep(delay)

    per_sample_results = [per_sample_results_map[i] for i in range(n)]

    # 汇总
    rule_detection_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    cpk_mae = sum(all_cpk_error) / len(all_cpk_error) if all_cpk_error else None
    cpk_found_rate = cpk_found_count / n if n > 0 else 0.0
    has_reasoning_rate = reasoning_count / n if n > 0 else 0.0

    # 推理时间
    sorted_ms = sorted(all_inference_ms)
    inf_mean = sum(sorted_ms) / len(sorted_ms) if sorted_ms else 0.0
    p95_idx = int(len(sorted_ms) * 0.95)
    inf_p95 = sorted_ms[min(p95_idx, len(sorted_ms) - 1)] if sorted_ms else 0.0

    per_rule_recall = {}
    for rule in per_rule_tp:
        total = per_rule_total[rule]
        per_rule_recall[rule] = round(per_rule_tp[rule] / total, 3) if total > 0 else None

    summary = {
        "model": model_name,
        "eval_condition": eval_condition,
        "n_train_samples": n_train_samples,
        "n_test_samples": n,
        "rule_detection_f1": round(rule_detection_f1, 3),
        "cpk_mae": round(cpk_mae, 3) if cpk_mae is not None else None,
        "cpk_found_rate": round(cpk_found_rate, 3),
        "has_reasoning_rate": round(has_reasoning_rate, 3),
        "inference_time_ms_mean": round(inf_mean, 1),
        "inference_time_ms_p95": round(inf_p95, 1),
        "skill_tokens": skill_tokens,
        "per_rule_recall": per_rule_recall,
    }

    result = {
        "summary": summary,
        "per_sample": per_sample_results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== 评测结果 ({eval_condition}) ===")
    print(f"  rule_detection_f1:       {summary['rule_detection_f1']}")
    print(f"  cpk_mae:                 {summary['cpk_mae']}")
    print(f"  cpk_found_rate:          {summary['cpk_found_rate']}")
    print(f"  has_reasoning_rate:      {summary['has_reasoning_rate']}")
    print(f"  inference_time_ms_mean:  {summary['inference_time_ms_mean']}")
    print(f"  inference_time_ms_p95:   {summary['inference_time_ms_p95']}")
    print(f"\n各规则 recall：")
    for rule, val in summary["per_rule_recall"].items():
        print(f"  {rule}: {val}")
    print(f"\n结果已写入：{output_path}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="SPC 模型评测")
    parser.add_argument("--model_url", required=True, help="vLLM 服务 URL，如 http://localhost:8000")
    parser.add_argument("--model_name", required=True, help="模型标识名（用于输出文件）")
    parser.add_argument("--test", required=True, help="测试集 JSONL 路径")
    parser.add_argument("--output", required=True, help="评测结果 JSON 输出路径")
    parser.add_argument("--n_train_samples", type=int, default=0, help="训练集样本数（用于 scaling 曲线 x 轴）")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评测样本数（调试用）")
    parser.add_argument("--delay", type=float, default=0.0, help="每个样本间隔秒数（限速）")
    parser.add_argument("--concurrency", type=int, default=8, help="并发请求数")
    parser.add_argument("--max_tokens", type=int, default=2048, help="最大输出 token 数（默认2048）")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="启用<think>推理链（更精准但更慢，默认关闭）")
    args = parser.parse_args()

    run_eval(
        model_url=args.model_url,
        model_name=args.model_name,
        test_path=args.test,
        output_path=args.output,
        n_train_samples=args.n_train_samples,
        max_samples=args.max_samples,
        delay=args.delay,
        concurrency=args.concurrency,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
