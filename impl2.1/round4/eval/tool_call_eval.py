"""
工具调用 F1 评估脚本 — 步骤级预测评估

评估已内化技能的 Qwen3-14B 对工厂数据查询的工具调用准确率。

评估方式：步骤预测（step prediction）
  - 对每条测试样本（ns格式），将最后一条 assistant tool_call 之前的所有消息作为前缀
  - 让模型预测下一个工具调用
  - 与 ground truth 比较

指标：
  - tool_name_acc：工具名称匹配率
  - arg_key_f1：参数 key 重叠 F1
  - arg_kv_f1：参数 key-value 完全匹配 F1
  - combined_f1：综合 F1（tool_name * 0.5 + arg_kv_f1 * 0.5）

用法：
  # 先启动 vLLM（训练后合并 adapter）
  python round4/scripts/merge_adapter.sh  # 先合并
  bash round4/scripts/deploy_vllm.sh       # 再部署

  # 运行评估
  python round4/eval/tool_call_eval.py \\
    --test_file round4/data/test_ns.jsonl \\
    --output round4/results/R4-base.json \\
    --model_url http://localhost:8035

  # 对比基线：Qwen3-14B base（无微调）
  python round4/eval/tool_call_eval.py \\
    --test_file round4/data/test_ns.jsonl \\
    --output round4/results/baseline_no_skill.json \\
    --model_url http://localhost:8036
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


DEFAULT_MODEL_URL = "http://localhost:8035"
DEFAULT_MODEL_NAME = "default"


# ── vLLM client ───────────────────────────────────────────────────────────────

def chat_completion(
    messages: list[dict],
    tools: list[dict],
    model_url: str,
    model_name: str = "default",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> dict:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    resp = requests.post(
        f"{model_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def get_served_model_name(model_url: str) -> str:
    """Get the first served model name from vLLM."""
    try:
        resp = requests.get(f"{model_url}/v1/models", timeout=10)
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "default"


# ── Metrics ───────────────────────────────────────────────────────────────────

def normalize_args(args_str: str | dict) -> dict:
    """Normalize arguments to dict."""
    if isinstance(args_str, dict):
        return args_str
    try:
        return json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def compute_arg_key_f1(gt_args: dict, pred_args: dict) -> float:
    """F1 of argument keys."""
    gt_keys = set(gt_args.keys())
    pred_keys = set(pred_args.keys())
    if not gt_keys and not pred_keys:
        return 1.0
    if not gt_keys or not pred_keys:
        return 0.0
    overlap = gt_keys & pred_keys
    precision = len(overlap) / len(pred_keys)
    recall = len(overlap) / len(gt_keys)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_arg_kv_f1(gt_args: dict, pred_args: dict) -> float:
    """F1 of argument key-value pairs (string comparison)."""
    def kv_set(d: dict) -> set:
        return {(k, json.dumps(v, ensure_ascii=False, sort_keys=True)) for k, v in d.items()}

    gt_kvs = kv_set(gt_args)
    pred_kvs = kv_set(pred_args)
    if not gt_kvs and not pred_kvs:
        return 1.0
    if not gt_kvs or not pred_kvs:
        return 0.0
    overlap = len(gt_kvs & pred_kvs)
    precision = overlap / len(pred_kvs)
    recall = overlap / len(gt_kvs)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_tool_call_metrics(
    gt_calls: list[dict],
    pred_calls: list[dict],
) -> dict:
    """
    Compare ground truth vs predicted tool calls.

    gt_calls / pred_calls: list of {"name": str, "arguments": str|dict}
    Returns dict with name_acc, arg_key_f1, arg_kv_f1, combined_f1.
    """
    if not gt_calls and not pred_calls:
        return {"name_acc": 1.0, "arg_key_f1": 1.0, "arg_kv_f1": 1.0, "combined_f1": 1.0}
    if not gt_calls or not pred_calls:
        return {"name_acc": 0.0, "arg_key_f1": 0.0, "arg_kv_f1": 0.0, "combined_f1": 0.0}

    # Compare first tool call (most critical)
    gt_first = gt_calls[0]
    pred_first = pred_calls[0]

    name_acc = float(gt_first["name"] == pred_first["name"])
    gt_args = normalize_args(gt_first.get("arguments", {}))
    pred_args = normalize_args(pred_first.get("arguments", {}))
    arg_key_f1 = compute_arg_key_f1(gt_args, pred_args)
    arg_kv_f1 = compute_arg_kv_f1(gt_args, pred_args)
    combined_f1 = name_acc * 0.5 + arg_kv_f1 * 0.5

    return {
        "name_acc": name_acc,
        "arg_key_f1": arg_key_f1,
        "arg_kv_f1": arg_kv_f1,
        "combined_f1": combined_f1,
    }


# ── Sample evaluation ─────────────────────────────────────────────────────────

def extract_tool_calls(msg: dict) -> list[dict]:
    """Extract tool calls from an assistant message."""
    calls = []
    for tc in msg.get("tool_calls", []):
        func = tc.get("function", {})
        calls.append({
            "name": func.get("name", ""),
            "arguments": func.get("arguments", "{}"),
        })
    return calls


def eval_sample(
    sample: dict,
    model_url: str,
    model_name: str,
    verbose: bool = False,
) -> dict | None:
    """
    Evaluate one test sample (step prediction).

    The sample's last message is ground truth (assistant + tool_calls).
    The prefix (all preceding messages) is given to the model.
    """
    messages = sample.get("messages", [])
    tools = sample.get("tools", [])

    if not messages:
        return None

    # Find the last assistant message with tool_calls
    target_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
            target_idx = i
            break

    if target_idx is None:
        return None

    # Prefix: all messages before the target
    prefix_messages = messages[:target_idx]
    gt_msg = messages[target_idx]
    gt_calls = extract_tool_calls(gt_msg)

    if not gt_calls:
        return None

    # Call model
    t0 = time.time()
    try:
        resp = chat_completion(prefix_messages, tools, model_url, model_name)
        latency_ms = int((time.time() - t0) * 1000)
        pred_msg = resp["choices"][0]["message"]
    except Exception as e:
        return {"error": str(e), "gt_calls": gt_calls, "pred_calls": [], "metrics": {
            "name_acc": 0.0, "arg_key_f1": 0.0, "arg_kv_f1": 0.0, "combined_f1": 0.0,
        }}

    pred_calls = extract_tool_calls(pred_msg)

    metrics = compute_tool_call_metrics(gt_calls, pred_calls)
    metrics["latency_ms"] = latency_ms

    if verbose:
        gt_name = gt_calls[0]["name"] if gt_calls else "(none)"
        pred_name = pred_calls[0]["name"] if pred_calls else "(none)"
        status = "✓" if gt_name == pred_name else "✗"
        print(f"  {status} GT={gt_name} PRED={pred_name} f1={metrics['combined_f1']:.2f}")

    return {
        "gt_calls": gt_calls,
        "pred_calls": pred_calls,
        "metrics": metrics,
        "prefix_len": len(prefix_messages),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="工具调用 F1 评估")
    parser.add_argument("--test_file", required=True, help="test_ns.jsonl 路径")
    parser.add_argument("--output", required=True, help="结果输出 JSON 路径")
    parser.add_argument("--model_url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--model_name", default="auto",
                        help="vLLM 服务模型名，'auto' 自动从 /v1/models 获取")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="限制评估样本数（调试用）")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Resolve model name
    model_name = args.model_name
    if model_name == "auto":
        model_name = get_served_model_name(args.model_url)
        print(f"Auto-detected model: {model_name}")

    # Load test data
    print(f"Loading test data from {args.test_file}...")
    with open(args.test_file) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Evaluating {len(samples)} samples on {args.model_url}...")
    print()

    results = []
    all_metrics: dict[str, list[float]] = defaultdict(list)
    n_errors = 0
    n_no_target = 0

    for i, sample in enumerate(samples):
        print(f"[{i+1:3d}/{len(samples)}]", end=" ", flush=True)
        r = eval_sample(sample, args.model_url, model_name, verbose=True)

        if r is None:
            n_no_target += 1
            print("(skip: no target tool call)")
            continue

        if "error" in r:
            n_errors += 1
            print(f"ERROR: {r['error']}")

        for k, v in r["metrics"].items():
            if k != "latency_ms":
                all_metrics[k].append(float(v))

        results.append(r)

    print()
    print("=" * 60)
    n_eval = len(results)
    print(f"Evaluated: {n_eval} samples  (errors: {n_errors}, skipped: {n_no_target})")
    print()

    summary: dict[str, Any] = {}
    for metric_name in ["name_acc", "arg_key_f1", "arg_kv_f1", "combined_f1"]:
        vals = all_metrics[metric_name]
        avg = sum(vals) / len(vals) if vals else 0.0
        summary[metric_name] = round(avg, 4)
        print(f"  {metric_name:20s}: {avg:.4f}  ({avg*100:.1f}%)")

    print()
    print(f"Primary metric (combined_f1 = 0.5*name_acc + 0.5*arg_kv_f1): {summary['combined_f1']:.4f}")
    if summary["combined_f1"] >= 0.70:
        print("  ✓ PASS: combined_f1 ≥ 0.70 — 目标达成！")
    elif summary["combined_f1"] >= 0.50:
        print("  ~ PARTIAL: 0.50 ≤ combined_f1 < 0.70 — 需要迭代优化")
    else:
        print("  ✗ FAIL: combined_f1 < 0.50 — 检查数据格式或增加训练数据")

    # Save results
    output = {
        "summary": summary,
        "n_samples": n_eval,
        "n_errors": n_errors,
        "n_skipped": n_no_target,
        "model_url": args.model_url,
        "model_name": model_name,
        "test_file": args.test_file,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
