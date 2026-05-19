"""
全轨迹工具调用评估 + 模型推理计时

评估方式：
  给模型 [system, user_question]（ns格式，无 skill doc），
  运行完整 agentic 循环（tool calls → mock responses → final answer），
  对比 GT 轨迹的工具调用序列。

指标：
  - trajectory_tool_f1: GT 工具集合 vs 预测工具集合的 set-level F1（名称匹配）
  - trajectory_kv_f1:   包含参数的 set-level F1
  - model_time_per_step_ms: 每步平均模型推理时间
  - n_steps_pred / n_steps_gt: 预测步骤数 vs GT 步骤数

用法：
  python round4/eval/trajectory_eval.py \\
    --test_file round4/data/test_ns_all_eval_ns.jsonl \\
    --ws_file round4/data/test_ns_all.jsonl \\
    --output round4/results/R4-trajectory-eval.json \\
    --model_url http://localhost:8035

对比基线 (base model without finetune):
  python round4/eval/trajectory_eval.py \\
    --model_url http://localhost:8036 \\
    --output round4/results/baseline-trajectory-eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

DEFAULT_MODEL_URL = "http://localhost:8035"
MAX_ITER = 15


# ── Simulated tool responses (GT replay + synthetic fallback) ─────────────────

from simulated_tools import ReplayToolEngine  # noqa: E402


# ── vLLM client ───────────────────────────────────────────────────────────────

def chat_completion(messages, tools, model_url, model_name, max_tokens=1024, temperature=0.0):
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
    resp = requests.post(f"{model_url}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def get_served_model_name(model_url):
    try:
        resp = requests.get(f"{model_url}/v1/models", timeout=10)
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "default"


# ── Full agentic loop ─────────────────────────────────────────────────────────

def run_agent(
    ns_messages: list[dict],
    tools: list[dict],
    model_url: str,
    model_name: str,
    tool_engine: "ReplayToolEngine | None" = None,
) -> dict:
    """
    Run full agentic loop. Uses ns_messages[:2] (system+user) as starting prompt.
    Returns detailed timing and tool call information.
    """
    # Start with just system + user (ns format, no skill doc)
    messages = [dict(m) for m in ns_messages[:2]]

    tool_calls_made = []  # list of {"name": str, "arguments": dict}
    step_times_ms = []  # model inference time per step
    final_answer = ""
    total_model_ms = 0

    for step in range(MAX_ITER):
        t0 = time.time()
        try:
            resp = chat_completion(messages, tools, model_url, model_name)
        except Exception as e:
            return {
                "error": str(e),
                "tool_calls": tool_calls_made,
                "final_answer": final_answer,
                "step_times_ms": step_times_ms,
                "n_steps": step,
            }
        step_ms = int((time.time() - t0) * 1000)
        step_times_ms.append(step_ms)
        total_model_ms += step_ms

        choice = resp["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason", "")
        messages.append(msg)

        tcs = msg.get("tool_calls") or []
        if tcs or finish_reason == "tool_calls":
            for tc in tcs:
                func = tc.get("function", {})
                name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls_made.append({"name": name, "arguments": args})
                # Use replay engine (GT data) or synthetic fallback
                if tool_engine is not None:
                    tool_resp = tool_engine.respond(name, args)
                else:
                    from simulated_tools import synthetic_response
                    tool_resp = synthetic_response(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{step}"),
                    "content": tool_resp,
                })
        else:
            final_answer = msg.get("content", "")
            break

    avg_ms = total_model_ms // max(len(step_times_ms), 1)
    return {
        "tool_calls": tool_calls_made,
        "final_answer": final_answer,
        "step_times_ms": step_times_ms,
        "n_steps": len(step_times_ms),
        "total_model_ms": total_model_ms,
        "avg_step_ms": avg_ms,
    }


# ── GT extraction from ws sample ─────────────────────────────────────────────

def extract_gt_tool_calls(ws_sample: dict) -> list[dict]:
    """Extract all tool calls from GT (ws format) conversation."""
    calls = []
    for m in ws_sample.get("messages", []):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                func = tc.get("function", {})
                name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if name:
                    calls.append({"name": name, "arguments": args})
    return calls


def extract_gt_final_answer(ws_sample: dict) -> str:
    """Extract GT final answer from ws sample (last text-only assistant message)."""
    msgs = ws_sample.get("messages", [])
    for m in reversed(msgs):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            return m.get("content", "")
    return ""


# ── Set-level F1 ─────────────────────────────────────────────────────────────

def normalize_args(args: dict | str) -> dict:
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {}
    return args or {}


def tool_call_to_str(tc: dict, include_args: bool = False) -> str:
    """Convert tool call to string for set comparison."""
    if not include_args:
        return tc["name"]
    args = normalize_args(tc.get("arguments", {}))
    return tc["name"] + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False)


def set_f1(gt_calls: list[dict], pred_calls: list[dict], include_args: bool = False) -> dict:
    """Compute set-level F1 of tool calls."""
    gt_set = [tool_call_to_str(tc, include_args) for tc in gt_calls]
    pred_set = [tool_call_to_str(tc, include_args) for tc in pred_calls]

    if not gt_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gt_set or not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Count overlapping (multiset intersection)
    from collections import Counter
    gt_counter = Counter(gt_set)
    pred_counter = Counter(pred_set)
    overlap = sum((gt_counter & pred_counter).values())

    precision = overlap / len(pred_set)
    recall = overlap / len(gt_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="全轨迹工具调用 F1 + 推理计时")
    parser.add_argument("--test_file", default="round4/data/test_ns_all_eval_ns.jsonl",
                        help="ns格式测试集（用于推理前缀）")
    parser.add_argument("--ws_file", default="round4/data/test_ns_all.jsonl",
                        help="ws格式测试集（用于GT工具调用序列）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--model_name", default="auto")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_name = args.model_name
    if model_name == "auto":
        model_name = get_served_model_name(args.model_url)
        print(f"Auto-detected model: {model_name}")

    # Load data
    ns_samples = [json.loads(l) for l in open(args.test_file) if l.strip()]
    ws_samples = [json.loads(l) for l in open(args.ws_file) if l.strip()]
    assert len(ns_samples) == len(ws_samples)

    if args.max_samples:
        ns_samples = ns_samples[:args.max_samples]
        ws_samples = ws_samples[:args.max_samples]

    print(f"Evaluating {len(ns_samples)} samples on {args.model_url} (model={model_name})")
    print()

    all_name_f1 = []
    all_kv_f1 = []
    all_step_ms = []
    all_n_steps_pred = []
    all_n_steps_gt = []
    results = []
    n_errors = 0

    # Per-skill tracking
    skill_metrics: dict[str, list] = defaultdict(list)

    for i, (ns_s, ws_s) in enumerate(zip(ns_samples, ws_samples)):
        tools = ns_s.get("tools", [])
        gt_calls = extract_gt_tool_calls(ws_s)
        gt_answer = extract_gt_final_answer(ws_s)

        # Detect skill from ws user message
        ws_user = ws_s["messages"][1].get("content", "")
        skill_match = re.search(r'name:\s*(\S+)', ws_user)
        skill = skill_match.group(1) if skill_match else "unknown"

        print(f"[{i+1:2d}/{len(ns_samples)}] skill={skill} gt_steps={len(gt_calls)}", end=" ... ", flush=True)
        t_start = time.time()

        engine = ReplayToolEngine(ws_s)
        result = run_agent(ns_s["messages"], tools, args.model_url, model_name, tool_engine=engine)
        elapsed = int((time.time() - t_start) * 1000)

        if "error" in result:
            n_errors += 1
            print(f"ERROR: {result['error'][:80]}")
            continue

        pred_calls = result["tool_calls"]
        name_metrics = set_f1(gt_calls, pred_calls, include_args=False)
        kv_metrics = set_f1(gt_calls, pred_calls, include_args=True)

        all_name_f1.append(name_metrics["f1"])
        all_kv_f1.append(kv_metrics["f1"])
        all_n_steps_pred.append(result["n_steps"])
        all_n_steps_gt.append(len(gt_calls))
        if result["step_times_ms"]:
            all_step_ms.extend(result["step_times_ms"])

        skill_metrics[skill].append({
            "name_f1": name_metrics["f1"],
            "kv_f1": kv_metrics["f1"],
            "n_steps_pred": result["n_steps"],
            "n_steps_gt": len(gt_calls),
        })

        status = "✓" if name_metrics["f1"] >= 0.75 else "✗"
        print(f"{status} name_f1={name_metrics['f1']:.2f} kv_f1={kv_metrics['f1']:.2f} "
              f"pred_steps={result['n_steps']} avg_ms={result['avg_step_ms']}")

        if args.verbose:
            gt_names = [tc["name"] for tc in gt_calls]
            pred_names = [tc["name"] for tc in pred_calls]
            print(f"     GT  tools: {gt_names}")
            print(f"     Pred tools: {pred_names}")
            if result.get("final_answer"):
                print(f"     Final: {result['final_answer'][:120]}")

        results.append({
            "skill": skill,
            "gt_tools": [tc["name"] for tc in gt_calls],
            "pred_tools": [tc["name"] for tc in pred_calls],
            "name_f1": name_metrics["f1"],
            "kv_f1": kv_metrics["f1"],
            "n_steps_pred": result["n_steps"],
            "n_steps_gt": len(gt_calls),
            "step_times_ms": result["step_times_ms"],
            "avg_step_ms": result["avg_step_ms"],
        })

    print()
    print("=" * 70)
    n = len(all_name_f1)
    print(f"TRAJECTORY EVAL  (n={n}, errors={n_errors})")
    print()

    avg_name_f1 = sum(all_name_f1) / n if n else 0
    avg_kv_f1 = sum(all_kv_f1) / n if n else 0
    avg_step_ms = sum(all_step_ms) / len(all_step_ms) if all_step_ms else 0
    avg_n_pred = sum(all_n_steps_pred) / n if n else 0
    avg_n_gt = sum(all_n_steps_gt) / n if n else 0

    print(f"  trajectory_tool_name_f1 : {avg_name_f1:.4f}  ({avg_name_f1*100:.1f}%)")
    print(f"  trajectory_kv_f1        : {avg_kv_f1:.4f}  ({avg_kv_f1*100:.1f}%)")
    print(f"  avg_model_ms_per_step   : {avg_step_ms:.0f} ms  ({avg_step_ms/1000:.2f} s)")
    print(f"  avg_steps_pred          : {avg_n_pred:.1f}")
    print(f"  avg_steps_gt            : {avg_n_gt:.1f}")

    # Production reference
    prod_ms_per_step = 6600  # 6.6s from evidence_timing_v1.md
    speedup = prod_ms_per_step / avg_step_ms if avg_step_ms > 0 else 0
    print()
    print(f"  Production model: ~{prod_ms_per_step}ms/step (from Langfuse logs)")
    print(f"  Speedup          : {speedup:.1f}×")
    print(f"  Target           : ≥2.7× (≤2444ms/step)")
    print()
    if avg_name_f1 >= 0.75:
        print("  ✓ trajectory_tool_name_f1 ≥ 75%")
    else:
        print(f"  ✗ trajectory_tool_name_f1 = {avg_name_f1*100:.1f}% (target: 75%)")
    if speedup >= 2.7:
        print(f"  ✓ speedup {speedup:.1f}× ≥ 2.7×")
    else:
        print(f"  ✗ speedup {speedup:.1f}× < 2.7×")

    print()
    print("Per-skill breakdown:")
    for sk, sm in sorted(skill_metrics.items()):
        n_sk = len(sm)
        avg_f1 = sum(x["name_f1"] for x in sm) / n_sk
        avg_steps = sum(x["n_steps_pred"] for x in sm) / n_sk
        print(f"  {sk:40s} n={n_sk:2d} name_f1={avg_f1:.2f} pred_steps={avg_steps:.1f}")

    # Save
    summary = {
        "trajectory_tool_name_f1": round(avg_name_f1, 4),
        "trajectory_kv_f1": round(avg_kv_f1, 4),
        "avg_model_ms_per_step": round(avg_step_ms),
        "avg_steps_pred": round(avg_n_pred, 1),
        "avg_steps_gt": round(avg_n_gt, 1),
        "speedup_vs_production": round(speedup, 2),
        "production_ref_ms_per_step": prod_ms_per_step,
    }
    output_data = {
        "summary": summary,
        "n_samples": n,
        "n_errors": n_errors,
        "model_url": args.model_url,
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
