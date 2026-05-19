"""
LLM-as-judge 最终答案质量评估

通过 Claude API 对模型最终回答质量打分（0-1），评估维度：
  - 事实准确性：对比 ground truth，信息是否正确
  - 完整性：是否覆盖了 GT 中提到的关键信息
  - 格式清晰度：答案是否易于理解

用法：
  # 先确保 vLLM 服务已启动

  # 生成预测答案并评分
  python round4/eval/llm_judge.py \\
    --grpo_file round4/data/grpo_raw.jsonl \\
    --output round4/results/R4-base-judge.json \\
    --model_url http://localhost:8035 \\
    --max_samples 50

  # 指定模型名
  python round4/eval/llm_judge.py \\
    --grpo_file round4/data/grpo_raw.jsonl \\
    --output round4/results/R4-base-judge.json \\
    --model_url http://localhost:8035 \\
    --model_name R4-base-merged
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


# ── API clients ───────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
JUDGE_MODEL = "ppio/pa/claude-sonnet-4-6"


def call_claude(prompt: str, system: str = "", max_tokens: int = 512) -> str:
    """Call Claude API for judging."""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": JUDGE_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    resp = requests.post(
        f"{ANTHROPIC_BASE_URL}/v1/messages",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def chat_completion_vllm(
    messages: list[dict],
    tools: list[dict],
    model_url: str,
    model_name: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> dict:
    """Call vLLM OpenAI-compatible endpoint."""
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
    try:
        resp = requests.get(f"{model_url}/v1/models", timeout=10)
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "default"


# ── Full agentic loop ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = "你是制造业数据分析助手，通过调用工具查询工厂数据来回答用户问题。"
MAX_ITER = 12


import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from simulated_tools import ReplayToolEngine, synthetic_response  # noqa: E402


def run_full_agent(
    question: str,
    tools: list[dict],
    model_url: str,
    model_name: str,
    tool_engine: "ReplayToolEngine | None" = None,
) -> tuple[str, list[tuple[str, dict]]]:
    """Run full agentic loop. Returns (final_answer, tool_calls_made)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_calls_made = []
    final_answer = ""

    for _ in range(MAX_ITER):
        resp = chat_completion_vllm(messages, tools, model_url, model_name)
        choice = resp["choices"][0]
        msg = choice["message"]
        finish_reason = choice["finish_reason"]
        messages.append(msg)

        if finish_reason == "tool_calls" or msg.get("tool_calls"):
            for tc in msg.get("tool_calls", []):
                func = tc["function"]
                name = func["name"]
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls_made.append((name, args))
                if tool_engine is not None:
                    tool_resp = tool_engine.respond(name, args)
                else:
                    tool_resp = synthetic_response(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_resp,
                })
        else:
            final_answer = msg.get("content", "")
            break

    return final_answer, tool_calls_made


# ── LLM judging ───────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """你是一个严格的质量评估员，专门评估工厂助手 AI 的回答质量。
你的任务是对比 AI 的预测回答和标准参考答案，从三个维度打分：

1. 事实准确性（factual）：AI 回答是否与参考答案在事实上一致？没有错误信息？
2. 完整性（completeness）：AI 回答是否涵盖了参考答案的关键信息点？
3. 格式清晰度（clarity）：回答是否结构清晰、易于用户理解？

每个维度 0~1 分（可用 0.0, 0.25, 0.5, 0.75, 1.0），最后给综合分（overall）。

输出格式（严格 JSON，不要其他文字）：
{"factual": 0.8, "completeness": 0.7, "clarity": 0.9, "overall": 0.8, "reason": "简短说明"}
"""


def judge_answer(
    question: str,
    gt_answer: str,
    pred_answer: str,
) -> dict:
    """Judge predicted answer quality vs ground truth using Claude."""
    prompt = f"""## 用户问题
{question}

## 标准参考答案（Ground Truth）
{gt_answer}

## AI 预测回答
{pred_answer if pred_answer else "(AI 没有给出任何文字回答)"}

请按要求打分，仅输出 JSON。"""

    try:
        result = call_claude(prompt, system=JUDGE_SYSTEM, max_tokens=256)
        # Extract JSON from result
        match = re.search(r'\{.*?\}', result, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(result)
    except Exception as e:
        return {
            "factual": 0.0, "completeness": 0.0, "clarity": 0.0,
            "overall": 0.0, "reason": f"Judge error: {e}",
        }


# ── Ground truth extraction from GRPO data ────────────────────────────────────

def extract_gt_from_ws(sample: dict) -> tuple[str, str, list[dict]]:
    """Extract (question, gt_final_answer, tools) from ws-format sample (messages + tools)."""
    messages = sample.get("messages", [])
    tools = sample.get("tools", [])

    question = ""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "") or ""
            if "## 用户问题" in content:
                idx = content.find("## 用户问题")
                content = content[idx + len("## 用户问题"):].strip()
            elif "## 技能指引" in content:
                idx = content.find("\n\n", content.find("## 技能指引"))
                content = content[idx:].strip()
            if content and not content.startswith("(以下内容为内部信息") and not content.startswith("<system-reminder>"):
                question = content
                break

    gt_answer = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls"):
            gt_answer = m["content"]
            break

    return question, gt_answer, tools


def extract_gt_from_grpo(sample: dict) -> tuple[str, str, list[dict]]:
    """
    Extract (question, gt_final_answer, tools) from grpo_raw.jsonl sample.

    GRPO format: {"prompt": [...messages...], "completion": [...], ...}
    The completion's last assistant text message is the ground truth answer.
    """
    prompt_msgs = sample.get("prompt", [])
    completion = sample.get("completion", [])

    # Question = first user message in prompt
    question = ""
    for m in prompt_msgs:
        if m.get("role") == "user":
            content = m.get("content", "")
            # Strip skill doc if present
            if "## 用户问题" in content:
                idx = content.find("## 用户问题")
                content = content[idx + len("## 用户问题"):].strip()
            elif "## 技能指引" in content:
                idx = content.find("\n\n", content.find("## 技能指引"))
                content = content[idx:].strip()
            question = content
            break

    # Tools
    tools = sample.get("tools", [])

    # GT answer = last text-only assistant message in completion
    gt_answer = ""
    if completion:
        # completion might be a list or a single dict
        comp_list = completion if isinstance(completion, list) else [completion]
        for m in reversed(comp_list):
            if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls"):
                gt_answer = m["content"]
                break
        if not gt_answer:
            # Fallback: last assistant message content
            for m in reversed(comp_list):
                if m.get("role") == "assistant":
                    gt_answer = m.get("content") or ""
                    break

    return question, gt_answer, tools


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge 最终答案质量评估")
    parser.add_argument("--grpo_file", default=None,
                        help="grpo_raw.jsonl 路径（含 GT 最终答案）")
    parser.add_argument("--ws_file", default=None,
                        help="ws格式 jsonl 路径（messages+tools，含 GT 最终答案）")
    parser.add_argument("--output", required=True, help="结果输出 JSON 路径")
    parser.add_argument("--model_url", default="http://localhost:8035")
    parser.add_argument("--model_name", default="auto")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="评估样本数（默认50，Claude judge 较慢）")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.grpo_file and not args.ws_file:
        print("ERROR: --grpo_file or --ws_file is required")
        return

    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_AUTH_TOKEN not set")
        return

    model_name = args.model_name
    if model_name == "auto":
        model_name = get_served_model_name(args.model_url)
        print(f"Auto-detected model: {model_name}")

    # Load data — each entry: (question, gt_answer, tools, ws_sample_for_replay)
    eval_samples = []
    if args.ws_file:
        print(f"Loading ws data from {args.ws_file}...")
        with open(args.ws_file) as f:
            ws_samples = [json.loads(l) for l in f if l.strip()]
        for s in ws_samples:
            q, gt, tools = extract_gt_from_ws(s)
            if q and gt:
                eval_samples.append((q, gt, tools, s))
        print(f"Samples with GT answers: {len(eval_samples)} / {len(ws_samples)}")
    else:
        print(f"Loading GRPO data from {args.grpo_file}...")
        with open(args.grpo_file) as f:
            grpo_samples = [json.loads(l) for l in f if l.strip()]
        for s in grpo_samples:
            q, gt, tools = extract_gt_from_grpo(s)
            if q and gt:
                eval_samples.append((q, gt, tools, None))
        print(f"Samples with GT answers: {len(eval_samples)} / {len(grpo_samples)}")
    eval_samples = eval_samples[:args.max_samples]
    print(f"Evaluating {len(eval_samples)} samples...")
    print()

    results = []
    score_sum = {"factual": 0.0, "completeness": 0.0, "clarity": 0.0, "overall": 0.0}

    for i, (question, gt_answer, tools, ws_sample) in enumerate(eval_samples):
        print(f"[{i+1:2d}/{len(eval_samples)}] Running agent...", end=" ", flush=True)
        t0 = time.time()

        engine = ReplayToolEngine(ws_sample) if ws_sample is not None else None
        try:
            pred_answer, tool_calls = run_full_agent(question, tools, args.model_url, model_name, tool_engine=engine)
        except Exception as e:
            print(f"ERROR (agent): {e}")
            results.append({"question": question[:60], "error": str(e)})
            continue

        print(f"→ Judging...", end=" ", flush=True)
        scores = judge_answer(question, gt_answer, pred_answer)
        elapsed = int((time.time() - t0) * 1000)

        for k in score_sum:
            score_sum[k] += scores.get(k, 0.0)

        if args.verbose:
            print(f"\n  Q: {question[:60]}")
            print(f"  GT: {gt_answer[:80]}")
            print(f"  PRED: {pred_answer[:80]}")
            print(f"  Scores: {scores}")
        else:
            print(f"overall={scores.get('overall', 0):.2f} ({elapsed}ms)")

        results.append({
            "question": question[:100],
            "gt_answer": gt_answer[:300],
            "pred_answer": pred_answer[:300],
            "tool_calls_count": len(tool_calls),
            "scores": scores,
        })

    n = len(results)
    non_error = [r for r in results if "error" not in r]
    n_ok = len(non_error)

    print()
    print("=" * 60)
    print(f"Completed: {n_ok}/{n} samples")
    print()
    if n_ok > 0:
        for k in ["factual", "completeness", "clarity", "overall"]:
            avg = score_sum[k] / n_ok
            print(f"  {k:15s}: {avg:.4f}  ({avg*100:.1f}%)")
        overall_avg = score_sum["overall"] / n_ok
        print()
        print(f"Primary metric (judge_overall_avg): {overall_avg:.4f}")
        if overall_avg >= 0.70:
            print("  ✓ PASS: judge_overall ≥ 0.70")
        else:
            print("  ~ PARTIAL: judge_overall < 0.70 — 需要分析答案错误类型")

    # Save
    output_data = {
        "summary": {k: round(score_sum[k] / n_ok, 4) if n_ok > 0 else 0.0 for k in score_sum},
        "n_samples": n,
        "n_ok": n_ok,
        "model_url": args.model_url,
        "model_name": model_name,
        "grpo_file": args.grpo_file or args.ws_file,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
