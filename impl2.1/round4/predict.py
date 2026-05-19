"""
工厂助手技能内化模型 — 预测程序

调用已内化技能的 Qwen3-14B 微调模型，无需 skill doc 即可完成工厂数据查询。

用法：
    # 启动 vLLM 服务（先执行）
    python common/tools/train/deploy_vllm.py \
        --model round4/checkpoints/R4-base-merged \
        --port 8035 --max_len 6144

    # 单次预测
    python round4/predict.py --question "S04线今天的良率是多少？"

    # 对比模式（带 skill doc vs 不带）
    python round4/predict.py --question "S04线今天的良率" \
        --skill_file round4/eval/skills/general-kpi-query.yaml \
        --compare

    # 批量预测（test set）
    python round4/predict.py --test_file round4/data/test_ns.jsonl \
        --output round4/results/R4-base-predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

SYSTEM_PROMPT = "你是制造业数据分析助手，通过调用工具查询工厂数据来回答用户问题。"
DEFAULT_MODEL_URL = "http://localhost:8035"
DEFAULT_SERVED_MODEL = "default"
MAX_ITER = 10  # max tool-calling rounds


# ── Tool mock (for demo without real API) ────────────────────────────────────

MOCK_RESPONSES: dict[str, Any] = {
    "list_object_types": {"code": 0, "message": "Success", "data": [
        {"apiName": "LineSite", "displayName": "线体", "description": "生产线体"},
        {"apiName": "Equipment", "displayName": "设备", "description": "生产设备"},
    ]},
    "get_object_type_metrics": {"code": 0, "message": "Success", "data": [
        {"metricApiName": "yield_rate", "metricDisplayName": "良率", "unit": "%"},
        {"metricApiName": "output", "metricDisplayName": "产量", "unit": "pcs"},
    ]},
    "get_idi_model_data": {"code": 0, "message": "Success", "data": {
        "records": [{"date": "2026-05-12", "value": 98.5, "unit": "%"}]
    }},
    "get_object_type_data": {"code": 0, "message": "Success", "data": []},
    "get_object_type_detail": {"code": 0, "message": "Success", "data": {}},
    "get_object_data": {"code": 0, "message": "Success", "data": []},
    "query_metric": {"code": 0, "message": "Success", "data": []},
    "query_data": {"code": 0, "message": "Success", "data": []},
}


def call_tool_mock(name: str, arguments: dict) -> str:
    """Return mock tool response for demo mode."""
    resp = MOCK_RESPONSES.get(name, {"code": -1, "message": f"Unknown tool: {name}", "data": None})
    return json.dumps(resp, ensure_ascii=False)


# ── vLLM client ──────────────────────────────────────────────────────────────

def chat_completion(
    messages: list[dict],
    tools: list[dict],
    model_url: str,
    model_name: str = "default",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    enable_thinking: bool = False,
) -> dict:
    """Call vLLM OpenAI-compatible chat completions endpoint."""
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
    if enable_thinking:
        payload["thinking"] = {"type": "enabled", "budget_tokens": 1024}

    resp = requests.post(
        f"{model_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def get_tools_from_sample(sample: dict) -> list[dict]:
    """Extract tools list from a test sample."""
    return sample.get("tools", [])


# ── Agentic loop ─────────────────────────────────────────────────────────────

def run_agent(
    question: str,
    tools: list[dict],
    model_url: str,
    model_name: str = "default",
    skill_content: str = "",
    mock: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run multi-turn agentic loop.

    Returns dict with:
      - messages: full conversation
      - final_answer: last assistant text response
      - tool_calls_made: list of (name, arguments) tuples
      - latency_ms: total time
    """
    t0 = time.time()

    # Build initial messages
    user_content = question
    if skill_content:
        user_content = f"## 技能指引\n{skill_content}\n\n## 用户问题\n{question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    tool_calls_made = []
    final_answer = ""

    for iteration in range(MAX_ITER):
        if verbose:
            print(f"\n[iter {iteration+1}] Calling model...", flush=True)

        resp = chat_completion(messages, tools, model_url, model_name)
        choice = resp["choices"][0]
        msg = choice["message"]
        finish_reason = choice["finish_reason"]

        # Add assistant message to history
        messages.append(msg)

        if finish_reason == "tool_calls" or msg.get("tool_calls"):
            # Process tool calls
            for tc in msg.get("tool_calls", []):
                func = tc["function"]
                name = func["name"]
                try:
                    args = json.loads(func["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {}

                if verbose:
                    print(f"  → Tool: {name}({json.dumps(args, ensure_ascii=False)[:100]})", flush=True)

                tool_calls_made.append((name, args))

                # Execute tool (mock or real)
                if mock:
                    result = call_tool_mock(name, args)
                else:
                    # Hook: replace with real tool execution
                    result = json.dumps({"error": "Real tool not connected"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        else:
            # Final answer
            final_answer = msg.get("content", "")
            if verbose:
                print(f"\n[Final Answer]\n{final_answer}", flush=True)
            break

    latency_ms = int((time.time() - t0) * 1000)
    return {
        "messages": messages,
        "final_answer": final_answer,
        "tool_calls_made": tool_calls_made,
        "latency_ms": latency_ms,
    }


# ── Evaluation mode ──────────────────────────────────────────────────────────

def eval_sample(
    sample: dict,
    model_url: str,
    model_name: str,
    mock: bool = True,
) -> dict:
    """Evaluate one test sample from test_ns.jsonl."""
    messages = sample.get("messages", [])
    tools = sample.get("tools", [])

    # Extract user question (ns format: no skill doc)
    question = ""
    for m in messages:
        if m.get("role") == "user":
            question = m.get("content", "")
            break

    # Ground truth: last function_call before final answer
    gt_calls = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                func = tc["function"]
                gt_calls.append({
                    "name": func["name"],
                    "arguments": func.get("arguments", "{}"),
                })

    # Run model
    result = run_agent(question, tools, model_url, model_name, mock=mock, verbose=False)

    # Compute tool_name accuracy (first call)
    pred_calls = result["tool_calls_made"]
    first_name_correct = (
        gt_calls and pred_calls and gt_calls[0]["name"] == pred_calls[0][0]
    ) if gt_calls and pred_calls else False

    return {
        "question": question[:80],
        "gt_tool_calls": gt_calls,
        "pred_tool_calls": [{"name": n, "arguments": a} for n, a in pred_calls],
        "first_tool_name_correct": first_name_correct,
        "latency_ms": result["latency_ms"],
        "final_answer": result["final_answer"][:200],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="工厂助手技能内化模型预测")
    parser.add_argument("--question", help="单条用户问题")
    parser.add_argument("--skill_file", help="可选：skill YAML 文件路径（with-skill 对比模式）")
    parser.add_argument("--test_file", help="批量评估：test_ns.jsonl 路径")
    parser.add_argument("--output", help="批量评估结果输出 JSONL")
    parser.add_argument("--model_url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--model_name", default=DEFAULT_SERVED_MODEL)
    parser.add_argument("--tools_file", help="工具定义 JSON 文件（单条预测时使用）")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="使用模拟工具响应（默认开启）")
    parser.add_argument("--no_mock", action="store_true",
                        help="连接真实工具 API")
    parser.add_argument("--compare", action="store_true",
                        help="同时运行 with-skill 和 no-skill 版本进行对比")
    args = parser.parse_args()

    mock = not args.no_mock

    # Load skill content if provided
    skill_content = ""
    if args.skill_file:
        skill_content = Path(args.skill_file).read_text(encoding="utf-8")

    # Load tools if provided
    default_tools: list[dict] = []
    if args.tools_file:
        default_tools = json.loads(Path(args.tools_file).read_text())

    if args.test_file:
        # Batch evaluation mode
        print(f"Running batch evaluation on {args.test_file}...")
        results = []
        n_correct = 0
        with open(args.test_file) as f:
            samples = [json.loads(l) for l in f if l.strip()]

        for i, sample in enumerate(samples):
            print(f"  [{i+1}/{len(samples)}]", end=" ", flush=True)
            try:
                r = eval_sample(sample, args.model_url, args.model_name, mock=mock)
                if r["first_tool_name_correct"]:
                    n_correct += 1
                print(f"✓" if r["first_tool_name_correct"] else "✗", flush=True)
                results.append(r)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)

        accuracy = n_correct / len(results) if results else 0
        print(f"\nFirst-tool-name accuracy: {n_correct}/{len(results)} = {accuracy:.2%}")

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Results saved to {args.output}")

    elif args.question:
        # Single question mode
        tools = default_tools

        print(f"\n{'='*60}")
        print(f"Question: {args.question}")
        print(f"Mode: {'with-skill' if skill_content else 'no-skill (internalized)'}")
        print(f"{'='*60}")

        result = run_agent(
            args.question, tools, args.model_url, args.model_name,
            skill_content=skill_content, mock=mock, verbose=True,
        )

        print(f"\n[Stats] {len(result['tool_calls_made'])} tool calls, {result['latency_ms']}ms")

        if args.compare and skill_content:
            print(f"\n{'='*60}")
            print("Comparison: no-skill (internalized)")
            print(f"{'='*60}")
            result_ns = run_agent(
                args.question, tools, args.model_url, args.model_name,
                skill_content="", mock=mock, verbose=True,
            )
            print(f"\n[Stats no-skill] {len(result_ns['tool_calls_made'])} calls, {result_ns['latency_ms']}ms")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
