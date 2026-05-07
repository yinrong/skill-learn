"""用 Claude API 对 SPC 测试集进行评测，支持 with_skill / no_skill 两种条件。

用法：
    # no-skill（基座基线，与 SFT ns 评测同等条件）
    python tools/eval/spc_eval_claude.py \
        --test common/data/test.jsonl \
        --output round3/results/R3-claude-noskill-json.json \
        --json_output

    # with-skill + JSON（公平上限：注入规则文档 + 强制 JSON 输出，消除提取器偏差）
    python tools/eval/spc_eval_claude.py \
        --test common/data/test.jsonl \
        --output round3/results/R3-claude-withskill-json.json \
        --with_skill --json_output
"""
from __future__ import annotations
import argparse
import json
import os
import re
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
from tools.spc.generator import SPC_SYSTEM_WITH_SKILL, SPC_SYSTEM_NO_SKILL, SPC_SKILL_DOC

# 去掉技能文档中的"四、输出格式要求"部分，用于 with-skill JSON 评测
# 避免技能文档的格式要求与 JSON 指令冲突
_SKILL_DOC_NO_FORMAT = SPC_SKILL_DOC[:SPC_SKILL_DOC.find("\n四、输出格式要求")] + "\n=========================="
_SYSTEM_WITH_SKILL_NO_FORMAT = "你是一名 SPC 工程师。\n\n" + _SKILL_DOC_NO_FORMAT

# JSON 输出格式约束（追加到 system prompt 末尾）
_JSON_FORMAT_INSTRUCTION = """

===== 输出格式要求 =====
文本输出部分必须是且仅是一个合法 JSON 对象：
{"violations": ["rule2", "rule7"], "cpk": 1.234}

字段说明：
- violations：触发的规则列表（小写英文标识符 rule1～rule8）；无违规则为 []
- cpk：CPK 数值（浮点数，保留3位小数）
JSON 前后不得有任何其他文字。
========================"""

# Claude API 通过环境变量配置
_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "http://model.mify.ai.srv/anthropic")
_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
_MODEL = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "ppio/pa/claude-sonnet-4-6")


def extract_from_json(text: str) -> tuple[Optional[list[str]], Optional[float]]:
    """从 JSON 格式输出中提取 violations 和 cpk。

    优先解析 </think> 之后的文本；若无 think 块则解析全文。
    返回 (violations_list, cpk_value)，解析失败返回 (None, None)。
    """
    think_end = text.find("</think>")
    scan_text = text[think_end + len("</think>"):].strip() if think_end >= 0 else text.strip()

    # 去除可能的 Markdown 代码块包裹
    scan_text = re.sub(r'```(?:json)?\s*', '', scan_text)
    scan_text = re.sub(r'```', '', scan_text)

    # 找到最后一个完整 JSON 对象
    # 模型先做分析再输出 JSON，JSON 在文本末尾；从后向前找最后一个 }
    brace_end_pos = scan_text.rfind('}')
    if brace_end_pos == -1:
        return None, None
    # 从 brace_end_pos 向前找匹配的 {
    depth = 0
    brace_start = -1
    for idx in range(brace_end_pos, -1, -1):
        if scan_text[idx] == '}':
            depth += 1
        elif scan_text[idx] == '{':
            depth -= 1
            if depth == 0:
                brace_start = idx
                break
    if brace_start == -1:
        return None, None
    candidate = scan_text[brace_start:brace_end_pos + 1]

    try:
        obj = json.loads(candidate)
        violations = [str(v).lower() for v in obj.get("violations", [])]
        # 验证格式合法性
        valid_rules = {f"rule{i}" for i in range(1, 9)}
        violations = [v for v in violations if v in valid_rules]
        cpk = float(obj["cpk"]) if obj.get("cpk") is not None else None
        return violations, cpk
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None, None


def call_claude(
    system: str,
    prompt: str,
    max_tokens: int = 3000,
    budget_tokens: int = 2000,
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
        body["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}

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
    use_json: bool = False,
) -> dict:
    if use_json:
        json_violations, json_cpk = extract_from_json(pred_text)
        if json_violations is not None:
            pred_violations = json_violations
            pred_cpk = json_cpk
        else:
            # JSON 解析失败，回退到 regex（并记录解析失败）
            pred_violations = extract_violations(pred_text)
            pred_cpk = extract_cpk(pred_text)
    else:
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
    with_skill: bool = False,
    json_output: bool = False,
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
    eval_condition = ("with_skill" if with_skill else "no_skill") + ("_json" if json_output else "")
    print(f"评测模型: {_MODEL}  样本数: {n}  thinking: {enable_thinking}")
    print(f"条件: {eval_condition}  with_skill={with_skill}  json_output={json_output}")

    all_f1 = []
    all_cpk_error = []
    all_inference_ms = []
    cpk_found_count = 0
    reasoning_count = 0
    per_rule_tp: dict[str, int] = {f"rule{i}": 0 for i in range(1, 9)}
    per_rule_total: dict[str, int] = {f"rule{i}": 0 for i in range(1, 9)}
    per_sample_results = []

    json_parse_fail_count = 0

    for i, sample in enumerate(samples):
        # 决定系统提示：JSON 模式下指令放最前面，建立身份认知后再附技能文档
        if json_output:
            if with_skill:
                system_text = _JSON_FORMAT_INSTRUCTION.strip() + "\n\n" + _SKILL_DOC_NO_FORMAT
            else:
                system_text = _JSON_FORMAT_INSTRUCTION.strip() + "\n\n" + SPC_SYSTEM_NO_SKILL
        else:
            system_text = SPC_SYSTEM_WITH_SKILL if with_skill else SPC_SYSTEM_NO_SKILL

        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        gt = sample.get("ground_truth", {})
        gt_violations = gt.get("violations", [])
        gt_cpk = gt.get("cpk")

        # JSON 模式：替换原始 instruction（原指令要求写全文报告，导致模型忽略 JSON 格式指令）
        # 让模型先做计算分析，最后输出 JSON（"最后一条指令"策略）
        # max_tokens=3000 足够完整分析 + 末尾 JSON；禁用 thinking 加速
        if json_output:
            json_instruction = "根据以下数据，检查 Nelson 8 条规则是否违规并计算 CPK。计算完成后，在回答最后输出 JSON 对象，无其他文字。"
            prompt = f"{json_instruction}\n\n{input_text}" if input_text else json_instruction
            max_tok, bgt = 3000, 0
            effective_thinking = False
        else:
            prompt = f"{instruction}\n\n{input_text}" if input_text else instruction
            max_tok, bgt = 3000, 2000
            effective_thinking = enable_thinking
        pred_text, elapsed_ms = call_claude(
            system=system_text,
            prompt=prompt,
            max_tokens=max_tok,
            budget_tokens=bgt,
            enable_thinking=effective_thinking,
        )

        metrics = compute_sample_metrics(pred_text, gt_violations, gt_cpk, use_json=json_output)
        if json_output:
            parsed_ok, _ = extract_from_json(pred_text)
            if parsed_ok is None:
                json_parse_fail_count += 1
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
        "eval_condition": eval_condition,
        "with_skill": with_skill,
        "json_output": json_output,
        "json_parse_fail_count": json_parse_fail_count if json_output else None,
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

    print(f"\n=== Claude 评测结果 [{eval_condition}] ===")
    print(f"  rule_detection_f1:  {summary['rule_detection_f1']}")
    print(f"  cpk_mae:            {summary['cpk_mae']}")
    print(f"  cpk_found_rate:     {summary['cpk_found_rate']}")
    print(f"  has_reasoning_rate: {summary['has_reasoning_rate']}")
    if json_output:
        print(f"  json_parse_fail:    {json_parse_fail_count}/{n}")
    print(f"\n各规则 recall：")
    for rule, val in summary["per_rule_recall"].items():
        print(f"  {rule}: {val}")
    print(f"\n结果已写入：{output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Claude API 评测 SPC 任务（支持 with_skill / json_output）")
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_thinking", action="store_true", help="禁用 extended thinking")
    parser.add_argument("--delay", type=float, default=1.0, help="请求间隔秒数（避免限速）")
    parser.add_argument("--with_skill", action="store_true",
                        help="注入完整技能文档（ws 条件），测量能力上限")
    parser.add_argument("--json_output", action="store_true",
                        help="强制 JSON 输出格式，消除提取器偏差")
    args = parser.parse_args()

    run_eval(
        test_path=args.test,
        output_path=args.output,
        max_samples=args.max_samples,
        enable_thinking=not args.no_thinking,
        delay=args.delay,
        with_skill=args.with_skill,
        json_output=args.json_output,
    )


if __name__ == "__main__":
    main()
