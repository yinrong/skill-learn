"""
从 ws_raw.jsonl 的完整轨迹中提取"最终答案生成"训练样本。

背景：
  当前 train_ns.jsonl 全部是 tool call 预测样本（最后一轮是 tool_call），
  没有覆盖"调用完工具后生成最终文字答案"的训练。

此脚本从 ws_raw.jsonl（完整 SFT 轨迹）中：
  1. 找到每条轨迹的最后一个"工具响应 → 最终答案"片段
  2. 截取 prefix = [system, user, ..., last_tool_result]
  3. target = 最终文字答案
  4. 去掉 user message 中的 skill YAML（ws→ns 转换）
  5. 过滤长度 ≤ cutoff_len

用法：
  python round4/scripts/extract_final_answer_samples.py \\
    --input round4/data/ws_raw.jsonl \\
    --output round4/data/final_answer_ns.jsonl \\
    --cutoff_len 6144
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CUTOFF_LEN = 6144
CHARS_PER_TOKEN = 3.0  # rough heuristic


def strip_skill_from_user(content: str) -> str:
    """Remove '## 技能指引\\n...' block from user message (ws → ns)."""
    pattern = re.compile(r'##\s*技能指引\s*\n.*?(?=##\s|\Z)', re.DOTALL)
    cleaned = pattern.sub('', content).strip()
    # Also handle "---" separator
    cleaned = re.sub(r'^---\s*\n', '', cleaned, flags=re.MULTILINE).strip()
    # If '## 用户问题' header is present, keep just the content after it
    if '## 用户问题' in cleaned:
        idx = cleaned.find('## 用户问题')
        cleaned = cleaned[idx + len('## 用户问题'):].strip()
    return cleaned if cleaned else content


def estimate_tokens(messages: list[dict]) -> int:
    total_chars = sum(
        len(str(m.get('content', ''))) +
        len(json.dumps(m.get('tool_calls', []), ensure_ascii=False))
        for m in messages
    )
    return int(total_chars / CHARS_PER_TOKEN)


def extract_final_answer_sample(sample: dict, cutoff_len: int) -> dict | None:
    """
    Extract final answer prediction sample from a full SFT trajectory.

    Returns OpenAI format sample where:
    - prefix = [system, user(ns), ...intermediate tool calls..., last_tool_result]
    - target = assistant final text answer
    """
    messages = sample.get('messages', [])
    tools = sample.get('tools', [])

    if not messages:
        return None

    # Find the last assistant text message (no tool_calls, non-empty content)
    final_answer_idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get('role') == 'assistant' and m.get('content') and not m.get('tool_calls'):
            final_answer_idx = i
            break

    if final_answer_idx is None or final_answer_idx == 0:
        return None

    prefix = list(messages[:final_answer_idx])
    final_answer = messages[final_answer_idx]

    # Apply ws→ns to the first user message
    for i, m in enumerate(prefix):
        if m.get('role') == 'user':
            stripped = strip_skill_from_user(m.get('content', ''))
            if stripped != m.get('content', ''):
                prefix[i] = {**m, 'content': stripped}
            break

    # Append final answer as target
    output_messages = prefix + [final_answer]

    # Length check
    if estimate_tokens(output_messages) > cutoff_len:
        # Try truncating middle: keep system + user + last 3 tool rounds + answer
        sys_msgs = [m for m in prefix if m['role'] == 'system']
        user_msgs = [m for m in prefix if m['role'] == 'user']
        # Last 4 messages of prefix (2 tool calls + 2 tool results)
        tail_prefix = prefix[-4:] if len(prefix) >= 4 else prefix[-2:]
        truncated_prefix = sys_msgs + user_msgs[:1] + tail_prefix
        output_messages = truncated_prefix + [final_answer]
        if estimate_tokens(output_messages) > cutoff_len:
            return None

    return {
        'messages': output_messages,
        'tools': tools,
        'sample_type': 'final_answer',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cutoff_len', type=int, default=CUTOFF_LEN)
    args = parser.parse_args()

    with open(args.input) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    print(f"Input: {len(samples)} samples from {args.input}")

    results = []
    n_skip_no_answer = 0
    n_skip_too_long = 0

    for s in samples:
        result = extract_final_answer_sample(s, args.cutoff_len)
        if result is None:
            # Check why it was skipped
            msgs = s.get('messages', [])
            has_final = any(
                m.get('role') == 'assistant' and m.get('content') and not m.get('tool_calls')
                for m in msgs
            )
            if not has_final:
                n_skip_no_answer += 1
            else:
                n_skip_too_long += 1
            continue
        results.append(result)

    print(f"Extracted: {len(results)} final answer samples")
    print(f"Skipped (no final answer): {n_skip_no_answer}")
    print(f"Skipped (too long): {n_skip_too_long}")

    if results:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"Saved to: {args.output}")
    else:
        print("No samples extracted.")


if __name__ == '__main__':
    main()
