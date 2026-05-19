"""
GRPO 格式 → SFT prefix 格式转换，同时应用 ws→ns 转换。

GRPO 样本结构：
  prompt: [system, user(含skill), prev_turns...]
  ground_truth: {tool_name: str, arguments: str}
  tools: [...]

SFT 输出格式（LlamaFactory openai 格式）：
  messages: [system, user(无skill), prev_turns..., assistant+tool_call]
  tools: [...]

用法：
  python round4/scripts/grpo_to_sft.py \
    --input round4/data/grpo_raw.jsonl \
    --output_train round4/data/train_ns.jsonl \
    --output_test round4/data/test_ns.jsonl \
    --cutoff_len 6144
"""
from __future__ import annotations

import argparse
import json
import re
import random
from collections import defaultdict
from pathlib import Path

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text) // 4


def strip_skill_from_user_message(content: str) -> str:
    """去掉 user message 中的技能指引块，只保留用户问题。"""
    if not isinstance(content, str):
        return content

    # Pattern: "## 用户问题\n" or "当前问题：" separator
    delim = re.search(r'\n##\s*用户问题\s*\n|当前问题[：:]\s*', content)
    if delim:
        return content[delim.end():].strip()

    # Pattern: starts with skill YAML block
    if '## 技能指引' in content or content.strip().startswith('---\nname:'):
        # Find last YAML closing --- and take everything after
        # Look for question marker after skill blocks
        question_re = re.search(
            r'(?:^|\n)(?!---\n)(?!name:)(?!version:)(?!description:)(?!content:)'
            r'(.+?)$',
            content,
            re.MULTILINE
        )
        # Simpler: split on \n\n after the YAML block ends
        # Find the end of the last --- block
        blocks = re.split(r'\n---\n', content)
        # Non-YAML blocks don't start with name:/version:/description:
        for block in reversed(blocks):
            if block.strip() and not re.match(
                r'^\s*(name:|version:|description:|content:|##\s*技能指引)',
                block.strip()
            ):
                cleaned = block.strip()
                if len(cleaned) > 5:
                    return cleaned

    return content


def get_skill_name(messages: list) -> str:
    for m in messages:
        if m.get('role') == 'user':
            content = m.get('content', '')
            if isinstance(content, str):
                names = re.findall(r'^name:\s*(.+)$', content, re.MULTILINE)
                if names:
                    return names[0].strip()
    return '(no skill)'


def grpo_to_sft(sample: dict) -> dict | None:
    """
    Convert GRPO sample to SFT format.

    Returns None if ground_truth cannot be converted to a valid tool_calls message.
    """
    prompt = sample.get('prompt', [])
    gt = sample.get('ground_truth', {})
    tools = sample.get('tools', [])

    if not prompt or not gt:
        return None

    tool_name = gt.get('tool_name', '')
    arguments = gt.get('arguments', '')
    if not tool_name or not arguments:
        return None

    # Validate arguments is valid JSON
    try:
        if isinstance(arguments, str):
            json.loads(arguments)
        args_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
    except (json.JSONDecodeError, TypeError):
        return None

    # Apply ws→ns: strip skill YAML from first user message
    new_messages = []
    first_user_done = False
    for m in prompt:
        if m.get('role') == 'user' and not first_user_done:
            new_m = dict(m)
            new_m['content'] = strip_skill_from_user_message(m.get('content', ''))
            new_messages.append(new_m)
            first_user_done = True
        else:
            new_messages.append(m)

    # Append the ground truth tool call as the assistant's next message
    tool_call_id = f"call_{hash(tool_name + args_str) % 10**12:012x}"
    gt_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": args_str
            }
        }]
    }
    new_messages.append(gt_message)

    return {
        "messages": new_messages,
        "tools": tools,
    }


def compute_total_tokens(sample: dict) -> int:
    total = 0
    for m in sample.get('messages', []):
        content = str(m.get('content') or '')
        tcs = m.get('tool_calls') or []
        total += count_tokens(content + json.dumps(tcs))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='round4/data/grpo_raw.jsonl')
    parser.add_argument('--output_train', default='round4/data/train_ns.jsonl')
    parser.add_argument('--output_test', default='round4/data/test_ns.jsonl')
    parser.add_argument('--cutoff_len', type=int, default=6144)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    samples = []
    filtered = {'too_long': 0, 'invalid': 0}

    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            sft = grpo_to_sft(raw)
            if sft is None:
                filtered['invalid'] += 1
                continue
            total_tok = compute_total_tokens(sft)
            if total_tok > args.cutoff_len:
                filtered['too_long'] += 1
                continue
            skill = get_skill_name(raw.get('prompt', []))
            samples.append((sft, total_tok, skill))

    print(f"Total GRPO samples: {sum(filtered.values()) + len(samples)}")
    print(f"  Invalid: {filtered['invalid']}")
    print(f"  Too long (> {args.cutoff_len}): {filtered['too_long']}")
    print(f"  Valid: {len(samples)}")

    # Stratified split by skill
    by_skill: dict[str, list] = defaultdict(list)
    for item in samples:
        by_skill[item[2]].append(item)

    print("\nSamples per skill:")
    for sk, items in sorted(by_skill.items(), key=lambda x: -len(x[1])):
        print(f"  {sk}: {len(items)}")

    train_items, test_items = [], []
    for skill, items in by_skill.items():
        random.shuffle(items)
        n_test = max(1, round(len(items) * args.test_ratio))
        n_test = min(n_test, len(items) - 1)
        test_items.extend(items[:n_test])
        train_items.extend(items[n_test:])

    # Stats
    train_tokens = sorted([t for _, t, _ in train_items])
    n = len(train_tokens)
    print(f"\nTrain token length stats (n={n}):")
    for pct in [50, 75, 90, 95, 99]:
        idx = min(int(n * pct / 100), n - 1)
        print(f"  p{pct}: {train_tokens[idx]}")

    # Write
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_train, 'w') as f:
        for s, _, _ in train_items:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    with open(args.output_test, 'w') as f:
        for s, _, _ in test_items:
            # Keep original GRPO for reference + add converted SFT
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\nTrain: {len(train_items)} → {args.output_train}")
    print(f"Test:  {len(test_items)} → {args.output_test}")
    print("\nTest skill distribution:")
    test_skill_cnt = defaultdict(int)
    for _, _, sk in test_items:
        test_skill_cnt[sk] += 1
    for sk, cnt in sorted(test_skill_cnt.items(), key=lambda x: -x[1]):
        print(f"  {sk}: {cnt}")


if __name__ == '__main__':
    main()
