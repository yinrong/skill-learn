"""
ws → ns 格式转换：去掉 user message 中的 ## 技能指引 段落。

这是技能内化方法论的关键步骤：
- ws (with_skill)：user message 含 skill YAML，模型学会"查文档"
- ns (no_skill)：user message 只含用户问题，模型须将技能知识内化到权重中

注意：不因 cutoff_len 丢弃任何样本。cutoff_len 参数仅用于统计，
训练时在 YAML config 中设置 cutoff_len=16384 以覆盖所有样本。

用法：
    python round4/scripts/convert_ws_to_ns.py \
        --input round4/data/ws_raw.jsonl \
        --output_train round4/data/train_ns_v2.jsonl \
        --output_test round4/data/test_ns_v2.jsonl
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
    # Rough fallback: ~4 chars per token
    def count_tokens(text: str) -> int:
        return len(text) // 4


def strip_skill_from_user_message(content: str) -> str:
    """
    去掉 user message 中的技能指引块。

    Vercel/agentscope 格式：
      ## 技能指引
      ---
      name: xxx
      ...
      ---
      content...
      ---

      ## 用户问题
      actual question

    ns 格式只保留 ## 用户问题 之后的内容（或整个 content 如果没有分隔符）。
    """
    if not isinstance(content, str):
        return content

    # Pattern 1: "## 用户问题" or "当前问题：" separator
    delim = re.search(r'\n##\s*用户问题\s*\n|当前问题[：:]\s*', content)
    if delim:
        return content[delim.end():].strip()

    # Pattern 2: "## 技能指引" at the top — remove everything up to the first non-skill section
    if content.strip().startswith('## 技能指引') or content.strip().startswith('---\nname:'):
        # Remove all YAML blocks at the start
        # Find the end of skill YAML blocks (last closing ---)
        # Then take everything after
        # Find last "---\n" that terminates a skill block
        parts = re.split(r'\n---\n', content)
        # YAML blocks come in pairs (opening --- and closing ---)
        # Find where non-YAML content starts
        remaining = []
        in_yaml = False
        for i, part in enumerate(parts):
            if i == 0 and (part.startswith('## 技能指引') or part.strip() == ''):
                in_yaml = True
                continue
            if in_yaml and re.match(r'^\s*name:', part.strip()):
                continue  # This is a YAML block
            in_yaml = False
            remaining.append(part)
        result = '\n---\n'.join(remaining).strip()
        if result:
            return result

    return content


def get_skill_name(sample: dict) -> str:
    """从样本的 user message 中提取技能名（第一个出现的）。"""
    for m in sample.get('messages', []):
        if m.get('role') == 'user':
            content = m.get('content', '')
            if isinstance(content, str):
                names = re.findall(r'^name:\s*(.+)$', content, re.MULTILINE)
                if names:
                    return names[0].strip()
    return '(no skill)'


def compute_total_tokens(sample: dict) -> int:
    """粗估样本总 token 数（prompt + all responses）。"""
    total = 0
    for m in sample.get('messages', []):
        content = m.get('content') or ''
        tcs = m.get('tool_calls') or []
        total += count_tokens(str(content) + json.dumps(tcs))
    return total


def convert_sample(sample: dict) -> dict:
    """将 ws 格式样本转换为 ns 格式（修改 user messages，去掉技能指引）。"""
    ns = dict(sample)
    new_messages = []
    for m in sample.get('messages', []):
        if m.get('role') == 'user':
            new_m = dict(m)
            content = m.get('content', '')
            new_m['content'] = strip_skill_from_user_message(content)
            new_messages.append(new_m)
        else:
            new_messages.append(m)
    ns['messages'] = new_messages
    return ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='round4/data/ws_raw.jsonl')
    parser.add_argument('--output_train', default='round4/data/train_ns_v2.jsonl')
    parser.add_argument('--output_test', default='round4/data/test_ns_v2.jsonl')
    parser.add_argument('--cutoff_len', type=int, default=16384,
                        help='Reference cutoff for stats only — NO samples are dropped')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_per_skill', type=int, default=1,
                        help='Ensure at least this many test samples per skill')
    args = parser.parse_args()

    random.seed(args.seed)

    # Load ALL samples — no length filtering
    samples = []
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            total_tok = compute_total_tokens(s)
            samples.append((s, total_tok, get_skill_name(s)))

    print(f"Loaded {len(samples)} samples (no filtering)")
    print(f"Reference cutoff_len={args.cutoff_len} — all samples kept regardless of length")

    # Group by skill for stratified split
    by_skill: dict[str, list] = defaultdict(list)
    for item in samples:
        by_skill[item[2]].append(item)

    print("\nSamples per skill:")
    for skill, items in sorted(by_skill.items(), key=lambda x: -len(x[1])):
        print(f"  {skill}: {len(items)}")

    # Stratified split: ensure each skill has ≥ min_per_skill in test
    train_items, test_items = [], []
    for skill, items in by_skill.items():
        random.shuffle(items)
        n_test = max(args.min_per_skill, round(len(items) * args.test_ratio))
        n_test = min(n_test, len(items) - 1)  # keep at least 1 in train
        test_items.extend(items[:n_test])
        train_items.extend(items[n_test:])

    # Token length stats for train set
    train_tokens = sorted([t for _, t, _ in train_items])
    n = len(train_tokens)
    if n:
        print(f"\nTrain token length stats (n={n}):")
        for pct in [50, 75, 90, 95, 99, 100]:
            idx = min(int(n * pct / 100), n - 1)
            print(f"  p{pct}: {train_tokens[idx]}")
        long_train = sum(1 for t in train_tokens if t > args.cutoff_len)
        print(f"  Samples > cutoff_len ({args.cutoff_len}): {long_train} ({100*long_train//n}%)")

    # Convert ws → ns and write
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_train, 'w') as f:
        for s, _, _ in train_items:
            f.write(json.dumps(convert_sample(s), ensure_ascii=False) + '\n')

    # Save test in both ws (reference) and ns (evaluation) formats
    with open(args.output_test, 'w') as f:
        for s, _, _ in test_items:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')  # test keeps ws for reference

    # Also save ns version for inference evaluation
    test_ns_path = args.output_test.replace('.jsonl', '_eval_ns.jsonl')
    with open(test_ns_path, 'w') as f:
        for s, _, _ in test_items:
            f.write(json.dumps(convert_sample(s), ensure_ascii=False) + '\n')

    print(f"\nTrain: {len(train_items)} samples → {args.output_train}")
    print(f"Test (ws ref):  {len(test_items)} samples  → {args.output_test}")
    print(f"Test (ns eval): {len(test_items)} samples  → {test_ns_path}")
    print("\nTest skill distribution:")
    test_skill_cnt = defaultdict(int)
    for _, _, sk in test_items:
        test_skill_cnt[sk] += 1
    for sk, cnt in sorted(test_skill_cnt.items(), key=lambda x: -x[1]):
        print(f"  {sk}: {cnt}")


if __name__ == '__main__':
    main()
