"""
OpenAI chat format → LlamaFactory sharegpt format 转换。

LlamaFactory qwen3 template 要求：
  conversations:
    - from: "system" / "human" / "gpt" / "function_call" / "observation"
      value: content_str
  tools: JSON string of tool list

OpenAI format → sharegpt mapping:
  role:system    → from:system
  role:user      → from:human
  role:assistant + tool_calls → from:function_call, value={"name":..,"arguments":{..}}
  role:assistant (no tools) → from:gpt
  role:tool      → from:observation

用法：
  python round4/scripts/to_llamafactory.py \
    --input round4/data/train_ns.jsonl \
    --output round4/data/train_lf.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

def strip_system_injections(content: str) -> str:
    """Remove <system-reminder> blocks and similar system injections from user messages."""
    # Remove <system-reminder>...</system-reminder> blocks
    content = re.sub(r'<system-reminder>.*?</system-reminder>', '', content, flags=re.DOTALL)
    # Remove (以下内容为内部信息...) context injections
    content = re.sub(r'\(以下内容为内部信息.*?\n\n', '', content, flags=re.DOTALL)
    return content.strip()


def is_system_injection(content: str) -> bool:
    """Return True if this user message is actually a system-injected context, not real user input."""
    if not content:
        return True
    s = content.strip()
    # Time context injection marker
    if s.startswith('(以下内容为内部信息'):
        return True
    # System reminder block
    if s.startswith('<system-reminder>'):
        return True
    # Pure system-reminder with no other content
    if re.match(r'^<system-reminder>', s):
        return True
    return False


def clean_messages(messages: list) -> list:
    """Remove system-injected user messages and merge consecutive real user messages.

    Production system injects time context + skill menu as user-role messages mid-conversation.
    For ns-format (no-skill) training:
    - Remove <system-reminder> (skill menu) injections entirely
    - Remove (以下内容为内部信息...) time context injections
    - Merge any remaining consecutive user messages
    """
    # Step 1: Remove injection messages
    filtered = []
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content') or ''
            # Strip system-injection sub-blocks from content
            cleaned = strip_system_injections(content)
            if is_system_injection(content):
                # Skip this message entirely (it's only an injection)
                continue
            if cleaned != content:
                # Content had injections stripped; use cleaned version
                msg = dict(msg)
                msg['content'] = cleaned
        filtered.append(msg)

    # Step 2: Merge consecutive user messages
    merged = []
    i = 0
    while i < len(filtered):
        msg = filtered[i]
        if msg.get('role') == 'user':
            user_parts = []
            while i < len(filtered) and filtered[i].get('role') == 'user':
                content = filtered[i].get('content') or ''
                if content.strip():
                    user_parts.append(content.strip())
                i += 1
            if user_parts:
                merged.append({'role': 'user', 'content': '\n'.join(user_parts),
                                'tool_calls': None})
        else:
            merged.append(filtered[i])
            i += 1
    return merged


def openai_to_sharegpt(sample: dict) -> dict | None:
    """Convert OpenAI chat format to LlamaFactory sharegpt format."""
    messages = sample.get('messages', [])
    tools = sample.get('tools', [])

    # Remove system-injected messages and merge consecutive user messages
    messages = clean_messages(messages)

    conversations = []
    system_content = ""

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content') or ''
        tool_calls = msg.get('tool_calls')

        if role == 'system':
            system_content = content
            # LlamaFactory can have system as first conversation turn
            conversations.append({'from': 'system', 'value': content})

        elif role == 'user':
            conversations.append({'from': 'human', 'value': content})

        elif role == 'assistant':
            if tool_calls:
                # Convert tool_calls to LlamaFactory function_call format
                # LlamaFactory expects: {"name": "...", "arguments": {...}} or list of them
                fc_list = []
                for tc in tool_calls:
                    func = tc.get('function', {})
                    name = func.get('name', '')
                    args = func.get('arguments', '{}')
                    # Parse arguments if string
                    if isinstance(args, str):
                        try:
                            args_obj = json.loads(args)
                        except json.JSONDecodeError:
                            args_obj = {}
                    else:
                        args_obj = args
                    fc_list.append({'name': name, 'arguments': args_obj})

                # LlamaFactory function_call value: single dict or list
                if len(fc_list) == 1:
                    fc_value = json.dumps(fc_list[0], ensure_ascii=False)
                else:
                    fc_value = json.dumps(fc_list, ensure_ascii=False)

                # Include any content (e.g., <think>...</think>) before tool call
                if content:
                    # Prepend thinking content to function_call message
                    # LlamaFactory FunctionFormatter handles <think> blocks
                    fc_value_with_think = content + fc_value if '<think>' in content else fc_value
                    conversations.append({'from': 'function_call', 'value': fc_value_with_think})
                else:
                    conversations.append({'from': 'function_call', 'value': fc_value})
            else:
                # Regular assistant message (final answer)
                conversations.append({'from': 'gpt', 'value': content})

        elif role == 'tool':
            # Tool result → observation
            conversations.append({'from': 'observation', 'value': content})

    if not conversations:
        return None

    # Post-process: enforce LlamaFactory's strict alternating role requirement.
    # After stripping system, positions should alternate: 0=human/obs, 1=gpt/fc, 2=human/obs, ...
    # Remove any human/observation turn found at an odd position (should be gpt/function_call).
    # These are always system re-injections of the original question before the final answer.
    non_system = [(i, c) for i, c in enumerate(conversations) if c['from'] != 'system']
    system_turns = [(i, c) for i, c in enumerate(conversations) if c['from'] == 'system']
    cleaned_non_system = []
    for idx, c in non_system:
        expected_pos = len(cleaned_non_system) % 2
        is_odd_role = c['from'] in ('human', 'observation')
        is_even_role = c['from'] in ('gpt', 'function_call')
        if expected_pos == 0 and is_even_role:
            # Missing human turn — skip this even turn
            continue
        if expected_pos == 1 and is_odd_role:
            # human/observation at wrong position (system re-injection) — skip it
            continue
        cleaned_non_system.append(c)

    # Rebuild conversations with system turns first
    conversations = [c for _, c in system_turns] + cleaned_non_system

    # Must end on an even-position turn (assistant/function_call = len%2==0 after removing system)
    non_sys_final = [c for c in conversations if c['from'] != 'system']
    if len(non_sys_final) % 2 != 0:
        # Last turn is at odd position — remove it if it's human/observation
        if non_sys_final and non_sys_final[-1]['from'] in ('human', 'observation'):
            conversations = [c for c in conversations if c != non_sys_final[-1]]

    # Tools: LlamaFactory expects a JSON string
    tools_str = json.dumps(tools, ensure_ascii=False) if tools else ""

    return {
        'conversations': conversations,
        'tools': tools_str,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip = 0, 0
    with open(args.input) as fin, open(args.output, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            sample = json.loads(line)
            converted = openai_to_sharegpt(sample)
            if converted is None:
                n_skip += 1
                continue
            fout.write(json.dumps(converted, ensure_ascii=False) + '\n')
            n_ok += 1

    print(f"Converted: {n_ok}, Skipped: {n_skip} → {args.output}")


if __name__ == '__main__':
    main()
