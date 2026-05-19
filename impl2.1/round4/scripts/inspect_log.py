#!/usr/bin/env python3
"""
日志阅读小工具

用法：
  python scripts/inspect_log.py data/ws_all.jsonl          # 查看第 1 条
  python scripts/inspect_log.py data/ws_all.jsonl 5        # 查看第 5 条
  python scripts/inspect_log.py data/ws_all.jsonl 5 10     # 查看第 5-10 条
"""
import json, sys
from pathlib import Path


def separator(label="", width=72):
    if label:
        pad = max(0, width - len(label) - 4)
        print(f"\n{'─'*2} {label} {'─'*pad}")
    else:
        print("─" * width)


def print_message(i, m):
    role = m.get("role", "?")
    content = m.get("content") or ""
    tool_calls = m.get("tool_calls") or []
    tool_call_id = m.get("tool_call_id", "")

    role_label = {
        "system":    "SYSTEM",
        "user":      "USER",
        "assistant": "ASSISTANT",
        "tool":      "TOOL",
    }.get(role, role.upper())

    header = f"[{i}] {role_label}"
    if tool_call_id:
        header += f"  (call_id={tool_call_id})"
    separator(header)

    # tool_calls（工具调用指令）
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            print(f"  ▶ {fn.get('name', '?')}")
            args_raw = fn.get("arguments", "")
            try:
                args = json.loads(args_raw)
                print(f"    {json.dumps(args, ensure_ascii=False, indent=4)}")
            except Exception:
                print(f"    {args_raw}")

    # content（\n 真实换行）
    if content:
        print(content)


def load_sample(path, idx):
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if i == idx:
                return json.loads(line)
    return None


def show(path, start, end):
    total = sum(1 for _ in open(path))
    end = min(end, total)

    for idx in range(start, end + 1):
        sample = load_sample(path, idx)
        if sample is None:
            print(f"第 {idx} 条不存在（文件共 {total} 条）")
            continue

        msgs = sample.get("messages", [])
        tools = sample.get("tools", [])

        separator(f"第 {idx}/{total} 条  │  {len(msgs)} 条消息  │  {len(tools)} 个工具")

        # 工具摘要
        if tools:
            names = [t.get("function", {}).get("name", "?") for t in tools]
            print(f"可用工具: {', '.join(names)}")

        for i, m in enumerate(msgs):
            print_message(i, m)

        print()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    path = Path(args[0])
    if not path.exists():
        print(f"文件不存在: {path}")
        sys.exit(1)

    start = int(args[1]) if len(args) >= 2 else 1
    end   = int(args[2]) if len(args) >= 3 else start

    show(path, start, end)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
