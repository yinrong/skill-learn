"""
fix_existing.py — Add thinking blocks to existing log samples (from ws_raw.jsonl).

For each sample:
  1. Identify the skill name from the user message (name: field after ## 技能指引)
  2. For each assistant message that has tool_calls but empty/null content:
     - Build context: everything before this message
     - Ask Claude to write a <think>...</think> block explaining the reasoning
     - Insert it as the content field
  3. For the final assistant text message:
     - If vague closing (no real data / digits), mark sample as invalid and skip
  4. Save valid samples to output file

Usage:
  python fix_existing.py \\
    --input /path/to/ws_raw.jsonl \\
    --output /path/to/fixed.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anthropic
import os

# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
)

MODEL = "ppio/pa/claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VAGUE_CLOSINGS = [
    "数据已整理如上",
    "报告已发送",
    "已发送",
    "已整理",
    "以上为",
    "如上所示",
    "请查收",
    "已完成",
    "已为您查询",
    "已帮您",
    "请参考以上",
    "以上数据",
]


def extract_skill_name(sample: dict) -> str:
    """Extract skill name from the first user message (name: field after ## 技能指引)."""
    for m in sample.get("messages", []):
        if m.get("role") == "user":
            content = m.get("content") or ""
            if isinstance(content, str):
                names = re.findall(r"^name:\s*(.+)$", content, re.MULTILINE)
                if names:
                    return names[0].strip()
    return "(unknown)"


def extract_skill_doc(sample: dict) -> str:
    """Extract the skill doc section from the first user message."""
    for m in sample.get("messages", []):
        if m.get("role") == "user":
            content = m.get("content") or ""
            if isinstance(content, str) and "## 技能指引" in content:
                # Return everything up to ## 用户问题 (or full content if not present)
                delim = re.search(r"\n##\s*用户问题\s*\n", content)
                if delim:
                    return content[: delim.start()].strip()
                return content.strip()
    return ""


def is_vague_closing(content: str) -> bool:
    """Return True if the final answer is a vague closing without real data."""
    if not content:
        return True
    stripped = content.strip()
    # Must contain at least one digit to have real data
    if not any(c.isdigit() for c in stripped):
        return True
    # Check for short vague phrases
    if len(stripped) <= 30:
        return True
    for phrase in VAGUE_CLOSINGS:
        if phrase in stripped and not any(c.isdigit() for c in stripped):
            return True
    return False


def has_real_final_answer(messages: list[dict]) -> bool:
    """Return True if the last non-tool-call assistant message has real data."""
    for m in reversed(messages):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            content = m.get("content") or ""
            if is_vague_closing(content):
                return False
            return len(content) > 50 and any(c.isdigit() for c in content)
    return False


def messages_to_context_str(messages: list[dict]) -> str:
    """Render a message list to a readable string for the Claude prompt."""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content") or ""
        tool_calls = m.get("tool_calls") or []

        if role == "system":
            parts.append(f"[SYSTEM]\n{content}")
        elif role == "user":
            parts.append(f"[USER]\n{content}")
        elif role == "assistant":
            if tool_calls:
                tc_str = json.dumps(tool_calls, ensure_ascii=False, indent=2)
                parts.append(f"[ASSISTANT tool_calls]\n{content}\n{tc_str}")
            else:
                parts.append(f"[ASSISTANT]\n{content}")
        elif role == "tool":
            parts.append(f"[TOOL RESULT]\n{content}")
        else:
            parts.append(f"[{role.upper()}]\n{content}")

    return "\n\n---\n\n".join(parts)


def ask_claude_for_think(
    context_messages: list[dict],
    skill_doc: str,
    tool_calls: list[dict],
) -> str:
    """
    Ask Claude to produce a <think>...</think> block for the given tool call,
    given the conversation context and the skill doc.
    Returns the raw thinking content (without the tags).
    """
    context_str = messages_to_context_str(context_messages)
    tool_calls_str = json.dumps(tool_calls, ensure_ascii=False, indent=2)

    prompt = f"""You are helping to add reasoning traces to training data for a manufacturing data assistant.

## Skill Documentation
{skill_doc}

## Conversation So Far
{context_str}

## Next Tool Call (what the assistant will do next)
{tool_calls_str}

Based on the conversation so far and the skill doc above, write the internal reasoning that should appear in a `<think>` block immediately before this tool call.

The think block should:
- Reference relevant parts of the skill doc (e.g., model_id mappings, step procedures)
- Explain why this specific tool call is needed
- Show how parameters were derived (e.g., "user asked for today → shift_date=2026-05-14")
- For model_id: if from a prior tool response, cite "from get_object_type_metrics response"; if from the skill doc mapping table, cite "from skill doc mapping: model_id=XXXX"

Output ONLY the thinking content, without the <think> tags, and without any other text."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_sample(sample: dict) -> dict | None:
    """
    Add think blocks to tool-calling assistant messages.
    Returns the fixed sample, or None if the sample should be discarded.
    """
    messages = sample.get("messages", [])

    # Check final answer quality first — skip samples with vague closings
    if not has_real_final_answer(messages):
        return None

    skill_doc = extract_skill_doc(sample)
    new_messages: list[dict] = []

    for idx, m in enumerate(messages):
        if m.get("role") != "assistant":
            new_messages.append(m)
            continue

        tool_calls = m.get("tool_calls") or []
        content = m.get("content") or ""

        # Only patch assistant messages that have tool_calls but empty/null content
        if tool_calls and not content.strip():
            context_so_far = messages[:idx]
            try:
                think_content = ask_claude_for_think(
                    context_so_far, skill_doc, tool_calls
                )
                new_content = f"<think>{think_content}</think>"
            except Exception as e:
                print(
                    f"  [WARN] Claude API error for msg idx={idx}: {e}",
                    file=sys.stderr,
                )
                new_content = content  # keep original (empty) rather than crashing

            new_m = dict(m)
            new_m["content"] = new_content
            new_messages.append(new_m)
        else:
            new_messages.append(m)

    fixed = dict(sample)
    fixed["messages"] = new_messages
    return fixed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Add thinking blocks to existing log samples via Claude."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file (ws_raw.jsonl format)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file for fixed samples",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_saved = 0
    n_skipped_vague = 0
    n_skipped_error = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            n_total += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno}: JSON parse error: {e}", file=sys.stderr)
                n_skipped_error += 1
                continue

            skill_name = extract_skill_name(sample)
            print(f"[{lineno}] skill={skill_name} ...", end=" ", flush=True)

            fixed = process_sample(sample)
            if fixed is None:
                print("SKIP (vague final answer)")
                n_skipped_vague += 1
                continue

            fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")
            n_saved += 1
            print("OK")

    print(f"\n{'='*50}")
    print(f"Input:         {n_total} samples")
    print(f"Saved:         {n_saved}")
    print(f"Skipped vague: {n_skipped_vague}")
    print(f"Skipped error: {n_skipped_error}")
    print(f"Output:        {args.output}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
