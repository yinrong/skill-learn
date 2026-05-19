"""
convert_ns_lf.py — Combined ns conversion + LF (LlamaFactory sharegpt) conversion.

Step 1: ns conversion
  - Strip the skill doc (## 技能指引 ... ## 用户问题) from the first user message
  - Filter out internal injection messages (content starts with "(以下内容为内部信息"
    or "<system-reminder>")
  - Keep everything else (system, tool calls, tool responses, final answer)

Step 2: LF (LlamaFactory sharegpt) conversion
  - Convert each message to sharegpt format (human/gpt/function_call/observation/system)
  - Thinking content in assistant messages is prepended to the function_call value

Step 3: Validation
  - Verify no observation → human pattern without a gpt turn between them

Step 4: Stats
  - Print sample counts at each stage
  - Print CRITICAL warning with LF file line count for LlamaFactory verification

Usage:
  python convert_ns_lf.py \\
    --input combined.jsonl \\
    --ns_output train_ns.jsonl \\
    --lf_output train_lf.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Step 1: ns conversion
# ---------------------------------------------------------------------------

def strip_skill_doc(content: str) -> str:
    """
    Remove the ## 技能指引 section from a user message, keeping only the
    text after ## 用户问题.

    If there is no ## 用户问题 marker the full message is returned unchanged
    (it is already ns format or has no skill doc).
    """
    if not isinstance(content, str):
        return content

    # Primary pattern: "## 用户问题" separator
    delim = re.search(r"\n##\s*用户问题\s*\n", content)
    if delim:
        return content[delim.end():].strip()

    # Also handle "当前问题：" style
    delim2 = re.search(r"当前问题[：:]\s*", content)
    if delim2:
        return content[delim2.end():].strip()

    # If the message starts with ## 技能指引 but has no ## 用户问题,
    # try to strip the YAML block and return what follows.
    if content.strip().startswith("## 技能指引") or content.strip().startswith("---\nname:"):
        parts = re.split(r"\n---\n", content)
        remaining = []
        in_yaml = False
        for i, part in enumerate(parts):
            if i == 0 and (
                part.startswith("## 技能指引") or part.strip() == ""
            ):
                in_yaml = True
                continue
            if in_yaml and re.match(r"^\s*name:", part.strip()):
                continue
            in_yaml = False
            remaining.append(part)
        result = "\n---\n".join(remaining).strip()
        if result:
            return result

    return content


def is_internal_injection(content: str) -> bool:
    """
    Return True if this message is an internal system injection that should
    be removed from the training sample.
    """
    if not content:
        return False
    s = content.strip()
    return s.startswith("(以下内容为内部信息") or s.startswith("<system-reminder>")


def convert_to_ns(sample: dict) -> dict:
    """
    Convert a ws-format sample to ns format:
      - Strip skill doc from the first user message
      - Remove internal injection messages
    """
    messages = sample.get("messages", [])
    new_messages: list[dict] = []
    first_user_seen = False

    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""

        # Filter out internal injection messages (any role)
        if is_internal_injection(content):
            continue

        if role == "user":
            if not first_user_seen:
                # Strip skill doc from the very first user message
                first_user_seen = True
                new_m = dict(m)
                new_m["content"] = strip_skill_doc(content)
                new_messages.append(new_m)
            else:
                new_messages.append(m)
        else:
            new_messages.append(m)

    ns = dict(sample)
    ns["messages"] = new_messages
    return ns


# ---------------------------------------------------------------------------
# Step 2: LF (LlamaFactory sharegpt) conversion
# ---------------------------------------------------------------------------

def to_lf(sample: dict) -> dict | None:
    """
    Convert a sample (ns format, OpenAI messages) to LlamaFactory sharegpt format.

    Key rules for LlamaFactory compatibility:
    - function_call MUST be immediately followed by observation (1-to-1 interleaved)
    - observation → human requires a gpt placeholder between them
    - Parallel tool calls (multiple tool_calls in one assistant message) are
      handled by matching each call to its tool response via tool_call_id,
      then emitting interleaved pairs: fc→obs, fc→obs, ...

    Returns None if the sample is empty after conversion.
    """
    msgs = sample.get("messages", [])
    tools = sample.get("tools", [])
    convs: list[dict] = []

    i = 0
    while i < len(msgs):
        m = msgs[i]
        role = m.get("role", "")
        content = m.get("content") or ""
        tool_calls = m.get("tool_calls") or []

        if role == "system":
            convs.append({"from": "system", "value": content})
            i += 1

        elif role == "user":
            # observation → human requires gpt placeholder
            if convs and convs[-1]["from"] == "observation":
                convs.append({"from": "gpt", "value": "继续..."})
            convs.append({"from": "human", "value": content})
            i += 1

        elif role == "assistant":
            if tool_calls:
                # Do NOT emit thinking content as gpt before function_call —
                # LlamaFactory does not allow gpt → function_call sequence.
                # The thinking is preserved in the ns format; in LF format
                # we drop it for tool-call turns to maintain valid sequences.
                pass  # skip intermediate thinking gpt

                # Look ahead: collect all consecutive tool messages that follow
                j = i + 1
                tool_resp_list = []
                while j < len(msgs) and msgs[j].get("role") == "tool":
                    tool_resp_list.append(msgs[j])
                    j += 1

                # Build tool_call_id → response map
                resp_by_id: dict[str, str] = {}
                for tr in tool_resp_list:
                    tc_id = tr.get("tool_call_id", "")
                    resp_by_id[tc_id] = tr.get("content") or "{}"

                # Emit interleaved function_call → observation pairs
                for tc in tool_calls:
                    tc_id = tc.get("id", "")
                    fn = tc.get("function", {})
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = (
                            json.loads(args_raw)
                            if isinstance(args_raw, str)
                            else args_raw
                        )
                    except json.JSONDecodeError:
                        args = args_raw

                    fc_value = json.dumps(
                        {"name": fn["name"], "arguments": args},
                        ensure_ascii=False,
                    )
                    convs.append({"from": "function_call", "value": fc_value})

                    # Emit the matching observation immediately after
                    obs_content = resp_by_id.get(tc_id) or "{}"
                    convs.append({"from": "observation", "value": obs_content})

                # Skip past the tool messages we already consumed
                i = j

            else:
                # Regular assistant message (final answer or intermediate text)
                convs.append({"from": "gpt", "value": content})
                i += 1

        elif role == "tool":
            # Standalone tool message (not already consumed above) — emit as-is
            convs.append({"from": "observation", "value": content})
            i += 1

        else:
            i += 1

    if not convs:
        return None

    return {
        "conversations": convs,
        "tools": json.dumps(tools, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Step 3: Validation
# ---------------------------------------------------------------------------

def check_no_obs_to_human(lf_sample: dict) -> list[str]:
    """
    Return a list of error descriptions if any observation → human transition
    exists without a gpt/function_call turn between them.
    """
    errors: list[str] = []
    convs = lf_sample.get("conversations", [])
    non_system = [c for c in convs if c["from"] != "system"]

    for i in range(len(non_system) - 1):
        cur = non_system[i]["from"]
        nxt = non_system[i + 1]["from"]
        if cur == "observation" and nxt == "human":
            errors.append(
                f"observation→human at non-system positions {i}→{i+1} "
                f"(values: {non_system[i]['value'][:40]!r} → {non_system[i+1]['value'][:40]!r})"
            )
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combined ns conversion + LF (LlamaFactory sharegpt) conversion "
            "with post-conversion validation."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSONL (ws or mixed format)")
    parser.add_argument("--ns_output", required=True, help="Output path for ns JSONL")
    parser.add_argument("--lf_output", required=True, help="Output path for LF JSONL")
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    Path(args.ns_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.lf_output).parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load and ns-convert
    # -----------------------------------------------------------------------
    ns_samples: list[dict] = []
    n_input = 0

    with open(args.input) as fin:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            n_input += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno}: JSON parse error: {e}", file=sys.stderr)
                continue
            ns_samples.append(convert_to_ns(sample))

    # Write ns output
    with open(args.ns_output, "w") as fout:
        for s in ns_samples:
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_after_ns = len(ns_samples)

    # -----------------------------------------------------------------------
    # Step 2: LF conversion
    # -----------------------------------------------------------------------
    lf_samples: list[dict] = []
    n_lf_skip = 0
    validation_errors: list[str] = []

    for sample in ns_samples:
        lf = to_lf(sample)
        if lf is None:
            n_lf_skip += 1
            continue

        # Step 3: validate no obs → human
        errs = check_no_obs_to_human(lf)
        if errs:
            for err in errs:
                validation_errors.append(err)
            # Still include the sample but warn

        lf_samples.append(lf)

    # Write LF output
    with open(args.lf_output, "w") as fout:
        for s in lf_samples:
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_after_lf = len(lf_samples)

    # -----------------------------------------------------------------------
    # Step 4: Stats
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("convert_ns_lf.py — Conversion Summary")
    print(f"{'='*60}")
    print(f"Input samples:        {n_input}")
    print(f"After ns conversion:  {n_after_ns}  → {args.ns_output}")
    print(f"After LF conversion:  {n_after_lf}  (skipped {n_lf_skip} empty)  → {args.lf_output}")

    if validation_errors:
        print(f"\n[WARN] {len(validation_errors)} observation→human pattern(s) found:")
        for err in validation_errors[:10]:
            print(f"  {err}")
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more")

    # Critical check: actual line count in LF file
    with open(args.lf_output) as f:
        actual_lf_lines = sum(1 for line in f if line.strip())

    print(f"\n{'!'*60}")
    print(
        f"  ⚠  CRITICAL: LF file has {actual_lf_lines} lines — "
        "verify this matches LlamaFactory 'Num examples'"
    )
    print(f"{'!'*60}\n")


if __name__ == "__main__":
    main()
