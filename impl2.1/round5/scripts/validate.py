"""
validate.py — Validate training samples for Round 5 skill internalization.

A sample is valid if:
  1. Last assistant message (final answer) has content > 50 chars and contains at least one digit
  2. At least one assistant message has a non-empty <think>...</think> block
  3. Type B skills only: all model_ids used in get_idi_model_data calls appear in a preceding
     get_object_type_metrics tool response

Type A skills (have static model_id mapping in skill doc):
  line-operation-skill, line-attendance-query, line-exemption-query,
  lineside-material-query, workstion-kpi-query, 分板机过站明细查询, object-data-query(no-idi)

Type B skills (model_id discovered dynamically via get_object_type_metrics):
  general-kpi-query, equipment-cpk-query

Usage:
  python validate.py --input file.jsonl --skill SKILL_NAME
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


TYPE_A_SKILLS = {
    "line-operation-skill",
    "line-attendance-query",
    "line-exemption-query",
    "lineside-material-query",
    "workstion-kpi-query",
    "分板机过站明细查询",
    "object-data-query",
}

TYPE_B_SKILLS = {
    "general-kpi-query",
    "equipment-cpk-query",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_think(content: str) -> str:
    """Return the text inside the first <think>...</think> block, or ''."""
    if not content:
        return ""
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    return m.group(1) if m else ""


def get_final_assistant_message(messages: list[dict]) -> dict | None:
    """Return the last assistant message that has no tool_calls (the final answer)."""
    for m in reversed(messages):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            return m
    return None


def get_idi_model_ids(messages: list[dict]) -> list[int]:
    """Collect all model_id values passed to get_idi_model_data tool calls."""
    ids: list[int] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name") != "get_idi_model_data":
                continue
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                continue
            model_id = args.get("model_id")
            if model_id is not None:
                try:
                    ids.append(int(model_id))
                except (TypeError, ValueError):
                    pass
    return ids


def get_metrics_model_ids_before(messages: list[dict], before_index: int) -> set[int]:
    """
    Collect all modelId values that appeared in get_object_type_metrics tool responses
    at positions strictly before `before_index`.
    """
    ids: set[int] = set()
    # We walk forward through messages. A tool response (role=tool) immediately follows
    # the assistant message that made the call, so we track the last seen tool name.
    last_tool_name: str | None = None
    for idx, m in enumerate(messages):
        if idx >= before_index:
            break
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                last_tool_name = fn.get("name")
        elif role == "tool":
            if last_tool_name == "get_object_type_metrics":
                content = m.get("content") or ""
                # Parse modelId values from the JSON response
                try:
                    data = json.loads(content) if isinstance(content, str) else content
                except json.JSONDecodeError:
                    data = {}
                for binding_list in _iter_model_bindings(data):
                    mid = binding_list.get("modelId")
                    if mid is not None:
                        try:
                            ids.add(int(mid))
                        except (TypeError, ValueError):
                            pass
            last_tool_name = None
    return ids


def _iter_model_bindings(data: dict):
    """Yield each modelBinding dict from a get_object_type_metrics response."""
    if not isinstance(data, dict):
        return
    inner = data.get("data") or []
    if isinstance(inner, list):
        for entry in inner:
            if isinstance(entry, dict):
                for binding in entry.get("modelBindings") or []:
                    if isinstance(binding, dict):
                        yield binding


def get_metrics_tool_response_index(messages: list[dict], idi_call_index: int) -> int:
    """
    For a get_idi_model_data call at `idi_call_index`, find the index of the
    get_object_type_metrics response that preceded it.  Returns the idi_call_index
    itself as upper bound so everything before that is "preceding".
    """
    return idi_call_index


# ---------------------------------------------------------------------------
# Per-sample validation
# ---------------------------------------------------------------------------

FAILURE_FINAL_TOO_SHORT = "final_answer_too_short_or_no_digit"
FAILURE_NO_THINK_BLOCK = "no_nonempty_think_block"
FAILURE_ORPHAN_MODEL_ID = "type_b_orphan_model_id_not_in_metrics"


def validate_sample(sample: dict, skill: str) -> list[str]:
    """
    Return a list of failure reasons.  Empty list means the sample is valid.
    """
    failures: list[str] = []
    messages = sample.get("messages") or []

    # --- Check 1: final answer quality ---
    final_msg = get_final_assistant_message(messages)
    final_content = (final_msg.get("content") or "") if final_msg else ""
    if len(final_content) <= 50 or not any(c.isdigit() for c in final_content):
        failures.append(FAILURE_FINAL_TOO_SHORT)

    # --- Check 2: at least one non-empty <think> block ---
    has_think = False
    for m in messages:
        if m.get("role") == "assistant":
            think = extract_think(m.get("content") or "")
            if think.strip():
                has_think = True
                break
    if not has_think:
        failures.append(FAILURE_NO_THINK_BLOCK)

    # --- Check 3: Type B model_id integrity ---
    if skill in TYPE_B_SKILLS:
        # For each get_idi_model_data call, the model_id must have appeared in a
        # preceding get_object_type_metrics response.
        for msg_idx, m in enumerate(messages):
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                if fn.get("name") != "get_idi_model_data":
                    continue
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                model_id = args.get("model_id")
                if model_id is None:
                    continue
                try:
                    model_id_int = int(model_id)
                except (TypeError, ValueError):
                    continue
                # Collect all metrics model_ids that appeared before this message
                preceding_ids = get_metrics_model_ids_before(messages, msg_idx)
                if model_id_int not in preceding_ids:
                    reason = (
                        f"{FAILURE_ORPHAN_MODEL_ID}(model_id={model_id_int}, "
                        f"available_from_metrics={sorted(preceding_ids)})"
                    )
                    failures.append(reason)
                    break  # one failure per sample is enough for this check

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate Round 5 training samples."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument(
        "--skill",
        required=True,
        help=(
            "Skill name, e.g. general-kpi-query. "
            "Used to determine Type A vs Type B validation rules."
        ),
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    skill = args.skill.strip()
    if skill not in TYPE_A_SKILLS and skill not in TYPE_B_SKILLS:
        print(
            f"[WARN] Skill '{skill}' not in known Type A or Type B lists. "
            "Applying Type A rules (skipping model_id check)."
        )

    n_valid = 0
    n_invalid = 0
    failure_counts: dict[str, int] = defaultdict(int)

    with open(args.input) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno}: JSON parse error: {e}")
                n_invalid += 1
                failure_counts["json_parse_error"] += 1
                continue

            failures = validate_sample(sample, skill)
            if failures:
                n_invalid += 1
                for reason in failures:
                    # Normalize reason key for the orphan check (strip variable part)
                    key = reason.split("(")[0]
                    failure_counts[key] += 1
            else:
                n_valid += 1

    total = n_valid + n_invalid
    print(f"\n{'='*50}")
    print(f"Skill:   {skill}")
    print(f"Input:   {args.input}")
    print(f"Total:   {total}")
    print(f"Valid:   {n_valid}  ({100*n_valid//total if total else 0}%)")
    print(f"Invalid: {n_invalid}  ({100*n_invalid//total if total else 0}%)")

    if failure_counts:
        print("\nFailure reasons:")
        for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
