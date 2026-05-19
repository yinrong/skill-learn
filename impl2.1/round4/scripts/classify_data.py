#!/usr/bin/env python3
"""
classify_data.py
Analyzes all ws_*.jsonl and grpo_*.jsonl files in round4/data/,
produces a classification report and merges ws files into ws_all.jsonl.
"""

import json
import re
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
REPORTS_DIR = SCRIPT_DIR.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

WS_FILES = sorted(
    p for p in DATA_DIR.glob("ws_*.jsonl") if p.name != "ws_all.jsonl"
)
GRPO_FILES = sorted(DATA_DIR.glob("grpo_*.jsonl"))

# ── Helpers ────────────────────────────────────────────────────────────────

def channel_from_filename(stem: str) -> str:
    """Map filename stem to channel label."""
    mapping = {
        "ws_raw":   "feishu",
        "grpo_raw": "feishu",
        "ws_web":   "web",
        "grpo_web": "web",
        "ws_cron":  "cron",
        "grpo_cron": "cron",
        "ws_alarm_crab_dolphin":  "alarm+crab+dolphin",
        "grpo_alarm_crab_dolphin":"alarm+crab+dolphin",
        "ws_alarm":  "alarm",
        "ws_crab":   "crab",
        "ws_dolphin":"dolphin",
        "grpo_alarm":  "alarm",
        "grpo_crab":   "crab",
        "grpo_dolphin":"dolphin",
    }
    for key, val in mapping.items():
        if stem == key:
            return val
    # fallback: strip ws_ / grpo_ prefix
    for prefix in ("ws_", "grpo_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def extract_skill(messages: list) -> str:
    """Extract skill name from the first user message `name: <skill>` line."""
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        match = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        break  # only check first user message
    return "unknown"


def token_estimate(record: dict) -> int:
    """Rough token count: total chars / 3 (covers CJK + ASCII mix)."""
    raw = json.dumps(record, ensure_ascii=False)
    return max(1, len(raw) // 3)


def tool_call_count(messages: list) -> int:
    """Count assistant turns that make tool calls.

    Supports two formats:
    1. OpenAI-style: message has 'tool_calls' key (list of call objects)
    2. Anthropic content-block style: content is a list with type='tool_use' blocks
    """
    count = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        # Format 1: top-level tool_calls field
        tc = m.get("tool_calls")
        if tc and isinstance(tc, list):
            count += len(tc)
            continue
        # Format 2: content list with tool_use blocks
        content = m.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    count += 1
    return count


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] {path.name} line {lineno}: {e}", file=sys.stderr)
    return records


def percentiles(values: list, ps=(25, 50, 75, 90, 99)) -> dict:
    if not values:
        return {p: 0 for p in ps}
    v = sorted(values)
    n = len(v)
    result = {}
    for p in ps:
        idx = max(0, int(n * p / 100) - 1)
        result[p] = v[min(idx, n - 1)]
    result["max"] = v[-1]
    return result


# ── Analyse WS files ───────────────────────────────────────────────────────

print("=== Analysing ws_*.jsonl ===")

ws_per_file = {}          # filename -> count
ws_skill_counter = Counter()
ws_channel_counter = Counter()
ws_token_values = []
ws_step_values = []
ws_all_records = []
ws_missing = []

for path in WS_FILES:
    if not path.exists():
        ws_missing.append(path.name)
        continue
    records = load_jsonl(path)
    ws_per_file[path.name] = len(records)
    channel = channel_from_filename(path.stem)
    print(f"  {path.name}: {len(records)} samples  [{channel}]")
    for rec in records:
        msgs = rec.get("messages", [])
        skill = extract_skill(msgs)
        ws_skill_counter[skill] += 1
        ws_channel_counter[channel] += 1
        ws_token_values.append(token_estimate(rec))
        ws_step_values.append(tool_call_count(msgs))
        ws_all_records.append(rec)

ws_token_pct = percentiles(ws_token_values)
ws_step_pct  = percentiles(ws_step_values)

# ── Analyse GRPO files ─────────────────────────────────────────────────────

print("\n=== Analysing grpo_*.jsonl ===")

grpo_per_file = {}
grpo_skill_counter = Counter()
grpo_channel_counter = Counter()
grpo_token_values = []
grpo_step_values = []
grpo_total = 0
grpo_missing = []

for path in GRPO_FILES:
    if not path.exists():
        grpo_missing.append(path.name)
        continue
    records = load_jsonl(path)
    grpo_per_file[path.name] = len(records)
    channel = channel_from_filename(path.stem)
    print(f"  {path.name}: {len(records)} samples  [{channel}]")
    for rec in records:
        msgs = rec.get("prompt", rec.get("messages", []))
        skill = extract_skill(msgs)
        grpo_skill_counter[skill] += 1
        grpo_channel_counter[channel] += 1
        grpo_token_values.append(token_estimate(rec))
        grpo_step_values.append(rec.get("step_index", 0))
        grpo_total += 1

grpo_token_pct = percentiles(grpo_token_values)
grpo_total_steps_pct = percentiles(
    [rec.get("total_steps", 0)
     for path in GRPO_FILES if path.exists()
     for rec in load_jsonl(path)]
)

# ── Merge ws_all.jsonl ─────────────────────────────────────────────────────

ws_all_path = DATA_DIR / "ws_all.jsonl"
with open(ws_all_path, "w", encoding="utf-8") as f:
    for rec in ws_all_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"\nMerged {len(ws_all_records)} records → {ws_all_path}")

# ── Build Markdown report ──────────────────────────────────────────────────

def pct_row(d: dict) -> str:
    return (f"p25={d[25]:,}  p50={d[50]:,}  p75={d[75]:,}  "
            f"p90={d[90]:,}  p99={d[99]:,}  max={d['max']:,}")


lines = []
lines.append("# Evidence Data Classification Report")
lines.append("")
lines.append("Generated by `scripts/classify_data.py`.")
lines.append("")

# ── 1. Per-file counts ──
lines.append("## 1. Per-file Sample Counts")
lines.append("")
lines.append("### SFT (ws_*.jsonl)")
lines.append("")
lines.append("| File | Samples | Channel |")
lines.append("|------|--------:|---------|")
for fname, cnt in sorted(ws_per_file.items()):
    stem = fname.replace(".jsonl", "")
    lines.append(f"| `{fname}` | {cnt:,} | {channel_from_filename(stem)} |")
lines.append(f"| **Total** | **{sum(ws_per_file.values()):,}** | — |")
if ws_missing:
    lines.append("")
    lines.append(f"_Missing / not yet generated: {', '.join(ws_missing)}_")
lines.append("")

lines.append("### GRPO (grpo_*.jsonl)")
lines.append("")
lines.append("| File | Step-samples | Channel |")
lines.append("|------|------------:|---------|")
for fname, cnt in sorted(grpo_per_file.items()):
    stem = fname.replace(".jsonl", "")
    lines.append(f"| `{fname}` | {cnt:,} | {channel_from_filename(stem)} |")
lines.append(f"| **Total** | **{sum(grpo_per_file.values()):,}** | — |")
if grpo_missing:
    lines.append("")
    lines.append(f"_Missing / not yet generated: {', '.join(grpo_missing)}_")
lines.append("")

# ── 2. Skill distribution ──
lines.append("## 2. Skill Distribution (SFT)")
lines.append("")
lines.append("| Skill | Count | % |")
lines.append("|-------|------:|--:|")
total_ws = sum(ws_skill_counter.values())
for skill, cnt in ws_skill_counter.most_common():
    pct = 100 * cnt / total_ws if total_ws else 0
    lines.append(f"| `{skill}` | {cnt} | {pct:.1f}% |")
lines.append("")

lines.append("### Skill Distribution (GRPO step-samples)")
lines.append("")
lines.append("| Skill | Count | % |")
lines.append("|-------|------:|--:|")
total_grpo = sum(grpo_skill_counter.values())
for skill, cnt in grpo_skill_counter.most_common():
    pct = 100 * cnt / total_grpo if total_grpo else 0
    lines.append(f"| `{skill}` | {cnt} | {pct:.1f}% |")
lines.append("")

# ── 3. Channel distribution ──
lines.append("## 3. Channel Distribution")
lines.append("")
lines.append("| Channel | SFT samples | GRPO step-samples |")
lines.append("|---------|------------:|------------------:|")
all_channels = sorted(set(list(ws_channel_counter.keys()) + list(grpo_channel_counter.keys())))
for ch in all_channels:
    lines.append(f"| {ch} | {ws_channel_counter.get(ch, 0):,} | {grpo_channel_counter.get(ch, 0):,} |")
lines.append("")

# ── 4. Sequence length distribution ──
lines.append("## 4. Sequence Length Distribution (estimated tokens ≈ chars/3)")
lines.append("")
lines.append("### SFT")
lines.append("")
lines.append(f"```\n{pct_row(ws_token_pct)}\n```")
lines.append("")
lines.append("### GRPO step-samples")
lines.append("")
lines.append(f"```\n{pct_row(grpo_token_pct)}\n```")
lines.append("")

# ── 5. Step count distribution ──
lines.append("## 5. Tool-call / Step Count Distribution")
lines.append("")
lines.append("### SFT — tool calls per conversation")
lines.append("")
lines.append(f"```\n{pct_row(ws_step_pct)}\n```")
lines.append("")

lines.append("### GRPO — total_steps per step-sample")
lines.append("")
lines.append(f"```\n{pct_row(grpo_total_steps_pct)}\n```")
lines.append("")

# breakdown table
step_hist = Counter(ws_step_values)
lines.append("#### SFT Step-count Histogram")
lines.append("")
lines.append("| Steps (tool calls) | Conversations |")
lines.append("|--------------------|--------------|")
for k in sorted(step_hist.keys()):
    lines.append(f"| {k} | {step_hist[k]} |")
lines.append("")

# ── 6. Merged dataset recommendation ──
lines.append("## 6. Merged Dataset Recommendation")
lines.append("")
lines.append(f"All existing `ws_*.jsonl` files have been merged into `data/ws_all.jsonl` "
             f"(**{len(ws_all_records):,} total SFT samples**).")
lines.append("")
lines.append("### Recommendation")
lines.append("")
lines.append("| Decision | Rationale |")
lines.append("|----------|-----------|")
lines.append("| Include all ws_* sources | Each source covers distinct skills/channels; no obvious overlap |")
lines.append("| Keep ws_all.jsonl as canonical SFT train file | Single file simplifies downstream training scripts |")
lines.append(f"| Total SFT samples: **{len(ws_all_records):,}** | Adequate for a fine-tuning run; consider oversampling minority skills |")
lines.append(f"| Total GRPO step-samples: **{sum(grpo_per_file.values()):,}** | Used for reward-model / GRPO training; not merged here |")
lines.append("")

# skill balance note
most_common_skill, mcs_count = ws_skill_counter.most_common(1)[0]
least_common_skill, lcs_count = ws_skill_counter.most_common()[-1]
lines.append(f"**Skill balance note**: most common skill `{most_common_skill}` has "
             f"{mcs_count} samples; least common `{least_common_skill}` has {lcs_count}. "
             f"Consider upsampling underrepresented skills if imbalance is >5×.")
lines.append("")

# ── Write report ──
report_path = REPORTS_DIR / "evidence_data_classification.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Report written → {report_path}")
