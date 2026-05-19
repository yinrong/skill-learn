---
name: Qwen3 thinking mode behavior in GRPO
description: enable_thinking=True vs False in apply_chat_template, and how to actually get real thinking chains in GRPO training
type: feedback
originSessionId: 6b507141-787a-4161-817d-c76dbc56b9af
---
`apply_chat_template` with `enable_thinking=True` gives a NORMAL prompt ending in `...assistant\n` — the model then decides whether to think.

`apply_chat_template` with `enable_thinking=False` inserts `<think>\n\n</think>\n\n` in the prompt — this SUPPRESSES thinking (model outputs empty think block).

**Key finding (experimentally confirmed on expYYY-merged, 2026-05-12):**
- The expYYY-merged model ALWAYS outputs `<think>\n\n</think>` (empty think, position 0) as a prefix, then puts ALL reasoning in the answer section after `</think>`.
- Appending `<think>\n` manually to the prompt makes it WORSE — model generates `</think>` at position 0 (immediately closes the block).

**How to apply:**
- Do NOT check for `</think>` position in GRPO reward function for this model.
- The model generates valid structured answers in the post-`</think>` section; `extract_violations()` and `extract_cpk()` work correctly on the full completion.
- Reward function should evaluate task quality (F1, CPK accuracy) without any thinking-format gate.
- Empty prediction penalty (-0.3) is still needed to block the "always predict empty" shortcut.
