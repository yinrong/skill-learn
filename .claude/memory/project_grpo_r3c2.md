---
name: GRPO R3-C2 training status
description: R3-C failure root causes, R3-C2 fixes applied, current training status
type: project
originSessionId: 6b507141-787a-4161-817d-c76dbc56b9af
---
R3-C failed (F1=0.003) due to reward hacking — model predicted empty violations + correct CPK for avg reward 0.6.

R3-C2 fixes:
1. Removed `end_think < 50` gate (Qwen3 model always outputs empty `<think>` prefix; real reasoning is after `</think>`)
2. Added empty prediction penalty: violations=[] but GT non-empty → -0.3
3. Increased KL beta: 0.01 → 0.05
4. Single GPU (CUDA_VISIBLE_DEVICES=0), num_generations=2

**Reward confirmed working 2026-05-12**: step 1 shows `mean=1.0000 vals=[0.9, 1.1]` — non-zero rewards, training is valid.

Training running: 513 steps, ~250s/step, ~35h total. Estimated completion ~10:20 AM 2026-05-14.

**Why:** R3-C had reward hacking AND all GRPO reward=0 due to `end_think` check blocking all completions. Both fixed in R3-C2.
**How to apply:** After training completes, run `bash round3/eval_grpo_r3c2.sh`. Then update INTERVIEW_QA.md and SKILL_INTERNALIZATION_METHODOLOGY_v2.md with results and re-upload to Feishu.
