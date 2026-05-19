---
name: Round 4 training status and resume guide
description: R4-sft-v3 training in progress, next steps after completion, known pitfalls
type: project
originSessionId: ab92bcb3-a038-4acc-8c10-307463441cfb
---
R4-sft-v3 训练于 2026-05-13 13:45 启动，预计 ~16:00 完成（132 steps，3 epochs，cutoff=24576）。

**Why:** R4-sft-v2（cutoff=8192）导致 judge=38.8%，根因是长对话被截断。v3 修复：标准LoRA，cutoff=24576，无量化，无DeepSpeed。

**How to apply:** 从全新会话恢复时，先读 `/home/yinrong/post-train/impl2.1/round4/RESUME.md`，包含完整的 Step 1-5 操作指南（merge → deploy vLLM → trajectory_eval → llm_judge → 更新报告）。

目标：tool_call_F1 ≥ 75% AND judge ≥ 0.75 AND speedup ≥ 2.7×（production 参考 6600ms/step）。

关键陷阱（已修复，下次恢复时核查）：
- vLLM max-model-len 必须 ≥ 16384（已改为 32768 in deploy_vllm.sh）
- judge 模型名必须用 `ppio/pa/claude-sonnet-4-6`（已在 llm_judge.py:42 修正）
- 评估必须用 `test_ns_all_eval_ns.jsonl`（ns格式），不能用含 skill doc 的 ws 版本
