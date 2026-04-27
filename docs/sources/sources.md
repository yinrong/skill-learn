# Sources（参考文献清单）

> 说明：本清单为 `doc/post-train/index.md` 的事实依据索引。
> 访问日期以首次纳入白皮书的日期为准；后续复核请在条目末尾追加“复核记录”。

## S1 — Claude Sonnet 4.6 发布说明

- URL: https://www.anthropic.com/news/claude-sonnet-4-6
- Accessed: 2026-04-14
- 摘要：
  - 说明 Sonnet 4.6 的定位与能力改进（computer use、agent planning、长上下文推理等）。
  - 提到 Sonnet 4.6 的 1M token context window（beta）。
  - 提供与其它模型对比的基准展示与方法学脚注（例如 SWE-bench Verified 的统计方式等）。

## S2 — Claude 4（Opus 4 / Sonnet 4）发布说明

- URL: https://www.anthropic.com/news/claude-4
- Accessed: 2026-04-14
- 摘要：
  - 给出 Claude Opus 4 与 Sonnet 4 在 SWE-bench Verified 等任务上的公开成绩与方法学说明。
  - 描述“extended thinking with tool use”“memory improvements”等能力方向。

## S3 — Anthropic 定价页（用于成本/ROI 讨论的公开参考）

- URL: https://claude.com/pricing
- Accessed: 2026-04-14
- 摘要：
  - 说明 Claude 产品与计划层级（Free/Pro/Max 等）及部分能力项（如 memory、connectors、MCP 等）。
  - 注：该页面对“API 每百万 tokens 单价”的信息可能需要跳转到 API 定价页或平台文档进一步核实；若白皮书引用到具体 $/MTok，请补充更精确来源。

## S4 — Qwen3 技术博客（模型规格、训练与部署建议）

- URL: https://qwenlm.github.io/blog/qwen3/
- Accessed: 2026-04-14
- 摘要：
  - 给出 Qwen3 系列（Dense/MoE）参数规模、激活参数、上下文长度等规格信息。
  - 描述 Qwen3 的 post-training pipeline（long CoT cold start → RL → thinking mode fusion → general RL）。
  - 提到部署建议（SGLang / vLLM）与 agentic/MCP 支持方向。

## S5 — MiMo 论文（推理能力与 RL 训练要点）

- URL: https://arxiv.org/abs/2505.07608
- Accessed: 2026-04-14
- 摘要：
  - 描述 MiMo-7B 的预训练与后训练方法（含 MTP、可验证数学/编程题 RL、difficulty-driven code reward、重采样稳定训练）。
  - 说明该路线如何提升推理能力，适合作为工业 RL-CoT 设计的参考。
  - 注：白皮书当前对标口径为 MiMo-v2-pro；若引用 MiMo-7B 指标/结论，需明确“来自论文的旧版本/公开研究结果”。

## S6 — LiveCodeBench 论文（代码评测与污染控制）

- URL: https://arxiv.org/abs/2403.07974
- Accessed: 2026-04-14
- 摘要：
  - 说明 LiveCodeBench 的设计目标：持续收集新题、降低评测污染、覆盖代码生成/自修复/执行等能力。
  - 支撑白皮书中“选择更抗污染的代码评测基线”的方法学依据。
