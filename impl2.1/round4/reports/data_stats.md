# Round 4 数据统计报告

生成时间：2026-05-12

## 数据来源

从 Langfuse（`http://aip.imp.xiaomi.com/platform`）抓取 `feishu_group` 和 `feishu_p2p` 标签的生产日志。

## ws_raw.jsonl — 完整 SFT 轨迹

| 指标 | 数值 |
|------|------|
| 总样本数 | 93 |
| 消息轮次（min） | 7 |
| 消息轮次（p50） | 17 |
| 消息轮次（p90） | 39 |
| 消息轮次（max） | 99 |
| 估算 tokens（min） | ~1,884 |
| 估算 tokens（p50） | ~5,634 |
| 估算 tokens（p90） | ~11,080 |
| 估算 tokens（p99） | ~15,507 |
| 估算 tokens（max） | ~15,507 |

### Skill 分布

| Skill | 样本数 |
|-------|-------|
| object-data-query | 25 |
| general-kpi-query | 20 |
| line-operation-skill | 16 |
| 分板机过站明细查询 | 10 |
| lineside-material-query | 9 |
| line-attendance-query | 6 |
| line-exemption-query | 3 |
| equipment-cpk-query | 3 |
| workstion-kpi-query | 1 |

## grpo_raw.jsonl — GRPO 步骤样本

| 指标 | 数值 |
|------|------|
| 总样本数 | 827 |

### Skill 分布

| Skill | 样本数 |
|-------|-------|
| general-kpi-query | 220 |
| line-operation-skill | 192 |
| object-data-query | 168 |
| lineside-material-query | 89 |
| 分板机过站明细查询 | 72 |
| line-attendance-query | 50 |
| equipment-cpk-query | 18 |
| line-exemption-query | 14 |
| workstion-kpi-query | 4 |

## 训练数据选择说明

**当前选择（已调整为）**：ws_raw.jsonl 全部 93 条完整 SFT 轨迹
- ws→ns 转换（去掉 skill doc，让模型内化而非查文档）
- cutoff_len = 16384（覆盖所有 93 条样本，p99 ≈ 15,507 tokens）
- 不丢弃任何样本

**之前的错误做法（已废弃）**：grpo_raw.jsonl 步骤样本转 SFT prefix（586 条）
- cutoff_len=6144 过滤导致丢失样本
- 只训练 tool call 预测，未覆盖最终答案生成
