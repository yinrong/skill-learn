# Round3 输出文件索引

> 实时维护，每产出新文件立即更新。  
> 根目录：`/home/yinrong/post-train/impl2.1/round3/`

---

## 阶段报告

| 文件 | 说明 |
|------|------|
| [progress.md](progress.md) | Round3 进度记录与 Round4 规划方向 |
| [reports/round3_summary.md](reports/round3_summary.md) | 完整实验汇总、规则难度分析、综合得分、Go/No-Go 评估 |

---

## 评测结果

| 实验 | F1 | rule2 | rule7 | CPK_MAE | 结果文件 |
|------|-----|-------|-------|---------|---------|
| R3-A | 0.403 | 0.379 | 0.500 | 0.077 | [results/R3-A.json](results/R3-A.json) |
| R3-B | 0.372 | 0.276 | 0.278 | 0.076 | [results/R3-B.json](results/R3-B.json) |
| R3-AB | 0.383 | 0.483 | 0.278 | 0.073 | [results/R3-AB.json](results/R3-AB.json) |
| R3-AB-v2 | 0.420 | 0.276 | 0.389 | 0.078 | [results/R3-AB-v2.json](results/R3-AB-v2.json) |
| R3-AB-v3 | 0.418 | 0.448 | 0.389 | 0.066 | [results/R3-AB-v3.json](results/R3-AB-v3.json) |
| R3-D1 (14B QLoRA int4) | 0.394 | 0.414 | 0.222 | 0.072 | [results/R3-D1.json](results/R3-D1.json) |
| R3-D2 (32B QLoRA FSDP 4卡) | 0.379 | 0.586 | 0.278 | 0.072 | [results/R3-D2.json](results/R3-D2.json) |
| R3-D3 (14B QLoRA int8) | ✅ 0.391 | 0.310 | 0.167 | 0.069 | [results/R3-D3.json](results/R3-D3.json) |
| R3-C (GRPO trl) | ✅ 0.000 | 0.000 | 0.000 | — | [results/R3-C-grpo.json](results/R3-C-grpo.json) |
| R3-AB-v4（rule7 3× 上采样） | ✅ 0.401 | 0.552 | 0.278 | 0.080 | [results/R3-AB-v4.json](results/R3-AB-v4.json) |
| R3-AB-v2-lowlr（lr=5e-5） | ✅ 0.409 | 0.310 | 0.222 | 0.058 | [results/R3-AB-v2-lowlr.json](results/R3-AB-v2-lowlr.json) |
| R3-E1 (32B bf16 8-GPU FSDP) | ⏳ 待 C 完成后自动启动 | — | — | — | [logs/train_R3-E1.log](logs/train_R3-E1.log) |
| **Claude-4.6 (公平条件 ns)** | **✅ 0.110** | **0.276** | **0.000** | **0.012** | [results/R3-claude-fair.json](results/R3-claude-fair.json) |

---

## 训练日志

| 实验 | 日志 |
|------|------|
| R3-A | [logs/train_R3-A.log](logs/train_R3-A.log) |
| R3-B | [logs/train_R3-B.log](logs/train_R3-B.log) |
| R3-AB | [logs/train_R3-AB.log](logs/train_R3-AB.log) |
| R3-AB-v2 | [logs/train_R3-AB-v2.log](logs/train_R3-AB-v2.log) |
| R3-AB-v3 | [logs/train_R3-AB-v3.log](logs/train_R3-AB-v3.log) |
| R3-C (GRPO，不支持) | [logs/train_R3-C.log](logs/train_R3-C.log) |
| R3-D1 (14B QLoRA) | [logs/train_R3-D1.log](logs/train_R3-D1.log) |
| R3-D2 (32B QLoRA FSDP 4卡) | [logs/train_R3-D2.log](logs/train_R3-D2.log) |

---

## Merge 日志

| 实验 | 日志 |
|------|------|
| R3-A | [logs/merge_R3-A.log](logs/merge_R3-A.log) |
| R3-B | [logs/merge_R3-B.log](logs/merge_R3-B.log) |
| R3-AB | [logs/merge_R3-AB.log](logs/merge_R3-AB.log) |
| R3-AB-v2 | [logs/merge_R3-AB-v2.log](logs/merge_R3-AB-v2.log) |
| R3-AB-v3 | [logs/merge_R3-AB-v3.log](logs/merge_R3-AB-v3.log) |
| R3-D1 | [logs/merge_R3-D1.log](logs/merge_R3-D1.log) |
| R3-D2 | [logs/merge_R3-D2.log](logs/merge_R3-D2.log) |
| R3-D3 | [logs/merge_R3-D3.log](logs/merge_R3-D3.log) |

---

## 评测日志

| 实验 | 日志 |
|------|------|
| R3-A | [logs/eval_R3-A.log](logs/eval_R3-A.log) |
| R3-B | [logs/eval_R3-B.log](logs/eval_R3-B.log) |
| R3-AB | [logs/eval_R3-AB.log](logs/eval_R3-AB.log) |
| R3-AB-v2 | [logs/eval_R3-AB-v2.log](logs/eval_R3-AB-v2.log) |
| R3-AB-v3 | [logs/eval_R3-AB-v3.log](logs/eval_R3-AB-v3.log) |
| R3-D1 | [logs/eval_R3-D1.log](logs/eval_R3-D1.log) |
| R3-D2 | [logs/eval_R3-D2_shard0.log](logs/eval_R3-D2_shard0.log) |
| R3-D3 | [logs/eval_R3-D3.log](logs/eval_R3-D3.log) |

---

## Checkpoints（合并后模型）

| 实验 | 路径 |
|------|------|
| R3-A-merged | [checkpoints/R3-A-merged/](checkpoints/R3-A-merged/) |
| R3-B-merged | [checkpoints/R3-B-merged/](checkpoints/R3-B-merged/) |
| R3-AB-merged | [checkpoints/R3-AB-merged/](checkpoints/R3-AB-merged/) |
| R3-AB-v2-merged | [checkpoints/R3-AB-v2-merged/](checkpoints/R3-AB-v2-merged/) |
| R3-AB-v3-merged | [checkpoints/R3-AB-v3-merged/](checkpoints/R3-AB-v3-merged/) |
| R3-D1-qlora-14b-merged | [checkpoints/R3-D1-qlora-14b-merged/](checkpoints/R3-D1-qlora-14b-merged/) |
| R3-D2-qlora-32b (adapter) | [checkpoints/R3-D2-qlora-32b/](checkpoints/R3-D2-qlora-32b/) |
| R3-D2-qlora-32b-merged | [checkpoints/R3-D2-qlora-32b-merged/](checkpoints/R3-D2-qlora-32b-merged/) |
| R3-D3-qlora-14b-int8-merged | [checkpoints/R3-D3-qlora-14b-int8-merged/](checkpoints/R3-D3-qlora-14b-int8-merged/) |

---

## 训练数据

| 数据集 | 文件 |
|--------|------|
| R3-A 训练集 | [data/train_R3-A.jsonl](data/train_R3-A.jsonl) |
| R3-B 训练集 | [data/train_R3-B.jsonl](data/train_R3-B.jsonl) |
| R3-AB 训练集 | [data/train_R3-AB.jsonl](data/train_R3-AB.jsonl) |
| R3-AB-v2（去截断）| [data/train_R3-AB-v2.jsonl](data/train_R3-AB-v2.jsonl) |
| R3-AB-v3（+640条扩充）| [data/train_R3-AB-v3.jsonl](data/train_R3-AB-v3.jsonl) |

---

## 数据生成日志

| 任务 | 日志 |
|------|------|
| 边界样本生成 | [logs/gen_boundary.log](logs/gen_boundary.log) |
| 额外边界样本 | [logs/gen_boundary_extra.log](logs/gen_boundary_extra.log) |
| ns_v5 扩充 | [logs/gen_ns_v5.log](logs/gen_ns_v5.log) |
| ns 额外扩充 | [logs/gen_ns_extra.log](logs/gen_ns_extra.log) |
| 多角色 ws | [logs/gen_multirole_ws.log](logs/gen_multirole_ws.log) |
| 多角色额外 | [logs/gen_multirole_extra.log](logs/gen_multirole_extra.log) |
| 合并扩充数据 | [logs/merge_extra_data.log](logs/merge_extra_data.log) |

---

## 附加实验（2026-04-29 晚追加）

| 实验 | 配置文件 | 训练日志 |
|------|---------|---------|
| R3-D3 (14B QLoRA int8) | [configs/R3-D3-qlora-14b-int8.yaml](configs/R3-D3-qlora-14b-int8.yaml) | [logs/train_R3-D3.log](logs/train_R3-D3.log) |
| R3-C (GRPO trl) | [tools/train/train_grpo_trl.py](tools/train/train_grpo_trl.py) | [logs/train_R3-C-grpo.log](logs/train_R3-C-grpo.log) |
| R3-E1 (32B bf16 8-GPU FSDP) | [configs/R3-E1-fsdp-32b-bf16.yaml](configs/R3-E1-fsdp-32b-bf16.yaml) | [logs/train_R3-E1.log](logs/train_R3-E1.log) |
| Claude-4.6 公平重测 | [logs/eval_R3-claude-fair.log](logs/eval_R3-claude-fair.log) | [results/R3-claude-fair.json](results/R3-claude-fair.json) |

## 并行消融实验（2026-04-30 追加，GPU 0/3）

| 实验 | 配置文件 | 训练日志 | 进度 |
|------|---------|---------|------|
| R3-AB-v4（rule7 3× 上采样，1935条） | [configs/R3-AB-v4.yaml](configs/R3-AB-v4.yaml) | [logs/train_R3-AB-v4.log](logs/train_R3-AB-v4.log) | ⏳ 训练中（GPU 0，~8h） |
| R3-AB-v2-lowlr（lr=5e-5 消融，1131条） | [configs/R3-AB-v2-lowlr.yaml](configs/R3-AB-v2-lowlr.yaml) | [logs/train_R3-AB-v2-lowlr.log](logs/train_R3-AB-v2-lowlr.log) | ⏳ 训练中（GPU 3，~4h） |

## 退化率测试

| 实验 | 日志 |
|------|------|
| R3-AB-v2 / R3-D1 / R3-D2 lm_eval | [logs/benchmark_sft-14B-R3-AB-v2.log](logs/benchmark_sft-14B-R3-AB-v2.log) |
| 退化率计算结果 | [common/benchmark_results/degradation_summary.json](../common/benchmark_results/degradation_summary.json) |

---

*最后更新：2026-04-30 22:57（R3-AB-v4 完成 F1=0.401，rule2=0.552，rule7=0.278；Round3 全部12个实验完成）*
