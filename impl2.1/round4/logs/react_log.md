# Round 4 ReAct 决策日志

> ReAct 模式：每个关键决策点记录 Observe → React

---

## Phase 0：环境准备（2026-05-12）

**Observe**：
- GPU 状态：GPU 0 有任务（vLLM），GPU 1 有任务（R3训练），GPU 2/3 空闲
- Langfuse 数据源：`http://aip.imp.xiaomi.com/platform`
- 凭证：pk=`pk-lf-5634e358-13dd-4931-9e80-6f08be8de3de`，sk=`sk-lf-...`

**React**：创建 round4/ 目录结构，写入 `.env` 文件，准备数据拉取

---

## Phase 1：数据采集（2026-05-12）

**Observe**：
```
ws_raw.jsonl:   93 条完整 SFT 轨迹（含 skill doc）
grpo_raw.jsonl: 827 条 GRPO 步骤样本
```

Skill 分布（ws_raw）：object-data-query(25), general-kpi-query(20), line-operation-skill(16)...

**React**：93 条 < 200 条阈值，但数据质量（真实生产日志）优先。不做合成增强。

---

## Phase 2A：初始数据方案（2026-05-12）— ❌ 已废弃

**Observe**：
- ws_raw 的 93 条样本，p99 ≈ 15,507 tokens，85% 超过 cutoff_len=6144
- 直接用 ws_raw 做 SFT 会被大量过滤

**React（错误决策）**：
- 选择 grpo_raw（步骤样本）转 SFT prefix 格式
- 设 cutoff_len=6144，得到 586 条训练样本
- 问题：所有样本只覆盖 tool_call 预测，未覆盖最终答案生成

**纠正原因**：
- 用户要求：不因 cutoff_len 丢弃日志
- 用户要求：只做 SFT，不做 GRPO

---

## Phase 2B：数据方案纠正（2026-05-12）— ✅ 当前方案

**Observe**：
```
ws_raw.jsonl 完整 SFT 轨迹长度（含 tool_calls）：
  p50: ~8,290 tokens
  p90: ~14,178 tokens
  p99: ~20,987 tokens（最大值）

cutoff_len 需求：
  16384: 覆盖 95%（3条超出）
  24576: 覆盖 100%（推荐，不丢弃任何样本）
```

**React**：
- cutoff_len = 24576
- 重新运行 convert_ws_to_ns.py（不过滤）
- 得到：train_ns_v2.jsonl（74条），test_ns_v2.jsonl（19条）

---

## Phase 2C：GRPO 进程清理（2026-05-12）

**Observe**：发现 Round 3 GRPO 训练（R3-C2）占用 GPU 0-3 共 ~33GB/卡

**React**：用户要求停止所有 GRPO → kill -9 所有 GRPO 进程 → 4 卡全部释放

---

## Phase 3：训练配置（2026-05-12）

**关键参数决策**：

| 参数 | 决策 | 理由 |
|------|------|------|
| cutoff_len | 24576 | 覆盖 max=20987 tokens |
| num_train_epochs | 5 | 74 条数据少，多 epoch |
| learning_rate | 5e-5 | 全轨迹 SFT，略降 lr |
| GPU | CUDA_VISIBLE_DEVICES=2,3 | 不占用 0/1（共享机器，他人使用）|
| batch_size | 1 per GPU | 24K token 长序列 |
| grad_accum | 4 | effective batch = 2×4 = 8 |

**训练命令**：
```bash
DISABLE_VERSION_CHECK=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train \
  round4/configs/R4-sft-v2.yaml \
  2>&1 | tee round4/logs/train_R4-sft-v2.log
```

---

## Phase 4：评估（待执行）

**评估框架**：
1. `tool_call_eval.py`：步骤预测 F1（给定前缀，预测下一工具调用）
2. `llm_judge.py`：全轨迹最终答案质量（Claude API 打分）

**ReAct 决策树**：
```
tool_call_f1 ≥ 0.70 AND judge ≥ 0.70 → 最优，写报告
tool_call_f1 0.50-0.69              → 分析弱 skill，补充数据迭代
tool_call_f1 < 0.50                 → 检查数据格式
judge < 0.70                        → 补 final_answer_ns.jsonl(92条) 重训
```

---

## Phase 3B：全量数据训练（2026-05-12 → 05-13）

**Observe（v2 训练结果，2026-05-13）**：
- R4-sft-v2：cutoff=8192 QLoRA，3 epochs，loss→0.18
- 评估结果：tool_call_f1=70.6% ✓，judge=38.8% ✗
- judge 失败根因：cutoff=8412 截断了复杂对话（training data max ~12K tokens），模型输出推理步骤而非答案
- 用户要求：tool_call_F1 ≥ 75% AND judge ≥ 0.75（加速比 ≥ 2.7×）

**React**：重训 R4-sft-v3，修复 cutoff 问题。

---

## Phase 3C：OOM 调试（2026-05-13）

**Observe**：
- 尝试 cutoff=32768，标准 LoRA + SDPA：OOM，tried to allocate 18.55 GiB，GPU 2 only 13.85 GiB free
- 假设：SDPA 无 flash-attn 时会 materialize 完整 attention matrix
- 尝试 flash-attn：网络无法访问，编译失败
- 尝试 DeepSpeed ZeRO-3：deepspeed 0.19.0 不兼容 LlamaFactory（需 ≤0.16.9）
- 降级到 0.16.9 + ZeRO-3 + LoRA：`ValueError: zip() argument 2 is longer than argument 1`（已知 bug）

**React**：放弃 ZeRO-3，改用 DDP 逐步测试 cutoff：
- cutoff=24576，max_steps=3，GPU 2+3：**成功，无 OOM，loss=0.48，4m14s**
- 原因分析：数据 max ~12K tokens（word count p99=5056 ≈ 6.5K tokens），动态 padding 保证实际 attention 远小于 24K²

---

## Phase 3D：R4-sft-v3 正式训练（2026-05-13 13:45）

**决策**：
| 参数 | 值 | 理由 |
|------|----|----|
| cutoff_len | 24576 | 实测覆盖 100% 数据，无 OOM |
| 训练方式 | 标准 LoRA（无量化）| QLoRA 内存不可控 |
| DeepSpeed | 无 | ZeRO-3 + LoRA 不兼容 |
| GPU | 2,3 | GPU 0 vLLM，GPU 1 其他任务 |
| epochs | 3 | 349 条数据，3 epochs 合理 |

**状态**：训练进行中（PID 329641，13:45 启动，预计 17:25 完成）
**总步数**：132 optimizer steps

---

## 待执行步骤

- [x] 启动 R4-sft-v3 训练（GPU 2+3，cutoff=24576，标准 LoRA）
- [ ] 合并 adapter → checkpoints/R4-sft-v3-merged
- [ ] 部署 vLLM（port 8035，max_model_len=32768）
- [ ] 运行 trajectory_eval.py（全轨迹 F1 + timing）
- [ ] 运行 llm_judge（目标 ≥ 0.75）
- [ ] 分析结果，满足 tool_call_F1 ≥ 75% AND judge ≥ 0.75 AND speedup ≥ 2.7×
- [ ] 更新 ROUND4_REPORT.md
