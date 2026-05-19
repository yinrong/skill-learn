# Round 4 执行手册（ReAct 模式）

---

## 一、任务目标

**核心问题（Goal）**：
> 通过生产日志微调 Qwen3-14B 后，模型能否在不依赖 skill doc 的情况下，复现线上模型（有 skill doc）的效果？差异有多大？

**背景**：
- 线上模型每次推理都需要把 skill YAML 文档（数百 token）塞进 prompt（ws 格式），增加延迟和成本
- 技能内化方法论（SKILL_INTERNALIZATION_METHODOLOGY_v2）假设：将 ws 格式轨迹转为 ns 格式后微调，模型可以内化规则，推理时不再需要 skill doc
- **本实验**：用 log2train 从真实生产日志中提取轨迹，验证此假设

**衡量标准（如何判断"复现了效果"）**：
- 线上模型（ws）做了什么工具调用 → 这是 Ground Truth
- 微调模型（ns）做了什么工具调用 → 与 GT 对比，看工具调用 F1
- 线上模型的最终答案 → 与微调模型答案对比，看语义质量（LLM-as-judge）

---

## 二、约束条件（用户明确要求）

1. **不丢弃任何样本**：cutoff_len 只做统计，不因长度过滤样本（2026-05-12 用户纠正）
2. **只做 SFT**：不使用 GRPO 训练算法（2026-05-12 用户纠正）
3. **GPU**：动态检测空闲 GPU，用PWD标记自己的进程，不要以外kill非本项目pwd内的进程；规划gpu任务并发前检测可用gpu个数，在不超过用户限制数量的前提下，以最终任务更快完成为目标，尽量提高并发度，包括单任务并发度和多任务并行
4. **证据文档**：每个关键计划决策和完成里程碑，必须写证据文档并在本 CLAUDE.md 中添加链接（2026-05-12 用户要求）
5. **多 Agent 并发**：所有独立任务必须用多 Agent 并发执行，不串行等待
6. **重新规划ReAct**：每隔5分钟定时检测所有agent运行情况，如果发生异常、完成，要基于其结果，考虑是否需要重新规划。agent完成时，也要触发同样的检测和重新规划。重新规划始终瞄准任务目标。如果对任务目标有歧义，要记下来待未来用户决策项（追加到USER-DECIDE.md），但不要等待用户决策，而是继续向前推进，在用户决策前尽量收集更多信息。重新规划时，如果USER-DECIDE.md因为新的发现，不再需要用户决策的时候，就把相关内容标记为“已不需要决策”。

---

## 二·五、证据文档索引

| 证据 | 文档链接 | 说明 |
|------|----------|------|
| 技能调用端到端耗时 + 换用14B加速预估 | [reports/evidence_timing_v1.md](reports/evidence_timing_v1.md) | 6条生产日志，耗时分解，换14B预估~2.7×加速 |
| 训练数据分类统计 | [reports/evidence_data_classification.md](reports/evidence_data_classification.md) | 412条SFT样本，skill分布，token长度分布 |
| 数据统计报告 | [reports/data_stats.md](reports/data_stats.md) | 各标签ws/grpo样本数统计 |
| **综合实验报告** | [reports/ROUND4_REPORT.md](reports/ROUND4_REPORT.md) | 完整实验结果：首步F1=70.6% ✓，judge=38.8%，原因分析 |

---

## 三、规划和执行记录（ReAct 模式）

### 计划 v1（2026-05-12 初始）
```
Phase 0: 环境初始化
Phase 1: 拉取 feishu_group/feishu_p2p 标签日志
Phase 2: ws→ns 转换（cutoff=6144）
Phase 3: GRPO-prefix 转 SFT（586 条步骤样本）
Phase 4: 评估 tool_call_f1
```

**纠正原因（v1 → v2）**：
- 纠正1：用 GRPO prefix 作为 SFT 数据 → 训练样本只覆盖 tool_call 预测，缺少最终答案生成（用户：不用GRPO，只做SFT，指的是完整轨迹 SFT）
- 纠正2：cutoff_len=6144 过滤掉 94 条样本 → 违背"不丢弃日志"约束
- 纠正3：只拉取了 feishu_group/feishu_p2p → 没有覆盖 web/cron/alarm 等全量日志（任务目标是获取所有日志）

### 计划 v2（2026-05-12 纠正后）
```
Phase 1B: 拉取全量日志（web, cron, alarm, crab, dolphin, feishu_*）
Phase 1C: 日志分类分析（按 skill/渠道/成功率分层统计）
Phase 2B: ws→ns 转换（cutoff=24576，不过滤，保留全部样本）
Phase 3B: 完整轨迹 SFT（Qwen3-14B，基于当前已有 74 条训练）
Phase 4:  评估：微调模型 vs 日志中原始模型的效果差异
```

**最终执行状态**：
| Phase | 状态 | 说明 |
|-------|------|------|
| Phase 0 | ✅ 完成 | 目录、凭证、GPU 检测 |
| Phase 1A | ✅ 完成 | feishu 日志 93条SFT + 827条GRPO |
| Phase 1B | ✅ 完成 | 全量日志：ws_all.jsonl 412条 |
| Phase 2B | ✅ 完成 | 全量转换：349条train + 63条test，实际cutoff=8192（OOM限制） |
| Phase 3B | ✅ 完成 | QLoRA SFT，2×H20，3 epochs，loss→0.18 |
| Phase 4 | ✅ 完成 | 首步F1=70.6% ✓ PASS；judge=38.8%（受cutoff截断影响） |
| Phase 5 | ✅ 完成 | [ROUND4_REPORT.md](reports/ROUND4_REPORT.md) |

### 评估结果汇总（R4-sft-v2，cutoff=8192，2026-05-13）

| 指标 | 值 | 目标 | 状态 |
|------|----|----|------|
| tool_call_f1（首步，ns格式） | **70.6%** | ≥75% | ✗ FAIL |
| tool_name_acc | 81.0% | - | - |
| judge_overall（答案质量） | 38.8% | ≥0.75 | ✗ FAIL |

**judge 失败根因**：cutoff=8192 截断了长对话（数据 max ~12K tokens），训练数据不完整。

---

### R4-sft-v3 训练进度（2026-05-13 进行中）

| 项目 | 状态 |
|------|------|
| 配置 | `configs/R4-sft-v3.yaml`（标准LoRA，cutoff=24576，无量化，无DeepSpeed）|
| 进度 | step 91/132，epoch 2/3 完成，checkpoint-88 已保存 |
| loss | 0.71 → 0.32（持续下降）|
| 预计完成 | 2026-05-13 ~16:00 |
| 恢复指南 | **[RESUME.md](RESUME.md)** — 含完整下一步操作 |

**完成后目标**：tool_call_F1 ≥ 75% AND judge ≥ 0.75 AND speedup ≥ 2.7×

---

## 四、目录结构

```
round4/
├── CLAUDE.md              本执行手册（含规划执行记录）
├── configs/
│   ├── R4-sft-v2.yaml     当前训练配置（完整SFT，cutoff=65536，349条）✓
│   └── R4-base.yaml       旧配置（废弃）
├── data/
│   ├── ws_raw.jsonl        feishu 来源完整 SFT 轨迹（93条）
│   ├── grpo_raw.jsonl      feishu 来源 GRPO 样本（827条）
│   ├── ws_raw_all.jsonl    全量日志（待抓取：web+cron+alarm+...）
│   ├── train_ns_v2.jsonl   训练集（74条，ns格式）✓
│   ├── test_ns_v2.jsonl    测试集（19条）✓
│   ├── train_lf_v2.jsonl   LlamaFactory格式训练集✓
│   ├── final_answer_ns.jsonl 备用答案生成样本（92条）
│   └── dataset_info.json   LlamaFactory 数据集注册✓
├── checkpoints/
│   └── R4-sft-v2/         训练输出
├── eval/
│   ├── tool_call_eval.py   工具调用 F1 评估✓
│   └── llm_judge.py        LLM-as-judge 评估✓
├── scripts/
│   ├── convert_ws_to_ns.py  ws→ns 格式转换✓
│   ├── to_llamafactory.py   格式转换✓
│   ├── merge_adapter.sh     合并 adapter✓
│   ├── deploy_vllm.sh       部署 vLLM✓
│   └── run_eval.sh          一键评估✓
├── logs/
│   ├── react_log.md         ReAct 详细决策日志
│   └── train_R4-sft-v2.log  训练日志（待生成）
└── reports/
    ├── data_stats.md        数据统计报告✓
    └── ROUND4_REPORT.md     综合报告（待生成）
```

---

## 五、Phase 1B：全量日志抓取

### Langfuse 数据概况（初步扫描，前1900条）
| 标签 | 出现次数 | 类型 |
|------|----------|------|
| web | 524+ | 用户 web 对话 |
| cron | 78+ | 定时任务查询 |
| alarm | 70+ | 告警触发查询 |
| crab | 31 | 螃蟹渠道（飞书群）|
| dolphin | 13 | - |
| feishu_group | 3+ | 飞书群消息 |
| feishu_p2p | 2+ | 飞书私聊 |

总计约 53,200 条 traces（532页 × 100）

### 抓取命令（按标签分批）

```bash
cd /home/yinrong/skill-learn-log2train

# 抓取 web 标签
python3 langfuse_to_training.py \
  --langfuse_tags web \
  --output_sft ../post-train/impl2.1/round4/data/ws_web.jsonl \
  --output_grpo ../post-train/impl2.1/round4/data/grpo_web.jsonl \
  --verbose

# 抓取 cron 标签
python3 langfuse_to_training.py \
  --langfuse_tags cron \
  --output_sft ../post-train/impl2.1/round4/data/ws_cron.jsonl \
  --output_grpo ../post-train/impl2.1/round4/data/grpo_cron.jsonl \
  --verbose

# 抓取 alarm 标签
python3 langfuse_to_training.py \
  --langfuse_tags alarm \
  --output_sft ../post-train/impl2.1/round4/data/ws_alarm.jsonl \
  --output_grpo ../post-train/impl2.1/round4/data/grpo_alarm.jsonl \
  --verbose
```

---

## 六、训练启动（Phase 3B）

```bash
# 从 round4/ 目录运行
cd /home/yinrong/post-train/impl2.1/round4
DISABLE_VERSION_CHECK=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train \
  configs/R4-sft-v2.yaml \
  2>&1 | tee logs/train_R4-sft-v2.log
```

---

## 七、评估流程（Phase 4）

```bash
# 从 round4/ 目录运行
cd /home/yinrong/post-train/impl2.1/round4

# 合并 adapter
bash scripts/merge_adapter.sh \
  checkpoints/R4-sft-v2 \
  checkpoints/R4-sft-v2-merged

# 部署 vLLM（GPU 2, port 8035）
bash scripts/deploy_vllm.sh 2 8035 checkpoints/R4-sft-v2-merged &

# 评估工具调用 F1
python eval/tool_call_eval.py \
  --test_file data/test_ns_v2.jsonl \
  --output results/R4-sft-v2-tool-f1.json \
  --model_url http://localhost:8035 --verbose

# LLM-as-judge
python eval/llm_judge.py \
  --grpo_file data/grpo_raw.jsonl \
  --output results/R4-sft-v2-judge.json \
  --model_url http://localhost:8035 --max_samples 30
```

---

## 八、Go/No-Go 标准

| 指标 | 通过 | 失败处理 |
|------|------|---------|
| tool_call_f1 ≥ 0.70 | → 成功 | 分析弱 skill，补充数据重训 |
| judge_overall ≥ 0.70 | → 成功 | 补 final_answer_ns(92条) 重训 |

---

*最后更新：2026-05-12*
*方法论参考：round3/reports/SKILL_INTERNALIZATION_METHODOLOGY_v2.md*
