# 恢复工作手册 — R3-C2 GRPO 训练

> 生成时间：2026-05-13  
> 用途：在全新 Claude Code / 新系统中恢复工作

---

## 当前状态（截至 2026-05-13 下午）

| 项目 | 状态 |
|------|------|
| 训练脚本 | `round3/train_grpo_trl.py` |
| 训练进度 | **234/513 步 (46%)**，~20h 剩余 |
| 当前 epoch | 1.35 |
| 最新 reward_fn/mean | 0.575（正常波动 0.0~1.1） |
| 检查点 | `round3/checkpoints/R3-C2/checkpoint-171`（epoch 1） |
| 训练日志 | `round3/logs/train_R3-C2.log` |
| GPU | CUDA_VISIBLE_DEVICES=0（单卡，55GB） |
| 预计完成 | ~2026-05-14 上午 10:00 |

---

## 训练进程确认

```bash
# 确认训练仍在运行
ps aux | grep train_grpo_trl | grep -v grep

# 查看最新进度
grep -E "[0-9]+/513\s+\[" round3/logs/train_R3-C2.log | tail -3

# 查看最新 reward
grep "rewards/reward_fn/mean" round3/logs/train_R3-C2.log | tail -3

# GPU 状态
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

---

## 训练完成判断

```bash
# 出现此文件 = 训练完成
ls round3/checkpoints/R3-C2/trainer_state.json && echo DONE || echo NOT_DONE
```

---

## 训练完成后：按顺序执行以下步骤

### 步骤 1：运行评测

```bash
cd /home/yinrong/post-train/impl2.1
bash round3/eval_grpo_r3c2.sh 2>&1 | tee round3/logs/eval_grpo_r3c2_run.log
```

eval 脚本做三件事：
1. 合并 LoRA adapter → `round3/checkpoints/R3-C2-merged`
2. 启动 vLLM（port 8034）
3. 跑 `common/tools/eval/spc_eval.py`（200样本），结果 → `round3/results/R3-C2.json`

### 步骤 2：读取评测结果

```bash
python3 -c "
import json
r = json.load(open('round3/results/R3-C2.json'))
print(f'F1={r[\"f1\"]:.3f}  P={r[\"precision\"]:.3f}  R={r[\"recall\"]:.3f}  EM={r[\"exact_match\"]:.3f}  CPK={r[\"cpk_found_rate\"]:.3f}')
"
```

### 步骤 3：更新报告

更新以下两个文件，填入 R3-C2 实测数值：

- `round3/reports/INTERVIEW_QA.md`
- `round3/reports/SKILL_INTERNALIZATION_METHODOLOGY_v2.md`

关键：将 "待填入" / "R3-C2 结果待定" 等占位符替换为实测的 F1/P/R/EM/CPK 数字。

### 步骤 4：上传飞书

将两个更新后的报告重新上传到飞书（用户自行操作，或通过飞书 MCP 工具）。

---

## Go/No-Go 标准

| 指标 | 通过标准 | 失败处理 |
|------|---------|---------|
| F1 | > 0.46（expYYY 0.430 + 0.03） | 调整 reward 权重，增大 num_generations，重跑 |
| reward hacking | reward 无突然跌至 0 | 检查 empty prediction penalty 是否生效 |

---

## 关键技术背景

### R3-C 失败原因（已修复）
- **奖励黑客**：模型预测空 violations + 正确 CPK → 平均 reward 0.6，F1 跌至 0.003
- **零奖励 bug**：`end_think < 50` 检查阻断所有 reward（Qwen3 模型输出 `<think>\n\n</think>` 在位置 0）

### R3-C2 修复
1. 移除 `end_think < 50` 检查（Qwen3 expYYY-merged 固定输出空 think block）
2. 空预测惩罚：`violations=[]` 但 GT 非空 → `-0.3`
3. KL beta：0.01 → 0.05
4. 单卡（CUDA_VISIBLE_DEVICES=0），num_generations=2

### Qwen3 Thinking 行为（重要）
- `apply_chat_template(enable_thinking=True)` → 正常 prompt，模型自行决定是否 think
- expYYY-merged 固定输出 `<think>\n\n</think>`（空 think）作为前缀，真正推理在 `</think>` 之后
- 不要在 reward 函数中检查 `</think>` 位置
- `extract_violations()` 和 `extract_cpk()` 在完整 completion 上能正常工作

---

## 文件路径速查

```
impl2.1/
├── round3/
│   ├── train_grpo_trl.py           # 训练脚本（当前运行）
│   ├── eval_grpo_r3c2.sh           # 评测脚本
│   ├── checkpoints/R3-C2/          # 训练输出
│   │   └── checkpoint-171          # epoch 1 检查点
│   ├── results/R3-C2.json          # 评测结果（训练完成后生成）
│   ├── logs/train_R3-C2.log        # 训练日志
│   └── reports/
│       ├── INTERVIEW_QA.md         # 待更新（填入 R3-C2 结果）
│       └── SKILL_INTERNALIZATION_METHODOLOGY_v2.md  # 待更新
├── round2/history-route2.1.1/checkpoints/expYYY-merged/  # 基座模型
└── common/
    ├── data/test.jsonl              # 评测集（200样本）
    └── tools/eval/extractor.py     # violations/cpk 提取器
```

---

## 如果训练意外中断

```bash
cd /home/yinrong/post-train/impl2.1

# 从最新检查点恢复（trl 自动支持 resume_from_checkpoint）
CUDA_VISIBLE_DEVICES=0 python3 round3/train_grpo_trl.py \
    --resume_from_checkpoint round3/checkpoints/R3-C2/checkpoint-171 \
    2>&1 | tee round3/logs/train_R3-C2_resume.log

# 注意：train_grpo_trl.py 目前不接受命令行参数，需手动在脚本中添加：
# trainer.train(resume_from_checkpoint="round3/checkpoints/R3-C2/checkpoint-171")
```

实际上 GRPOTrainer 会自动检测 output_dir 中的最新 checkpoint 并恢复，
直接重新运行即可：

```bash
CUDA_VISIBLE_DEVICES=0 python3 round3/train_grpo_trl.py 2>&1 | tee round3/logs/train_R3-C2_resume.log
```
