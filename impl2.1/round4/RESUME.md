# Round 4 恢复指南（2026-05-13）

> 从全新会话恢复时，先读本文件，再读 CLAUDE.md。

---

## 当前状态（2026-05-13 15:30 快照）

### 训练进度

| 项目 | 状态 |
|------|------|
| 模型 | Qwen3-14B |
| 训练配置 | `round4/configs/R4-sft-v3.yaml` |
| cutoff_len | 24576（覆盖 100% 数据，无 OOM）|
| 进度 | **step 91/132，epoch 2.05/3 完成** |
| loss 趋势 | 0.71 → 0.58 → 0.42 → 0.32（持续下降，正常）|
| checkpoint-44 | epoch 1 保存 ✓ |
| checkpoint-88 | epoch 2 保存 ✓（当前最优）|
| 预计完成 | 2026-05-13 ~16:00 |
| 日志 | `round4/logs/train_R4-sft-v3.log` |
| 训练进程 PID | 329641（llamafactory-cli）|

**检查训练是否仍在运行**：
```bash
ps aux | grep "llamafactory-cli train" | grep -v grep
tail -5 /home/yinrong/post-train/impl2.1/round4/logs/train_R4-sft-v3.log
```

**检查训练是否已完成**：
```bash
grep "train metrics" /home/yinrong/post-train/impl2.1/round4/logs/train_R4-sft-v3.log
```

---

## 训练完成后：执行顺序

### Step 1：合并 adapter

```bash
cd /home/yinrong/post-train/impl2.1/round4
bash scripts/merge_adapter.sh \
  checkpoints/R4-sft-v3 \
  checkpoints/R4-sft-v3-merged
```

- 脚本会自动找到最新 checkpoint（`checkpoint-132` 或 epoch 3 保存的）
- 输出：`checkpoints/R4-sft-v3-merged/`（完整权重，可直接用 vLLM 加载）
- 日志：`logs/merge_adapter.log`

### Step 2：部署 vLLM（port 8035）

检查 GPU 空闲情况（训练占用 GPU 2,3 会释放）：
```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits
```

部署（选择空闲 GPU，通常训练结束后 GPU 2 或 3 可用）：
```bash
cd /home/yinrong/post-train/impl2.1/round4
bash scripts/deploy_vllm.sh 2 8035 checkpoints/R4-sft-v3-merged &
# 等待 "Application startup complete" 出现（约 2-3 分钟）
```

验证服务：
```bash
curl -s http://localhost:8035/v1/models | python3 -m json.tool
```

### Step 3：全轨迹评估（trajectory_eval.py）

```bash
cd /home/yinrong/post-train/impl2.1

python round4/eval/trajectory_eval.py \
  --test_file round4/data/test_ns_all_eval_ns.jsonl \
  --ws_file round4/data/test_ns_all.jsonl \
  --output round4/results/R4-sft-v3-trajectory.json \
  --model_url http://localhost:8035 \
  --verbose
```

**目标指标**：
- `trajectory_tool_name_f1` ≥ 75%
- `speedup_vs_production` ≥ 2.7×（production 参考：6600ms/step）

### Step 4：LLM-as-judge 答案质量评估

```bash
cd /home/yinrong/post-train/impl2.1

python round4/eval/llm_judge.py \
  --test_file round4/data/test_ns_all_eval_ns.jsonl \
  --ws_file round4/data/test_ns_all.jsonl \
  --output round4/results/R4-sft-v3-judge.json \
  --model_url http://localhost:8035
```

**注意**：judge 模型已修正为 `ppio/pa/claude-sonnet-4-6`（`eval/llm_judge.py:42`）

**目标指标**：`judge_overall` ≥ 0.75

### Step 5：更新报告

根据评估结果更新 `round4/reports/ROUND4_REPORT.md`，对比 v2 和 v3 结果。

---

## 关键文件索引

| 文件 | 说明 |
|------|------|
| `round4/configs/R4-sft-v3.yaml` | 当前训练配置（cutoff=24576，标准LoRA，无QLoRA，无DeepSpeed）|
| `round4/data/train_ns_all.jsonl` | 训练集（349条，ns格式）|
| `round4/data/test_ns_all.jsonl` | 测试集（63条，ws格式，用于GT）|
| `round4/data/test_ns_all_eval_ns.jsonl` | 测试集（63条，ns格式，用于模型推理）|
| `round4/eval/trajectory_eval.py` | 全轨迹 F1 + 计时评估 |
| `round4/eval/llm_judge.py` | LLM-as-judge 答案质量评估（model: ppio/pa/claude-sonnet-4-6）|
| `round4/scripts/merge_adapter.sh` | 合并 LoRA adapter |
| `round4/scripts/deploy_vllm.sh` | 部署 vLLM（已修正 max-model-len=32768）|
| `round4/logs/train_R4-sft-v3.log` | 训练日志 |
| `round4/reports/ROUND4_REPORT.md` | 综合报告（含 v2 结果，待更新 v3）|

---

## 历史结果（R4-sft-v2，cutoff=8192）

| 指标 | 值 | 目标 | 状态 |
|------|----|------|------|
| tool_call_f1（首步）| 70.6% | ≥75% | ✗ |
| judge_overall | 38.8% | ≥0.75 | ✗ |

**失败根因**：cutoff=8192 截断了长对话（数据 max ~12K tokens），导致训练数据不完整。
**R4-sft-v3 修复**：cutoff=24576 覆盖 100% 数据。

---

## 已知问题 / 陷阱

1. **vLLM max-model-len**：必须 ≥ 16384，否则长 prompt 返回 400。已在 `deploy_vllm.sh` 中修正为 32768。
2. **judge 模型名**：`ANTHROPIC_BASE_URL=http://model.mify.ai.srv/anthropic` 需用 `ppio/pa/claude-sonnet-4-6`，已修正。
3. **DeepSpeed ZeRO-3 + LoRA**：不兼容（zip() 长度不匹配 bug），R4-sft-v3 已去掉 deepspeed。
4. **flash-attn**：未安装（网络无法访问 PyPI wheels），用 SDPA 替代（功能一致，稍慢）。
5. **评估格式**：必须用 `test_ns_all_eval_ns.jsonl`（ns格式），不能用 `test_ns_all.jsonl`（ws格式，含 skill doc）。

---

*生成时间：2026-05-13 15:30*
