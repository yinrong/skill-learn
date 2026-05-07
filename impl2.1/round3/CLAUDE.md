# Round3 执行手册

> Round3 目标：F1 ≥ 0.50，补齐 goal.md 要求的量化/多卡/GRPO 维度，弱规则 rule2/rule7 召回 ≥ 0.60
> 起点：expYYY（Qwen3-14B，F1=0.430，round2 最优）

---

## 快速索引

| 阶段 | 脚本 | 说明 |
|------|------|------|
| 数据准备 | `bash round3/prepare_data.sh` | 生成边界/扩充数据，合并，注册到 LLaMA-Factory |
| P0 实验 | `bash round3/run_experiments.sh p0` | R3-A + R3-B（方向A弱规则 + 方向B扩数据） |
| P0 组合 | `bash round3/run_experiments.sh ab` | R3-AB（A+B 最优组合） |
| P1 GRPO | `bash round3/run_experiments.sh grpo` | R3-C GRPO 接续训练（基于 expYYY-merged） |
| P1 QLoRA | `bash round3/run_experiments.sh qlora` | R3-D1（14B int4）+ R3-D2（32B int4） |
| P1 72B | `bash round3/run_experiments.sh 72b` | R3-E1 8卡 ZeRO-3（需先下载 72B） |
| 评测报告 | `bash round3/eval_and_report.sh` | 对所有 merged 模型评测，生成对比表格 |

**所有脚本必须从 `impl2.1/` 目录运行：**

```bash
cd /home/yinrong/post-train/impl2.1
```

---

## 一、P0：方向A + 方向B（优先执行）

### 步骤 1：数据生成（~2~3小时）

```bash
cd /home/yinrong/post-train/impl2.1
bash round3/prepare_data.sh
```

生成内容：
- `round3/data/boundary_ws.jsonl` — rule2/rule7 边界样本（180条，ws格式）
- `round3/data/boundary_ns.jsonl` — rule2/rule7 边界样本（180条，ns格式）
- `round3/data/ns_v5.jsonl` — 扩充 ns 池（300条）
- `round3/data/multirole_ws.jsonl` — 多角色 ws（200条，4角色各50条）
- `round3/data/train_R3-A.jsonl` — R3-A 训练集（~680条）
- `round3/data/train_R3-B.jsonl` — R3-B 训练集（~1000条）
- `round3/data/train_R3-AB.jsonl` — R3-AB 训练集（~1180条）

如果数据已存在，跳过生成步骤：

```bash
SKIP_GEN=1 bash round3/prepare_data.sh
```

### 步骤 2：训练 R3-A（方向A，弱规则专项）

```bash
bash round3/run_experiments.sh a
```

完成后检查 `round3/results/R3-A.json`：
- rule2 召回 ≥ 0.60？
- rule7 召回 ≥ 0.60？
- 整体 F1 ≥ 0.45？

### 步骤 3：训练 R3-B（方向B，扩数据）

```bash
bash round3/run_experiments.sh b
```

### 步骤 4：基于最优方向训练 R3-AB（A+B 组合）

```bash
bash round3/run_experiments.sh ab
```

---

## 二、P1：GRPO（方向C）

前置条件：`round2/history-route2.1.1/checkpoints/expYYY-merged` 存在

```bash
bash round3/run_experiments.sh grpo
```

GRPO 训练基于 LLaMA-Factory，reward_func 来自：
`common/tools/train/grpo_reward.py`

奖励组成：
- rule_f1 (0~1.0)：规则引擎 F1 主奖励
- cpk_bonus (0~0.1)：|pred_cpk - gt_cpk| < 0.1
- format_bonus (0~0.05)：含英文 rule 标识符

---

## 三、P1：QLoRA 量化（方向D）

```bash
bash round3/run_experiments.sh qlora
```

| 实验 | 基座 | 量化 | 预期显存 | 对比基准 |
|------|------|------|---------|---------|
| R3-D1 | Qwen3-14B | int4 NF4 | ~10GB/卡 | R3-AB（bf16 LoRA） |
| R3-D2 | Qwen3-32B | int4 NF4 | ~22GB/卡 | expHHH（bf16，F1=0.400） |

通过标准：量化 F1 损失 ≤ 2%

---

## 四、P1：8卡 72B（方向E）

### 步骤 1：下载 Qwen3-72B（~140GB，约2小时）

```bash
modelscope download --model Qwen/Qwen3-72B --local_dir /home/yinrong/models/Qwen3-72B
```

### 步骤 2：运行 8卡 ZeRO-3 训练

```bash
bash round3/run_experiments.sh 72b
```

---

## 五、目录结构

```
round3/
├── data/                    训练数据（生成后填充）
│   ├── boundary_ws.jsonl    方向A：rule2/rule7 边界样本（ws格式）
│   ├── boundary_ns.jsonl    方向A：rule2/rule7 边界样本（ns格式）
│   ├── ns_v5.jsonl          方向B：扩充 ns 池
│   ├── multirole_ws.jsonl   方向B：多角色 ws 数据
│   ├── train_R3-A.jsonl     R3-A 训练集
│   ├── train_R3-B.jsonl     R3-B 训练集
│   ├── train_R3-AB.jsonl    R3-AB 训练集
│   └── train_R3-grpo.jsonl  R3-C GRPO 数据集
├── configs/
│   ├── R3-A.yaml            R3-A 训练配置
│   ├── R3-B.yaml            R3-B 训练配置
│   ├── R3-AB.yaml           R3-AB 训练配置
│   ├── R3-C-grpo.yaml       R3-C GRPO 配置
│   ├── R3-D1-qlora-14b.yaml QLoRA 14B int4
│   ├── R3-D2-qlora-32b.yaml QLoRA 32B int4
│   ├── R3-E1-zero3-72b.yaml ZeRO-3 72B
│   └── deepspeed_zero3.json ZeRO-3 DeepSpeed 配置
├── checkpoints/             模型检查点（训练后生成）
│   ├── R3-A/
│   ├── R3-A-merged/
│   └── ...
├── results/                 评测结果 JSON
├── reports/                 汇总报告
├── logs/                    训练/评测日志
├── tools/data/
│   ├── gen_boundary_rule27.py   方向A：边界样本生成
│   ├── gen_expanded_ns.py       方向B：扩充ns/多角色ws
│   ├── merge_datasets.py        数据集合并
│   └── prepare_grpo_dataset.py  GRPO 数据格式转换
├── prepare_data.sh          数据准备主脚本
├── run_experiments.sh       实验运行主脚本
└── eval_and_report.sh       评测报告脚本
```

---

## 六、Go/No-Go 标准

| 实验 | 通过标准 | 失败处理 |
|------|---------|---------|
| R3-A（弱规则） | rule2/rule7 召回 ≥ 0.60 | 增加边界样本，调高 --n_rule2 --n_rule7 至 120 |
| R3-C（GRPO） | F1 > 0.46（expYYY + 0.03） | 调整 reward 权重，增大 num_generations |
| R3-D（QLoRA） | F1 损失 ≤ 2% vs 同规模 bf16 | 换 int8（quantization_bit: 8） |
| R3-E（72B）| 训练可完成，F1 ≥ 0.45 | 降到 4 卡，或改用 QLoRA 模式 |

**进入 round4 条件**：任意方向 F1 ≥ 0.50，或所有 goal.md 维度均有实测数据。

---

## 七、关键路径

```
prepare_data.sh
    ↓ 并行
    [A] gen_boundary_rule27.py
    [B1] gen_expanded_ns.py (ns)
    [B2] gen_expanded_ns.py (multirole_ws)
    ↓ 合并
    train_R3-A.jsonl / train_R3-B.jsonl / train_R3-AB.jsonl
    ↓ 注册
    spc_r3_A / spc_r3_B / spc_r3_AB → LLaMA-Factory
    ↓
run_experiments.sh p0 → R3-A + R3-B → 评测
    ↓ 若 R3-A 或 R3-B 更优
run_experiments.sh ab → R3-AB → 评测
    ↓（并行）
run_experiments.sh grpo  → R3-C → 评测
run_experiments.sh qlora → R3-D1 / R3-D2 → 评测
run_experiments.sh 72b   → R3-E1 → 评测
```

---

*生成时间：2026-04-28*
*起点：expYYY（F1=0.430，round2 最优）*
