# 技能内化项目面试问答

**项目**：SPC Nelson 规则 LLM 内化（Route 2.1.1，Round 1–3）  
**最后更新**：2026-05-11

---

## Q1：这个项目做了什么，解决了什么问题？

**背景**：公司有 40+ 篇超长技能文档（SPC/质检/工艺规则），每次推理都要把文档塞进上下文，Token 成本极高，且有文档泄露风险。

**方案**：通过监督微调（SFT）将技能文档"烧入"模型权重，使模型推理时无需任何文档，直接从权重中调用技能。

**验证任务**：SPC Nelson 规则检测。给定 25 个工序数据点，判断哪些 Nelson 规则被触发（8 条规则的多标签分类），并计算过程能力指数 CPK。

**核心结论**：  
SFT 14B 模型（无技能文档推理）在规则检测 F1 上超越 Claude Sonnet + 有技能文档（0.420 vs 0.266），CPK 提取率从 62% 提升至 100%。SFT 的核心优势不在规则知识量（两者持平），而在**格式一致性**——输出格式对齐评测标准，可程序化解析。

---

## Q2：训练数据是怎么生成的？整个流水线是什么样子的？

**两阶段流水线**：

**阶段 1：输入场景生成器**

用随机种子程序化生成多样化输入，例如 SPC：

```python
rng = random.Random(seed)
mu    = rng.uniform(45, 55)       # 随机过程均值
sigma = rng.uniform(2, 5)         # 随机标准差
data  = [rng.gauss(mu, sigma) for _ in range(25)]  # 25 个数据点
# 程序注入目标规则触发条件（如让第 8–16 点偏向均值同侧 → 触发 rule2）
```

生成约 1000 种不同的输入场景，覆盖规则触发的各种组合。训练集和测试集严格使用不同种子池。

**阶段 2：教师模型调用（with_skill 格式）**

```
System: [完整技能文档，~1200 tokens，含 Nelson 8 条规则定义 + CPK 公式]
User:   [阶段 1 生成的数据场景]
Assistant: <think>
  逐点分析数据，依次检查 rule1–rule8 触发条件，
  计算均值/标准差/连续点数……
</think>
{"violations": ["rule2", "rule5"], "cpk": 1.23, "report": "..."}
```

Claude Sonnet 4.6 作为教师模型，3 路并发 API 调用。

**阶段 3：质量验证**

每条生成的样本都用规则引擎（Python 确定性计算）验证答案正确性，不合格直接丢弃。这一步过滤了约 5–10% 的样本。

**阶段 4：格式转换（关键步骤）**

```python
def convert_to_noskill(example):
    messages[0]["content"] = "你是一名 SPC 工程师。"  # 移除技能文档
    return example
```

模型**必须**从 `<think>` 推理链中学习规则，而不能依赖系统提示中的文档。这步决定了内化是否成功：with_skill 格式训练的模型推理时无文档 F1 接近 0。

---

## Q3：如何微调？LoRA 参数怎么理解？

**微调框架**：LlamaFactory（round1–3 SFT），trl GRPOTrainer（round3 GRPO）  
**基座模型**：Qwen3-14B（14B 参数，支持 `<think>` 扩展思维链）

**LoRA 设置**：

```yaml
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_target: all       # 对所有线性层挂载
flash_attn: sdpa
bf16: true
```

**参数含义**：

LoRA 冻结原始权重矩阵 W（d×k），添加低秩分解 `ΔW = (alpha/rank) × B·A`：
- `B`：d×r 矩阵（可训练）
- `A`：r×k 矩阵（可训练）  
- `r=128`：控制表达能力，128 比 64 更能拟合复杂规则
- `alpha=256`：实际更新幅度 = alpha/rank = 2.0，值越大更新越激进

`lora_target: all` 对所有线性层（注意力 Q/K/V/O + FFN up/gate/down）挂载 LoRA，确保规则知识能写入足够多的参数位置。

**训练超参**：

```yaml
num_train_epochs: 5
per_device_train_batch_size: 2
gradient_accumulation_steps: 4   # effective batch = 8
learning_rate: 1.0e-4
cutoff_len: 5120                  # ← 最关键配置
```

---

## Q4：cutoff_len 是什么？为什么这么重要？

`cutoff_len` 是 LlamaFactory 训练时对每条样本（输入+输出）的截断长度。

**为什么是静默杀手**：SPC 任务的模型输出很长（推理链 ~4000 tokens + JSON 答案），而 cutoff_len 不够时，输出尾部会被截断，但**训练不报错**，loss 看起来也在下降。

**Round 1 实验**：cutoff_len=4096，SPC 输出均值 4191 tokens，94% 的样本推理链被截断。最终结果：rule8（在输出末尾判断）召回=0.00，其他规则 F1 也偏低（0.358）。

**修复**：cutoff_len=5120，Rule8 召回从 0.00 恢复正常。

**设置方法**：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
lengths = [len(tokenizer(ex["output"])["input_ids"]) for ex in data]
# cutoff_len 设为 max(lengths) 或 p100
print(f"max={max(lengths)}, p95={np.percentile(lengths,95):.0f}")
```

---

## Q5：如何评测？指标是什么？

**核心原则**：始终无技能文档评测。

```python
system_prompt = "你是一名 SPC 工程师。"  # 不含任何规则
```

评测时用 vLLM 部署 merged 模型，发 200 条测试样本（与训练集不同种子生成），用规则引擎提取并比较结果。

**主要指标**：

| 指标 | 含义 | Round 3 最优 |
|------|------|------------|
| **rule_detection_F1** | 规则预测集合 macro F1 | 0.420 |
| per_rule_recall | 每条规则单独召回率 | rule7 最低 0.333 |
| CPK_found_rate | CPK 值可提取比例 | 1.000 |
| CPK_MAE | CPK 数值误差 | <0.05 |

**F1 计算方式**（多标签集合匹配）：

```
Ground Truth: {rule2, rule5}
预测:         {rule2, rule7}

TP=1(rule2), FP=1(rule7多报), FN=1(rule5漏报)
Precision = 1/2 = 0.50
Recall    = 1/2 = 0.50
F1 = 0.50
```

---

## Q6：GRPO 是怎么做的？样本什么样子？怎么打分？

**动机**：SFT 的上限由教师数据质量决定。GRPO 直接优化任务指标（rule F1），理论上可突破教师数据上界。

### 训练样本格式

GRPO 输入是纯问题（无答案），不携带技能文档：

```
System: "你是一名 SPC 工程师。"   ← 无技能文档，测量内化程度
User:   "请分析以下采样数据，判断是否触发 Nelson 规则，计算 CPK：
         数据：[45.2, 46.8, 47.3, ...]
         USL=55.0, LSL=45.0"
```

模型为每条 prompt 采样 **G=4 个候选输出**，每个候选是完整的推理链+答案：

```
候选 1: <think>逐点分析 rule1...</think> {"violations": ["rule1"], "cpk": 1.23, ...}
候选 2: <think>分析后认为无异常...</think> {"violations": [], "cpk": 1.20, ...}
候选 3: ...（共 4 个）
```

### 打分（Reward 函数）

每个候选输出用程序化 Reward 函数打分，最高约 1.15 分：

```python
def reward_fn(prompts, completions, ground_truth=None, **kwargs):
    for completion, gt in zip(completions, ground_truth):
        # 1. 主奖励：规则检测 F1（0~1.0）
        pred_violations = extract_violations(completion)  # 正则提取 rule1~rule8
        rule_f1 = compute_f1(pred_violations, gt["violations"])

        # 2. CPK 精度奖励（0~0.1）
        pred_cpk = extract_cpk(completion)
        if pred_cpk and abs(pred_cpk - gt["cpk"]) < 0.1:
            cpk_bonus = 0.1
        elif pred_cpk:
            cpk_bonus = 0.05  # cpk 存在但不精确

        # 3. 格式奖励（0~0.05）
        format_bonus = 0.05 if re.search(r'\brule[1-8]\b', completion) else 0.0

        total = rule_f1 + cpk_bonus + format_bonus  # 最高约 1.15
```

### 策略梯度更新

```
advantage_i = reward_i - mean(rewards_for_this_prompt)
```

同一 prompt 的 4 个候选互相比较：高于均值的候选被强化，低于均值的被抑制。  
KL 惩罚（beta=0.01）防止模型偏离参考模型太远。

### R3-C 实际结果

| 指标 | R3-C (GRPO) | R3-AB-v2 (SFT 最优) |
|------|------------|---------------------|
| rule_detection_F1 | **0.003** | 0.420 |
| has_reasoning_rate | **0.0** | — |
| cpk_found_rate | 0.99 | 1.000 |
| 训练时长 | 18h02m / 255步 | — |

**F1=0.003，GRPO 完全失败，SFT 能力被灾难性覆盖。**

### R3-C 失败根因：Reward Hacking（奖励函数捷径）

`has_reasoning_rate=0.0` 是关键诊断信号——模型完全放弃了 `<think>` 推理链。

**模型发现的捷径**：始终预测 `violations=[]`（空）+ 正确计算 CPK：
- 约 50% 无违规样本：F1=1.0，reward ≈ 1.0+0.1 = 1.1
- 约 50% 有违规样本：F1=0.0，reward ≈ 0+0.1 = 0.1
- 平均 reward ≈ 0.6 —— 与训练 log 中观测到的 reward_mean=0.57~0.65 完全吻合

这个"懒惰策略"得到稳定的中等奖励，比真正检测规则（高风险高方差）更容易被强化。经过 3 epoch，模型完全放弃规则检测，SFT 学到的能力被覆盖。

### 之前修复的四个工程 bug（已验证）

| Bug | 现象 | 原因 | 修复 |
|-----|------|------|------|
| max_completion_length=2048 | clipped_ratio=1.0，reward=0 | SPC 答案均值 4191 tokens，全截断 | 改为 5000 |
| 使用 LlamaFactory 框架 | 无法运行 | LlamaFactory 0.9.3 依赖损坏 | 改用 trl 1.3.0 GRPOTrainer |
| 输入含技能文档（with_skill） | 测量的是"有文档"性能 | prepare_grpo_dataset.py 未清除 system | 强制替换为 no_skill system |
| 1460 样本 × 8 候选 | 单步 280s，预计 115h | 生成量过大 | 分层采样 171 样本 + 4 候选，降至 18h |

这四个 bug 已在 R3-C 中全部修复，训练正常完成。但 reward 函数设计缺陷（允许空预测捷径）导致新的失败——说明 GRPO 工程问题和 reward 设计问题是两类独立风险，需分别解决。

---

## Q7：有哪些关键发现和反直觉结论？

**1. 数据质量 > 数据数量**  
R3-AB（1180 条，含截断样本，F1=0.383）→ R3-AB-v2（去截断后 1131 条，F1=0.420）。  
减少了 49 条样本，F1 提升 9.7%。之后又加 640 条（R3-AB-v3），F1 几乎没变（0.418）。

**2. 上采样对反直觉规则无效**  
rule7（"连续 15 点全在 ±1σ 内 = 异常"）是反直觉规则。3× 上采样（R3-AB-v4）对 rule7 召回无效（0.278），但意外提升了 rule2（0.552），因为 rule7 样本中包含连续点判断逻辑，间接帮助了 rule2。

**3. 量化损失远超预期**  
int4/int8 量化在 SPC 这类精确数值任务上损失 7–10%（而一般文本任务通常 <2%）。原因：CPK 需要精确浮点计算，量化精度敏感。

**4. 技能文档在 Claude 中反而降低 F1**  
Claude + 有技能文档（0.266 F1）< Claude 无技能文档（0.424 F1）。  
原因：有文档时 Claude 被驱动输出冗长分析报告，JSON 解析失败率从 14% 升至 38%。

**5. 提取器偏差掩盖了真实性能**  
原始测试中 Claude 无技能 F1=0.110（用为 SFT 设计的 regex 提取器）。换成 JSON 提取器后重测，Claude 无技能 F1=0.424。**SFT 的优势不在规则知识，而在格式可靠性**（CPK 提取率 100% vs 88%）。

**6. GRPO reward 捷径可灾难性覆盖 SFT 能力**  
R3-C：修复四个工程 bug 后，GRPO 训练正常完成（18h），但 F1 从 0.420 跌至 0.003。  
原因：reward 函数允许"始终预测空违规+正确 CPK"这一捷径（期望 reward≈0.6），比真正学规则更稳定。模型在 3 epoch 内完全收敛到该捷径，has_reasoning_rate=0.0。  
**结论：GRPO reward 函数必须封堵所有高收益捷径，否则比不训练更差。**

---

## Q8：模型选型和量化怎么做决策的？

| 配置 | F1 | 推理显存 | 综合得分 |
|------|-----|---------|---------|
| 14B bf16 LoRA（推荐） | **0.420** | 96GB | 0.093 |
| 14B QLoRA int4 | 0.394（-6.2%） | ~20GB | **0.427** |
| 14B QLoRA int8 | 0.391（-6.9%） | ~20GB | **0.428** |
| 32B QLoRA int4 | 0.379（-9.8%） | ~45GB | 0.085 |

综合得分 = `任务F1 × (1-退化率) / (推理时间_s × 推理卡数 × 单卡显存_GB) × 1000`

**结论**：若对精度要求高（如本 SPC 任务），用 14B bf16 LoRA；若显存受限（部署 20GB 卡），接受 6% 精度损失用 14B int4。**不推荐 32B int4**：更大参数量被量化损失完全抵消，F1 反而更低。

---

## Q9：退化率是什么？SFT 对通用能力有多大影响？

**退化率**：领域 SFT 后通用基准的平均得分下降幅度：

```
退化率 = Σ (SFT前基准得分 - SFT后基准得分) / 基准数
```

测试基准：gsm8k（数学）、hellaswag（常识推理）、mmlu（综合知识）等 6 项。

| 实验 | F1 | 退化率 | 主要来源 |
|------|-----|--------|---------|
| R3-AB-v2（14B bf16） | 0.420 | 3.08% | gsm8k -18.7% |
| R3-D1（14B int4） | 0.394 | 2.33% | gsm8k -14.5% |
| R3-D2（32B int4） | 0.379 | **0.45%** | 极小，均匀分布 |

**关键发现**：
- 领域 SFT 主要冲击数学推理（gsm8k），原因是 SPC 训练数据的思维模式与数学推理的思维模式存在一定竞争
- 更大模型（32B）退化率更小，通用能力更鲁棒
- 约 200–1000 条领域数据的 SFT 带来约 3% 的退化，在可接受范围内

---

## Q10：如果 F1 还是达不到目标（0.50），下一步怎么做？

**P0（直接提升 F1）**：

1. **rule7 对比样本对**：构造正例（15 点全在 ±1σ 内 → 触发）和负例（第 15 点略超出 → 不触发）成对训练。上采样对反直觉规则无效，需要对比学习。

2. **数据质量审核**：人工审查 R3-AB-v2 基础数据，对截断样本补充完整推理链，而非简单扩充。

3. **GRPO reward 函数重设计**（Round 4）：R3-C 因 reward 捷径完全失败（F1=0.003）。修复方向：
   - 空预测惩罚：`violations=[]` 但 ground truth 非空时，额外 -0.3
   - 强制推理链：无 `<think>` 块时 reward 乘零
   - 提高 KL beta（0.01→0.05），防止过度偏离 SFT 基础

**P1（补齐维度）**：

4. **72B 大模型**：Qwen3-72B，8 卡 ZeRO-3，验证参数量上界对 F1 的影响。

5. **反直觉规则知识注入**：在推理链中显式写入"连续 15 点全在 ±1σ 内的概率约为 0.683^15 ≈ 0.23%，统计上属于异常"这类统计知识，让模型从原理层面理解 rule7。

---

*基于 Route 2.1.1 Round 1–3 全部实验（20+ 次训练）*  
*更新：2026-05-12（含 R3-C GRPO 实际结果与失败诊断）*
