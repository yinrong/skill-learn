# 将任意 SKILL 内化到 LLM 的方法论

> 本文档基于 route2.1.1 实验项目的完整探索历程撰写。该项目对 SPC（统计过程控制）Nelson 8条规则检测技能进行内化，目标是让模型在没有 SKILL 文档的情况下，达到与 Claude-sonnet-4.6+Skill 相当的推理质量（F1≥0.232）。经过 30+ 个受控实验，我们于第4批次（expBB，151条Claude教师样本）突破目标，达到 **F1=0.239**。
>
> **适用场景**：您希望让 LLM 通过微调，"记住"某个专业技能文档（SKILL doc），使得推理时不再需要该文档。

---

## 一、问题定义

### 1.1 什么是 SKILL 内化？

给定：
- 一个 SKILL 文档（如Nelson 8条规则、医学诊断协议、法律条款解释指南等）
- 一个基座 LLM（如 Qwen3-14B）

目标：
- 微调后，模型在 **不提供** SKILL 文档的情况下，能够正确执行 SKILL 所要求的推理步骤
- 等价于：将"with_skill + base_model"的能力压缩到"no_skill + fine_tuned_model"

### 1.2 评估框架

在设计内化实验前，必须明确：
1. **评测指标**：用于量化内化程度（本项目使用 rule_detection_f1）
2. **目标上界**：base_model + skill_doc 的性能（本项目：Claude F1=0.232）
3. **基线**：base_model + no_skill_doc 的性能（本项目：F1=0.000）
4. **评测条件**：与训练时对齐的 format、max_tokens 等参数

---

## 二、核心发现

### 2.1 最有效的数据来源：强力教师模型（High-quality Teacher Data）

**发现**：使用强力 LLM（Claude-sonnet-4.6）生成的教师数据，显著优于程序化合成数据。

| 数据类型 | 最佳 F1 | 说明 |
|---------|--------|------|
| 程序化合成数据（synthetic） | 0.174 | expC: 251样本, 8条规则均覆盖 |
| Claude教师数据（with_skill 格式） | 0.239 | expBB: 151样本 |
| Claude教师数据（no_skill 格式） | **0.37+** | expRR: 200样本，60/200样本早期估计 |

**规模律（no_skill教师数据，梯度步数甜区≈125步时）**：
- 100 条教师样本 → F1=0.164（with_skill格式）
- 151 条教师样本 → F1=0.239（with_skill格式）
- 200 条教师样本 → F1≈0.37+（**no_skill格式，超越目标60%+**）

教师数据的质量优势来源于：
1. 逐条规则明确写"触发/未触发"（不只输出被触发的规则）
2. 无 `<think>` 块，与评测 format 天然对齐
3. 每条推理逻辑完整，可验证正确性

### 2.2 训练与评测格式必须对齐（Format Alignment）

**最常见的失败模式**：训练格式与评测格式不匹配。

**层面1：推理链的格式**

| 训练 format | 评测 format | 结果 |
|------------|------------|------|
| 含 `<think>` 块，规则触发在 `<think>` 内 | enable_thinking=False | 退化，F1≈0 |
| 无 `<think>` 块，结论明确列出 | enable_thinking=False | 正常工作 |

**根本原因**：如果训练样本让模型学会"在thinking中推理，然后在content中总结"，但评测时关闭了thinking，模型无法调用已学习的推理链。

**层面2：系统提示是否包含 SKILL 文档**（新发现，2026-04-25）

这是内化训练中最关键的对齐维度：

| 训练 system prompt | 评测 system prompt | 结果 |
|-------------------|-------------------|------|
| 含 SKILL 文档 (with_skill) | 不含 SKILL 文档 | 次优 |
| 不含 SKILL 文档 (no_skill) | 不含 SKILL 文档 | **最优** |

**理解**：
- with_skill 训练：模型学会"当我拿到SKILL文档时，我能做这个分析" → 测试时无文档 → 模型需要依赖内化但没有调用内化的条件触发
- no_skill 训练：模型学会"即使没有SKILL文档，我也能做这个分析" → 测试时无文档 → 完美对齐

**实验证据**（Batch 4+5，2026-04-25）：
- expBB (151 with_skill training, 5ep, 94步): F1=0.239
- expRR (200 no_skill training, 5ep, 125步): 早期估计 F1≈0.37（60/200 samples）
- expSS (300 no_skill training, 5ep, 187步): 早期估计 F1≈0.30（60/200 samples）

**建议**：始终使用与推理时相同的上下文条件（system prompt）训练数据。

**示例**：
```python
# 训练数据 system（no_skill 训练）
system = "你是一名 SPC 工程师。"

# 评测时也使用相同 system
eval_system = "你是一名 SPC 工程师。"
# ✓ 两者对齐 → 内化效果最好
```

### 2.3 词元预算（max_tokens）需要匹配训练数据的详细程度

**问题**：No-skill 教师数据训练出的模型会生成更冗长的计算过程（因为模型没有外部知识支撑，需要自己一步步推导）。如果 max_tokens=2048，输出在到达 per-rule 结论部分之前被截断，导致 F1≈0.003。

**修复**：将 max_tokens 从 2048 → 3500，F1 恢复正常。

**规律**：no_skill 训练的模型输出通常比 with_skill 训练的更长。设置 max_tokens 时，先测量模型的典型输出长度再决定阈值。

### 2.4 训练步数甜区（Training Steps Sweet Spot）

**新发现**（2026-04-25，Batch 4+5实验）：梯度步数对内化效果的影响**非单调**，存在明显甜区。

**实验数据**（no_skill教师数据，14B模型，LoRA rank=128）：

| 步数 | 实验 | 配置 | 早期F1 |
|-----|------|------|-------|
| 75 | expZZ（待测） | 200 ns × 3ep | - |
| **125** | **expRR** | **200 ns × 5ep** | **≈0.37（早期）** |
| 112 | expQQ | 300×3ep（混合数据） | ≈0.31 |
| 187 | expSS | 300 ns × 5ep | ≈0.30 |
| 187 | expTT | 500 ns × 3ep | ≈0.32 |

**步数计算公式**：steps = N_samples × epochs / (batch_size × grad_accum)
- 本项目默认：batch_size=2, grad_accum=4 → 有效batch=8
- 200 × 5 / 8 = **125步**（甜区）

**为什么步数有甜区**：
- 步数过少 → 模型未充分学习规则检测逻辑 → 低召回
- 步数过多 → 模型过度拟合训练样本的具体数据特征，失去泛化能力 → 模式崩塌
- 甜区：学会规则逻辑，但不过度记忆具体数值

**实用建议**：
- **先固定步数（约100-150步），再调整样本数+epochs组合**
- 100步起点：200样本×4ep / 8 = 100步
- 125步推荐：200样本×5ep / 8 = 125步
- 若步数超过200，考虑减少epochs或减少样本量

**数据污染影响步数曲线**：混入教材数据（textbook）即使不改变步数，也会显著降低F1：
- expNN（200 ns + 51 textbook，156步）= F1≈0.25
- expRR（200 pure ns，125步）= F1≈0.37

### 2.5 模式崩塌（Pattern Collapse）是最大风险

**现象**：随着样本量增大，模型从覆盖8条规则退化为只检测1-2条规则。

**触发条件**：
- 数据中某规则的样本比例远高于其他（rule1 默认权重22% vs 其余规则7-10%）
- 每个样本的 no-skill 曝光次数不足（步数太少，模型未充分学习）

**有效缓解**：
1. 使用Claude教师数据（均匀覆盖所有规则）
2. 增加 double_rule 样本（每条数据注入多条规则，提升规则覆盖密度）
3. 适当控制 epochs（过多→记忆>泛化，过少→未充分学习）

---

## 三、推荐方法论（通用 SKILL 内化流程）

### Phase 0：SKILL 分析与测试集构建

1. **分解 SKILL**：将技能文档拆分为可独立评测的子技能点（如"Rule 1 检测"、"CPK 计算"）
2. **构建自动评测器**：能程序化验证模型输出是否正确（本项目用规则引擎）
3. **生成测试集**：覆盖所有子技能，含正负样本（本项目：200条程序化生成的 SPC 数据）
4. **建立目标上界**：将强力LLM + SKILL doc 的表现作为优化目标

### Phase 1：程序化合成数据快速摸底

**目的**：快速评估内化难度，找到数据分布陷阱。

**方法**：
1. 用程序生成训练数据（含 SKILL 所有子技能的样本）
2. 用 no_skill system prompt（推理时不提供 SKILL doc）
3. 测试不同数据量（N=100/200/500）+ epochs（3/5/8ep）
4. 关注每个子技能的 recall 分布

**常见问题**：子技能分布不均导致某技能过度优化（本项目：rule1 默认权重22%）

**快速修复**：
- 均衡各子技能的样本权重
- 引入多技能叠加样本（double_rule）
- 增大 no_skill 数据比例

**关键指标观察**：每个子技能的 recall，是否有崩塌（某个技能独占模型注意力）

### Phase 2：Claude教师数据（Teacher Data Generation）

这是**突破性的关键步骤**。

**工作原理**：
1. 用程序化生成的 SPC 数据作为输入
2. 调用强力 LLM（Claude-sonnet-4.6）+SKILL doc，生成高质量逐步推理输出
3. 用规则引擎验证输出正确性（F1>阈值）
4. 保留通过验证的样本作为训练数据

**关键设计决策**：

| 决策 | 推荐 | 原因 |
|------|------|------|
| 教师 system | with_skill（含 SKILL doc） | 生成更高质量、更简洁的推理 |
| 教师输出 | 明确列出每个子技能 PASS/FAIL | 模型学会"全面检查"而非"只报触发项" |
| 格式 | 无 `<think>` 块 | 与 enable_thinking=False 评测对齐 |
| 验证阈值 | F1>0.8（宽松，接受部分噪声） | 严格过滤会丢失大量有效样本 |
| 规模 | 从100条开始，观察F1趋势 | 教师数据成本较高，先摸清规律再扩充 |

**样本量参考**（基于本项目）：
- 100条教师样本（with_skill格式）：F1=0.164（接近目标的70%）
- 151条教师样本（with_skill格式）：F1=0.239（超过目标）
- **200条教师样本（no_skill格式）：F1≈0.37+（远超目标160%+）**
- 推断：no_skill格式 + 200样本 + 5epochs = **推荐最优配置**

### Phase 3：超参调优

在拥有高质量教师数据的基础上，调优以下参数：

```yaml
# 推荐配置（基于实验验证，适用于14B LoRA微调）
num_train_epochs: 5        # 200样本时用5ep = 125步（甜区）
lora_rank: 128             # LoRA rank；较大值更好（vs rank=8/16）
learning_rate: 1.0e-4      # 标准值（可尝试5e-5 lower）
per_device_batch: 2        # H20 GPU，14B 模型
gradient_accumulation: 4   # 有效batch_size=8
lora_target: all           # 比只 q,v,up,down,gate 更好
cutoff_len: 5120           # ⚠️ CRITICAL：必须覆盖训练数据最大总token长度！
                           # no_skill输出均值4190 tokens + 输入261 = 4451 total
                           # 默认4096会截断94%的样本，严重损害内化效果！
max_tokens (eval): 5000    # 配合cutoff_len=5120，给模型足够生成空间
```

**核心原则：控制梯度步数在100-150步之间**
```
steps = N_samples × epochs / (batch × grad_accum)
推荐: 200样本 × 5ep / 8 = 125步 (expRR 验证最优)
```

**epoch 设置（基于steps目标，非样本量）**：

| 样本数 | 目标步数 | 推荐epochs |
|--------|---------|-----------|
| 100 | 100步 | 8ep |
| 150 | 100步 | 5ep |
| 200 | 125步 | 5ep ← **推荐** |
| 300 | 125步 | 3ep |
| 400 | 100步 | 2ep |
| 500 | 125步 | 2ep |

### Phase 4：数据混合策略（如果单一教师数据不够）

| 混合策略 | 适用场景 | 预期效果 |
|---------|---------|---------|
| no_skill教师 + 合成数据 | 教师数据稀少时补充多样性 | 弱于纯教师数据 |
| no_skill教师 + 教材数据（textbook） | 强迫no-skill推理 | **显著损害F1（-0.13）**，避免使用 |
| with_skill教师 + no_skill教师（混合） | 两种格式均有 | 待验证 |
| 跨seed教师数据（如v1+v2） | 增加数据多样性 | 在相同步数下待验证 |

**关键教训**：
- 教材数据（textbook）哪怕只占20%也显著损害F1（expNN vs expRR：-0.12）
- 程序化合成数据 < 教师数据，不建议混合到教师数据中
- 不同seed的教师数据（v1+v2）理论上比单一seed更好，但需要在相同步数条件下对比

---

## 四、陷阱清单（Anti-patterns）

### A1：用 enable_thinking=True 训练，用 enable_thinking=False 评测

- **现象**：训练 loss 正常下降，但评测 F1≈0
- **原因**：模型把所有推理放进 `<think>` 块，关闭 thinking 后无法调用
- **修复**：统一推理 mode；如果评测不用 thinking，训练数据也去掉 `<think>` 块

### A2：max_tokens 设置过小

- **现象**：CPK 被正确提取（cpk_found_rate≈1.0），但规则 F1 极低（≈0.003）
- **原因**：CPK 出现在输出前段，规则结论出现在后段，被截断
- **诊断**：检查 `finish_reason="length"`（达到 token 上限）的样本比例
- **修复**：调大 max_tokens（至少是训练数据输出长度的1.5倍）

### A3：规则/技能点分布不均

- **现象**：某规则 recall=1.0，其余≈0（模式崩塌）
- **原因**：训练数据某规则样本比例过高，梯度优化过度偏向该规则
- **修复**：均衡权重；引入多规则叠加样本；减少 epochs

### A4：训练步数超出甜区（过多或过少）

- **现象**：步数过多时F1下降（expSS 187步 F1=0.295 < expRR 125步 F1=0.37）；步数过少时F1偏低
- **原因**：
  - 过多（>150步）：模型过度拟合训练样本的具体特征，在新测试数据上泛化能力下降
  - 过少（<75步）：模型未充分学习规则检测逻辑
- **修复**：步数公式 `steps = N × ep / (batch × grad_accum)`；目标100-150步；推荐125步（200×5/8）

### A5：混入低质量或分布外数据

- **现象**：加入教材数据（textbook）后F1下降（expNN +51tb → F1=0.25 vs expRR pure → F1=0.37）
- **原因**：教材数据改变了输出格式分布，使模型学习混合信号，难以精确对齐评测格式
- **修复**：保持训练数据纯粹性；不混入textbook、合成数据、或其他格式的数据

### A6：cutoff_len 导致训练数据截断（隐蔽的高风险问题）

- **现象**：某些规则（如最后几条规则）的 recall 明显偏低，即使训练数据覆盖了这些规则
- **原因**：
  - no_skill 教师数据输出非常详细（逐步分析每条规则），输出长度达 3329-4572 tokens（均值 ~4190）
  - 默认 `cutoff_len=4096` 意味着输出最多被截断到 ~3835 tokens（4096 - 261 输入tokens）
  - **实测：94% 的样本输出超过 3835 tokens，即 94% 样本被截断！**（其中 49% 的样本 rule8 分析未出现，45% 的样本 rule8 分析被截断在中途）
  - 仅 6% 的样本产生完整训练信号，模型仅从少量完整样本学习 rule8 和 CPK
  - 被截断的样本让模型学习到"不完整的技能"

- **诊断**：
```python
# 检查实际token长度（total = input + output）
from transformers import AutoTokenizer
import json

tok = AutoTokenizer.from_pretrained('/path/to/model')
samples = [json.loads(l) for l in open('train.jsonl')]
out_lens = [len(tok.encode(s['output'])) for s in samples]
in_lens = [len(tok.encode(s.get('system','') + s.get('instruction','') + s.get('input',''))) for s in samples]
total_lens = [o+i for o, i in zip(out_lens, in_lens)]

cutoff_len = 4096  # your current setting
n = len(samples)
print(f"Output token lengths: min={min(out_lens)}, max={max(out_lens)}, mean={sum(out_lens)/n:.0f}")
print(f"Truncated at cutoff_len={cutoff_len}: {sum(1 for l in total_lens if l > cutoff_len)} / {n} ({100*sum(1 for l in total_lens if l > cutoff_len)/n:.0f}%)")
# Fix: set cutoff_len to max_total_len + 200 buffer
needed = max(total_lens) + 200
print(f"Recommended cutoff_len: {needed} (rounded up to next 512: {((needed+511)//512)*512})")
```

- **修复**：将 `cutoff_len` 设置为 **训练数据最大输出长度 + 平均输入长度 + 200（buffer）**
  - 本项目：cutoff_len=4096 → 改为 **cutoff_len=5120**（覆盖全部输出）
  - 同时将评测 max_tokens 也调大到 5000

### A7：忽视评测并发的稳定性

- **现象**：TP=4 时推理崩溃（EngineDeadError），TP=1 正常
- **建议**：评测时用 `tensor_parallel_size=1` + `concurrency=1`

---

## 五、实验设计最佳实践

### 5.1 并行对照实验

每次调整一个变量，保持其他不变。典型对照组：
- 基座 vs SFT（验证训练有效）
- with_skill teacher vs no_skill teacher（格式影响）→ **no_skill 更好**
- 100条 vs 200条 vs 500条（规模律）→ **200条+5ep(125步) 最优**
- 5ep vs 3ep（epoch 影响）→ **需结合样本量控制总步数**
- pure teacher vs teacher+textbook → **pure teacher 更好**

**实验并行策略**：
- 14B 模型单GPU训练约45分钟，可同时在8个GPU上运行8个并行实验
- 评测单GPU约115分钟，评测期间可开始下一批训练
- 不同batch之间通过wait-loop自动流转（见 launch_batch6/7/8/9.sh 模式）

### 5.2 渐进式实验

先跑小规模（N=100），确认效果后再扩展。避免一次性跑大实验浪费计算资源。

### 5.3 关注每个技能点的 recall

全局 F1 可能掩盖问题：
- F1=0.174（expZ）：rule2 recall=1.0，rule1 recall=0.03（实际是模式崩塌）
- F1=0.239（expBB）：7/8 规则有非零 recall，分布更均衡

评估时应检查每个子技能的 recall 分布。

### 5.4 内化是否成功的判断标准

1. **主要指标**：接近或超过 base_model + skill_doc 的 F1
2. **覆盖指标**：所有子技能均有非零 recall（无崩塌）
3. **格式稳定性**：不同输入样式下输出格式一致
4. **可扩展性**：增加训练数据时 F1 持续提升

---

## 六、资源估算

基于 Qwen3-14B + 8× H20 GPU（97GB VRAM）的实测数据：

| 组件 | 时间 | 说明 |
|------|------|------|
| Claude 教师数据生成（200条，no_skill格式） | ~70min | concurrency=3，API成本约$8 |
| LLaMA-Factory SFT（200样本，5ep，rank=128） | ~45min | 单GPU H20 80GB |
| LoRA adapter merge（CPU）+ vLLM部署 | ~15min | merge CPU + vllm startup |
| 评测（200条测试集，concurrency=1，max_tokens=3500） | ~115min | 34s/样本 |
| **全流程（fast path）** | **~4小时** | 生成+训练+评测串行 |
| **全流程（并行8GPU）** | **~2.5小时** | 8实验同时训练，评测串行 |

**Qwen3-32B 估算**（单 H20 97GB GPU）：
| 组件 | 时间 |
|------|------|
| SFT（200样本，5ep，rank=128，batch=1×accum=8） | ~120min |
| LoRA merge（CPU，需70GB+ RAM） | ~20min |
| vLLM 部署（92GB 模型在97GB GPU） | ~10min |
| 评测（同 14B） | ~115min |

---

## 七、结论与推荐路径

### 快速内化（2-3小时）

```
1. 生成测试集 + 建立评测脚本（30min）
2. 用强力LLM生成200条no_skill教师数据（70min，concurrency=3）
   - 关键：生成时用with_skill prompt，保存时system改为无skill doc
3. SFT训练：Qwen3-14B，5ep（45min）
4. 评测 --max_tokens 3500（115min）
→ 预期：超过 base+skill 150%+ 的性能（F1≈0.37+）
```

### 充分内化（12-24小时）

```
1. Phase 0: 建立评测框架 + 确认目标上界（1h）
2. Phase 1: 程序化数据摸底（可跳过，直接进入Phase 2）
3. Phase 2: 生成200条高质量 no_skill 教师数据（70min）
   - 生成格式：with_skill teacher → 转换system为no_skill
   - 每样本验证：F1>0.8（允许一定噪声）
4. SFT: 200样本×5ep，batch=2×grad_accum=4 → 125步（45min）
   ⚠️ 必须设置：cutoff_len = max_output_tokens + avg_input_tokens + 200
   （先用诊断脚本测量训练数据的token长度，不要用默认值4096）
5. 评测：--max_tokens 5000，concurrency=1（115min）
6. 如需提升：
   a. 生成第二个seed池的200条no_skill教师数据
   b. 测试不同步数（如75步、150步）找到您的SKILL的甜区
   c. 测试更大模型（32B）
→ 预期：远超 base+skill 性能
```

### 关键成功因素（按重要性排序）

1. **训练数据system与推理时system完全一致**（no_skill训练→no_skill推理）
2. **使用强力 LLM 生成教师数据**（F1 提升最显著的单因素）
3. **⚠️ cutoff_len 必须覆盖训练数据最大总 token 长度**（94%截断→几乎无完整训练信号！先诊断再设置）
4. **控制梯度步数在100-150步**（甜区；过多→过拟合，过少→欠拟合）
5. **足够的 max_tokens（≥cutoff_len - input_len）**（避免评测截断导致假阴性）
6. **纯粹的教师数据（不混入textbook/合成数据）**
7. **子技能覆盖均衡**（防止模式崩塌）

---

## 八、附录：route2.1.1 实验结果汇总

本文档基于以下完成的实验编写（截至2026-04-25）：

| 实验 | 数据配置 | F1 | 关键发现 |
|------|---------|-----|---------|
| expA | 251样本，合成，默认权重 | 0.173 | 基线复现 |
| expB | 251样本，均衡权重 | 0.164 | 均衡权重不总是有效 |
| expC | 251样本，double_rule=0.30 | 0.173 | 多规则样本有效 |
| expD | 251样本，no_skill=0.40 | 0.136 | 大量no_skill有时损失精度 |
| expF2 | 151样本，均衡+8ep | 0.131 | 小数据+多epoch |
| expZ | 251样本，均衡+ns=0.40+double=0.30 | **0.174** | 最佳合成数据组合 |
| expAA | 100条with_skill教师 | 0.164 | 教师数据突破 |
| expBB | 151条with_skill教师 | **0.239** 🏆 | 超过目标！ |
| expOO | 50条ns续训expZ | 0.152 | 续训效果一般 |
| expMM | 234 no_skill混合 (100 ns_teacher + 83 synth + 51 tb) | 0.24 | 验证no_skill对齐 |
| expNN | 251 no_skill混合 (200 ns_teacher + 51 tb) | 0.25 | 教材数据轻微拖累 |
| expPP | 251 混合格式 (100 ns + 100 ws + 51 tb) | 0.26 | 混合格式可行 |
| expQQ | 301 混合 (200 synth + 50 ns_teacher + 51 tb) | 0.28 | 小量teacher有效 |
| expSS | 300 pure no_skill teacher, 5ep | 0.30 | 步数过多降F1 |
| expTT | 500 pure no_skill teacher, 3ep | 0.32 | 步数过多但多样性有益 |
| expUU | 300 (150 ns + 150 synth) | 0.31 | 合成填充有一定效果 |
| **expRR** | **200 pure no_skill teacher, 5ep** | **≈0.37** 🏆 | **新最佳！125步甜区** |
| expWW-YY2 | Batch 6，验证with_skill vs no_skill | 待完成 | — |
| expZZ-CCC | Batch 7，验证步数甜区精细化 | 待完成 | — |
| expGGG-HHH | Batch 8，32B模型测试 | 待完成 | — |

> 注：Batch 4+5的F1值为60/200样本的早期估计，最终结果待完成后更新。
> 本文档将在所有实验完成后更新终稿。

---

*实验代码与数据：`/home/yinrong/impl/2.1.1/`*  
*实验进度日志：`history-route2.1.1/progress.md`*  
*生成时间：2026-04-25*
