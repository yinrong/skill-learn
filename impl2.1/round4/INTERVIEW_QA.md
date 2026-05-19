# LLM 技能内化：踩坑面试题集

> 基于 Round 1–4 真实实验教训整理
> 覆盖：数据准备、训练配置、评估设计、强化学习、工程实施、方法论

---

## 一、数据准备

---

**Q1：用生产日志做 SFT 训练数据，有什么隐患？**

**简明答案**：生产日志可能包含捷径行为（如结果缓存命中），这些行为的参数来源在训练时不可见，模型学到的是记忆而非流程。

**详细讲解**

生产系统往往有记忆注入或缓存机制。以工厂数据查询为例，线上模型在第二次查询同类指标时，会直接从缓存取出上次的 `model_id`，跳过 `list_object_types → get_object_type_metrics` 的发现步骤。日志里的轨迹因此呈现：

```
user: 查 UPH
assistant: get_idi_model_data(model_id=12202)   ← 12202 从缓存来，日志里无记录
```

做 ns 格式转换（删除 skill doc 和记忆注入）后，训练样本变成"参数来源不明的工具调用"。模型无法从中学到发现流程，只能死记 UPH→12202 这个映射，遇到新场景就失败。

**验证方法**：训练前对每条样本逐步检查，确保每个工具调用的参数能从当前可见的上下文（用户输入 + 前序工具返回值）中找到来源。来源不明的样本过滤或重建。

---

**Q2：训练数据中同类查询的 model_id 始终是同一个值，有什么风险？**

**简明答案**：模型会记住这个固定值，而非学会通过工具发现值。换一个 model_id 就失败。

**详细讲解**

如果 UPH 查询在所有训练样本里都用 `model_id=12202`，模型会把"UPH 查询 → 12202"作为一个固定规律记入权重，而不是学到"调 get_object_type_metrics，读返回值里的 modelId 字段"这个流程。

这种记忆化在训练集上表现完美（loss 极低），但换一个环境（model_id 变成 99999）立即失效。

**解决方法**：同类查询的训练样本必须覆盖 ≥3 个不同 model_id。生产日志多样性不足时，用教师数据生成补充，生成时随机化 model_id 并保证与工具响应一致。

---

**Q3：SFT 训练时，output 中缺少推理链（只有最终答案），会导致什么？**

**简明答案**：ns 格式下模型无法内化 skill 知识，推理时遇到新问题会直接凭直觉答而非按流程走。

**详细讲解**

SFT 只对 output 计算 loss。skill doc 中的规则和流程知识，如果没有在 output 的推理链里展开，删除 skill doc（ns 转换）后这些知识就彻底消失了：

```
output（知识未展开）：
  get_idi_model_data(model_id=12202)   ← 模型只学到"调这个函数"

output（知识已展开）：
  <think>
    list_object_types 返回 apiName=line_operation
    get_object_type_metrics 返回 modelId=12202（UPH 模型）
    因此用 model_id=12202 查询
  </think>
  get_idi_model_data(model_id=12202)   ← 模型学到了完整推理链
```

教师数据生成时，必须要求 Claude 在 thinking block 中完整展开决策过程，不能只输出工具调用指令。

---

**Q4：训练集和测试集用了相同的生产日志来源，有什么问题？**

**简明答案**：同一批日志中相似问题可能有重叠，导致测试集泄露，高估模型泛化能力。

**详细讲解**

工厂生产系统中，同类查询（如"今天 S04 线 UPH"）在不同时间会反复出现。如果不按时间或来源隔离，训练集里的某个查询可能和测试集里的查询高度相似，模型直接"背题"就能得高分。

正确做法：训练集和测试集来自不同时间段的日志，或使用不同随机种子生成（教师数据方案）。

---

## 二、训练配置

---

**Q5：cutoff_len 设置不够大，会产生什么症状？你怎么排查？**

**简明答案**：output 末尾内容被截断，相应的规则/步骤从未被训练，导致模型对"靠后的规则"召回率接近 0。

**详细讲解**

LlamaFactory 等框架会将超过 cutoff_len 的样本截断而非过滤。如果一条样本的 output 是 5000 tokens，而 cutoff_len=4096，那么后 904 tokens 的内容对应的 loss 永远是 0，模型永远学不到这部分。

典型症状：
- 某些特定规则（如 rule8）的召回率始终为 0
- 整体 F1 不随数据量增加而提升
- 训练 loss 下降正常，但特定指标异常

排查步骤：
```python
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_path)
lengths = [len(tokenizer(s["output"])["input_ids"]) for s in train_data]
print(f"max={max(lengths)}, p95={np.percentile(lengths, 95):.0f}")
# cutoff_len 必须 ≥ max(lengths)
```

---

**Q6：训练框架实际加载的样本数比文件行数少，你怎么发现，怎么处理？**

**简明答案**：训练开始后立即检查日志中的 `Num examples`，必须等于文件行数；如不一致，找 `WARNING` 日志中的过滤原因。

**详细讲解**

LlamaFactory 等框架在数据加载阶段会静默过滤不符合格式要求的样本，不报错、不抛异常，仅打印 WARNING 级别的提示。如果不主动检查，可能整个实验都在不完整的数据上跑：

Round 4 实例：文件有 949 条，实际只加载 600 条（原始 349 条全被过滤），原因是多轮对话中 `observation → human` 的角色序列不符合 LlamaFactory 的期望格式，但日志只有几行 WARNING，很容易忽略。

检查方式：
```bash
grep "Num examples" train.log
# 应等于 wc -l train_data.jsonl
```

常见过滤原因：
- `Invalid role tag`：多轮对话中间出现意外的角色序列
- `Empty content`：某轮消息内容为空
- cutoff 过滤（若框架配置为过滤而非截断）

---

**Q7：重新训练时，不小心用了上一次训练的 output_dir，会发生什么？**

**简明答案**：框架会从上次的 checkpoint 继续训练（resume），而非从基础模型重新开始，导致在错误的模型基础上叠加训练。

**详细讲解**

LlamaFactory 等框架发现 output_dir 中有 checkpoint 时会自动 resume。如果上一次训练的模型已经出现问题（如 Round 4 只用 600 条合成数据训练，导致模型遗忘了大量非 KPI 技能），在它基础上继续训练会污染结果。

症状：第一步的 loss 异常低（不符合从头训练的预期），或模型行为与预期基础模型不符。

解决方式：每次独立实验使用独立的 output_dir 命名（如 `checkpoints/R4-sft-v4b/` 而非复用 v4 的目录）。

---

**Q8：量化（int4/int8）对 SFT 效果的影响，通常被低估在哪里？**

**简明答案**：量化对精确数值（如 model_id、参数值）的记忆能力损失远大于对文本生成的影响，通常损失 7–10%。

**详细讲解**

量化将权重压缩到低精度，本质上引入了信息损失。对自然语言生成任务（如摘要、翻译），这种损失通常 <2%，可接受。但对需要精确记忆数值的任务（如工具调用参数、规则编号），量化引入的误差会导致模型"记错数字"。

Round 3 实验数据：同等训练数据下，bf16 LoRA F1=0.430，int8 量化 F1 损失约 7–10%。

使用量化时的注意事项：
- 量化降低的是权重内存，不降低激活内存（长序列训练时仍可能 OOM）
- 如果任务涉及精确数值，优先尝试标准 LoRA（bf16）而非 QLoRA

---

## 三、评估设计

---

**Q9：评估工作流类任务（多步工具调用）时，用端到端模拟工具和用步骤预测，有什么本质区别？**

**简明答案**：步骤预测用真实日志上下文，不需要模拟工具，结果可靠；端到端模拟引入工具响应的准确性问题，干扰结论。

**详细讲解**

对于多步工具调用任务，评估核心问题是："给定正确的上下文，模型能否输出正确的下一步？"

**步骤预测**（正确方式）：
```
input:  来自日志的真实上下文（system + user + 前N步真实工具调用和响应）
output: 模型预测的第N+1步工具调用
对比:   日志中的真实第N+1步
```
全程使用真实日志数据，不需要任何模拟。

**端到端模拟**（Round 4 走弯路的方式）：
让模型从头完整跑一遍，每次工具调用都用模拟响应。问题：模拟响应和真实响应不一致时（如 api_name 不同），后续步骤的失败是工具模拟的问题，而非模型的问题，无法区分。

步骤预测的另一个优势：直接复用日志，每条日志可拆出 N 个独立评估对，数据利用率高。

---

**Q10：首步工具准确率（first-step accuracy）和全轨迹 F1，分别适合衡量什么？**

**简明答案**：首步准确率衡量意图理解是否正确；全轨迹 F1 受步数差异影响大，不适合作为主指标。

**详细讲解**

**首步准确率**：只评估第一个工具调用的名称是否正确（context = system + user），不依赖任何后续工具响应。这个指标稳定、可靠，直接反映模型是否理解了用户意图并选对了处理路径。

**全轨迹 multiset F1**：将 GT 工具调用序列和预测序列做集合层面的 F1。严重受步数影响：GT 走 17 步，模型走 7 步，即使 7 步全对，recall 也只有 7/17=41%，F1 约 58%。如果模型学会了更高效的路径（少走无效步骤），反而会被这个指标惩罚。

实际使用建议：
- 主指标：首步工具名准确率 + 关键工具（如 `get_idi_model_data`）的被调用率
- 辅助指标：步骤预测 combined_f1（逐步对比，排除步数差异）
- 全轨迹 F1 仅作参考，不作决策依据

---

## 四、强化学习（GRPO）

---

**Q11：GRPO 训练后，模型的 F1 从 0.42 暴跌到 0.003，has_reasoning_rate=0。是什么原因？**

**简明答案**：reward 函数存在高收益捷径：始终预测空违规 + 猜对 CPK 值可以获得约 0.6 的稳定奖励，比正确检测规则更容易，模型在 3 个 epoch 内完全收敛到这条捷径。

**详细讲解**

Round 3-C 实验的 reward 函数：
```python
reward = rule_f1 + cpk_bonus(+0.1) + format_bonus(+0.05)
```

当模型始终预测"无违规（空集）"：
- rule_f1 = 0（如果真实有违规）或 1.0（如果真实无违规，约占样本 40%）
- cpk_bonus：只要 CPK 值猜对就有 +0.1
- 期望 reward ≈ 0.40 × 1.0 + 0.6 × 0 + cpk_bonus ≈ 0.5–0.6

而正确检测违规的 reward 才 0.4–0.8（不稳定）。模型发现"装死"更稳，于是 GRPO 将其强化到极限。

**修复方式**：
```python
# 空预测惩罚：预测无违规但实际有违规时，减 0.3
if pred_violations == [] and gt_violations != []:
    reward -= 0.3
# 无推理链乘零
if "<think>" not in output:
    return 0.0
```

设计 reward 函数时必须枚举所有"低成本高奖励"的捷径并逐一封堵。

---

**Q12：GRPO 训练过程中出现 `clipped_ratio=1.0`，意味着什么？**

**简明答案**：所有生成的补全都被截断了，模型无法得到任何有效奖励，梯度为 0，训练等同于空转。

**详细讲解**

`clipped_ratio=1.0` 表示 100% 的生成补全长度超过了 `max_completion_length` 参数设置，全部在末尾被截断。被截断的补全：
- 答案不完整，无法被 reward 函数正确解析
- 几乎所有样本 reward ≈ 0
- 所有样本的 advantage ≈ 0（因为所有人得分相近）
- 梯度近似于 0，参数不更新

诊断方式：
```python
# 训练前必查
p95_len = np.percentile(output_lengths, 95)
print(f"建议 max_completion_length ≥ {int(p95_len * 1.1)}")
```

特别注意：GRPO 的生成长度通常比 SFT 的 output 长（因为包含完整推理链），不能直接复用 SFT 的 cutoff_len 作为 max_completion_length。

---

**Q13：GRPO 训练时，KL 惩罚系数（kl_coef/beta）设置过小会导致什么？**

**简明答案**：模型偏离参考模型（SFT 基础）过远，SFT 阶段学到的通用能力被灾难性覆盖。

**详细讲解**

GRPO 通过策略梯度强化高分输出，KL 项限制模型不能偏离参考模型太远：
```
loss = -reward_advantage + kl_coef × KL(policy || ref_policy)
```

kl_coef 过小时，KL 约束失效，模型可以自由向"得高分"的方向漂移，不顾 SFT 能力。常见后果：
- 模型学会了特定任务格式，但通用对话能力退化
- 在评估集上某些指标提升，但实际部署表现变差

建议值：`kl_coef = 0.01–0.1`，从 0.01 开始，观察 KL 散度是否稳定在合理范围（<5）。

---

## 五、工程实施

---

**Q14：训练进程被 kill 后，GPU 显存没有释放，新训练启动失败，怎么处理？**

**简明答案**：找到占用显存的残留进程并 kill，等待 GPU 显存完全归零后再启动新训练。

**详细讲解**

`kill` 命令发送 SIGTERM，PyTorch 进程可能需要数秒到数十秒才能释放 GPU 显存（需等 CUDA context 清理）。如果立即启动新训练：
- 新进程申请显存时可用显存不足，OOM
- 如果用 SIGKILL（-9），清理可能不完整

正确流程：
```bash
# 1. kill 旧进程
kill $(ps aux | grep "llamafactory" | grep -v grep | awk '{print $2}')

# 2. 等待显存释放（轮询）
while nvidia-smi | grep -q "MiB  |"; do sleep 2; done

# 3. 确认后启动新训练
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
```

---

**Q15：DeepSpeed ZeRO-3 和 LoRA 结合使用时，为什么会出现"zip() 长度不匹配"的报错？**

**简明答案**：ZeRO-3 会将参数分片到不同 GPU，而 LoRA 在某些框架版本中假设参数是完整的，导致参数数量不匹配。

**详细讲解**

ZeRO-3 将模型参数在所有 GPU 上分片存储，每个 GPU 只持有部分参数。LoRA 在合并 adapter 时需要将 LoRA 权重与原始权重对齐，某些版本的 LlamaFactory 在这个过程中调用了 `zip(原始参数, LoRA参数)`，而两者的长度因分片不一致而报错。

**解决方式**（Round 4 采用）：去掉 DeepSpeed，改用 DDP（数据并行）+ 标准 LoRA。对 H20 97GB 显存的机器，Qwen3-14B 的 SFT 不需要 ZeRO-3，单卡或双卡 DDP 即可。若确实需要 ZeRO，使用 ZeRO-1 或 ZeRO-2，不使用 ZeRO-3 + LoRA 的组合。

---

## 六、方法论

---

**Q16："ns 格式转换"（删除 skill doc）的核心机制是什么？为什么删了还能学到 skill 知识？**

**简明答案**：SFT 只对 output 计算 loss，skill 知识通过 output 的推理链传递到模型权重，input 中的 skill doc 只是生成教师数据时的辅助，训练时可以删除。

**详细讲解**

```
训练时（ns 格式）：
  Input（不计 loss）：  system="", user="查 UPH"
  Output（计 loss）：   <think>list→metrics 返回 modelId=12202, 因此调用...</think>
                        get_idi_model_data(12202) → UPH=185
```

模型从 output 学到的是：
1. 遇到 UPH 查询，需要先调 list_object_types 发现 api_name
2. 再调 get_object_type_metrics 获取 modelId
3. 用得到的 modelId 调 get_idi_model_data

这些知识来自 output 的推理链，不来自 input。所以删掉 skill doc 后，只要 output 里的推理链完整，知识就会进入权重。

**前提条件**：教师模型（Claude）在生成时必须在 thinking block 里展开完整推理，把"为什么这样调、参数从哪来"说清楚，不能只输出工具调用指令。

---

**Q17：skill doc 在线上（ws 格式）工作正常，直接把线上日志的 skill doc 删掉做训练，会有什么问题？**

**简明答案**：线上日志的 output 是在 skill doc 完整存在的情况下产生的，output 里可能没有展开推理链（因为模型靠 input 里的 skill doc 就够了），删掉 skill doc 后训练数据的 output 缺乏可学习的推理过程。

**详细讲解**

线上 ws 模型的工作方式：
```
Input:  skill doc（含详细步骤）+ user question
Output: [直接调工具，不在 thinking 里解释为什么]
```

模型不需要在推理链里解释，因为 skill doc 已经在 input 里。删掉 skill doc 后：
```
Input（已删 skill doc）：user question
Output：[直接调工具，无解释]   ← 来源不明
```

模型无从学习"为什么这样调"，只能死记结果。

正确做法：用 Claude + skill doc 重新生成教师数据，要求 Claude 在 thinking 里显式展开推理，再做 ns 转换。

---

**Q18：单轮对话技能内化（如 SPC 规则）和工作流技能内化（如多步工具查询），在数据验证上有什么核心区别？**

**简明答案**：单轮验证 output 是否可从 input 独立推导；工作流验证每步工具参数是否有追溯来源（来自用户输入或前序工具响应），不允许凭空出现。

**详细讲解**

**单轮对话验证**：
```python
# 遮住 skill doc，output 是否仍然可以从 input 推导出来？
assert can_derive_without_skill_doc(input=user_question, output=answer)
```
如果 output 只依赖 user_question 中的数据（如 SPC 的数据点），就通过。

**工作流验证**：
```python
for step in tool_calls:
    for param, value in step.arguments.items():
        source = find_source(
            value,
            visible_sources=[user_input] + previous_tool_responses
        )
        assert source is not None, f"参数 {param}={value} 来源不明"
```

工作流验证更严格，因为每一步的参数必须来自已知来源。"来源不明"通常意味着参数来自被删除的记忆注入或 skill doc，这类样本不能直接使用。

---

*整理自 Round 1–4 实验，2026-05-14*
