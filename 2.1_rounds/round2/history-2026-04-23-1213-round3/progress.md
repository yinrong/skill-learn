# Route 2.1 SPC Skill Internalization Demo — 进度记录

---

## 2026-04-23 10:24 — 第一轮训练结果总结（失败）

### 实验设置
- 模型: Qwen3-8B (base instruct)
- 训练方式: LoRA rank=128 alpha=256 target=all 3 epochs
- 训练条件: WITH Skill document in system prompt
- 评测条件: WITHOUT Skill document (no-skill eval)

### 结果
| 条件 | N | rule_f1 | 说明 |
|------|---|---------|------|
| base-no-skill | 0 | 0.000 | 符合预期 |
| base-with-skill | 0 | 0.219 | 上界参考 |
| sft-N1-no-skill | 100 | 0.000 | 未内化 |
| sft-N2-no-skill | 200 | 0.000 | 未内化 |
| sft-N3-no-skill | 300 | 0.000 | 未内化 |
| sft-N4-no-skill | 500 | 0.000 | 未内化 |
| sft-N4-with-skill | 500 | 0.256 | 有Skill时略有提升 |

### 失败原因分析
1. **格式不一致**: 无Skill时模型输出"规则1/规则2"中文格式，提取器只识别"rule1/rule2" → F1=0
2. **规则定义错误**: 无Skill时模型用错误的Nelson规则定义（UCL/LCL混淆为USL/LSL，窗口长度错误等）
3. **预训练知识强势**: 中文SPC先验知识未被500样本覆盖
4. **训练样本单一**: 全部由模板生成，格式雷同，多样性不足
5. **enable_thinking=False**: 评测时禁用思考，但训练包含\<think\>块；即使开启thinking后F1仍=0

### 现有模型资产
- checkpoints/demo-N1~N4 (LoRA adapters)
- checkpoints/demo-N1~N4-merged (合并后模型)
- vLLM: port 8020 (base), 8023 (N3), 8024 (N4) 运行中

---

## 2026-04-23 10:25 — 用户反馈 & 下一步方向

### 用户建议
a. 使用更大模型（Qwen3-14B/32B 已在 /home/yinrong/models/）
b. 调整训练参数：
   - 更少参数部分 + 更多 epoch（如只训 attention，10+ epochs）
   - 样本包含思考过程（已有，但多样性不足）
   - 增加样本多样性，手工编写部分样本
   - Skill长度保持40-500行（当前40行，偏短）
   - 对Skill内容作更详细/多风格解释，每种作为不同样本

### 当前Skill文档状态
- 长度: 40行, 913字符, ~162词
- 格式: 简短表格 + CPK公式 + 输出格式要求
- 问题: 过于简洁，没有对每条规则的详细说明、例子、边界情况

---

## 2026-04-23 10:26 — 下一步计划（待执行）

### 方案设计
1. **扩充Skill文档** 到 80-120行：增加每条规则的详细解释、计算步骤、边界情况示例
2. **多样化训练样本格式**：
   - Style A: 详细逐步分析（现有格式）
   - Style B: 简洁结论先行格式
   - Style C: 中英双语标注
   - Style D: 带边界值计算的精确格式
3. **手工编写10-20条高质量示范样本**（每条规则至少2条）
4. **调整训练参数**：
   - Option X: lora_target=q_proj,v_proj (只训attention), 10 epochs
   - Option Y: lora_rank=64 lora_target=all, 5 epochs
5. **使用Qwen3-14B**（GPUs 1+2 TP=2，但需确认稳定性）

### 关键问题（需研究）
- 为什么LoRA微调无法覆盖预训练知识？需要多少样本？
- 数据多样性 vs 数据量，哪个更重要？
- LoRA target selection对知识注入的影响？

**状态: 正在进行 — 搜索相关研究 + 设计新方案**

---

## 2026-04-23 12:11 — 第2轮训练归档，第3轮设计开始

### 归档
- 第2轮数据和partial checkpoints → history-2026-04-23-1211-round2/
- 原因：训练在 v2 数据上尚未完成，但发现根本性问题：
  1. 样本缺乏"教材级"讲解，全是数值模板样本
  2. lora_target 应包含 FFN 层（up/down/gate_proj）才能注入知识
  3. 训练模型应换 Qwen3-14B（8B 预训练 SPC 知识太弱）

### 第3轮方案
- 数据策略：
  1. 教材级样本（gen_textbook.py）：~100-120条，每条500-3000字
     - 标签体系: industry × process × role × style × topic × subtopic
     - 行业: 手机/车电/整车/大家电/小家电/半导体
     - 角色: 质量/工艺/装备/算法/生管/总监/新人/投资人等
  2. 混合数据集 N1/N2/N3/N4 重新生成（mixed=True，加入教材样本）
- 训练模型: Qwen3-14B
- lora_target: q_proj,v_proj,up_proj,down_proj,gate_proj
- Hook路径已修复: ~/.claude-mem-xiaomi → /home/yinrong/.claude-mem-xiaomi

**状态：正在编写 gen_textbook.py**

---

## 2026-04-23 17:30 — 第3轮评测进行中（进度简报）

### 当前执行状态
4个评测作业同时运行，使用 vLLM（GPU 1/2/3，TP=1，port 8040/8041/8044）：

| 作业 | 进度 | 当前 rule_f1 | 状态 |
|------|------|------------|------|
| base-14b-no-skill  | 20/200 | 0.000 | 运行中（慢，port 8040 竞争） |
| base-14b-with-skill | 40/200 | 0.163 | 运行中 |
| sft-r3-N1-no-skill | 120/200 | **0.102** | 运行中，非零突破 |
| sft-r3-N4-no-skill | **200/200** | 0.005 | **已完成** |

### N4 最终结果（sft_r3_N4.json）
```
rule_detection_f1:  0.005   ← 失败，本质等于0
cpk_mae:            0.795
cpk_found_rate:     0.240
has_reasoning_rate: 0.000   ← 完全没有<think>推理链！
per_rule_recall: rule1=0.025, rule2~rule8≈0
```

### N4 失败模式分析
抽查样本输出发现两种极端行为：
1. **全输出**：对某些样本预测 rule1~rule8 全部触发（海量误报）
2. **空输出**：大多数样本预测为空（漏报）
3. `has_reasoning_rate=0.000`：模型没有生成 `<think>` 推理链

**根本原因假设**：
- N4 训练了 8 epoch × 552 steps（N1 只有 152 steps），总梯度步数是 N1 的 3.6 倍
- N4 train_loss=0.226（N1=0.559）：过拟合训练分布
- 过拟合导致模型在有 Skill 文档时才能正常输出，无 Skill 时输出崩塌
- 不同于 N1（训练不足、但部分内化），N4 彻底过拟合成"上下文依赖"行为

### N1 当前结果（interim）
- F1 ≈ 0.10~0.13，稳定非零
- 相比第1轮（全为0.000）是**突破性进展**
- 说明 Qwen3-14B + FFN LoRA + 混合数据策略方向正确

### 正在等待的结果
- N1 完整结果（约20分钟后）
- base-14b-no-skill 完整结果（约40分钟后）
- base-14b-with-skill 完整结果（同上）

### N1 最终结果（sft_r3_N1.json）
```
rule_detection_f1:  0.100   ← 非零！第1轮为0.000
cpk_mae:            0.206   ← 良好
cpk_found_rate:     0.625
has_reasoning_rate: 0.000   ← 仍无<think>链（模型平输出）
per_rule_recall: rule1=0.525, rule2=0.345, rule3=0.536,
                 rule4=0.333, rule5=0.409, rule6=0.394,
                 rule7=0.556, rule8=0.308
```
**所有8条规则 recall 均非零**，说明知识确实被内化了。

### 下一步（已执行）
1. ✅ N2（200样本）+ N3（300样本）→ 4 epoch，GPU1/GPU2 同步训练中
2. ✅ base-with-claude 评测脚本（tools/eval/spc_eval_claude.py）已创建并运行中
3. ⏳ base-14b-no-skill / with-skill vLLM 评测继续运行中（慢）
4. ⏳ N4 将用 4 epoch 重新训练（N2/N3 完成后）

---

## 2026-04-23 18:00 — 第3轮阶段性更新

### 已完成评测

| 条件 | rule_f1 | cpk_found | notes |
|------|---------|-----------|-------|
| base-14b-no-skill（预估）| 0.000 | ? | 基于第1轮8B结果推算，评测仍在进行 |
| base-14b-with-skill（interim） | ~0.163 | ? | 40/200 采样 |
| sft-r3-N1-no-skill | **0.100** ✓ | 0.625 | 第3轮突破 |
| sft-r3-N4-no-skill（8 epoch）| 0.005 | 0.240 | 过拟合失败 |
| base-with-claude（运行中）| ? | ? | 上限参考 |

### 当前运行任务
- **训练**：N2（GPU1，128步，~25min）、N3（GPU2，176步，~30min）—— 4 epoch
- **评测**：base-14b-no/with-skill（vLLM port 8040，慢，估2小时）
- **评测**：base-with-claude（50条，Claude API，无thinking）

### 关键结论
1. **方向正确**：N1 F1=0.100，所有规则均有部分内化
2. **过拟合警告**：8 epoch 导致 N4 崩塌（loss=0.226 → F1≈0）；4 epoch 更合适
3. **下一步**：N2/N3 完成后合并→评测→对比 N1/N2/N3 三点拟合增长曲线
4. **上限参考**：Claude 自身表现（理论最高，因生成了训练样本）

### 文件路径
- progress.md: `history-2026-04-23-1213-round3/progress.md`（本文件）
- 根目录 progress.md 仅作索引指针

---

## 2026-04-23 18:45 — N2/N3 四轮诊断 + 根本原因

### N2/N3 评测（4 epoch）结果：F1=0
- N2（200样本，4epoch，128步）：F1=0.000，输出"第X条"中文格式
- N3（300样本，4epoch，176步）：F1=0.000，输出"全部通过，过程受控"

### 关键发现：no-skill曝光次数决定内化效果

| 模型 | 总样本 | no-skill样本数 | epochs | no-skill曝光次数 | F1 |
|------|--------|---------------|--------|-----------------|-----|
| N1   | 151    | 25+51=76      | 8      | 608             | **0.100** |
| N2   | 251    | 50+51=101     | 4      | 404             | 0.000 |
| N3   | 351    | 75+51=126     | 4      | 504             | 0.000 |
| N4   | 551    | 125+51=176    | 8      | 1408            | 0.005 |

**结论**：no-skill样本曝光次数需要≥600次才能形成稳定的英文rule标识符输出。  
N4过训（1408次曝光 + 552步）→ 过拟合，N1恰好在甜区（608次曝光 + 152步）。

### N2 诊断细节
- WITH Skill文档：输出正确格式 rule1/rule2 ✓（Skill doc是触发器）
- WITHOUT Skill文档：输出"第1条、第2条"中文格式 ✗
- 说明：模型学会了"有Skill才用English identifier"的条件关联，而非真正内化

### 修复方案：调整各规模的训练轮次

目标：~200梯度步（N1=152步效果良好，N4=552步过拟合）

| 模型 | 样本数 | 目标步数 | 步/epoch | 推荐epochs | 预期F1 |
|------|--------|---------|---------|-----------|--------|
| N2   | 251    | 188步    | 31步/ep  | **6 epoch** | ~0.1 |
| N3   | 351    | 220步    | 44步/ep  | **5 epoch** | ~0.1 |
| N4   | 551    | 207步    | 69步/ep  | **3 epoch** | ~0.1 |

### 已修复：extractor识别"Rule 1"（带空格）格式
- 旧版：`\b(rule[1-8])\b` → 只识别 "rule1"
- 新版：`\brule\s*([1-8])\b` → 识别 "rule1"、"Rule 1"、"rule 4"等
- 同时增加"未触发"行跳过逻辑，避免false positive

### 已完成所有first-pass评测（4ep版本）

| 模型 | epochs | steps | F1 | CPK found | 状态 |
|------|--------|-------|----|-----------|------|
| sft-r3-N1 | 8 | 152 | **0.100** | 0.625 | 最佳 |
| sft-r3-N2 | 4 | 128 | 0.000 | 0.445 | 欠训练 |
| sft-r3-N3 | 4 | 176 | 0.005 | 0.350 | 欠训练 |
| sft-r3-N4 | 8 | 552 | 0.005 | 0.240 | 过训练 |

### 正在执行：第二次训练（修正epochs）

目标：~600次no-skill曝光 + ~200梯度步

| 模型 | epochs | 预期steps | no-skill曝光 | 状态 |
|------|--------|---------|------------|------|
| N2 6ep | 6 | 192 | 606 | **GPU1运行中 (32/192)** |
| N3 5ep | 5 | 220 | 630 | **GPU2运行中（刚开始）** |
| N4 3ep | 3 | 207 | 528 | ⏳等GPU空闲 |

### 其他评测进行中
- base-14b-with-skill vLLM eval: 160/200, F1=0.120
- Claude with-skill: **[10/50] F1=0.251** ← 上限确认！（比base-14b-with-skill高2倍）
- base-14b-no-skill: 还在跑（很慢，port 8040竞争）

---

## 2026-04-23 20:00 — 第二次训练 + 关键修复进度

### 已完成

**训练完成：**
- N2 6ep：train_loss=0.469，39分钟，GPU1完成
- N3 5ep：train_loss=0.401，46分钟，GPU2完成

**评测完成：**
| 条件 | rule_f1 | 说明 |
|------|---------|------|
| base-14b-no-skill | 0.000 | 符合预期，基线 |
| base-14b-with-skill | 0.109 | 有Skill doc时的基座表现 |
| base-claude-with-skill | **0.232** | 上限（n=50），rule1~6 recall≈1.0 |

### 关键修复：评测器兼容中英双语

**背景**：N2 6ep在`enable_thinking=False`时输出"第1条"中文格式，`enable_thinking=True`时输出"rule3"英文格式。旧提取器仅识别英文，导致F1=0。

**修复**：
1. `tools/eval/extractor.py`：增加 `_CHINESE_RULE_PATTERN = re.compile(r'第\s*([1-8一二三四五六七八])\s*条')` 同时识别中英文
2. `tools/eval/spc_eval.py`：保持 `enable_thinking=False`（快速），依赖更新后的提取器

**验证**：测试用例通过
- `"第1条触发"` → `['rule1']` ✓
- `"rule3 触发"` → `['rule3']` ✓
- `"未触发第2条"` → `[]` ✓

### 当前运行状态（20:00）

**训练：**
- N4 3ep：GPU1，155/207步（~7分钟剩余）

**评测（port 8042=N2，port 8043=N3，port 8040=base）：**
| 模型 | 进度 | running_F1 | 备注 |
|------|------|-----------|------|
| N2 6ep no-skill | [60/200] | **0.147** | 非零！比N1(0.100)更高 |
| N3 5ep no-skill | [60/200] | 0.000 | 仍在评测中 |
| base-14b-no-skill | ~180/200 | 0.000 | 接近完成 |

**N2初步结论**：F1≈0.15，高于N1的0.100，说明N=200时有提升 → scaling趋势存在！

### 会话持久化说明
当前chat session位于 `/root/.claude/projects/-home-yinrong/ac47ff5c-1060-4bc9-a266-57b8cbb39f9c.jsonl`。
如机器重置（仅/home/yinrong保留），session丢失，但所有工作产物（模型/日志/结果/代码）在/home/yinrong中安全保存。
恢复方法：新session打开本文件，从最后一个"## "节点继续。

---

## 2026-04-23 20:30 — 第3轮最终结果汇总

### 全量评测完成

| 条件 | N | epochs | steps | rule_f1 | cpk_found | per_rule_recall摘要 |
|------|---|--------|-------|---------|-----------|-------------------|
| base-14b-no-skill | - | - | - | 0.000 | ? | - |
| base-14b-with-skill | - | - | - | 0.109 | ? | - |
| base-claude-with-skill | - | - | - | 0.232 | 1.0* | rule1~6≈1.0, rule7=0.43, rule8=0 |
| sft-r3-N1 | 100 | 8 | 152 | **0.100** | 0.625 | 全部8条均有召回 (0.31~0.56) |
| sft-r3-N2-6ep | 200 | 6 | 192 | **0.119** | 0.525 | rule1=0.675 (主导) + rule2=0.069 |
| sft-r3-N3-5ep | 300 | 5 | 220 | 0.064 | 0.735 | rule2=0.517 (主导) + rule4=0.083 |
| sft-r3-N4-3ep | 500 | 3 | 207 | 0.090 | 0.095 | rule1=0.5, rule2=0.14, rule5=0.18 |

### Scaling曲线结论

幂律拟合：f1 = 0.215 × N^(-0.157)，R²=0.168 → **非单调，无法外推**

趋势：N1(0.100) → N2(0.119) → N3(0.064) → N4(0.090) — 非单调下降趋势

**核心结论**：
1. ✅ **阶段性成功**：所有4个SFT模型 F1 > 0，第1/2轮全为0.000的问题已解决
2. ⚠️ **scaling不清晰**：N3 dip（F1=0.064）打断了趋势，非单调性说明当前设置下数据量scaling不稳定
3. ✅ **N1最均衡**：唯一覆盖全部8条规则的模型，152步/8epoch/608曝光次数是甜区
4. ❌ **模式崩塌**：N2/N3/N4均出现单一规则主导（分别集中于rule1/rule2/rule1），而非均匀覆盖

### per_rule详细对比

| rule | N1(8ep) | N2(6ep) | N3(5ep) | N4(3ep) |
|------|---------|---------|---------|---------|
| rule1 | 0.525 | **0.675** | 0.000 | 0.500 |
| rule2 | 0.345 | 0.069 | **0.517** | 0.138 |
| rule3 | 0.536 | 0.000 | 0.000 | 0.000 |
| rule4 | 0.333 | 0.000 | 0.083 | 0.000 |
| rule5 | 0.409 | 0.000 | 0.045 | 0.182 |
| rule6 | 0.394 | 0.000 | 0.000 | 0.000 |
| rule7 | 0.556 | 0.000 | 0.000 | 0.000 |
| rule8 | 0.308 | 0.000 | 0.000 | 0.000 |

N1是唯一8条规则全覆盖的模型。N2~N4均有不同程度的模式崩塌。

### 关键文件
- 曲线图：`history-2026-04-23-1213-round3/reports/scaling_curve_r3.png`
- 旧结果归档：`history-2026-04-23-1213-round3/results/archive/`
- Extractor修复：支持"rule1"/"Rule 1"/"第1条"/"规则②"所有格式

### 第3轮 Go/No-Go 评估

**判断：Conditional Go**（有条件通过）
- 已证明：Qwen3-14B + FFN LoRA + 混合训练 → 知识内化可行
- 未证明：清晰的 N^β 幂律 scaling（因模式崩塌打断趋势）
- 下一步方向：
  1. 分析N1为什么8条规则均有覆盖而N2~N4没有 → 可能需要更均衡的规则覆盖训练样本
  2. 考虑在eval时使用enable_thinking=True（虽然慢），避免中文格式崩塌问题
  3. 或：统一训练样本的输出格式（所有样本强制使用英文rule标识符）
  4. route2.2方向：探索更大模型/更长训练/更高质量教材样本
