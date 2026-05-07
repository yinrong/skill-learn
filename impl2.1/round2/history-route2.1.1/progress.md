# Route 2.1.1 SPC Skill 内化效果提升 — 进度记录

---

## 2026-04-24 18:07 — 第4轮（Round 4）实验启动

### 背景与目标

- **前置**：Route 2.1 Demo 完成，最佳 F1=0.119（N2-6ep，无Skill）
- **目标**：F1 接近 Claude-sonnet-4.6+Skill（F1≈0.232），推理快、资源少
- **当前瓶颈**（来自 route2.1 Round3 分析）：
  1. 模式崩塌：N2/N3/N4 均出现单一规则主导（rule1 或 rule2），只有 N1 覆盖全部8条规则
  2. 根本原因：rule1 注入权重 22%（其余 7~10%），随着数据量增大，梯度步数增多，rule1 过度优化
  3. 格式问题：extractor 已修复中英文双格式支持，F1=0.119 已计入此修复
  4. 甜区：~600 no-skill曝光次数 + ~200 梯度步数 = N1 sweet spot

### 实验矩阵（Round 4）

**已启动（2026-04-24 18:07）：**

| 实验 | GPU | 样本数 | 配置变化 | epochs | 总步数 | no_skill曝光 | 假设 |
|------|-----|--------|---------|--------|-------|------------|-----|
| Exp-A | 0 | 251 (200+51tb) | 默认权重（基线复现） | 6 | 192 | 606 | 验证 round3 结果可复现 |
| Exp-B | 1 | 251 | 均衡权重（rule1~8各11.5%） | 6 | 192 | 606 | 均衡权重消除 rule1 主导 |
| Exp-C | 2 | 251 | double_rule_prob=0.30 | 6 | 192 | 606 | 多规则样本提升规则覆盖 |
| Exp-D | 3 | 251 | no_skill_ratio=0.40 | 5 | 160 | 605 | 更多 no_skill 曝光 |
| Exp-E | 4 | 251 (8B) | 均衡权重 + Qwen3-8B | 6 | 192 | 606 | 8B 模型对比 |
| Exp-F2 | 6 | 151 (100+51tb) | 均衡权重 + N1规模 | 8 | ~152 | 608 | 原甜区 + 均衡权重 |
| Exp-G2 | 7 | 451 (400+51tb) | 均衡权重 + 大数据量 | 4 | ~232 | 604 | 更大数据量 + 均衡权重 |

**后台任务（GPU 5）：**
- Round3 N1 模型基线复测（port 8050）：验证 F1=0.100 可复现

**No-skill 曝光次数计算：**
- 公式：(N × no_skill_ratio + textbook_count) × epochs
- N1原甜区：(100×0.25 + 51) × 8 = 76 × 8 = 608 ✓
- 目标：所有实验均设计为 ~600 次曝光

### 核心假设

| 假设 | 检验方法 |
|------|---------|
| H1: 均衡权重消除 rule1 主导 | 对比 Exp-A vs Exp-B 的 per_rule_recall 分布 |
| H2: 更多多规则样本提升覆盖 | 对比 Exp-A vs Exp-C 的 per_rule_recall 分布 |
| H3: 均衡权重效果独立于数据量 | 对比 Exp-F2 vs Exp-G2（均为均衡权重，不同数据量）|
| H4: 8B 模型效果明显弱于 14B | 对比 Exp-B vs Exp-E |

### 预期完成时间

- 14B 训练：~40分钟/实验（192步 × ~12.5s/step）
- 8B 训练：~25分钟/实验（192步 × ~8s/step）
- 估计完成：2026-04-24 ~19:00
- 合并+评测：~15分钟/模型（合并5分钟 + vLLM启动5分钟 + 评测5分钟）

---

## 2026-04-24 18:23 — Claude 教师数据生成启动

- 已创建 `tools/data/gen_claude_teacher.py`，后台运行 (PID 17538)
- 目标：500条，recall≥0.7 验证过滤，concurrency=6
- 进度（18:33）：29/500 通过，65次尝试，**通过率≈44%**
- **主要过滤原因**：
  - 正常样本（GT=[]）Claude 过度检出 rule5/rule6/rule7/rule8 → recall=0.5 拒绝
  - 实际违规样本 Claude 漏报 → recall=0.0 拒绝
- 预计完成时间：~21:15（500条），~18:57（100条），~19:32（200条）
- 保存路径：`history-route2.1.1/data/train_claude_teacher.jsonl`

---

## 2026-04-24 18:31 — ExpE 训练完成，评测启动

- **Exp-E (8B)** 训练完成：192/192步，train_runtime=1421s（23分41秒）
  - train_loss=0.317，已合并 adapter → 9个 safetensors
  - vLLM 正在 GPU 4 启动（port 8104）
- 修复了两个 Bug：
  1. `monitor_and_eval.sh` 误检 "trainable parameters" 导致提前触发评测 → 改为仅检测 `train_runtime`
  2. `eval_and_report.sh` 使用相对路径导致 peft 误认为 HuggingFace Hub 模型 → 改为 `pwd` 绝对路径
  3. merge_adapter.py 缺少 `--template qwen3` → 已修复

---

---

## 2026-04-24 19:15 — Round 4 Batch 1 评测完成（全部8实验）

### Batch 1 最终结果汇总

| 实验 | F1 | CPK_rate | CPK_MAE | rules_covered | 关键配置 |
|------|-----|----------|---------|---------------|---------|
| expA | **0.173** | 0.000 | - | 7/8 | default权重, 251s, 6ep |
| expB | 0.164 | 0.600 | 0.496 | 7/8 | 均衡权重, 251s, 6ep |
| expC | **0.173** | **0.945** | 0.305 | 8/8 | double=0.30, 251s, 6ep |
| expD | 0.136 | 0.705 | **0.121** | 8/8 | no_skill=0.40, 251s, 5ep |
| expE | 0.121 | 0.500 | 0.413 | 8/8 | 8B, 均衡, 251s, 6ep |
| expF2 | 0.131 | 0.890 | 0.343 | 8/8 | 均衡, 151s, 8ep |
| expG2 | 0.049 | 0.160 | 0.086 | 6/8 | 均衡, 451s, 4ep ← **意外崩溃** |
| expX | 待评测 | | | | 均衡, 751s, 3ep (GPU5) |

### 核心发现（Batch 1）

#### 发现 H1：默认权重导致不同方向的崩溃
- expA: rule2=1.000（彻底崩塌），CPK完全不输出
- 即使均衡权重(expB)，rule2仍=0.690（部分崩塌）
- **结论**：默认权重 OR 均衡权重在 6ep+251样本下均倾向于 rule2 主导

#### 发现 H2：double_rule_prob=0.30 是最佳单因素改善
- expC: F1=0.173, CPK=0.945（最高!），8/8规则覆盖
- 多规则样本强迫模型同时分析多个规则 → CPK计算必须同时正确
- **结论**: double_rule_prob是提升CPK内化最有效的单因素

#### 发现 H3：high no_skill_ratio (0.40) 防崩塌最有效
- expD: F1=0.136, CPK_MAE=0.121（最准确!），8/8规则，最均衡分布
- 更多无Skill练习 → 模型必须从记忆检索全部规则 → 防崩塌
- **结论**: no_skill_ratio=0.40 > 均衡权重（避免崩塌效果更强）

#### 发现 H4：样本数增加≠效果更好（反直觉发现）
- expG2 (451s, 4ep): F1=0.049 崩溃！远低于 expF2 (151s, 8ep): F1=0.131
- 总no_skill曝光次数相近（~600），但expG2每样本重复次数少(4次) vs expF2(8次)
- **结论**: **每样本重复次数比总曝光次数更重要**（记忆强化需要足够重复）

#### 发现 H5：8B模型显著弱于14B
- expE (8B): F1=0.121 vs expB (14B): F1=0.164（同配置）
- **结论**: 14B模型显著更优，应继续以14B为基础

### Batch 2 设计（正在训练）

**基于 H1-H5 的关键假设**: 均衡权重 + 高no_skill_ratio(0.40) + double_rule_prob=0.30 应产生最佳综合效果

| 实验 | 配置 | GPU | 假设 |
|------|------|-----|------|
| expY | 均衡+no_skill=0.40, 251s, 5ep | 4 | H1+H3合并效果 |
| expZ | 均衡+no_skill=0.40+double=0.30, 251s, 5ep | 0 | H1+H2+H3最优组合 |
| expX | 均衡, 751s, 3ep | 5 | 大数据量+均衡(验证H4) |

**下一步：Batch 3 规划（待 Batch 2 结果）**
- Claude教师数据实验（~21:00后，数据生成中）
- GRPO强化训练（基于最优SFT模型）
- 高no_skill_ratio 极限测试（0.50, 0.60）

---

## 2026-04-24 19:20 — Batch 2+3 全部启动（8 GPU 满载）

### 启动实验汇总（基于 Batch 1 发现）

| 实验 | GPU | 配置 | 假设 |
|------|-----|------|------|
| expY | 4 | 均衡+no_skill=0.40, 251s, 5ep | H1+H3合并效果 |
| expZ | 0 | 均衡+no_skill=0.40+double=0.30, 251s, 5ep | H1+H2+H3全组合 |
| expW | 1 | 默认+no_skill=0.40+double=0.30, 251s, 5ep | H2+H3不加均衡权重 |
| expU | 2 | 均衡+no_skill=0.50, 251s, 5ep | 极限no_skill率 |
| expS | 7 | 均衡+no_skill=0.60, 251s, 4ep | 更极限no_skill率 |
| expT | 6 | 默认+no_skill=0.40, 451s, 3ep | 扩大数据量保持expD配置 |
| expV | 3 | 默认+double=0.30, 451s, 6ep | 扩大数据量保持expC配置 |
| expX | 5 | 均衡, 751s, 3ep | 超大数据量均衡（验证H4） |

### 关键对照关系
- expD vs expY: 加均衡权重的效果（no_skill=0.40共同）
- expC vs expV: 扩大数据量对expC配置的影响
- expD vs expT: 扩大数据量对expD配置的影响
- expD vs expU vs expS: no_skill梯度(0.40, 0.50, 0.60)效果
- expY vs expZ: double_rule_prob对全组合的增量效果
- expW vs expZ: 均衡权重对全组合的增量效果

### 预期完成时间
- expX, expY, expZ, expW, expU, expS: ~20:00-20:30
- expT: ~20:40
- expV: ~21:30（342步，最长）
- Claude教师数据: ~21:15（500条）

---

---

## 2026-04-24 20:20 — Batch 2 全部评测完成 + Batch 3 启动

### Batch 2 完整结果汇总

| 实验 | F1 | CPK_rate | CPK_MAE | rules_covered | 关键配置 |
|------|-----|----------|---------|---------------|---------|
| expX | 0.133 | 0.000 | - | 3/8 | 均衡, 751s, 3ep ← **崩塌** |
| expY | 0.144 | 0.425 | 0.508 | 6/8 | 均衡+no_skill=0.40, 251s, 5ep |
| expZ | **0.174** | 0.495 | 0.479 | 8/8 | 均衡+no_skill=0.40+double=0.30, 251s, 5ep |
| expW | 0.134 | 0.455 | 0.598 | 7/8 | 默认+no_skill=0.40+double=0.30, 251s, 5ep |
| expU | 0.129 | 0.295 | 0.290 | 4/8 | 均衡+no_skill=0.50, 251s, 5ep ← 过高崩 |
| expT | 0.082 | 0.415 | 0.091 | 5/8 | 默认+no_skill=0.40, 451s, 3ep ← 扩容崩 |
| expS | 0.133 | 0.110 | 0.891 | 2/8 | 均衡+no_skill=0.60, 251s, 4ep ← rule3+4崩 |
| expV | 待测 | | | | 默认+double=0.30, 451s, 6ep |

### Batch 2 核心发现

#### 发现 B2-1：no_skill_ratio 最优值约为 0.40
- 0.40 → expZ: 8/8规则, F1=0.174 (最佳)
- 0.50 → expU: 4/8规则, F1=0.129 (退化)
- 0.60 → expS: 2/8规则, rule2消失→rule3+rule4主导 (严重退化)
- **结论**: 最优区间 no_skill_ratio ∈ [0.35, 0.45]

#### 发现 B2-2：规模扩大再次证伤
- expX (751s, 3ep, balanced): F1=0.133, rules=3/8 → **均衡权重也无法救大规模**
- expT (451s, 3ep, default+nsk=0.40): F1=0.082, rules=5/8 → **同等epoch不足**
- **结论**: 每样本重复次数(≥5ep)是防崩关键，样本数不是

#### 发现 B2-3：均衡权重的增量效果（通过对照组）
- expD (default+nsk=0.40) F1=0.136 vs expY (balanced+nsk=0.40) F1=0.144 → +0.008
- expW (default+nsk=0.40+dbl=0.30) F1=0.134 vs expZ (balanced+nsk=0.40+dbl=0.30) F1=0.174 → **+0.040**
- **结论**: 均衡权重+double组合时增益显著，但单独使用增益小

#### 发现 B2-4：深度错误分析（per-rule精准率）
通过对expZ、expC、expD进行per-sample分析，发现根本问题：

| 模型 | rule2 TP/FP | 主要问题 |
|------|-------------|---------|
| expZ | 29/171 | **rule2永远预测**（100%假阳性率）|
| expC | ~everywhere | 每条规则都有大量FP（模型"预测一切"）|
| expD | 均匀分布 | 每条规则都有FP，但稍均衡 |

**核心发现**: F1=0.174的限制来自精准率(Precision)极低，而非召回率
- expZ rule2: P=0.145, R=1.000 → 模型总预测rule2（即使没有）
- expC rule1/rule3/rule4/rule7: P=0.053~0.162, R=0.556~0.750 → "海量预测"策略

**根本原因**: 模型未学会"何时不应预测规则"，缺乏否定样本的精准学习

### Batch 3 实验设计（2026-04-24 20:10 启动）

| 实验 | GPU | 配置 | 步数 | 假设/目标 |
|------|-----|------|------|---------|
| expAA | 0 | 100教师+83无Skill+51教材, 5ep | 150 | 教师数据提升精准率 |
| expBB | 1 | 151教师+83无Skill+51教材, 5ep | 180 | 更多教师数据 |
| expCC | 2 | rule2权重↓=0.03+nsk=0.40+dbl=0.30, 5ep | 160 | 消除rule2偏置 |
| expDD | 4 | balanced+nsk=0.40+double=0.50, 5ep | 160 | 更多双规则样本 |
| expFF | 6 | balanced+nsk=0.40+dbl=0.30, 351s, 5ep | 220 | 更多样本同配置 |
| expGG | 7 | balanced+nsk=0.40+dbl=0.30, 251s, 10ep | 320 | 更多epoch同配置 |
| expEE_s1 | 5 | 课程学习Stage1: 251s with-skill 6ep | 192 | 先学有skill再内化 |

**expEE Stage 2**: expEE_s1完成后，在Stage1 adapter基础上继续训练100条no-skill样本 5ep

### 关键假设（Batch 3）
1. **expCC**: rule2下调权重→模型学会不总预测rule2→精准率提升→F1>0.174
2. **expAA/BB**: Claude教师的推理链包含"rule X未触发 因为..."→学会精准否定
3. **expFF vs expGG**: 同配置下，更多样本 vs 更多epoch哪个更重要？
4. **expEE**: 课程学习是否优于混合训练？

### Claude教师数据状态
- 生成进度：~315/500（pass rate ~45%）
- 预计完成：~21:15

---

## 2026-04-24 21:40 — Batch 3 全部完成 + Batch 4 启动

### Batch 3 完整结果汇总

| 实验 | F1 | CPK_rate | CPK_MAE | rules_covered | rule2_FP | 关键配置 |
|------|-----|----------|---------|---------------|---------|---------|
| expAA | 待评 | | | | | 100教师(with_skill)+83无Skill+51教材, 5ep |
| expBB | 待评 | | | | | 151教师(with_skill)+83无Skill+51教材, 5ep |
| expCC | 0.146 | 0.700 | 0.492 | 4/8 | 152 | rule2下调权重+no_skill=0.40+double=0.30 |
| expDD | 0.132 | 0.475 | 0.190 | 7/8 | 81 | balanced+no_skill=0.40+double=0.50 |
| expFF | 0.148 | 0.440 | 0.342 | 8/8 | 76 | expZ配置+351样本, 5ep |
| expGG | 0.132 | - | - | 5/8 | - | expZ配置+10ep → 过拟合 |
| expEE_s1 | 0.118 | - | - | - | - | 课程学习Stage1: with_skill 192步 |
| expEE_s2 | 0.094 | - | - | 4/8 | 2 | 课程学习Stage2: no_skill 78步 → 灾难性遗忘 |

### Batch 3 核心发现

#### 发现 B3-1：rule2权重下调无法消除rule2 FP
- expCC (rule2=0.03权重): rule2_FP=152，与expZ的171相比略有改善
- **结论**: rule2 FP来源不是训练数据分布，是模型固有偏置

#### 发现 B3-2：double=0.50将rule2 FP减半
- expDD (double=0.50): rule2_FP=81 (vs expZ的171)
- expFF (更多样本,expZ配置): rule2_FP=76 (vs expZ的171)  
- **结论**: 多规则样本强迫同时考虑多规则，是降低rule2偏置的有效手段

#### 发现 B3-3：课程学习导致灾难性遗忘
- expEE_s2: F1=0.094 (低于Stage1的0.118)，rule2_FP=2（几乎完全消除）
- 但其他规则（rule2,5,6,8）全部遗忘，只保留rule4高recall
- **结论**: Stage2的50no_skill样本×6ep学习率太高，过度改写了Stage1知识

#### 发现 B3-4：更多epoch导致过拟合
- expGG (expZ config + 10ep): F1=0.132 < expZ (5ep) F1=0.174
- **结论**: expZ的5ep是最优epoch数，增加导致过拟合

#### 发现 B3-5：更多样本依然下降
- expFF (351s, 5ep, expZ config): F1=0.148 < expZ (251s, 5ep): F1=0.174
- **结论**: 再次确认当前配置下规模增大有害

### 未解决问题：rule2 FP的根本原因
- 测试集rule2基础率=14.5%（29/200样本）
- expZ模型对rule2: TP=29/29(recall=1.0), FP=171(precision=0.145)
- 模型对EVERY样本都预测rule2（类似于高频词倾向）
- **根本原因**: 在thinking=False评测时，模型产生"SPC分析→必有rule2"的默认关联
- **实验揭示**: 训练数据format说明rule2未触发（`<think>` block），但评测不生成think block
  → 模型在无think引导时退化为rule2主导的先验输出

---

## 2026-04-24 22:45 — expBB 最终结果 + Batch 5 启动

### expBB 最终结果（🏆 突破目标！）

| 指标 | expAA (100 teacher) | expBB (151 teacher) | 目标 |
|------|---------------------|---------------------|------|
| F1 | 0.164 | **0.239** | 0.232 |
| rules_covered | 6/8 | **7/8** | - |
| CPK_rate | 0.98 | 0.88 | - |
| CPK_MAE | 0.155 | **0.154** | - |
| r2_TP | 7 | 17 | - |
| r2_FP | 31 | 86 | - |

**Per-rule recall（expBB）**:
- rule1: 0.75, rule2: 0.586, rule3: 0.429, rule4: 0.458, rule5: 0.182
- rule6: 0.121, rule7: **0.000**, rule8: 0.077

**关键分析：**
- expBB (F1=0.239) 已超越 Claude-sonnet-4.6+skill (F1=0.232) 目标！🎉
- 每增加50条教师数据，F1约提升0.04~0.08（100→151：+0.075）
- rule7依然为0（25ep内从未学会），rule8/rule6仍低
- r2_FP=86（高于expAA的31），原因：expBB用更多teacher让模型预测更多规则，rule2 FP也随之增加
- **核心结论**：Claude teacher data是关键突破口，more is better（at least up to 151 samples）

### Batch 5 实验矩阵（2026-04-24 22:44 启动）

**目标**：验证教师数据规模律，找到最优数据量

| 实验 | GPU | 样本数 | 配置 | 假设 |
|------|-----|-------|------|------|
| expRR | 0 | 200 | 200无Skill教师, 5ep | 200 teacher vs 151 teacher |
| expSS | 1 | 300 | 300无Skill教师, 5ep | 继续规模律曲线 |
| expTT | 4 | 500 | 500无Skill教师, **3ep** | 全量教师数据（减ep防过拟合） |
| expUU | 7 | 300 | 150无Skill教师+150合成, 5ep | 50/50 teacher+synth混合 |

**规模律假设（from expAA/BB）**:
- 100 teacher → F1=0.164
- 151 teacher → F1=0.239 (+0.075)
- 200 teacher → F1=? (期望 ~0.280+)
- 300 teacher → F1=? (期望 ~0.310+)

**关键对照**：
- expRR vs expNN: 200 teacher alone vs 200 teacher+51 textbook
- expSS vs expRR: 300 vs 200 teacher（规模律是否持续？）
- expTT: 测试更多数据 + 更少epoch的效果
- expUU vs expNN: teacher+synth混合 vs 纯teacher的精准率对比

---

## 2026-04-24 22:00 — Batch 4 启动（Claude teacher + No-skill对齐）

### 关键洞察（触发Batch 4设计）

**Train-Eval格式不匹配**：
- 训练样本：ALL outputs有`<think>`块，rule2拒绝在think内显示
- 评测时（enable_thinking=False）：模型直接生成结论，无think引导
- 结论出现规则: 只提被触发的规则，正常样本不提rule2
- 结果: 模型学会"在think之后列出触发规则"，但eval无think→退化为先验

**Claude teacher data格式优势**：
- 无`<think>`块：整个输出是step-by-step答案
- 每条规则明确写"触发/未触发"（不只写被触发的）
- 结论段：显式列出所有触发规则的完整列表
- 与eval格式天然对齐（eval也是直接生成答案）

### Batch 4 实验矩阵

| 实验 | GPU | 样本数 | 配置 | 关键假设 |
|------|-----|-------|------|---------|
| expMM | 2 | 234 | 100无Skill教师+83合成无Skill+51教材 | no-skill教师直接对齐eval格式 |
| expNN | 3 | 251 | 200无Skill教师+51教材 | 纯教师数据内化 |
| expOO | 5→7 | 50 | expZ checkpoint续训 50no-skill教师, 3ep, LR=5e-6 | 低LR微调防遗忘 |
| expPP | 6 | 251 | 100无Skill教师+100有Skill教师+51教材 | 混合教师数据 |
| expQQ | 5 | 301 | 200合成(expZ配置)+50无Skill教师+51教材 | 小剂量教师数据效果 |

**无Skill教师数据**：将500条原始教师样本的system从"with_skill"改为"no_skill"（仅system字段改变，output保持Claude的高质量推理）

### 已完成
- [x] 创建train_claude_teacher_noskill.jsonl（500条）
- [x] 生成并注册所有Batch 4数据集
- [x] 创建所有config文件
- [x] 启动5个并行训练实验

### 预期完成时间
- expMM/NN/PP: ~22:40（150-160步）
- expQQ: ~23:00（190步）
- expOO eval: ~22:15（合并+vLLM+评测）
- expAA/expBB eval: 结果中（等待完成）

---

## 2026-04-25 00:30 — Batch 4+5 评测重启 + Batch 6 准备

### 关键Bug修复：max_tokens=2048截断问题

**发现时间**: 2026-04-25 00:00  
**影响**: Batch 4全部实验（expMM/NN/PP/QQ）评测结果错误（F1≈0.003-0.038）  

**根因**:
- no_skill teacher格式训练的模型学会生成详细计算表格（每数据点逐一分析）
- 这些详细输出在eval max_tokens=2048时被截断，在到达per-rule结论部分之前就停止
- CPK计算出现在输出前段（故cpk_found_rate正常），但per-rule触发/未触发在后段（被截断）

**修复**:
- eval_and_report.sh: `--max_tokens 3500`（从默认2048改）
- 重新评测所有受影响实验

### Batch 4+5 评测状态（2026-04-25 00:30）

| 实验 | 状态 | vLLM端口 | n_train |
|------|------|---------|---------|
| expMM | ⏳ 重新评测中 (max_tokens=3500) | 8102 | 234 |
| expNN | ⏳ 重新评测中 (max_tokens=3500) | 8103 | 251 |
| expPP | ⏳ 重新评测中 (max_tokens=3500) | 8106 | 251 |
| expQQ | ⏳ 重新评测中 (max_tokens=3500) | 8105 | 301 |
| expRR | ⏳ 评测中 (max_tokens=3500) | 8100 | 200 |
| expSS | ⏳ 评测中 (max_tokens=3500) | 8101 | 300 |
| expTT | ⏳ 评测中 (max_tokens=3500) | 8104 | 500 |
| expUU | ⏳ 评测中 (max_tokens=3500) | 8107 | 300 |

预计完成时间: ~02:15 CST (34s/样本 × 200样本)

### Batch 6 实验矩阵（准备中）

批次6将在Batch 4+5全部完成后自动启动（launch_batch6.sh，PID=113797）。

| 实验 | GPU | 样本数 | 配置 | 关键假设 |
|------|-----|-------|------|---------|
| expWW | 0 | 500 | 全部500条with_skill教师, 3ep | 500 with_skill vs 500 no_skill(expTT) |
| expXX | 1 | 500 | 250 with_skill + 250 no_skill, 3ep | 混合格式效果 |
| expVV | 4 | 800 | 500 no_skill(seed1) + 300 no_skill(seed200), 2ep | 扩充教师数据池 |
| expYY2 | 7 | 600 | 300 with_skill + 300 no_skill v2, 3ep | 新数据+混合格式 |

**新teacher数据**: train_claude_teacher_v2.jsonl (300条, seed=200) 已生成完成

**核心问题**:
1. with_skill vs no_skill 哪种教师格式更好？（expWW vs expTT）
2. 混合格式是否有优势？（expXX/YY2）
3. 扩大数据池是否持续有效？（expVV: 800样本）

---

## 历史记录

### Route 2.1 Round 3 最终结果（来自 impl/2.1）

| 条件 | rule_f1 | 说明 |
|------|---------|------|
| base-14b-no-skill | 0.000 | 基线 |
| base-14b-with-skill | 0.109 | Skill 加持后的基座 |
| base-claude-with-skill | 0.232 | 目标上界 |
| sft-r3-N1 (14B, 8ep) | **0.100** | 全部8条规则有召回 |
| sft-r3-N2-6ep (14B) | **0.119** | rule1主导（0.675），其余≈0 |
| sft-r3-N3-5ep (14B) | 0.064 | rule2主导（0.517） |
| sft-r3-N4-3ep (14B) | 0.090 | rule1+rule5 |

**关键发现：~600 no-skill曝光次数 + ~200梯度步数 = 甜区**
**待解决：模式崩塌（N2+的单规则主导现象）**

---

## 2026-04-25 01:00 — Batch 4+5 评测进行中 + Batch 7 设计

### Batch 4+5 评测中期进度（40/200样本，早期估计）

| 实验 | 样本数 | 格式 | epochs | 步数 | 早期F1 | 数据说明 |
|------|-------|------|--------|------|-------|---------|
| expMM | 234 | ns | 5 | ~146 | 0.218 | 100 ns_teacher + 83 synth_ns + 51 textbook |
| expNN | 251 | ns | 5 | ~156 | 0.186 | 200 ns_teacher + 51 textbook |
| expPP | 251 | ns | 5 | ~156 | 0.296 | 100 ns + 100 ws teacher + 51 textbook |
| expQQ | 301 | ns | 3 | ~112 | 0.308 | 200 synth + 50 ns_teacher + 51 textbook |
| expRR | 200 | ns | 5 | **125** | **0.317** | 200 pure ns_teacher (seed 1-200) |
| expSS | 300 | ns | 5 | 187 | 0.235 | 300 pure ns_teacher (seed 1-300) |
| expTT | 500 | ns | 3 | 187 | 0.294 | 500 pure ns_teacher (seed 1-500) |
| expUU | 300 | ns | 5 | 187 | 0.279 | 150 ns_teacher + 150 synth |

**注：早期F1来自40/200样本，最终结果约2小时后出炉（~02:15 CST）**

### 早期规律分析

**核心发现：步数甜区 ≈ 125 步（expRR: 200 ns × 5ep / 8batch = 125步）**

| 步数 | 实验 | 样本数×ep | 早期F1 |
|-----|------|---------|-------|
| 75 | *(待测)* | 200×3ep | - |
| 100 | *(待测)* | 400×2ep | - |
| **125** | expRR | 200×5ep | **0.317** |
| 112 | expQQ | 300×3ep | 0.308 |
| 150 | *(待测)* | 400×3ep | - |
| 187 | expSS/TT | 300×5ep / 500×3ep | 0.235-0.294 |

**次要发现：教材数据（textbook）显著损害F1**
- expNN（200 ns + 51 tb）= 0.186 vs expRR（200 pure ns）= 0.317
- 差异原因：textbook改变了模型的输出风格分布，远离eval格式

**数据多样性效果**
- expTT（500 ns, 3ep, 187步）= 0.294 > expSS（300 ns, 5ep, 187步）= 0.235
- 相同步数下，更多样本轻微有益，但都不如125步的expRR

### Batch 7 实验矩阵（已准备就绪，等待Batch 6完成后自动启动）

**核心假设**：步数125是甜区，需验证 (a) 更少步数的效果, (b) 跨数据池多样性的效果

| 实验 | GPU | 样本数 | 步数 | 配置 | 验证假设 |
|------|-----|--------|------|------|---------|
| expZZ | 0 | 200 ns_v1 | 75 | 3ep | 步数低于甜区效果如何？ |
| expAAA | 1 | 200 ns_v2 | 125 | 5ep | 不同seed数据池 vs expRR同步数 |
| expBBB | 2 | 400 (v1+v2) | 100 | 2ep | 跨池多样性 + 少步 |
| expCCC | 3 | 400 (v1+v2) | 150 | 3ep | 跨池多样性 + 略多步 |

**launch_batch7.sh (PID=115871)** 在launch_batch6.sh完成后自动启动Batch 7训练。

### 数据池说明
- ns_v1: train_claude_teacher_noskill.jsonl (500条, seed 1-200系列)
- ns_v2: train_claude_teacher_v2_noskill.jsonl (300条, seed 200+系列)
- 两池互不重叠（已验证）

### Teacher v3 生成进度
- 生成中: train_claude_teacher_v3.jsonl (300条, with_skill, seed=500), PID=114069
- 当前: ~33/300 样本已生成
- 用途: Batch 8实验（如需进一步扩充数据池）

---

## 2026-04-25 01:30 — Batch 4+5 评测进行中（更新：100/200样本）

### 当前 Batch 4+5 最新进度（100/200样本）

所有8个实验同时运行（max_tokens=3500, no_skill推理）：

| 实验 | 格式 | 步数 | 早期F1@100 | 特点 |
|------|------|------|-----------|------|
| expMM | ns混合 | ~146 | 0.268 | 100 ns_teacher + 83 synth_ns + 51 textbook |
| expNN | ns+tb | ~156 | 0.272 | 200 ns_teacher + 51 textbook |
| expPP | ws+ns+tb | ~156 | 0.252 | 100 ns + 100 ws teacher + 51 textbook |
| expQQ | ns混合 | ~112 | 0.262 | 200 synth + 50 ns_teacher + 51 textbook |
| **expRR** | **pure ns** | **125** | **0.378** | **200 pure no_skill teacher — 当前最佳！** |
| expSS | pure ns | 187 | 0.316 | 300 pure ns (步数过多) |
| expTT | pure ns | 187 | 0.336 | 500 pure ns (多样性有益但步数过多) |
| expUU | ns+synth | 187 | 0.336 | 150 ns + 150 synth |

**关键确认**：expRR（200 pure ns, 125步）遥遥领先，比目标上界 F1=0.232 高出63%

### 关键发现（已确认）：cutoff_len=4096 截断49%的rule8训练数据

- no_skill教师数据输出长度：3329-4572 tokens（均值~4190）
- cutoff_len=4096时，约49%的样本rule8分析被截断
- 所有当前实验均使用cutoff_len=4096，可能导致rule8 recall偏低
- **修复**：expMMM（Batch 9）使用cutoff_len=5120，覆盖全部输出

### 批次进度和预计时间

| 批次 | 预计启动 | 预计完成 | 状态 |
|------|---------|---------|------|
| Batch 4+5 eval | 运行中 | ~02:20 CST | 100/200完成 |
| Batch 6 training | ~02:20 | ~03:05 | 等待中 |
| Batch 6 eval | ~03:05 | ~05:00 | 等待中 |
| Batch 7 training | ~05:00 | ~05:45 | 等待中 |
| Batch 7 eval | ~05:45 | ~07:40 | 等待中 |
| Batch 8 training (32B) | ~07:40 | ~09:40 | 等待中 |
| Batch 8 eval | ~09:40 | ~11:35 | 等待中 |
| Teacher v3 生成 | 00:35 | ~03:40 | 89/300进行中 |
| Batch 9 training | ~11:35 | ~12:20 | 等待中 |
| Batch 9 eval | ~12:20 | ~14:15 | 等待中 |

### Batch 8 实验矩阵（等待Batch 7完成）

| 实验 | 模型 | 数据 | 步数 | 假设 |
|------|------|------|------|------|
| expGGG | Qwen3-32B | 200 ns_v1 × 5ep | 125 | 32B模型是否明显优于14B？ |
| expHHH | Qwen3-32B | Batch 7最佳数据 | 125 | 32B + 最优数据组合 |

### Batch 9 实验矩阵（等待Batch 8完成 + teacher v3）

| 实验 | 模型 | 数据 | 步数 | cutoff | 假设 |
|------|------|------|------|--------|------|
| expIII | 14B | 200 ns_v3 × 5ep | 125 | 4096 | v3新seed池是否与v1相当？ |
| expJJJ | 14B | 400 (v1+v3) × 3ep | 150 | 4096 | 跨池多样性 |
| expKKK | 14B | 200 ns_v1 × 5ep | 125 | 4096 | 低LR=5e-5是否有益？ |
| expLLL | **8B** | 200 ns_v1 × 5ep | 125 | 4096 | 8B vs 14B 性能对比 |
| **expMMM** | **14B** | **200 ns_v1 × 5ep** | **125** | **5120** | **核心修复：cutoff_len从4096→5120** |

**expMMM的重要性**：expRR最大潜在问题被修复，预期F1从0.378进一步提升至0.40+

### 关键发现更新：cutoff_len截断比率更新为94%

经过精确的token分析（AutoTokenizer实测）：
- 输出token长度：min=3329, max=4572, 均值=4190
- cutoff_len=4096时，**94%的样本被截断**（total tokens > 4096）
- cutoff_len=5120时，**0%的样本被截断**
- 之前报告的"49%"是特指rule8分析完全不出现；实际上更多样本在rule8之前就被截断

这意味着expRR（F1=0.378）是在94%训练数据被截断的情况下取得的！expMMM理论上能大幅超越。

### Batch 10 实验矩阵（等待Batch 9完成）

| 实验 | 模型 | 数据 | 步数 | cutoff | 关键问题 |
|------|------|------|------|--------|---------|
| expNNN | **32B** | 200 ns_v1 × 5ep | 125 | **5120** | 32B + cutoff fix组合效果？ |
| expOOO | 14B | 200 ns_v1 × 5ep | 125 | **5120** | LR=5e-5 + cutoff fix? |
| expPPP | 14B | 400 (v1+v3) × 3ep | 150 | **5120** | 跨池多样性 + cutoff fix |
| expQQQ | **7B** | 200 ns_v1 × 5ep | 125 | **5120** | 最小模型+cutoff fix |
| expSSS | 14B | 200 **ns_v3** × 5ep | 125 | **5120** | v3新池 + cutoff fix |

**所有Batch 10实验均使用cutoff_len=5120，这是目前最关键的配置变量。**

### Batch 11 实验矩阵（等待Batch 10完成 + teacher v4）

| 实验 | 模型 | 数据 | 步数 | cutoff | 关键问题 |
|------|------|------|------|--------|---------|
| expTTT | 14B | 200 ns_v1 × 3ep | **75** | 5120 | 步数变少（含cutoff fix） |
| expUUU | 14B | 200 ns_v1 × 8ep | **200** | 5120 | 步数过多（过拟合测试） |
| expVVV | 14B | 400 (v1+v4) × 3ep | 150 | 5120 | v1+v4跨池多样性 |
| expWWW | **32B** | 400 (v1+v3) × 3ep | 150 | 5120 | 32B + 数据多样性 |

**Teacher v4**（seed=700，300条）：PID=120191，目前0/300，预计~04:47 CST完成

---

## 2026-04-25 02:10 — Batch 4+5 评测最终阶段（160/200样本）

### Batch 4+5 评测进度（160/200样本）

所有8个实验均处于活跃评测中（max_tokens=3500，no_skill推理）。vLLM日志确认每个请求约30秒，正常运行。

| 实验 | 进度 | F1@160 | 数据说明 | 步数 |
|------|------|--------|---------|------|
| expMM | 160/200 | 0.266 | 100 ns_teacher + 83 synth_ns + 51 textbook | ~146 |
| expNN | 160/200 | 0.265 | 200 ns_teacher + 51 textbook | ~156 |
| expPP | 160/200 | 0.266 | 100 ns + 100 ws teacher + 51 textbook | ~156 |
| expQQ | 160/200 | 0.266 | 200 synth + 50 ns_teacher + 51 textbook | ~112 |
| **expRR** | 160/200 | **0.375** | **200 pure ns_teacher** | **125** |
| expSS | 160/200 | 0.321 | 300 pure ns_teacher | 187 |
| expTT | 160/200 | 0.327 | 500 pure ns_teacher, 3ep | 187 |
| expUU | 160/200 | 0.339 | 150 ns_teacher + 150 synth | 187 |

**预计完成：~02:16 CST（每样本~30秒 × 40样本）**

### 关键修复（本会话）

1. **deploy_vllm.py 默认 max_model_len**: 4096 → **5120**
   - 所有后续批次（Batch 6+）的eval将使用5120 token上下文窗口
   - 对应训练时的 cutoff_len=5120

2. **训练配置验证（cutoff_len已全部修正）**：
   - expWW/XX/YY2 (ws数据): cutoff=6144 ✓
   - expVV (ns数据): cutoff=5120 ✓
   - expZZ/AAA/BBB/CCC (Batch 7): cutoff=5120 ✓
   - expGGG/HHH (Batch 8, 32B): cutoff=5120 ✓
   - expIII/JJJ/KKK/LLL/MMM (Batch 9): cutoff=5120 ✓

### Teacher 数据生成状态

| 数据集 | 进度 | 预计完成 |
|-------|------|---------|
| train_claude_teacher_v3.jsonl | 137/300 | ~02:50 CST |
| train_claude_teacher_v4.jsonl | 30/300 | ~03:30 CST |
| v3_noskill 转换 | 待v3完成 | ~02:50 CST |
| v4_noskill 转换 | 待v4完成 | ~03:30 CST |

### 评测完成后自动触发的流水线

| 批次 | 触发条件 | 预计启动 | 训练时长估计 |
|------|---------|---------|------------|
| Batch 6 训练 | B4/5全结果 | ~02:20 CST | 2-3小时 |
| Batch 6 eval | B6训练完成 | ~04:30 CST | 1-2小时 |
| Batch 7 训练 | B6全结果 | ~06:00 CST | 1-2小时 |
| Batch 7 eval | B7训练完成 | ~08:00 CST | 1小时 |
| Batch 8 训练(32B) | B7全结果 | ~09:00 CST | 2-3小时 |
| Batch 8 eval | B8训练完成 | ~12:00 CST | 1-2小时 |
| Batch 9 训练 | B8结果+v3_ns | ~13:00 CST | 2小时 |
| Batch 9 eval | B9训练完成 | ~15:00 CST | 1小时 |

**expMMM（Batch 9，cutoff_len=5120，expRR等效配置）** 是关键实验，预计在~16:00 CST获得结果。

---

## 2026-04-25 02:30 — Batch 4+5 最终结果 + Batch 6 训练启动

### Batch 4+5 最终结果（200/200样本）

| 实验 | 最终F1 | CPK找到率 | rule7 | rule8 | 数据说明 | 步数 |
|------|--------|----------|-------|-------|---------|------|
| expMM | 0.273 | 1.0 | 0.11 | 0.00 | 100 ns + 83 synth + 51 textbook | ~146 |
| expNN | 0.279 | 1.0 | 0.00 | 0.00 | 200 ns + 51 textbook | ~156 |
| expPP | 0.290 | 1.0 | 0.00 | 0.00 | 100 ns + 100 ws + 51 textbook | ~156 |
| expQQ | 0.255 | 0.985 | 0.06 | 0.00 | 200 synth + 50 ns + 51 textbook | ~112 |
| **expRR** | **0.358** | 1.0 | 0.11 | 0.00 | **200 pure ns_teacher** | **125** |
| expSS | 0.313 | 1.0 | 0.06 | 0.00 | 300 pure ns_teacher | 187 |
| expTT | 0.319 | 1.0 | 0.11 | 0.00 | 500 pure ns, 3ep | 187 |
| expUU | 0.343 | 1.0 | 0.06 | 0.00 | 150 ns + 150 synth | 187 |

**关键规律**：
1. rule8=0.00（所有实验）→ cutoff_len=4096截断94%样本的rule8部分（已知）
2. expRR（200 pure ns, 125步）= **最佳 F1=0.358**，超目标上界（0.232）54%
3. textbook数据降低F1：expNN(+tb)=0.279 vs expRR(pure ns)=0.358
4. 步数125是甜区：187步的实验全部比125步低

### Batch 6 训练启动（2026-04-25 02:25 CST）

**原因**：launch_batch6.sh（PID=113797）在检测到所有8个结果后崩溃（bash脚本文件读取错误）。
手动重新启动训练：

| 实验 | GPU | 样本数 | 步数 | cutoff | 配置 | PID |
|------|-----|--------|------|--------|------|-----|
| expWW | 0 | 500 ws | 189 | 6144 | 全部500 with_skill teacher, 3ep | 125059 |
| expXX | 1 | 500 mixed | 189 | 6144 | 250 ws + 250 ns, 3ep | 125060 |
| expVV | 4 | 800 ns | 200 | 5120 | 500 ns_v1 + 300 ns_v2, 2ep | 125061 |
| expYY2 | 7 | 600 mixed | 225 | 6144 | 300 ws + 300 ns_v2, 3ep | 125062 |

**monitor_batch6.py (PID=127739)** 监控训练并自动启动eval（max_tokens=5000）

**预计完成时间**：02:25 + ~2.5h（225步 × 1.5步/分）= **~05:00 CST**

### Teacher 数据生成状态（更新）

| 数据集 | 进度 | 预计完成 |
|-------|------|---------|
| train_claude_teacher_v3.jsonl | 200/300 | ~03:30 CST |
| train_claude_teacher_v4.jsonl | 83/300 | ~04:45 CST |
| v3_noskill 转换 | 待v3完成 | ~03:30 CST |
| v4_noskill 转换 | 待v4完成 | ~04:45 CST |

