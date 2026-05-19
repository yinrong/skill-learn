# Round 5 数据规划

> 基于 ws_raw.jsonl（93条 feishu 日志）细致分析
> 目标：构建可学习、参数来源可追溯的高质量训练集

---

## 一、数据可用性分析

### 1.1 当前数据规模与 Skill 分布

| Skill | 样本数 | 可用（估计）|
|-------|--------|-----------|
| object-data-query | 25 | ~20 |
| general-kpi-query | 20 | ~15 |
| line-operation-skill | 16 | ~8 |
| 分板机过站明细查询 | 10 | ~8 |
| lineside-material-query | 9 | ~7 |
| line-attendance-query | 6 | ~5 |
| line-exemption-query | 3 | ~2 |
| equipment-cpk-query | 3 | ~2 |
| workstion-kpi-query | 1 | ~1 |
| **合计** | **93** | **~68** |

---

### 1.2 可用性问题：必须过滤的样本

**问题一：最终答案不完整（18%，约17条）**

最后一条 assistant 消息是过渡性文字，不包含实际数据：

```
"数据已整理如上 🐙"              ← 无数据
"报告已通过飞书卡片发送完毕"      ← 无数据
"（此为系统上下文注入，等待下一条用户消息。）" ← 无数据
```

这类样本训练后，模型会学到"最后说一句无意义的话即可"的捷径。**必须过滤**。

**问题二：明显损坏样本（3条）**

- `sample 2`：最后一条 assistant 消息内容是用户的问题原文
- `sample 27/28`：最后一条 assistant 消息是原始 `<tool_call>` XML 标签，未被格式化

**必须过滤**。

**问题三：Skill 标签错误（约 12 条）**

- `object-data-query` 中有 10 条调用了 `get_idi_model_data`，违反该 skill 的定义（"不经过指标模型"）
- `lineside-material-query` 中有 2 条问的是 UPH/良率，应属于 `general-kpi-query`

标签错误会给模型发送矛盾信号。**必须过滤或重新分类**。

**问题四：高度重复样本（line-operation-skill 中 7/16 条相同问题）**

"查询S03和S04线体前一天夜班（21:00-09:00）的运营对比报告" 出现 7 次（占该 skill 44%）。大量重复导致模型过拟合这一特定问法。**保留2条，其余5条过滤**。

---

### 1.3 可用性小结

| 问题类型 | 数量 | 处理方式 |
|---------|------|---------|
| 最终答案不完整 | ~17 | 过滤 |
| 损坏样本 | 3 | 过滤 |
| Skill 标签错误 | ~12 | 过滤或重分类 |
| 高度重复 | ~5 | 去重，保留1-2条 |
| **可用样本（过滤后）** | **~56** | — |

---

## 二、可学习性分析

### 2.1 致命问题：model_id 来源不可追溯（98.8%）

这是 Round 4 失败的核心原因，数据中有确凿证据。

**事实**：93条样本中共 258 次 `get_idi_model_data` 调用，其中 **255次（98.8%）使用的 model_id 在任何前序工具响应中均未出现过**。

工具响应里出现的 model_id：`6668, 7720, 8742, 9923...`（旧系统绑定）

模型实际调用的 model_id：`12198, 12202, 12204, 12223, 12385, 12389...`（来自模型记忆）

```
训练数据里的轨迹实际上是：
  list_object_types → [返回 HXZB_PE01_C 等]
  get_object_type_metrics(api_names="line") → [返回 modelId=6668]
  get_idi_model_data(model_id=12202)   ← 12202 从未出现在任何响应里！
```

**如果用这些数据训练，模型学到的是：无论工具返回什么，UPH 就用 12202。**
这与 skill doc 明确禁止的行为完全一致：`❌ 根据历史经验或硬编码猜测 model_id`。

**各 skill 受影响程度：**

| Skill | idi 调用次数 | 可追溯 | 孤立（硬编码）|
|-------|------------|--------|--------------|
| equipment-cpk-query | 3 | 0 | 3（全部用12365）|
| general-kpi-query | 64 | 2 | 62 |
| line-attendance-query | 12 | 0 | 12（12385/12389）|
| line-operation-skill | 111 | 0 | 111 |
| lineside-material-query | 22 | 0 | 22 |
| object-data-query | 37 | 1 | 36 |
| **合计** | **258** | **3(1.2%)** | **255(98.8%)** |

---

### 2.2 次要问题：无思考链（100%）

93条样本的 assistant 消息均无 `<think>` block。模型在生产中关闭了思考输出。

这意味着训练数据中**工具选择和参数构造的推理过程完全缺失**——模型看不到"我为什么选这个工具、这个参数从哪来"的示范。ns 格式训练时，这部分知识无法从 output 传递到权重。

---

### 2.3 问题三：问法多样性不足

| Skill | 独立问法数 | 样本数 | 重复率 |
|-------|-----------|--------|--------|
| line-operation-skill | 9 | 16 | 44% 重复 |
| general-kpi-query | 14 | 20 | 30% 重复 |
| object-data-query | 19 | 25 | 24% 重复 |
| equipment-cpk-query | 2 | 3 | 1条重复 |

问法单一会导致模型对特定措辞过拟合，遇到"S04线今天UPH如何"就答不出（训练里只有"查询S04线今天的UPH"）。

---

### 2.4 可学习性小结

| 问题 | 影响程度 | 能否用现有数据训练 |
|------|---------|----------------|
| model_id 来源不可追溯（98.8%）| 致命 | **不能**，会学到错误行为 |
| 无思考链 | 严重 | 可以训练，但效果有上限 |
| 问法多样性不足 | 中等 | 可以训练，但泛化弱 |

**结论：现有56条可用样本，直接训练会重蹈 Round 4 覆辙。必须在数据增强后再训练。**

---

## 三、数据增强方案

### 方案一：修复 metrics 响应（修复参数来源问题）【优先级：最高】

**目标**：让每条 get_idi_model_data 调用前，都有一个 get_object_type_metrics 响应包含该 model_id。

**做法**：对每条样本，扫描所有 get_idi_model_data 调用，找到它用的 model_id，然后修改最近的 get_object_type_metrics 响应，在 modelBindings 里注入该 model_id。

**修复前（真实日志原样）**：
```json
// get_object_type_metrics 响应
{"data": [{"apiName": "line", "modelBindings": [
  {"modelId": 6668, "modelName": "昌平排产数据", ...}
]}]}

// 后续 get_idi_model_data 调用
{"model_id": 12202, "params_json": "[...]"}   ← 12202 从未出现
```

**修复后（注入 model_id）**：
```json
// get_object_type_metrics 响应（注入真实绑定）
{"data": [{"apiName": "line", "modelBindings": [
  {"modelId": 6668, "modelName": "昌平排产数据", ...},
  {"modelId": 12202, "modelName": "线体小时别UPH",     ← 注入
   "joinKey": "line_code",
   "params": [{"name": "start_date", "type": "date", "optional": false},
               {"name": "line", "type": "string", "optional": true}]}
]}]}

// 后续 get_idi_model_data 调用
{"model_id": 12202, ...}   ← 现在有明确来源
```

这样训练，模型学到的是"我看到 metrics 返回了 12202，所以我用 12202"——而不是硬编码。

**实现**：写脚本 `round5/scripts/fix_metrics_responses.py`，对每条样本自动注入。

---

### 方案二：添加思考链（修复无推理链问题）【优先级：高】

**目标**：在每条 assistant 消息里（尤其工具调用前）加入推理过程，让模型学到决策依据。

**做法**：用 Claude + skill doc，对每条样本重新生成 assistant 消息（保留工具响应原样，只重写 assistant 部分），要求 Claude 输出完整推理链。

**增强前（原始日志）**：
```
[assistant]
TOOL_CALL: get_idi_model_data(model_id=12202, ...)
```

**增强后（Claude 重写 + 保留 tool responses 原样）**：
```
[assistant]
<think>
  用户问 S04 线今天的 UPH。
  上一步 get_object_type_metrics 返回了 line 对象的 modelBindings：
    - modelId=12202, modelName="线体小时别UPH"
  这正好是 UPH 指标对应的模型。
  参数：line=S04，start_date=今天（已从 execute_shell_command 获取）。
  构造参数 params_json。
</think>
TOOL_CALL: get_idi_model_data(model_id=12202, params_json="[...]")
```

ns 转换后（删 skill doc），这条推理链留在 output 里，模型能从中学到"我是因为看到了 metrics 响应才用 12202"。

---

### 方案三：多样化问法变体（修复问法单一问题）【优先级：中】

**目标**：每条原始样本生成 3-5 个问法变体，保持工具调用和答案不变，只改问法措辞。

**做法**：用 Claude 对原始用户问题做改写，覆盖不同表达风格。

**原始问题**：
```
查询S03和S04线体前一天夜班（21:00-09:00）的运营对比报告
```

**变体生成**（实际内容示例）：
```
变体1：昨晚夜班S03/S04线的综合运营怎么样
变体2：S03跟S04昨天夜班对比一下，看看产出、UPH、良率
变体3：帮我出一份上个夜班（21点到早上9点）S03和S04的运营情况对比
变体4：昨夜S03/04两条线的运营数据对比
```

每个变体复用原样本的完整工具调用链和最终答案，只替换第一条 user message 中的问题文字。

---

### 方案四：补充稀缺 Skill 的教师合成数据【优先级：中】

`equipment-cpk-query`（3条）和 `workstion-kpi-query`（1条）样本太少，泛化能力极弱。

**做法**：用 Claude + skill doc，针对这两个 skill 从零生成训练样本。生成要求：
- 包含完整推理链（方案二的要求）
- get_object_type_metrics 响应必须包含对应 model_id（方案一的要求）
- 问法覆盖：不同装备编码、不同日期表达（今天/昨天/2026-04-24）、不同班次

**equipment-cpk-query 合成样本示例**：

```
[user]
## 技能指引
name: equipment-cpk-query
...（完整 skill doc）...
## 用户问题
查一下昨天回焊炉的CPK

[assistant]
<think>
  Step 0: entity_type=装备, entity_code=待查（用户未提供具体编码）
  Step 1: 无记忆注入，memory_hit=false，执行 Step 2
</think>
TOOL_CALL: list_object_types(keyword="装备")

[tool]
{"data": [{"apiName": "device", "displayName": "装备", ...}]}

[assistant]
<think>
  apiName=device，继续查 metrics
</think>
TOOL_CALL: get_object_type_metrics(api_names="device")

[tool]
{"data": [{"apiName": "device", "modelBindings": [
  {"modelId": 54321, "modelName": "装备CPK统计",
   "params": [{"name": "device_code", "type": "string", "optional": true},
               {"name": "startTm", "type": "datetime", "optional": false},
               {"name": "endTm", "type": "datetime", "optional": false}]}
]}]}

[assistant]
<think>
  找到 CPK 模型：modelId=54321。
  entity_code=待查，执行 Step 3 查装备列表
</think>
TOOL_CALL: get_object_type_data(api_name="device", filter_json="...")

[tool]
{"data": {"data": [{"device_code": "M-SMT-RF-A-HCY-0301",
                    "device_name": "回焊炉A", ...}], ...}}

[assistant]
<think>
  device_code=M-SMT-RF-A-HCY-0301
  昨天=2026-05-13，startTm=2026-05-13 09:00:00，endTm=2026-05-14 09:00:00
</think>
TOOL_CALL: get_idi_model_data(model_id=54321,
  params_json='[{"name":"device_code","value":"M-SMT-RF-A-HCY-0301"},
               {"name":"startTm","value":"2026-05-13 09:00:00"},
               {"name":"endTm","value":"2026-05-14 09:00:00"}]')

[tool]
{"data": {"data": [{"device_code": "M-SMT-RF-A-HCY-0301",
                    "cpk": 1.67, "cp": 1.72, "date": "2026-05-13"}]}}

[assistant]
回焊炉A（M-SMT-RF-A-HCY-0301）昨天（2026-05-13）的CPK数据：

| 装备 | CPK | CP | 日期 |
|------|-----|----|------|
| 回焊炉A | 1.67 | 1.72 | 2026-05-13 |

CPK=1.67 > 1.33，**过程能力良好**。
```

注意：model_id=54321 是合成的随机值，**不同样本使用不同随机 model_id**，防止记忆化。

---

## 四、增强后预期数据量

| 来源 | 方法 | 预计样本数 |
|------|------|-----------|
| 原始 feishu 日志（过滤后） | 方案一+二（修复+加 think）| ~56 |
| 问法变体扩充 | 方案三（每条 ×3 变体）| ~168 |
| equipment-cpk-query 合成 | 方案四 | 30 |
| workstion-kpi-query 合成 | 方案四 | 20 |
| line-exemption-query 补充 | 方案四 | 20 |
| **合计** | | **~294 条** |

---

## 五、执行顺序

```
Step 1: 过滤损坏、不完整、标签错误样本
        → round5/data/ws_feishu_clean.jsonl（~56条）

Step 2: 修复 metrics 响应（注入 model_id）
        → round5/scripts/fix_metrics_responses.py

Step 3: 用 Claude 补全思考链（重写 assistant 消息）
        → round5/scripts/add_thinking.py

Step 4: 生成问法变体
        → round5/scripts/gen_question_variants.py

Step 5: 合成稀缺 skill 教师数据
        → round5/scripts/gen_synthetic.py

Step 6: 合并、去重、验证参数可追溯性（100% 通过才入库）
        → round5/data/train_final.jsonl

Step 7: ns 格式转换（删 skill doc，保留工具声明）
        → round5/data/train_ns_v5.jsonl
```

---

*2026-05-15*
