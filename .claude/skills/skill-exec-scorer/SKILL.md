---
name: skill-exec-scorer
description: 评估微调后 14B 模型的 skill 执行质量。输入一条对话记录和对应的 skill doc，逐条核查自检清单，输出结构化评分。当用户说"评估模型输出"、"给这条对话打分"、"跑 skill 执行评估"时触发。
allowed-tools: Read, Write, Bash
---

# Skill 执行质量评分

对微调后模型（无 skill doc 推理）的单次 skill 执行结果进行合规评分。

---

## 输入格式

需要提供两样东西：

1. **对话记录**（JSON 或文件路径）：包含 messages（含 tool_calls 和 tool 响应）
2. **Skill doc**（文本或文件路径）：对应技能的完整文档，含自检清单

---

## 评分流程

### Step 1：解析对话

从 messages 中提取：
- 调用的工具序列（按顺序）
- 每次 `get_idi_model_data` 使用的 model_id
- 每次 `get_object_type_metrics` 的响应中出现的 modelId 列表
- 最终 assistant 文字答案
- 所有 `<think>` 块内容

### Step 2：逐条核查自检清单

对 skill doc 中 `## 自检清单` 的每一条，判断：

| 状态 | 含义 |
|------|------|
| ✓ PASS | 满足 |
| ✗ FAIL | 违规，附说明 |
| — SKIP | 对话中信息不足以判断（跳过，不计入分母）|

**通用核查逻辑（适用所有 skill）**：

1. **发现步骤是否完整**：
   - 若 skill 要求调用 `get_idi_model_data`，检查是否先调了 `list_object_types` 和 `get_object_type_metrics`
   - 若跳过任一步骤 → FAIL

2. **model_id 来源合法性**：
   - 提取所有 `get_idi_model_data` 的 model_id 参数
   - 检查每个 model_id 是否出现在前序 `get_object_type_metrics` 的工具响应中
   - 若工具响应被截断（JSON 不完整）→ SKIP
   - 若完整但 model_id 不在其中 → FAIL，说明"model_id 未从工具响应获取，疑似硬编码"

3. **禁止调用的工具**：
   - 检查 skill doc 中 `❌` 禁止项
   - 若调用了禁止的工具 → FAIL

4. **必填参数完整性**：
   - 对最后一次 `get_idi_model_data` 调用，检查 `params_json` 中是否包含 skill doc 规定的所有 `optional=false` 参数
   - 若缺失任何必填参数 → FAIL

5. **输出数值与工具响应一致性**：
   - 从最终答案中提取数字（UPH、良率、CPK 等）
   - 在工具响应中查找对应数值
   - 若答案中的关键数值在所有工具响应中均找不到 → FAIL，说明"数值疑似推导或捏造"
   - 若工具响应不完整 → SKIP

6. **最终答案可见性**：
   - 检查最后一条 assistant 消息是否有非空的可见文本（非仅 thinking block）
   - 若最终输出全在 `<think>` 里 → FAIL

### Step 3：计算得分

```
合规得分 = PASS 数 / (PASS + FAIL) × 100%
（SKIP 不计入分母）
```

---

## 输出格式

```json
{
  "skill": "equipment-cpk-query",
  "overall": "PASS" | "FAIL",
  "score": 85,
  "checklist": [
    {
      "item": "model_id 来源合法，非猜测",
      "status": "FAIL",
      "reason": "get_idi_model_data 使用了 model_id=12365，但前序 get_object_type_metrics 响应中未出现此 ID"
    },
    {
      "item": "所有 optional=false 参数已传值",
      "status": "PASS",
      "reason": "startTm 和 endTm 均已提供"
    },
    {
      "item": "输出数值来自工具响应",
      "status": "SKIP",
      "reason": "工具响应截断，无法验证"
    }
  ],
  "tool_sequence": ["list_object_types", "get_object_type_metrics", "get_object_type_data", "get_idi_model_data"],
  "idi_called": true,
  "model_id_traceable": false
}
```

---

## 批量评估用法

当需要对多条样本批量评估时，生成 `round{N}/results/exec_scores.jsonl`，每行一条结果，最后汇总：

```python
# 汇总统计
results = [json.loads(l) for l in open("exec_scores.jsonl")]
pass_rate   = sum(1 for r in results if r["overall"] == "PASS") / len(results)
idi_rate    = sum(1 for r in results if r["idi_called"]) / len(results)
trace_rate  = sum(1 for r in results if r["model_id_traceable"]) / len(results)

print(f"合规通过率: {pass_rate:.0%}")
print(f"get_idi_model_data 调用率: {idi_rate:.0%}")
print(f"model_id 可追溯率: {trace_rate:.0%}")
```

---

## 铁律

1. **工具响应截断 → SKIP，不 FAIL**：信息不足时不惩罚
2. **只评估 assistant 的行为，不评估工具响应内容的准确性**：工具响应是外部系统给的，不是模型的责任
3. **自检清单以 skill doc 中的版本为准**：不同 skill 的清单不同，不能混用
4. **批量评估时，工具响应不完整的样本仍参与统计**：idi_called 等结构性指标不受截断影响
