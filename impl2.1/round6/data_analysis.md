# Round 6 数据分析：R5 失败根因

> 基于 R5 评估结果（round5/results/R5-trajectory.json）

---

## 失败 Skill 模式分析

### equipment-cpk-query（0% idi，9/9 样本失败）

**模型行为**：`list_object_types → get_object_type_data`（完全跳过 metrics）

**根因**：R5 训练数据 30 条中，Claude 为 CPK 查询生成了"先查实体列表再查数据"的模式，而非"先查 metrics 找 model_id 再调 idi"。模型记住了 CPK→直接查实体 的捷径。

**修复**：生成 40 条强制经过 `list→metrics(device)→idi` 路径的样本，验证函数保证每条样本都有 `get_idi_model_data`。

---

### line-attendance-query（0% idi，2/2 样本失败）

**模型行为**：`list_object_types → get_object_type_metrics`（正确调了 metrics，但停止）

**根因**：评估时 `simulated_tools` 的 `_METRICS["line"]` 只返回 UPH/良率/产出（12202/12198/12204），没有出勤模型（12385/12389）。模型看到所有 model 都是 UPH 类的，无法匹配"出勤"语义，放弃调用 idi。

**修复**：
1. 评估修复：`_METRICS["line"]` 返回全部 8 个维度（含 12385/12389）
2. 训练补强：生成 30 条 metrics 响应包含 12385/12389 的样本

---

### line-exemption-query（0% idi，5/5 样本失败）

**模型行为**：`list_object_types → get_object_type_data`（跳过 metrics）

**根因**：与 equipment-cpk 类似——模型记住了直接查异常数据的捷径，没有走 metrics→idi 路径。R5 只有 20 条训练数据，量不足以覆盖所有异常查询类型。

**修复**：生成 30 条强制经过 `list→metrics→idi` 路径的样本。

---

### 分板机过站明细查询（0% idi，2/2 样本失败）

**模型行为**：`list_object_types → get_object_type_metrics → get_object_type_data × 多次`（循环查实体）

**根因**：GT 轨迹从 `get_object_type_data` 开始（先查设备编码），而模型训练数据统一从 `list_object_types` 开始，导致模式不匹配。模型进入了循环查询状态，从未达到 idi 调用。

**修复**：生成 20 条从 `get_object_type_data` 开始的样本，匹配 GT 模式。

---

## simulated_tools 修复清单

| 工具 | api_names 参数 | 修复内容 |
|------|--------------|---------|
| get_object_type_metrics | "line" | 新增 modelId 12385/12389/12194/12223/12203 |
| get_object_type_metrics | "pcb_router" | 新增 modelId 12328 |

---

*2026-05-20*
