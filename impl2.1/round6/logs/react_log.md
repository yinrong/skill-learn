# Round 6 ReAct 决策日志

---

## 规划阶段（2026-05-20）

### 决策 R6-D1：双轨修复策略

- **观察**：R5 4 个失败 skill 有 2 种不同失败模式：
  1. **跳过 metrics**（equipment-cpk, line-exemption）：模型调 `list→data`，不经过 `metrics→idi` 路径
  2. **停在 metrics**（line-attendance）：模型调 `list→metrics` 但停止，因为 metrics 返回的 model_id 是 UPH（12202）而非出勤（12385），模型判断"模型不匹配"后放弃

- **决策**：双轨并行修复：
  - **轨道 A（评估修复）**：更新 simulated_tools，让 `get_object_type_metrics(api_names="line")` 返回全部 Type A model_ids（8 个维度），模型能找到正确的 model_id
  - **轨道 B（训练数据修复）**：为 4 个失败 skill 增加 2 倍训练数据，重点展示完整的 `list→metrics→idi` 路径

- **依据**：[../round5/results/R5-trajectory.json](../round5/results/R5-trajectory.json) 失败模式分析

---

### 决策 R6-D2：分板机单独处理

- **观察**：分板机 GT 轨迹从 `get_object_type_data` 开始（不是 `list_object_types`），但模型训练数据用 `list→metrics→idi` 模式，导致模式不匹配
- **决策**：为分板机生成符合 GT 模式的训练数据（`get_object_type_data→list→metrics→idi`），而非统一使用 `list→metrics→idi` 模板
- **依据**：R5 分板机 pred 总是从 list 开始而 GT 从 data 开始

---

*执行阶段记录追加于此*
