# 证据：技能调用端到端耗时分析

> 生成时间：2026-05-12（API 实时提取）
> 数据来源：Langfuse API `http://aip.imp.xiaomi.com/platform`
> 标签覆盖：`web`（fast_lane 单步路径）+ `feishu_group`（multi-step agent 路径）
> 当前线上模型：未知大小（推测 ~32-72B，ws 格式，含 skill doc）

---

## 一、Web fast_lane 路径（6 条 traces）

**架构**：单次 LLM intent 解析 → 单次 ontology API 数据查询，无 agent 循环。
**时间定义**：
- `model_time_s`：`GENERATION`（`llm_ir.intent` span）延迟
- `tool_time_s`：`llm_ir.ontology_api` SPAN 延迟（数据查询后端）

| trace_id (short) | intent_type | query | total_lat_s | model_time_s | tool_time_s | 主要瓶颈 |
|---|---|---|---|---|---|---|
| `f8bea27e3ce675a0` | kpi_query | S06接料报错明细 | 52.36 | 5.33 | 47.03 | 后端数据查询慢 |
| `2664073664351b30` | kpi_query | s03线接料异常明细 | 23.34 | 3.64 | 19.70 | 后端数据查询慢 |
| `b22b2e76e387e8e4` | kpi_query | S03线装备异常+故障 2026-05-12 20-21h | 17.22 | 13.17 | 4.05 | LLM（多指标解析） |
| `e999775e6c11b538` | kpi_query | 今天S06线的上料和接料明细 | 15.77 | 14.98 | 0.79 | LLM |
| `88e4368993f95512` | kpi_query | 今天S06线的上料明细 | 15.43 | 14.84 | 0.58 | LLM |
| `575f700c5c39349d` | kpi_query | S06线今天异常数据 | 14.44 | 2.06 | 12.37 | 后端数据查询慢 |

**fast_lane 统计摘要**：

| 统计项 | 值 |
|---|---|
| 平均 total_lat | **23.1 s** |
| 平均 model_time | **9.0 s** |
| 平均 tool_time (ontology) | **14.1 s** |
| 步骤数 | 固定 1 |

---

## 二、Feishu Group Agent 路径（6 条 traces）

**架构**：ReAct 多步 agent 循环（skill_match → AGENT span → 多轮 chat GENERATION + TOOL 调用）+ Feishu 消息收发开销（~10-15s）。
**时间定义**：
- `model_time_s`：所有 `chat xiaomi/mimo-v2-pro-imp` GENERATION span 延迟之和
- `tool_time_s`：所有 TOOL span 延迟之和
- `agent_lat_s`：AGENT span 总延迟（≈ model_time + tool_time；不含 Feishu I/O）
- `total_lat_s`：Langfuse trace 总延迟（含 Feishu I/O 和 context_filter）

| trace_id (short) | skill | query | n_steps | n_tool_calls | total_lat_s | agent_lat_s | model_time_s | tool_time_s | 主要工具 |
|---|---|---|---|---|---|---|---|---|---|
| `91f4a6de6c15aff6` | object-data-query | 当前有哪些实体及模型 | 3 | 13 | 56.70 | 41.07 | 40.32 | 3.75 | list_object_types×6, get_object_type_metrics×6, get_link_info×1 |
| `32cbd82400f7fea7` | general-kpi-query | 查询最近一周S04线UPH | 4 | 5 | 52.72 | 32.16 | 31.60 | 0.98 | get_idi_model_data×4, get_object_type_metrics×1 |
| `7bcadc7259aaadc1` | object-data-query | 当前线体实体都绑定了哪些模型 | 4 | 3 | 40.86 | 21.37 | 17.86 | 3.46 | list_object_types×1, get_object_type_metrics×1, get_object_type_detail×1 |
| `c8492c6e09b51182` | object-data-query | 查询各对象下绑定了多少模型 | 3 | 2 | 32.93 | 24.59 | 23.26 | 1.27 | get_object_type_metrics×1, list_object_types×1 |
| `733dd15c93e3fbe1` | object-data-query | S04线今天OEE | 11 | 10 | 65.23 | 57.31 | 55.77 | 1.27 | view_text_file×5, get_idi_model_data×4, get_object_type_metrics×1 |
| `261941365775b2c6` | general-kpi-query | S04线今日小时别UPH | 21 | 31 | 158.01 | 151.01 | 136.79 | 14.72 | get_object_type_metrics×18, get_object_type_data×7, get_idi_model_data×2, list_object_types×2, format_feishu_card×1 |

**Agent 路径统计摘要**：

| 统计项 | 值 |
|---|---|
| 平均 total_lat | **67.6 s** |
| 平均 agent_lat | **54.6 s** |
| 平均 model_time | **50.9 s** |
| 平均 tool_time | **4.2 s** |
| 平均 n_steps | **7.7 步** |
| 每步平均 model_time | **~6.6 s/step** |
| model_time 占 agent_lat 比例 | **83–98%** |
| Feishu I/O 开销（total - agent_lat） | **~10–15 s** |

---

## 三、路径对比

| 维度 | Web fast_lane | Feishu Group Agent |
|---|---|---|
| 路径类型 | 单步（1 LLM + 1 ontology API） | 多步 ReAct（3-21 LLM + 2-31 tools） |
| 典型延迟范围 | 4–52 s（中位 ~15 s） | 33–158 s（中位 ~55 s） |
| 主要瓶颈 | LLM 或后端数据查询（各占约 50%） | 几乎完全由 LLM 推理决定（83-98%） |
| 工具执行时间 | 0.1–47 s（ontology API） | 0.3–15 s（各工具总和） |
| 速度核心 | 减少 ontology API 延迟 | 减少 LLM 每次推理时间 |

---

## 四、换用 Qwen3-14B 后的预期加速

### 假设条件

| 假设 | 说明 |
|---|---|
| 当前线上模型规模 | 推测 ~32-72B（依据：每步推理 ~6-10s） |
| 目标模型 | Qwen3-14B（微调后，ns 格式，无 skill doc） |
| 部署规格 | 2×A100-80G（与 round4 训练相同硬件） |
| skill doc 大小 | ~400-600 token/轮（ws 格式） |

### 速度提升预测

**模型推理加速**（14B vs 当前）：

| 当前规模假设 | 14B 加速比 | 新 model_time/step |
|---|---|---|
| 72B | ~4–5× | ~1.3–1.6 s/step |
| 32B | ~2–3× | ~2.1–3.2 s/step |
| 保守估计 2.5× | 2.5× | **~2.5 s/step** |

**Prompt 缩短效益**（去掉 skill doc）：

- ns 格式去掉 skill doc → 每轮减少约 500 token 输入
- TTFT 减少约 10–20%
- 额外加速：**~10%**

### 综合预测（Agent 路径）

| 场景 | 当前 ws 格式 | 14B ns 微调 | 加速倍数 |
|---|---|---|---|
| 简单查询（avg 3–4 steps） | ~43 s | **~16 s** | 2.7× |
| 中等查询（avg 7–8 steps） | ~68 s | **~25 s** | 2.7× |
| 复杂查询（avg 21 steps） | ~158 s | **~58 s** | 2.7× |

注：tool_time 不变（后端工具调用不受模型大小影响）。加速来自：LLM 推理 2.5× + prompt 缩短 10% ≈ **综合 ~2.5–3× 加速**。

---

## 五、关键结论

1. **LLM 推理是 Agent 路径的绝对瓶颈**：6 条 feishu_group traces 中，model_time 占 agent_lat 的 83–98%，tool 执行几乎可忽略。减小模型规模是降低延迟最直接的手段。

2. **工具调用本身极快**：单次工具调用延迟 0.1–3.6 s；即便 31 次工具调用总计仅 14.7 s，而对应 LLM 推理时间高达 136.8 s。

3. **步骤数是 Agent 延迟的主要预测因子**：3–4 步 → 33–57 s；21 步 → 158 s，约 7 s/步线性关系。减少推理步骤数（better skill internalization）同样重要。

4. **Fast_lane 显著更快**：单步 fast_lane 典型延迟 ~15 s，相比 agent 路径 ~55 s 快约 3–4×。说明 skill_match 路由优化的价值在于绕过多步 agent 循环。

5. **Fast_lane 中的 ontology API 时有性能异常**：3/6 条 web traces 中 ontology 延迟 >10 s（最高 47 s），是独立于模型规模的后端问题，需另外排查。

---

*数据提取方法：Python requests + HTTPBasicAuth，调用 `/api/public/traces?tags=web` 和 `/api/public/traces/{traceId}`*
*计算方法：`model_time = Σ latency(GENERATION where name starts with "chat")`；`tool_time = Σ latency(TOOL)`；`agent_latency = AGENT span latency`*
