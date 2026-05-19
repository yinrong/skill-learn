# Round 5 执行手册（ReAct 模式）

---

## 一、任务目标

**核心问题（Goal）**：
> 通过 Claude 生成的高质量教师数据微调 Qwen3-14B，模型能否在不依赖 skill doc 的情况下，正确执行多步工具调用（包括 get_idi_model_data），达到与线上 ws 模型相当的答案质量？

**Round 4 的教训与 Round 5 的关键改进**：
- Round 4 失败根因：ns 转换删掉了 skill doc，但训练数据的 output 里没有推理链，配置知识（model_id 映射）无法传递到模型权重
- Round 5 修复：用 Claude + skill doc 生成带完整思考链的教师数据，思考链展开 skill doc 的配置知识

**成功标准**：
| 指标 | 目标 | 说明 |
|------|------|------|
| get_idi_model_data 被调用率 | ≥ 70% | 主指标，Round 4 为 0% |
| 步骤预测 combined_f1 | ≥ 0.60 | 每步对比 GT |
| 类型 B 鲁棒性 | 通过 | 模型能读 metrics 响应取新 model_id |

---

## 二、约束条件

1. **只做 SFT**：不使用 GRPO
2. **GPU 限制 ≤ 5 块**：训练用 GPU 0,1,2,3（4块），vLLM 用 GPU 4,5（2块）
3. **GPU 进程标记**：启动前检测空闲 GPU，用 PWD 标记自己的进程，不误 kill 其他项目进程
4. **并发最大化**：数据生成 10 路并发；评估 2 路并发；单任务内尽量利用并行
5. **多 Agent 并发**：所有独立任务必须用多 Agent 并发执行，不串行等待
6. **证据文档**：每个关键决策和里程碑，必须写证据文档并在本文件中添加链接
7. **ReAct 重新规划**：
   - 每 5 分钟检测所有 Agent 运行情况
   - Agent 完成或异常时，触发检测 + 是否重新规划
   - 重新规划始终瞄准任务目标
   - 对任务目标的歧义，记入 USER-DECIDE.md，不等待用户，继续推进
   - 若 USER-DECIDE.md 中的问题已有答案，标记"已不需要决策"

---

## 三、证据文档索引

| 证据 | 文档链接 | 说明 |
|------|----------|------|
| 数据分析报告 | [data_analysis.md](data_analysis.md) | 93条feishu日志的可用性和可学习性分析 |
| 执行计划 | [plan.md](plan.md) | 完整执行方案（含并发拓扑、GPU分配）|

*执行过程中新增证据文档后，在此追加。*

---

## 四、规划与执行记录（ReAct 日志）

详细决策日志见：[logs/react_log.md](logs/react_log.md)

---

## 五、待用户决策项

详见：[USER-DECIDE.md](USER-DECIDE.md)
