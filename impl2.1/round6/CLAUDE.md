# Round 6 执行手册（ReAct 模式）

---

## 一、任务目标

**核心问题（Goal）**：
> 修复 R5 中 equipment-cpk、line-attendance、line-exemption、分板机 4 个 skill 的 get_idi_model_data 调用率为 0%，使全体 skill 总调用率从 43% 提升到 ≥ 70%。

**成功标准**：

| 指标 | 目标 | R5 现状 |
|------|------|---------|
| get_idi_model_data 全体调用率 | ≥ 70% | 43% |
| general-kpi-query idi 率 | 保持 ≥ 80% | 84% ✓ |
| equipment-cpk-query idi 率 | ≥ 60% | 0% ✗ |
| line-attendance-query idi 率 | ≥ 60% | 0% ✗ |
| line-exemption-query idi 率 | ≥ 50% | 0% ✗ |
| trajectory_tool_name_f1 | ≥ 55% | 51.8% |
| speedup | ≥ 2.7× | 8.9× ✓ |

---

## 二、约束条件

1. **只做 SFT**（不使用 GRPO）
2. **GPU 限制 ≤ 5 块**（当前机器 4 块 H20）
3. **GPU 进程标记**：启动前 `nvidia-smi` 检测空闲，用 PWD 标记进程
4. **并发最大化**：数据生成多路并发，评估 2 路并发
5. **证据文档**：每个关键决策和里程碑写证据并在本文件索引
6. **ReAct 重新规划**：每 5 分钟检测，异常/完成时触发重新规划，歧义记入 USER-DECIDE.md

---

## 三、证据文档索引

| 证据 | 链接 | 说明 |
|------|------|------|
| R5 评估结果 | [../round5/results/R5-trajectory.json](../round5/results/R5-trajectory.json) | trajectory_f1=51.8%, idi=43% |
| R5 失败模式分析 | [data_analysis.md](data_analysis.md) | 4个失败skill的根因 |
| R6 执行计划 | [plan.md](plan.md) | 完整执行方案 |

---

## 四、ReAct 日志

见 [logs/react_log.md](logs/react_log.md)

---

## 五、待用户决策项

见 [USER-DECIDE.md](USER-DECIDE.md)
