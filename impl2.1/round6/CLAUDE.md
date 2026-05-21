# Round 6 执行手册（ReAct 模式）

---

## 一、任务目标

**核心问题（Goal）**：
> 修复 R5 中 equipment-cpk、line-attendance、line-exemption、分板机 4 个 skill 的 get_idi_model_data 调用率为 0%，使全体 skill 总调用率从 43% 提升到 ≥ 70%。

**实验目标**：在不依赖 skill doc 的情况下，微调后的 Qwen3-14B 模型能正确执行多步工具调用，效果等同甚至超过线上 ws 模型（有 skill doc）。Round 3 已验证此目标可达（SPC 任务 ns 模型超越 Claude + skill doc）。

**成功标准**：

| 指标 | R5 现状 | R6 目标 |
|------|---------|--------|
| **全体 get_idi_model_data 调用率** | 43% | **100%** |
| general-kpi-query idi 调用率 | 84% | **100%** |
| equipment-cpk-query idi 调用率 | 0% ✗ | **100%** |
| line-attendance-query idi 调用率 | 0% ✗ | **100%** |
| line-exemption-query idi 调用率 | 0% ✗ | **100%** |
| 分板机过站明细查询 idi 调用率 | 0% ✗ | **100%** |
| line-operation-skill idi 调用率 | 50% | **100%** |
| trajectory_tool_name_f1 | 51.8% | ≥ 65% |
| judge_overall | 23.8% | ≥ 30% |
| speedup_vs_production | 8.9× ✓ | ≥ 2.7× |

**方法**：主动学习循环——训练→评估→逐样本分析失败→精准生成修复数据→重训，直到 100%。

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
