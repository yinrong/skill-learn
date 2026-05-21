# Round 5 ReAct 决策日志

> 记录执行过程中的关键决策、观察和重新规划节点

---

## 2026-05-15 规划阶段

### 决策 R5-D1：数据来源选择
- **观察**：ws_raw.jsonl 93 条 feishu 日志全部格式正确；其余 319 条（cron/alarm/web）因协程对象 bug 作废
- **决策**：只用 feishu 日志，用 Claude 生成补充数据（约 250 条）
- **依据**：[data_analysis.md](../data_analysis.md) 第一、二节

### 决策 R5-D2：model_id 分两类处理
- **观察**：line-operation-skill skill doc 含完整映射表（12202/12198/12204...）；general-kpi-query skill doc 无映射表
- **决策**：类型 A（有映射表）用真实 model_id + 引用映射表的 think；类型 B（无映射表）用随机 model_id 注入 metrics + think 引用工具返回
- **依据**：本次对话中对 skill doc 的逐条检查

### 决策 R5-D3：并发方案
- **决策**：10 路数据生成并发；训练 4 块 GPU；评估 2 路并发
- **预期收益**：总耗时约 12-16 小时

---

*执行阶段：每次 Agent 完成或异常时，在此追加记录*

### 决策 R5-D4：Phase 2 提前触发条件调整
- 观察：fix_existing 每条调用 Claude API 约 30-60s，93 条需 2+ 小时；gen_* 将在 ~30 分钟内完成
- 决策：gen_* 全部完成后立即触发 Phase 2，使用 fixed_existing 当时的实际产出（预计 20-25 条），不等待 ≥40 目标
- 依据：250+ 条合成数据覆盖所有 skill，不存在覆盖缺口；等待 2 小时不符合"最快完成"原则
- USER-DECIDE.md 更新：此决策不需要用户决策（技术判断）

### 决策 R5-D5：LF 格式修复（三轮迭代）
- 观察1：training error "Invalid JSON in function message" — thinking 文字被前缀拼接到 function_call JSON 里
- 修复1：thinking 内容改为单独 gpt 消息
- 观察2：Num examples=13，"Invalid role tag" — gpt→function_call 序列非法
- 修复2：工具调用前的思考内容不再作为 gpt 输出，LF 层面仅保留 function_call 链
- 观察3：并行多工具调用产生 fc,fc,obs,obs 顺序非法
- 修复3：按 tool_call_id 配对，改为 fc→obs, fc→obs 交错输出
- 最终结果：272/272 样本全部加载，51 步训练正常启动

### 决策 R5-D6：R5-sft-v1 评估完成
- 观察：trajectory_f1=51.8%, get_idi_model_data=43%, judge=23.8%
- 关键突破：idi调用率从v4的0%提升到43%（general-kpi-query达84%）
- 根因发现：simulated_tools._METRICS缺少"line"和"device"的model_id绑定（只有"line_operation"）
- 修复：补充"line"→[12202,12198,12204]和"device"→[12365]映射
- 剩余问题：
  1. equipment-cpk-query idi=0%（虽然"device"已修复，可能还有其他问题）
  2. line-attendance/exemption idi=0%（这些skill的api_name可能不匹配）
  3. judge仍然低（23.8%）—— 评估框架限制 + 部分idi查询的simulated data值不准确
- 下一步：分析idi=0%的skill，检查其metrics返回是否包含正确model_id
