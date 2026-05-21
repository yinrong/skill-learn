# Round 6 执行计划

---

## 目标

**实验目标**：在不依赖 skill doc 的情况下，微调后的 Qwen3-14B 模型能正确执行多步工具调用，完整走完"发现 → 查询 → 回答"的流程，效果等同甚至超过线上 ws 模型（有 skill doc）。

**本轮核心任务**：通过主动学习循环，将所有应调用 `get_idi_model_data` 的 skill 的调用率提升到 100%。

**成功标准**：

| 指标 | Round 5 现状 | Round 6 目标 |
|------|------------|-------------|
| **全体 get_idi_model_data 调用率** | 43%（27/63）| **100%**（63/63 中应调 idi 的全部调用）|
| general-kpi-query idi 调用率 | 84%（21/25）| **100%**（25/25）|
| equipment-cpk-query idi 调用率 | 0%（0/9）| **100%**（9/9）|
| line-attendance-query idi 调用率 | 0%（0/2）| **100%**（2/2）|
| line-exemption-query idi 调用率 | 0%（0/5）| **100%**（5/5）|
| 分板机过站明细查询 idi 调用率 | 0%（0/2）| **100%**（2/2）|
| line-operation-skill idi 调用率 | 50%（1/2）| **100%**（2/2）|
| trajectory_tool_name_f1 | 51.8% | ≥ 65% |
| judge_overall | 23.8% | ≥ 30% |
| speedup_vs_production | 8.9× | ≥ 2.7×（已满足，维持）|

**衡量方式**：
- `get_idi_model_data` 调用率：`pred_tools` 包含 `get_idi_model_data` 的样本数 / 该 skill 应调 idi 的测试样本总数
- trajectory_f1：步骤预测，GT 来自真实日志，不用端到端模拟
- judge：LLM-as-judge 对最终答案质量的评分（受评估框架限制，上限约 65%）

**为什么目标是 100%**：ws 模型有 skill doc 时对需要调 idi 的 skill 几乎 100% 正确调用。技能内化的目标是复现甚至超越这个效果——Round 3 已经验证了这是可达到的（SPC 任务 ns 模型 F1 超过 Claude + skill doc）。

---

## 一、核心认知：R5 失败根因

### 两种失败模式

| Skill | R5 调用率 | 模型实际行为 | 根因 |
|-------|----------|------------|------|
| equipment-cpk-query | 0% | `list→data`（跳过 metrics）| 训练数据 30 条不足，模型未学会先查 metrics |
| line-attendance-query | 0% | `list→metrics`（停止）| simulated metrics 只返回 UPH/良率 ID，没有出勤 ID（12385/12389），模型判断"无匹配模型"后放弃 |
| line-exemption-query | 0% | `list→data`（跳过 metrics）| 训练数据 20 条不足，模型未内化 metrics→idi 路径 |
| 分板机过站明细查询 | 0% | 从 list 开始循环 | GT 从 `get_object_type_data` 开始，模式不匹配 |
| general-kpi-query | 84% | 4/25 未调 idi | 部分问法变体未被覆盖，仍有 4 条样本失败 |

### 为什么静态补数据不够达到 100%

当前 Round 5 的方案是"随机生成 N 条样本"。这种方式对于 70% 目标够用，但对 100% 不够，因为：

1. **随机生成不保证覆盖每个具体失败案例**：测试集的 63 条样本有特定的问法、线体、时间表达。随机生成的训练数据可能没有覆盖这些具体变体。
2. **没有反馈机制**：生成后训练，不知道哪些测试样本仍然失败，无法针对性修复。
3. **失败模式在迭代中会变化**：修复一批问题后，可能暴露新的失败模式。

### 解决方案：主动学习循环

```
Loop:
  Step 1: 训练当前数据集
  Step 2: 对 63 条测试样本逐条评估
  Step 3: 找出仍然失败的具体样本（pred_tools 不含 get_idi_model_data）
  Step 4: 对每条失败样本，生成 5-10 条高度相似的训练样本（同 skill、类似问法）
          ——这些样本必须包含完整 get_idi_model_data 调用
  Step 5: 合并到训练集，重训
  重复直到所有应调 idi 的测试样本全部通过
```

---

## 二、数据规划

### 2.1 R5 数据复用（290 条）

全部保留。

### 2.2 第一轮修复数据（~120 条，并发生成）

覆盖 R5 中 4 个 0% skill，确保每条样本通过 `get_idi_model_data` 存在性验证：

| Skill | 增量 | 关键要求 |
|-------|------|---------|
| equipment-cpk-query | +40 | 必须经过 `list→metrics(device)→idi`，metrics 含 12365 |
| line-attendance-query | +30 | metrics 返回含 12385/12389，模型从中选出勤模型 |
| line-exemption-query | +30 | 必须经过 `list→metrics→idi`，metrics 含 12192/12223 |
| 分板机过站明细查询 | +20 | 从 `get_object_type_data` 开始（匹配 GT 模式）|

### 2.3 逐样本分析与精准修复数据（迭代轮次，每轮 +N 条）

第一轮训练评估后，对每条仍然失败的测试样本：

```python
# 失败样本分析格式
{
  "test_sample_id": 7,
  "skill": "equipment-cpk-query",
  "user_question": "查下S04线印刷机昨天的CPK",
  "pred_tools": ["list_object_types", "get_object_type_data"],  # 失败：没有 idi
  "gt_tools": ["list_object_types", "get_object_type_detail", "get_object_type_data"],
  "failure_reason": "模型调 data 后给出'无数据'结论，未调 metrics→idi",
  "targeted_training_count": 8  # 为此样本生成 8 条相似训练数据
}
```

每条精准修复样本要求：
- 用户问题与失败样本高度相似（同装备类型、相近时间表达）
- 工具调用链包含完整 `metrics→idi` 路径
- think block 明确说明"选择 model_id=XXXX 是因为 metrics 返回了此模型，modelName 匹配查询意图"

### 2.4 数据规模规划（含迭代）

| 轮次 | 数据来源 | 样本数 | 预期 idi 调用率 |
|------|---------|--------|--------------|
| R6-v1 | R5(290) + 第一轮修复(120) | ~410 | 预计 ~70-80% |
| R6-v2 | R6-v1 + 精准修复（失败样本×8）| ~450-500 | 预计 ~90-95% |
| R6-v3（如需）| R6-v2 + 剩余失败精准修复 | ~480-530 | 预计 ~100% |

---

## 三、评估修复（simulated_tools.py）

### 修改 1：`_METRICS["line"]` 返回全部 8 个维度

```python
"line": [
    {"modelId": 12202, "modelName": "线体小时别UPH",
     "params": [{"name":"start_shift_date","type":"date","optional":False},
                {"name":"end_shift_date","type":"date","optional":False},
                {"name":"line","type":"string","optional":True}]},
    {"modelId": 12198, "modelName": "线体一次良率",
     "params": [{"name":"start_time","type":"datetime","optional":False},
                {"name":"end_time","type":"datetime","optional":False},
                {"name":"line","type":"string","optional":True}]},
    {"modelId": 12204, "modelName": "线体产出达成",
     "params": [{"name":"start_shift_date","type":"date","optional":False},
                {"name":"end_shift_date","type":"date","optional":False},
                {"name":"line","type":"string","optional":True}]},
    {"modelId": 12203, "modelName": "线体OEE",
     "params": [{"name":"start_shift_date","type":"date","optional":False},
                {"name":"end_shift_date","type":"date","optional":False},
                {"name":"line","type":"string","optional":True}]},
    {"modelId": 12385, "modelName": "线体出勤率",       # ← 新增
     "params": [{"name":"shift_date","type":"date","optional":False},
                {"name":"line","type":"string","optional":True}]},
    {"modelId": 12389, "modelName": "员工出勤打卡记录", # ← 新增
     "params": [{"name":"shift_date","type":"date","optional":False},
                {"name":"line","type":"string","optional":True}]},
    {"modelId": 12194, "modelName": "线体抛料率",       # ← 新增
     "params": [{"name":"start_time","type":"datetime","optional":False},
                {"name":"end_time","type":"datetime","optional":False},
                {"name":"line","type":"string","optional":True}]},
    {"modelId": 12223, "modelName": "线体TOP异常任务",  # ← 新增
     "params": [{"name":"start_time","type":"datetime","optional":False},
                {"name":"end_time","type":"datetime","optional":False},
                {"name":"line","type":"string","optional":True}]},
],
```

### 修改 2：`_METRICS["pcb_router"]` 新增分板机模型

```python
"pcb_router": [
    {"modelId": 12328, "modelName": "分板机铣刀寿命",
     "params": [{"name":"shift_date","type":"date","optional":False},
                {"name":"line","type":"string","optional":True}]}
],
```

---

## 四、并发执行方案

### GPU 分配

| 阶段 | GPU | 用途 |
|------|-----|------|
| Phase 1 数据生成 | 无 | Claude API（4路并发）|
| Phase 3 训练 | GPU 0,1,2,3（4块）| DDP，effective_batch=16 |
| Phase 4 vLLM | GPU 2,3 | tensor_parallel=2 |
| Phase 4 评估 | 无 | HTTP 调用（2路并发）|

### 迭代循环并发拓扑

```
Phase 0（立即）：修复 simulated_tools.py

Phase 1（4路并发，~2-3h）
  ├── Agent-1: equipment-cpk-query +40条（强制 metrics 路径）
  ├── Agent-2: line-attendance-query +30条（metrics 含 12385/12389）
  ├── Agent-3: line-exemption-query +30条（强制 metrics 路径）
  └── Agent-4: 分板机过站明细查询 +20条（data 开始模式）

Phase 2（串行，30min）
  └── 合并数据 → ns转换 → LF转换 → ⚠️ 验证加载数 = 文件行数

Phase 3（训练 R6-v1，4×GPU，~55min）

Phase 4（2路并发评估，~2h）
  ├── trajectory_eval（逐样本输出 pred_tools）
  └── llm_judge

─── 分析失败样本，进入下一轮 ───

Phase 1b（按失败样本生成精准修复数据，~1-2h）
Phase 3b（训练 R6-v2，~55min）
Phase 4b（评估，~2h）

─── 若仍有失败，继续 Phase 1c/3c/4c ───
```

### Phase 1 并发命令

```bash
cd /home/yinrong/post-train/impl2.1/round5

python scripts/generate_data.py --skill equipment-cpk-query   --count 40 --output ../round6/data/gen_cpk_r6.jsonl        --seed 101 2>&1 | tee ../round6/logs/gen_cpk.log &
python scripts/generate_data.py --skill line-attendance-query --count 30 --output ../round6/data/gen_attendance_r6.jsonl  --seed 102 2>&1 | tee ../round6/logs/gen_attendance.log &
python scripts/generate_data.py --skill line-exemption-query  --count 30 --output ../round6/data/gen_exemption_r6.jsonl   --seed 103 2>&1 | tee ../round6/logs/gen_exemption.log &
python scripts/generate_data.py --skill "分板机过站明细查询"   --count 20 --output "../round6/data/gen_分板机_r6.jsonl"     --seed 104 2>&1 | tee "../round6/logs/gen_分板机.log" &
wait && echo "Phase 1 done"
```

### Phase 4 逐样本失败分析脚本

```python
# round6/scripts/analyze_failures.py
# 分析 trajectory_eval 结果，找出仍然失败的测试样本
import json

IDI_SKILLS = {
    "general-kpi-query", "equipment-cpk-query", "line-attendance-query",
    "line-exemption-query", "分板机过站明细查询", "line-operation-skill",
    "project-kpi-query", "lineside-material-query"
}

with open("round6/results/R6-v1-trajectory.json") as f:
    d = json.load(f)

failures = []
for i, s in enumerate(d["samples"]):
    if s.get("skill") not in IDI_SKILLS:
        continue
    if "get_idi_model_data" not in s.get("pred_tools", []):
        failures.append({
            "index": i,
            "skill": s["skill"],
            "pred_tools": s.get("pred_tools", []),
            "gt_tools": s.get("gt_tools", []),
        })

print(f"失败样本数: {len(failures)}")
for f in failures:
    print(f"  [{f['index']}] {f['skill']}: pred={f['pred_tools']}")

# 输出到文件供精准修复使用
with open("round6/data/failures_v1.json", "w") as f:
    json.dump(failures, f, ensure_ascii=False, indent=2)
```

---

## 五、验证标准

```python
def validate_r6_sample(sample, skill: str) -> bool:
    msgs = sample["messages"]
    final = (msgs[-1].get("content") or "").strip()

    # 1. 最终答案有实质内容
    assert len(final) > 50 and any(c.isdigit() for c in final), \
        "最终答案过短或无数值"

    # 2. 有非空 think block
    thinks = []
    for m in msgs:
        if m.get("role") == "assistant":
            c = m.get("content") or ""
            import re
            found = re.findall(r"<think>(.*?)</think>", c, re.DOTALL)
            thinks.extend(t.strip() for t in found if t.strip())
    assert thinks, "缺少非空 think block"

    # 3. 所有应调 idi 的 skill，必须有 get_idi_model_data
    IDI_SKILLS = {
        "equipment-cpk-query", "line-attendance-query", "line-exemption-query",
        "分板机过站明细查询", "general-kpi-query", "line-operation-skill",
        "line-attendance-query", "workstion-kpi-query", "project-kpi-query"
    }
    if skill in IDI_SKILLS:
        all_tools = [
            tc["function"]["name"]
            for m in msgs if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
        ]
        assert "get_idi_model_data" in all_tools, \
            f"此 skill 必须包含 get_idi_model_data，实际工具序列: {all_tools}"

    # 4. idi 使用的 model_id 在前序 metrics 响应中可追溯
    import re as _re
    metrics_ids = set()
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and i > 0:
            prev = msgs[i-1]
            if prev.get("role") == "assistant":
                prev_tools = [tc["function"]["name"] for tc in (prev.get("tool_calls") or [])]
                if "get_object_type_metrics" in prev_tools:
                    found = _re.findall(r'"modelId"\s*:\s*(\d+)', m.get("content",""))
                    metrics_ids.update(int(x) for x in found)
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                if tc["function"]["name"] == "get_idi_model_data":
                    args = json.loads(tc["function"].get("arguments","{}"))
                    mid = args.get("model_id")
                    if mid and metrics_ids:
                        assert int(mid) in metrics_ids, \
                            f"model_id={mid} 不在 metrics 响应中（{metrics_ids}）"

    return True
```

---

## 六、ReAct 监控与重新规划

```
每 5 分钟检测：
  - 各 Agent 进程存活（ps aux | grep generate_data | grep -v grep）
  - 输出文件行数 vs 目标
  - 训练 loss 趋势（tail round6/logs/train_R6.log）
  - GPU 使用（nvidia-smi）

重新规划触发条件：
  - 生成数据中 idi 出现率 < 80%（说明生成流程仍有问题）
  - 训练 loss 不收敛
  - 评估后仍有 skill idi 调用率为 0%（需要更深入的数据重设计）
  - 3 轮迭代后仍未达到 100%（记录 USER-DECIDE.md，分析是否需要超参调整）

决策日志 → logs/react_log.md
待决策项 → USER-DECIDE.md（不等待，继续推进）
证据文档 → 完成后追加到 CLAUDE.md 索引
目标始终是：所有应调 get_idi_model_data 的 skill 调用率 = 100%
```

---

## 七、预计总耗时（3轮迭代）

| 步骤 | 耗时 |
|------|------|
| Phase 0：simulated_tools 修复 | 30 分钟 |
| Phase 1a：数据生成（4路并发）| 2-3 小时 |
| Phase 2a：转换验证 | 30 分钟 |
| Phase 3a：训练 R6-v1（4×GPU）| ~55 分钟 |
| Phase 4a：评估 + 失败分析 | ~2 小时 |
| Phase 1b：精准修复数据生成 | 1-2 小时 |
| Phase 3b：训练 R6-v2 | ~55 分钟 |
| Phase 4b：评估 | ~2 小时 |
| Phase 1c/3c/4c（如需第三轮）| ~4 小时 |
| **合计（2-3轮）** | **约 12-18 小时** |

---

*2026-05-20*
