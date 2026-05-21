# Round 6 执行计划

> 目标：修复 4 个失败 skill，全体 get_idi_model_data 调用率 43% → ≥ 70%

---

## 一、核心认知：R5 失败根因

### 两种失败模式

| Skill | R5 调用率 | 模型实际行为 | 根因 |
|-------|----------|------------|------|
| equipment-cpk-query | 0% | `list→data`（跳过 metrics）| 训练数据 30 条不足，模型未学会先查 metrics；simulated tools 已有 device→12365 绑定但模型不走这条路 |
| line-attendance-query | 0% | `list→metrics`（停止）| simulated metrics 只返回 UPH/良率 ID，没有出勤 ID（12385/12389），模型找不到合适模型后放弃 |
| line-exemption-query | 0% | `list→data`（跳过 metrics）| 训练数据 20 条不足，模型未内化 metrics→idi 路径 |
| 分板机过站明细查询 | 0% | 从 list 开始循环 | GT 从 `get_object_type_data` 开始，模型训练数据用了不同的起始模式 |

### 两轨修复策略

**轨道 A：评估修复（simulated_tools）**
- `get_object_type_metrics(api_names="line")` 当前只返回 [12202, 12198, 12204]（UPH/良率/产出）
- 需要补全所有 Type A 维度：[12202, 12198, 12204, 12203, 12385, 12389, 12194, 12223]
- 这样 line-attendance 调 `metrics(line)` 就能找到 12385/12389

**轨道 B：训练数据修复**
- 为失败 skill 生成 2× 数量、强制经过 `metrics→idi` 路径的训练样本
- 分板机单独处理：生成从 `get_object_type_data` 开始的正确模式

---

## 二、数据规划

### 2.1 R5 保留复用（290 条）

R5 数据质量良好（general-kpi 84% idi），全部保留。

### 2.2 失败 Skill 增量数据（新增 ~120 条）

| Skill | R5 数量 | R6 增量 | 总计 | 关键要求 |
|-------|--------|--------|------|---------|
| equipment-cpk-query | 30 | **+40** | 70 | 必须经过 `list→metrics(device)→idi` 路径，metrics 返回 12365 |
| line-attendance-query | 20 | **+30** | 50 | metrics 返回 12385/12389，模型从中选择出勤模型 |
| line-exemption-query | 20 | **+30** | 50 | 必须经过 `list→metrics→idi` 路径，metrics 返回 12192/12223 |
| 分板机过站明细查询 | 15 | **+20** | 35 | 从 `get_object_type_data` 开始（匹配 GT 模式）|

### 2.3 训练数据样本规范（增量部分）

每条增量样本必须满足：
1. ✓ 调用链包含 `get_object_type_metrics → get_idi_model_data`
2. ✓ `get_idi_model_data` 使用的 model_id 出现在前序 metrics 响应里
3. ✓ 有非空 `<think>` block 解释为何选择此 model_id
4. ✓ 最终答案包含实际数据（非空结论）

### 2.4 数据量规划

| 来源 | 数量 |
|------|------|
| R5 数据（复用）| 290 |
| equipment-cpk 增量 | 40 |
| line-attendance 增量 | 30 |
| line-exemption 增量 | 30 |
| 分板机 增量 | 20 |
| **合计** | **410** |

### 2.5 失败 Skill 训练样本示例

**equipment-cpk-query（展示 list→metrics→idi 必须路径）**：

```
[user] S04线印刷机今天的CPK

[assistant]
<think>
Step 0: entity_type=装备, entity_code=待查（"印刷机"需查询编码）
Step 1: 无记忆，memory_hit=false → Step 2
Step 2: 调 list_object_types(keyword="装备") 获取 apiName
</think>
TOOL: list_object_types(keyword="装备")

[tool] {"data": [{"apiName": "device", "displayName": "装备"}]}

[assistant]
<think>
apiName=device，调 get_object_type_metrics 获取 CPK 模型 ID
</think>
TOOL: get_object_type_metrics(api_names="device")

[tool] {"data": [{"apiName": "device", "modelBindings": [
  {"modelId": 12365, "modelName": "装备CPK统计",
   "params": [{"name":"startTm","type":"datetime","optional":false},
              {"name":"endTm","type":"datetime","optional":false},
              {"name":"device_code","type":"string","optional":true}]}
]}]}

[assistant]
<think>
找到 CPK 模型：modelId=12365（装备CPK统计）
params: startTm(datetime), endTm(datetime), device_code(optional)
entity_code 未知，先查装备列表找印刷机编码
</think>
TOOL: get_object_type_data(api_name="device", filter_json=...)

[tool] {"data": {"data": [{"device_code":"M-SMT-SPI-A-JUZ-0401","device_name":"印刷机","line_code":"S04"}]}}

[assistant]
<think>
device_code=M-SMT-SPI-A-JUZ-0401
今天=2026-05-20，全天 startTm=2026-05-20 09:00:00，endTm=2026-05-21 09:00:00
</think>
TOOL: get_idi_model_data(model_id=12365, params_json='[...]')

[tool] {"data": {"data": [{"device_code":"M-SMT-SPI-A-JUZ-0401","cpk":1.54,"cp":1.61}]}}

[assistant] S04线印刷机（M-SMT-SPI-A-JUZ-0401）今天CPK=1.54，CP=1.61，过程能力良好（CPK≥1.33）。
```

**line-attendance-query（metrics 需包含 12385/12389）**：

```
[user] S04线今天的出勤情况

[assistant]
<think>
查询线体出勤数据，Step 2 先获取 apiName 和指标模型
</think>
TOOL: list_object_types(keyword="线体")

[tool] {"data": [{"apiName": "line", "displayName": "线体"}]}

[assistant]
<think>
apiName=line，调 metrics 查找出勤模型
</think>
TOOL: get_object_type_metrics(api_names="line")

[tool] {"data": [{"apiName": "line", "modelBindings": [
  {"modelId": 12202, "modelName": "线体小时别UPH", ...},
  {"modelId": 12198, "modelName": "线体一次良率", ...},
  {"modelId": 12385, "modelName": "线体出勤率",       ← 关键：必须包含
   "params": [{"name":"shift_date","type":"date","optional":false},
              {"name":"line","type":"string","optional":true}]},
  {"modelId": 12389, "modelName": "员工出勤打卡记录",
   "params": [{"name":"shift_date","type":"date","optional":false},
              {"name":"line","type":"string","optional":true}]}
]}]}

[assistant]
<think>
用户问出勤情况 → 选择 modelId=12385（线体出勤率）和 12389（员工打卡）
今天=2026-05-20，shift_date=2026-05-20
</think>
TOOL: get_idi_model_data(model_id=12385, params_json='[{"name":"shift_date","value":"2026-05-20"},{"name":"line","value":"S04"}]')
...
```

---

## 三、评估修复（simulated_tools.py）

### 修改 1：扩充 "line" metrics 返回全部维度

```python
# simulated_tools.py _METRICS["line"] 增加到 8 个 model_id
"line": [
    {"modelId": 12202, "modelName": "线体小时别UPH", "params": [...]},
    {"modelId": 12198, "modelName": "线体一次良率",   "params": [...]},
    {"modelId": 12204, "modelName": "线体产出达成",   "params": [...]},
    {"modelId": 12203, "modelName": "线体OEE",       "params": [...]},
    {"modelId": 12385, "modelName": "线体出勤率",     "params": [...]},  ← 新增
    {"modelId": 12389, "modelName": "员工出勤打卡",   "params": [...]},  ← 新增
    {"modelId": 12194, "modelName": "线体抛料率",     "params": [...]},  ← 新增
    {"modelId": 12223, "modelName": "线体TOP异常",    "params": [...]},  ← 新增
],
```

### 修改 2：分板机的 pcb_router metrics

```python
"pcb_router": [
    {"modelId": 12328, "modelName": "分板机铣刀寿命",
     "params": [{"name": "shift_date", "type": "date", "optional": False},
                {"name": "line", "type": "string", "optional": True}]}
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

### 并发拓扑

```
Phase 0（串行，2h）
  ├── 修复 simulated_tools.py（线立即可做）
  └── 准备 generate_data.py 脚本（更新失败 skill 的问题模板）

Phase 1（4路并发，约 2-3h）
  ├── Agent-1: equipment-cpk-query +40条（强制 metrics 路径）
  ├── Agent-2: line-attendance-query +30条（metrics 含 12385/12389）
  ├── Agent-3: line-exemption-query +30条（强制 metrics 路径）
  └── Agent-4: 分板机过站明细查询 +20条（data 开始模式）

Phase 2（串行，30min）
  └── 合并 R5+增量 → ns转换 → LF转换 → 验证加载数

Phase 3（训练，GPU 0,1,2,3，~45min）
  └── R6-sft-v1（410条，3 epochs）

Phase 4（2路并发，~2h）
  ├── trajectory_eval
  └── llm_judge
```

### Phase 1 并发命令

```bash
cd /home/yinrong/post-train/impl2.1/round5

python scripts/generate_data.py --skill equipment-cpk-query    --count 40 --output ../round6/data/gen_equipment-cpk-extra.jsonl   --seed 101 2>&1 | tee ../round6/logs/gen_cpk.log &
python scripts/generate_data.py --skill line-attendance-query  --count 30 --output ../round6/data/gen_line-attendance-extra.jsonl  --seed 102 2>&1 | tee ../round6/logs/gen_attendance.log &
python scripts/generate_data.py --skill line-exemption-query   --count 30 --output ../round6/data/gen_line-exemption-extra.jsonl   --seed 103 2>&1 | tee ../round6/logs/gen_exemption.log &
python scripts/generate_data.py --skill "分板机过站明细查询"    --count 20 --output "../round6/data/gen_分板机-extra.jsonl"           --seed 104 2>&1 | tee "../round6/logs/gen_分板机.log" &
wait && echo "Phase 1 done"
```

### Phase 2 合并与转换

```bash
# 合并 R5 全量 + R6 增量
cat /home/yinrong/post-train/impl2.1/round5/data/ws_combined.jsonl \
    /home/yinrong/post-train/impl2.1/round6/data/gen_*.jsonl \
    > /home/yinrong/post-train/impl2.1/round6/data/ws_combined_r6.jsonl

wc -l /home/yinrong/post-train/impl2.1/round6/data/ws_combined_r6.jsonl
# 期望：~410 条

cd /home/yinrong/post-train/impl2.1/round5
python scripts/convert_ns_lf.py \
  --input ../round6/data/ws_combined_r6.jsonl \
  --ns_output ../round6/data/train_ns_v6.jsonl \
  --lf_output ../round6/data/train_lf_v6.jsonl

# ⚠️ Round 4/5 铁律：验证加载数
cp ../round6/data/train_lf_v6.jsonl ../round4/data/train_lf_v6.jsonl
```

### Phase 3 训练配置

```yaml
# round6/configs/R6-sft-v1.yaml
model_name_or_path: /home/yinrong/models/Qwen3-14B
dataset: skill_r6_ns_v1
dataset_dir: /home/yinrong/post-train/impl2.1/round4/data
cutoff_len: 32768
output_dir: /home/yinrong/post-train/impl2.1/round6/checkpoints/R6-sft-v1
num_train_epochs: 3
# 其余同 R5-sft-v1.yaml
```

```bash
cd /home/yinrong/post-train/impl2.1/round4 && \
DISABLE_VERSION_CHECK=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
  ../round6/configs/R6-sft-v1.yaml \
  2>&1 | tee ../round6/logs/train_R6-sft-v1.log
```

---

## 五、ReAct 监控与重新规划

```
每 5 分钟检测：
  - 各 Agent 进程存活（ps aux | grep generate_data | grep -v grep）
  - 输出文件行数 vs 目标（wc -l round6/data/gen_*.jsonl）
  - 训练 loss 趋势（tail round6/logs/train_R6.log）
  - GPU 使用（nvidia-smi）

重新规划触发条件：
  - equipment-cpk 生成数据 idi 比例 < 50%（说明训练数据仍然不走 metrics 路径）
  - 训练 loss 连续 3 checkpoint 不降
  - GPU OOM（调小 batch 或 cutoff_len）
  - 评估后 equipment-cpk idi 仍为 0%（需要更深入的训练数据重设计）

决策日志 → logs/react_log.md
待决策项 → USER-DECIDE.md（不等待，继续推进）
证据文档 → 完成后追加到 CLAUDE.md 索引
目标始终是：全体 get_idi_model_data 调用率 ≥ 70%
```

---

## 六、验证标准

```python
def validate_r6_sample(sample, skill: str) -> bool:
    msgs = sample["messages"]
    final = msgs[-1].get("content") or ""
    pred_tools = [m.get("tool_calls",[]) for m in msgs if m.get("role")=="assistant"]

    # 1. 最终答案有实质内容
    assert len(final) > 50 and any(c.isdigit() for c in final)

    # 2. 有非空 think block
    all_thinks = [extract_think(m) for m in msgs if m.get("role")=="assistant"]
    assert any(t.strip() for t in all_thinks)

    # 3. 关键：对于应该调 idi 的 skill，必须有 get_idi_model_data
    IDI_SKILLS = {"equipment-cpk-query","line-attendance-query",
                  "line-exemption-query","分板机过站明细查询",
                  "general-kpi-query","line-operation-skill"}
    if skill in IDI_SKILLS:
        all_tool_names = [tc["function"]["name"]
                          for m in msgs if m.get("role")=="assistant"
                          for tc in (m.get("tool_calls") or [])]
        assert "get_idi_model_data" in all_tool_names, \
            f"增量样本必须包含 get_idi_model_data，当前工具序列: {all_tool_names}"

    return True
```

**⚠️ 增量数据的额外要求**：对 4 个失败 skill，验证函数必须检查 `get_idi_model_data` 存在，不通过则重试生成，不入库。

---

## 七、预计总耗时

| Phase | 耗时 |
|-------|------|
| Phase 0 simulated_tools 修复 | 30 分钟 |
| Phase 1 数据生成（4路并发）| 2-3 小时 |
| Phase 2 转换验证 | 30 分钟 |
| Phase 3 训练（4块GPU，410条）| ~55 分钟 |
| Phase 4 评估（2路并发）| 2-3 小时 |
| **合计** | **约 6-8 小时** |

---

## 八、新模型上线后的服务改造

同 R5，无额外变化。核心服务改造仍是下掉 skill 路由逻辑。

---

*2026-05-20*
