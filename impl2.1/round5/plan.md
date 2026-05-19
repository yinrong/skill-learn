# Round 5 执行计划

> 策略：用 Claude 生成高质量训练数据，解决 Round 4 的根本问题
> 不等待日志服务修复

---

## 一、核心认知

### skill doc 承载两类知识，ns 转换会同时删除

| 知识类型 | 例子 | ns 转换后的状态 |
|---------|------|---------------|
| 流程知识 | "Step 2: call list_object_types" | 已从 input 消失，需从训练 output 内化 |
| 配置知识 | "UPH = model_id 12202" | 已从 input 消失，需从训练 output 内化 |

Round 4 失败的根因：训练数据的 assistant output 里没有思考链，配置知识和流程知识都无法在 ns 转换后传递到模型权重。

### 不同 skill 的 model_id 来源不同

对 plan 设计有根本影响，必须分别处理：

| 类型 | Skill | model_id 来源 | 生成数据时的处理 |
|------|-------|-------------|----------------|
| **A：有映射表** | line-operation-skill, line-attendance-query, line-exemption-query, lineside-material-query, workstion-kpi-query, 分板机过站明细查询 | skill doc 第三节映射表（固定值）| 使用真实 model_id；think block 引用映射表 |
| **B：无映射表** | general-kpi-query, equipment-cpk-query | 通过 get_object_type_metrics 动态发现 | 使用随机 model_id 注入 metrics 响应；think block 说"从工具响应读取" |

类型 B 使用随机 model_id 的原因：模型必须学会"读 metrics 响应取 model_id"，而非记住特定值。在生产部署中，真实的 get_object_type_metrics 会返回真实 model_id，模型只需正确读取即可。

---

## 二、数据生成架构

```
对每个 Skill：
  ┌──────────────────────────────────────────────────────┐
  │  Step 1: 准备                                         │
  │    - 从 ws_raw.jsonl 提取该 skill 的完整 skill doc    │
  │    - 判断类型（A：有映射表 / B：无映射表）             │
  │    - 若类型 B：预分配本样本唯一的随机 model_id         │
  │      (10000–99999，不与其他样本重复)                   │
  │    - 随机生成问题场景（线体/日期/表达方式）            │
  ├──────────────────────────────────────────────────────┤
  │  Step 2: 构造初始消息                                 │
  │    messages = [                                        │
  │      {"role": "system", "content": "你是制造业..."},  │
  │      {"role": "user", "content": skill_doc + 问题}   │
  │    ]                                                   │
  ├──────────────────────────────────────────────────────┤
  │  Step 3: 多轮 Claude API 调用                         │
  │    Loop:                                               │
  │      → 调用 Claude，得到 <think>推理</think>+工具调用  │
  │      → 脚本生成工具响应（按下方规则）                  │
  │      → 追加到 messages                                 │
  │      → 直到 Claude 输出纯文字最终答案                  │
  ├──────────────────────────────────────────────────────┤
  │  Step 4: 验证（按类型分别验证）                        │
  │    类型 A: think block 引用了映射表中的 model_id       │
  │    类型 B: idi 使用的 model_id 在 metrics 响应中出现   │
  │    共同:   最终答案含实际数据；think block 非空        │
  └──────────────────────────────────────────────────────┘
```

---

## 三、工具响应生成规则

### list_object_types(keyword)

| keyword 含关键词 | 返回 apiName |
|----------------|-------------|
| 线体/line | `line` |
| 装备/device/设备 | `device` |
| 物料/material | `mat_fail_detail` |
| 工站/station | `work_station` |
| 项目/project | `project` |
| 分板机/铣刀 | `pcb_router` |
| 线边仓/material | `lineside_material` |

```json
{"code": 0, "data": [{"apiName": "line", "displayName": "线体", "desc": ""}]}
```

### get_object_type_metrics(api_names)

**类型 A（有映射表）**：返回 skill doc 映射表中该 api_name 对应的真实 model_id 和 params：

```json
// line-operation-skill，api_names="line"
{"code": 0, "data": [{"apiName": "line", "modelBindings": [
  {"modelId": 12202, "modelName": "线体小时别UPH",     "params": [{"name":"start_shift_date","type":"date",...}]},
  {"modelId": 12198, "modelName": "线体一次良率",       "params": [{"name":"start_time","type":"datetime",...}]},
  {"modelId": 12204, "modelName": "线体产出达成",       "params": [{"name":"start_shift_date","type":"date",...}]},
  {"modelId": 12203, "modelName": "线体OEE",           "params": [{"name":"start_shift_date","type":"date",...}]},
  {"modelId": 12385, "modelName": "线体出勤率",         "params": [{"name":"shift_date","type":"date",...}]},
  {"modelId": 12389, "modelName": "员工出勤打卡记录",   "params": [{"name":"shift_date","type":"date",...}]},
  {"modelId": 12194, "modelName": "线体抛料率",         "params": [{"name":"start_time","type":"datetime",...}]},
  {"modelId": 12223, "modelName": "线体TOP异常任务",    "params": [{"name":"start_time","type":"datetime",...}]}
]}]}
```

**类型 B（无映射表）**：返回本样本预分配的随机 model_id，params 根据 skill 类型模拟：

```json
// general-kpi-query，api_names="line"，随机 model_id=73421
{"code": 0, "data": [{"apiName": "line", "modelBindings": [
  {"modelId": 73421, "modelName": "线体UPH指标",
   "params": [{"name":"start_shift_date","type":"date","optional":false},
              {"name":"end_shift_date","type":"date","optional":false},
              {"name":"line","type":"string","optional":true},
              {"name":"shift","type":"string","optional":true}]}
]}]}
```

### get_idi_model_data(model_id, params_json)

返回与 skill 对应的真实结构数据，数值随机（保持合理范围）：

```json
// UPH 类
{"code": 0, "data": {"data": [
  {"line_code": "S04", "shift_date": "2026-05-14", "shift": "D",
   "uph": 438, "target_uph": 500}], "count": 1}}

// 良率类
{"code": 0, "data": {"data": [
  {"line_code": "S04", "process_section_code": "SMT",
   "first_pass_rate": 0.941, "target_first_pass_rate": 0.80}], "count": 1}}

// CPK 类
{"code": 0, "data": {"data": [
  {"device_code": "M-SMT-SPI-A-JUZ-0401", "cpk": 1.54, "cp": 1.61}
], "count": 1}}
```

### get_object_type_data(api_name, ...)

返回 2-3 条实体数据（线体列表、装备列表等）。

---

## 四、think block 模板（两种类型）

### 类型 A：skill doc 有映射表

```
<think>
  用户问 S04 线今天 UPH。
  skill doc 第三节映射表：UPH 对应 model_id=12202，时间参数类型为 date。
  Step 4 已通过 get_object_type_metrics 确认：
    - model_id=12202 对应 "线体小时别UPH"
    - params: start_shift_date(date,必填), end_shift_date(date,必填), line(string,可选)
  Step 5 execute_shell_command 获取系统时间：2026-05-14 10:23:45
  工厂今天=2026-05-14（09:00后，使用当天日期）
  构造：start_shift_date=2026-05-14, end_shift_date=2026-05-14, line=S04
</think>
```

### 类型 B：skill doc 无映射表，从工具响应发现

```
<think>
  用户问 S04 线今天 UPH。
  Step 2: list_object_types(keyword="线体") → apiName=line
  Step 4: get_object_type_metrics(api_names="line") → 
    发现 modelId=73421, modelName="线体UPH指标"
    params: start_shift_date(date,必填), line(string,可选)
  使用 modelId=73421（来自工具返回，非猜测）
  时间参数类型=date，今天=2026-05-14
</think>
```

---

## 五、各 Skill 生成目标与问题模板

### general-kpi-query（类型 B，目标：60条）

问法模板：
```python
entities  = ["S03线", "S04线", "S03/S04线"]
metrics   = ["UPH", "良率", "OEE", "产出", "抛料率"]
times     = ["今天", "昨天", "今天白班", "昨晚夜班", "2026-04-{d:02d}"]
templates = [
    "{e}今天的{m}",
    "查一下{t}{e}的{m}情况",
    "{e} {t}的{m}数据是多少",
    "帮我看下{e}的{m}，{t}的",
]
```
每条样本使用不同随机 model_id。

---

### line-operation-skill（类型 A，目标：40条）

使用 skill doc 真实 model_id（8个维度）。每条样本完整展示 8 维度查询，需要 8 次 idi 调用。

问法模板：
```
"查询{line}今天的综合运营情况"
"出一份{line}昨晚夜班的运营报告"
"{line}{date}各指标汇总"
```

---

### equipment-cpk-query（类型 B，目标：30条）

每条样本使用不同随机 model_id。装备编码未知时，需先 get_object_type_data 查装备列表。

问法模板：
```
"查{device_type}的CPK"     # device_type: 印刷机/回焊炉/AOI
"看下{date}{device_type}的SPC数据"
```

---

### object-data-query（无 idi 调用，目标：30条）

工具序列：`list_object_types → get_object_type_detail → get_object_type_data → 答案`

---

### lineside-material-query（类型 A，目标：20条）

### line-attendance-query（类型 A，目标：20条）

### line-exemption-query（类型 A，目标：20条）

### 分板机过站明细查询（类型 A，目标：15条）

### workstion-kpi-query（类型 A，目标：15条）

---

## 六、对现有 56 条干净日志的处理

**只需补全思考链，不需要修改工具响应。**

- 类型 A skill（有映射表）：Claude 重写 assistant 消息，在 `<think>` 里展开"从 skill doc 映射表读取 model_id=XXXX"
- 类型 B skill（无映射表）：Claude 重写 assistant 消息，在 `<think>` 里展开"从 get_object_type_metrics 返回值读取 model_id=XXXX"
- 最终答案无实际数据的样本：丢弃

修复后预计可用 ~40 条。

---

## 七、数据规模规划

| 来源 | 样本数 |
|------|--------|
| general-kpi-query（Claude 生成）| 60 |
| line-operation-skill（Claude 生成）| 40 |
| equipment-cpk-query（Claude 生成）| 30 |
| object-data-query（Claude 生成）| 30 |
| lineside-material-query（Claude 生成）| 20 |
| line-attendance-query（Claude 生成）| 20 |
| line-exemption-query（Claude 生成）| 20 |
| 分板机过站明细查询（Claude 生成）| 15 |
| workstion-kpi-query（Claude 生成）| 15 |
| 现有日志（补 think block 后）| ~40 |
| **合计** | **~290 条** |

---

## 八、验证标准

```python
def validate_sample(sample, skill_type: str) -> bool:
    msgs = sample["messages"]
    final = msgs[-1].get("content") or ""

    # 共同：最终答案有实际内容
    assert len(final) > 50 and any(c.isdigit() for c in final)

    # 共同：有非空 think block
    thinks = [extract_think(m) for m in msgs if m["role"] == "assistant"]
    assert any(t.strip() for t in thinks)

    idi_ids = extract_idi_model_ids(msgs)
    if not idi_ids:
        return True  # 无 idi 调用（object-data-query 类）

    if skill_type == "A":  # 有映射表
        # think block 引用了 skill doc 映射表中的 model_id
        for m in msgs:
            if m["role"] == "assistant":
                think = extract_think(m)
                assert any(str(mid) in think for mid in idi_ids), \
                    "think block 未引用 model_id"
                break
    else:  # 类型 B
        # idi 使用的 model_id 在 metrics 响应中出现
        metrics_ids = extract_metrics_model_ids(msgs)
        assert idi_ids.issubset(metrics_ids), \
            f"孤立 model_id: {idi_ids - metrics_ids}"

    return True
```

---

## 九、并发执行方案（最快完成）

### GPU 分配

| 阶段 | GPU | 用途 |
|------|-----|------|
| Phase 0（准备）| 无 | 纯 CPU 脚本 |
| Phase 1（数据生成）| 无 | Claude API 调用，无 GPU |
| Phase 3（训练）| GPU 0,1,2,3（4块）| DDP 训练，4 块比 2 块快约 50% |
| Phase 4（vLLM）| GPU 4,5（2块）| tensor_parallel=2，与训练 GPU 隔离 |
| Phase 4（评估）| 无 | CPU 推理脚本，调用 vLLM HTTP |

### 并发拓扑

```
Phase 0（串行，1天）
  └── 准备脚本、提取 skill doc、配置环境
          │
          ▼
Phase 1（9路并发 + 1路并发，约 4-6 小时）
  ┌──────────────────────────────────────────────┐
  │  Agent-1: general-kpi-query  (60条)          │
  │  Agent-2: line-operation-skill (40条)        │
  │  Agent-3: equipment-cpk-query (30条)         │
  │  Agent-4: object-data-query  (30条)          │
  │  Agent-5: lineside-material-query (20条)     │
  │  Agent-6: line-attendance-query (20条)       │
  │  Agent-7: line-exemption-query (20条)        │
  │  Agent-8: 分板机过站明细查询 (15条)           │
  │  Agent-9: workstion-kpi-query (15条)         │
  │                                               │
  │  Agent-10: 修复现有 56 条日志（补 think）     │ ← 与上述并行
  └──────────────────────────────────────────────┘
          │ 全部完成后
          ▼
Phase 2（串行，1小时）
  └── 合并 → ns 转换 → LF 转换 → 验证加载数
          │
          ▼
Phase 3（训练，4块 GPU，约 30 分钟）
  └── CUDA_VISIBLE_DEVICES=0,1,2,3
      effective_batch = 1 × 4(GPU) × 4(acc) = 16
      steps ≈ 290×3/16 = 54（vs 2块 GPU 的 109 步）
          │
          ▼
Phase 4（评估，2路并发）
  ├── vLLM: CUDA_VISIBLE_DEVICES=4,5，tensor_parallel=2
  │
  ├── Agent-A: trajectory_eval（步骤预测）     ┐ 同时启动
  └── Agent-B: llm_judge（答案质量）           ┘
```

### 各 Phase 执行细则

**Phase 1：数据生成（10路并发）**

每个 Agent 独立运行 `generate_data.py --skill <SKILL_NAME>`，互不依赖：

```bash
# 同时启动10个后台进程
python round5/scripts/generate_data.py --skill general-kpi-query   --count 60  &
python round5/scripts/generate_data.py --skill line-operation-skill --count 40  &
python round5/scripts/generate_data.py --skill equipment-cpk-query  --count 30  &
python round5/scripts/generate_data.py --skill object-data-query    --count 30  &
python round5/scripts/generate_data.py --skill lineside-material    --count 20  &
python round5/scripts/generate_data.py --skill line-attendance      --count 20  &
python round5/scripts/generate_data.py --skill line-exemption       --count 20  &
python round5/scripts/generate_data.py --skill 分板机过站明细查询   --count 15  &
python round5/scripts/generate_data.py --skill workstion-kpi        --count 15  &
python round5/scripts/fix_existing.py  --input data/ws_raw.jsonl               &
wait && echo "All generation done"
```

**Phase 2：合并 + 转换**

```bash
# 合并所有生成结果 + 修复后日志
cat round5/data/gen_*.jsonl round5/data/fixed_existing.jsonl > round5/data/ws_combined.jsonl

# ns 转换（删 skill doc，保留工具声明）
python round5/scripts/convert_to_ns.py \
  --input round5/data/ws_combined.jsonl \
  --output round5/data/train_ns_v5.jsonl

# LF 转换
python round5/scripts/to_lf.py \
  --input round5/data/train_ns_v5.jsonl \
  --output round5/data/train_lf_v5.jsonl

# ⚠️ Round 4 教训：验证加载数量
wc -l round5/data/train_lf_v5.jsonl
# 然后检查 LlamaFactory 的 "Num examples" 日志，必须 = 上面的行数
```

**Phase 3：训练（4块 GPU）**

```yaml
# round5/configs/R5-sft-v1.yaml
output_dir: checkpoints/R5-sft-v1   # 新目录，不 resume 旧 checkpoint
cutoff_len: 32768
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
# 启动：CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train configs/R5-sft-v1.yaml
# effective_batch=16，约 54 步，预计 30-40 分钟
```

**Phase 4：评估（2路并发）**

```bash
# vLLM（GPU 4,5）
CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
  --model checkpoints/R5-sft-v1-merged \
  --tensor-parallel-size 2 --port 8035 &

# 等 vLLM 就绪后并发启动评估
python round4/eval/trajectory_eval.py --output round5/results/R5-trajectory.json &
python round4/eval/llm_judge.py       --output round5/results/R5-judge.json      &
wait

# 类型 B 鲁棒性测试：替换 metrics 里的 model_id 为新随机值，验证模型仍能正确使用
python round5/scripts/robustness_test.py --skill general-kpi-query
```

### 预计总耗时

| Phase | 耗时 |
|-------|------|
| Phase 0 准备 | 4-6 小时 |
| Phase 1 数据生成（10并发）| 4-6 小时 |
| Phase 2 转换验证 | 30 分钟 |
| Phase 3 训练（4块GPU）| 30-40 分钟 |
| Phase 4 评估（2并发）| 2-3 小时 |
| **合计** | **约 12-16 小时** |

### ReAct 监控与重新规划

每 5 分钟执行一次检测，由 Cron Job 驱动：

```
检测内容：
  1. 各 Agent 进程是否存活（ps aux | grep generate_data）
  2. 各 skill 的输出文件行数 vs 目标数量
  3. 训练 loss 趋势（tail logs/train_R5.log）
  4. GPU 使用情况（nvidia-smi）

重新规划触发条件：
  - 某 skill 生成失败率 > 30% → 调整 prompt 或换模型
  - 训练 loss 不收敛（3 个 checkpoint loss 不降）→ 调超参
  - GPU OOM → 减小 cutoff_len 或 batch_size
  - 评估结果不达标 → 分析失败 skill，补充数据，启动迭代

目标始终是：get_idi_model_data 调用率 ≥ 70%
```

**决策日志**：每次重新规划记入 `logs/react_log.md`
**待决策项**：歧义追加到 `USER-DECIDE.md`，不等待用户，继续推进
**证据文档**：每个里程碑完成后，在 `CLAUDE.md` 证据索引中追加链接

---

## 十、对日志服务的改进建议（发给维护人）

**建议1：修复 cron/alarm/web tag 日志格式**

**证据**：ws_cron.jsonl 第1条 user message 末尾包含：
```
<coroutine object OpenAIChatModelCached.__call__ at 0x7fa363526560>
```
Python 异步调用未 `await`，协程对象被转成字符串存入日志。cron/alarm/web 共 319 条因此全部作废，仅剩 93 条 feishu 数据可用。修复后训练数据来源可扩大 4 倍以上。

**建议2：在 Langfuse 中记录完整 thinking block**

**证据**：ws_raw.jsonl 全部 93 条 feishu 日志中，含非空 `<think>` block 的样本：**0 条**（100% 无思考链）。Qwen3 在生产中 thinking 被关闭或输出被过滤。开启记录后日志可直接作为高质量训练数据，无需 Claude 二次生成。

---

## 十一、新模型上线后的服务改造

工具调用 **API 格式不变**，服务编排逻辑需简化：

| 环节 | 当前 ws 模型 | 新 ns 模型 |
|------|------------|-----------|
| Skill 路由 | 服务根据意图选择 skill doc 注入 | **下掉**，模型已内化所有 skill |
| 消息构造 | system + skill_doc + 用户问题 + 记忆注入 + SKILL_MENU | system + 用户问题（大幅简化）|
| 工具定义 | 17 个工具随请求传入 | 不变 |
| 工具执行 | 服务执行工具并返回结果 | 不变 |
| 响应处理 | 直接输出可见文本 | **需过滤 `<think>...</think>`**（建议服务侧过滤后再展示用户）|

---

*2026-05-15*
