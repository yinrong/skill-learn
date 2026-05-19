# 技能内化方法论 v3：单轮对话 + 工作流分解

**适用范围：将结构化专业技能编入语言模型权重**
**版本：v3.0（2026-05-14）**
**基于：SKILL_INTERNALIZATION_METHODOLOGY_v2 + Round 1–4 实验复盘**

---

## 一、背景

### 1.1 什么是技能内化

| 术语 | 定义 |
|------|------|
| **技能（Skill）** | 一套结构化的程序性知识，使智能体能执行特定领域任务 |
| **内化（Internalization）** | 推理时不依赖外部技能文档，直接从权重调用技能 |
| **ws 格式** | 训练用的带 skill doc 格式（with skill） |
| **ns 格式** | 推理用的无 skill doc 格式（no skill） |

### 1.2 为什么要内化

| 方案 | 推理时 skill doc | 优势 | 局限 |
|------|----------------|------|------|
| 直接注入（ws） | 每次携带完整文档 | 无需训练 | Token 成本高、延迟大 |
| **技能内化（ns SFT）** | **不需要** | 低延迟、低成本 | 技能变更需重训 |
| GRPO 强化 | 不需要 | 可突破教师数据上限 | 需设计 reward，工程复杂 |

### 1.3 ns 格式训练的核心机制

SFT 只对 **output** 计算 loss，不对 input 计算 loss：

```
Input（不计 loss）：  [system="", user="查 UPH"]
Output（计 loss）：   [<think>调 list→metrics→idi，model_id 来自工具返回</think>
                      TOOL_CALL list_object_types(...)
                      TOOL_RESULT ...
                      TOOL_CALL get_idi_model_data(model_id=X)
                      答：UPH=185]
```

**关键推论**：
- skill doc 删除前，必须确保所有 skill 知识已经**展开到 output** 里（reasoning chain 或工具调用链中）
- 如果 skill 知识只存在于 input（skill doc），删除后 output 里看不到这些知识，模型无法学习

### 1.4 两类技能的本质差异

| 类型 | 特征 | 例子 | 核心挑战 |
|------|------|------|---------|
| **单轮对话** | 输入→确定性输出，无工具 | SPC 规则、合规检查 | 规则覆盖完整性 |
| **工作流分解** | 多步工具调用，结果依赖上一步 | 工厂数据查询 | 每步来源可追溯，无硬编码 |

两类技能训练流程有根本差异，**不可混用同一套方法**。

---

## 二、单轮对话学习

适用于：输入 → 推理 → 输出，全程不调用外部工具。

### 2.1 训练数据来源

**优先使用日志**，满足条件时直接用；不满足时才用教师数据补充。

**日志可用条件**：
- output 包含完整推理链（不只是最终答案）
- output 不依赖将被删除的 skill doc 内容（即：遮住 skill doc，output 仍然能从 input 推导出来）
- 可用规则引擎验证答案正确性

**日志不满足时，使用教师数据生成**：

```
[1] 定义 skill doc（规则 + 步骤 + 输出格式）
[2] 用代码生成多样化输入场景（覆盖所有规则组合）
[3] Claude + skill doc → 生成推理链 + 答案（ws 格式）
[4] 规则引擎验证答案正确性，过滤错误样本
[5] 删除 skill doc（ns 转换）→ 得到训练数据
```

教师数据生成的典型触发条件：无相关日志；日志 output 仅有结论无推理链；日志覆盖的输入场景不足（边界条件缺失）。

**数量建议**：200–1000 条，从 200 开始，评估后决定是否扩充。

**训练集/测试集隔离**：必须使用不同来源或不同随机种子，禁止共享场景。

### 2.2 ns 转换验证

```python
# 转换后必须验证：skill 知识是否已进入 output
for sample in ns_data:
    # output 中必须包含规则推理，不能只有最终答案
    assert "<think>" in sample["output"] or 规则标识符 in sample["output"]
```

### 2.3 训练配置

```yaml
cutoff_len: max(输出 token 数) × 1.2   # 必须覆盖最长样本
num_train_epochs: 3–5
```

**静默截断检查**：
```bash
# 训练前必做
python -c "
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_path)
data = [json.loads(l) for l in open('train.jsonl')]
lengths = [len(tok(s['output'])['input_ids']) for s in data]
print(f'max={max(lengths)}, p95={sorted(lengths)[int(len(lengths)*0.95)]:.0f}')
"
```

### 2.4 评估

```
评估格式：system=""，无任何 skill 内容
评估指标：F1（规则检测）或 Exact Match
对比基准：Claude + skill doc 的 F1
```

---

## 三、工作流分解学习

适用于：多步工具调用，每步结果依赖上一步工具返回值。

**核心原则：每个参数必须有可追溯的来源（工具返回或用户输入），禁止硬编码。**

### 3.1 数据来源的根本约束

**生产日志不能直接作为训练数据**，原因：

1. **记忆捷径（memory_hit=true）**：生产系统往往有记忆注入机制，模型直接用上次缓存的参数值（如 `model_id=12202`）跳过发现步骤。ns 转换删掉记忆注入后，训练数据出现"来源不明的参数值"。

2. **无法验证自洽性**：生产日志的输出是在 skill doc + 记忆注入等完整上下文下产生的。这些上下文被删除后，同样的输出可能无法从仅有的用户输入推导出来。

**使用生产日志前，必须逐条验证**：

```python
def is_valid_ns_sample(sample):
    """
    验证每个工具调用参数是否有合法来源：
    - 来自用户输入（user message 中可提取）
    - 来自前序工具返回值（tool response 中可找到）
    - 不允许凭空出现
    """
    tool_responses_so_far = []
    user_content = extract_user_content(sample)
    
    for step in sample["tool_calls"]:
        for param_name, param_value in step["arguments"].items():
            source = find_param_source(
                param_value, 
                user_content, 
                tool_responses_so_far
            )
            if source is None:
                return False, f"参数 {param_name}={param_value} 来源不明"
        
        tool_responses_so_far.append(step["response"])
    
    return True, "OK"
```

通过验证的样本才能入训练集。不通过的样本有两种处理方式：
- **丢弃**：如果来源不明的参数是捷径行为（memory_hit）
- **重建**：用 Claude + skill doc 重新生成该样本的完整轨迹

### 3.2 教师数据生成（日志不足时的补充路径）

当生产日志中大量样本不满足自洽性要求（§3.1 验证失败），或某类 skill 的 model_id 多样性不足时，用教师数据生成补充：

```
[1] 从生产日志中提取用户问题（去掉工具调用和答案）
[2] Claude + 完整 skill doc → 生成完整工具调用链
    - 教师模型必须展示完整发现步骤（不走记忆捷径）
    - 每个参数来源必须在推理链中明确说明
[3] 验证教师生成的参数与工具返回值一致
[4] ns 转换：删除 skill doc，保留工具声明
```

教师数据的关键质量要求：

```
✓ 有效推理链（知识展开到 output）：
  <think>
    list_object_types 返回 apiName=device
    get_object_type_metrics 返回 modelId=12365（CPK 模型）
    因此调用 get_idi_model_data(model_id=12365)
  </think>

✗ 无效推理链（参数来源不透明）：
  <think>查询装备 CPK</think>
  get_idi_model_data(model_id=12365)   ← 12365 从哪来？
```

### 3.3 评估方法：步骤预测

**工作流评估的正确方式是步骤预测，不需要任何模拟工具。**

原理：生产日志已包含完整的真实上下文和 GT 输出。将轨迹拆成独立的（输入, 输出）对：

```
step 0 评估对：
  input:  [system, user_question]
  output: GT_step_0（如 list_object_types(keyword="装备")）

step 1 评估对：
  input:  [system, user_question, tool_call_0, tool_resp_0]  ← 全来自真实日志
  output: GT_step_1（如 get_object_type_metrics(api_names="device")）

step 2 评估对：
  input:  [system, user_question, ..., tool_resp_1]
  output: GT_step_2（如 get_idi_model_data(model_id=12365, ...)）
```

评估指标：
- `tool_name_acc`：工具名称正确率（主指标）
- `arg_kv_f1`：参数 key-value 匹配 F1
- `combined_f1 = 0.5 × tool_name_acc + 0.5 × arg_kv_f1`

**不需要、也不应该写任何模拟工具**。端到端模拟（给模型完整跑一遍）无法与步骤预测提供额外信息，却引入工具模拟的准确性问题。

### 3.4 训练格式

训练数据格式（单步预测）：

```jsonl
{
  "messages": [
    {"role": "system", "content": "你是制造业数据分析助手..."},
    {"role": "user", "content": "查询S04线今天的UPH"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "list_object_types", "arguments": "{\"keyword\": \"线体\"}"}}]},
    {"role": "tool", "content": "{\"data\": [{\"apiName\": \"line_operation\", ...}]}"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_object_type_metrics", ...}}]},
    {"role": "tool", "content": "{\"data\": [{\"modelId\": 12202, \"modelName\": \"UPH\", ...}]}"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_idi_model_data", "arguments": "{\"model_id\": 12202, ...}"}}]},
    {"role": "tool", "content": "{\"data\": [{\"uph\": 185, ...}]}"},
    {"role": "assistant", "content": "S04线今天UPH为185..."}
  ],
  "tools": [...]
}
```

### 3.5 多样性要求（防止记忆化）

同类型查询（如 UPH 查询）必须覆盖**不同 model_id**，防止模型记住固定映射：

```python
# 验证训练集多样性
from collections import Counter
model_ids_per_skill = defaultdict(set)
for sample in train_data:
    skill = extract_skill(sample)
    model_ids = extract_idi_model_ids(sample)
    model_ids_per_skill[skill].update(model_ids)

for skill, ids in model_ids_per_skill.items():
    if len(ids) < 3:
        print(f"警告：{skill} 只有 {len(ids)} 个不同 model_id，可能导致记忆化")
```

如果某个 skill 对应的 model_id 在生产日志里总是同一个，**必须通过生成数据引入变化**（见 3.2 的教师数据生成，生成时随机化 model_id）。

### 3.6 训练框架对接

**必须验证实际加载样本数**：

```bash
# 训练开始后立即检查
grep "Num examples" train.log
# 必须等于 wc -l train_data.jsonl
```

不一致的常见原因及处理：

| 现象 | 原因 | 处理 |
|------|------|------|
| 加载数 < 文件行数 | 格式验证失败，样本被静默过滤 | 检查 WARNING 日志，修复格式 |
| `Invalid role tag` 警告 | 多轮对话中 `observation→human` 序列 | 在每个中间 human 前插入占位 gpt 消息 |
| 加载 0 条 | 数据集名称未注册或文件路径错误 | 检查 dataset_info.json |

---

## 四、附录

### A. 各阶段核心检查清单

#### 数据准备

- [ ] 训练数据来源已确认（教师生成 or 验证过的生产日志）
- [ ] 每个工具调用参数的来源可追溯（来自用户输入或前序工具返回）
- [ ] 同一 skill 的训练样本覆盖 ≥3 个不同 model_id（工作流类型）
- [ ] 训练集与测试集使用不同来源/种子，无泄露
- [ ] ns 转换后，output 中包含完整推理链，不依赖已删除的 skill doc

#### 训练

- [ ] `cutoff_len` ≥ 最长样本 token 数
- [ ] 训练框架实际加载样本数 = 文件行数（无静默过滤）
- [ ] output_dir 是新目录，不会意外 resume 已有 checkpoint
- [ ] 训练开始后 5 分钟内有正常 loss 输出（非 0 也非 NaN）

#### 评估（工作流类型）

- [ ] 使用步骤预测（not 端到端模拟）
- [ ] 评估 context 来自真实日志（not 合成）
- [ ] 首步 tool_name_acc 作为主指标（不受后续步骤影响）
- [ ] 检查 key tool（如 `get_idi_model_data`）的被调用率

---

### B. 静默失败识别

| 失败类型 | 表象 | 发现方式 |
|---------|------|---------|
| cutoff 截断 | 特定规则 recall=0，其他正常 | 检查 output token 长度分布 |
| 样本静默过滤 | 训练收敛异常快，loss 极低 | 对比文件行数 vs 加载样本数 |
| 记忆化而非学习 | 训练集 F1 高，不同 model_id 测试集 F1 低 | Schema-change 测试 |
| memory_hit 污染 | 参数来源无法追溯 | 对每条样本运行参数来源验证脚本 |
| Resume 旧 checkpoint | 模型行为异常（已有技能退化） | 检查 output_dir 是否存在历史 checkpoint |

---

### C. 技能类型判断树

```
用户问题到答案的过程中需要外部数据吗？
├── 否 → 单轮对话学习（见第二节）
└── 是
    ├── 每次调用工具的参数可以从用户输入直接确定吗？
    │   ├── 是 → 简单工具调用（参照单轮对话，增加工具声明）
    │   └── 否（需要上一步工具返回值确定下一步参数）
    │       → 工作流分解学习（见第三节）
    │         ⚠️ 必须满足参数可追溯性要求
```

---

### D. Round 1–4 典型错误与对应条款

| Round | 错误 | 本文档对应条款 |
|-------|------|--------------|
| R1–2 | cutoff_len 过小，output 末尾内容被截断 | §2.3 静默截断检查 |
| R3-C | GRPO reward 存在捷径，模型收敛到空预测 | v2 附录 B.5（未变更）|
| R4 | 用生产日志替代教师数据，memory_hit 污染训练集 | §3.1 数据来源约束 |
| R4 | 348/349 条样本被 LlamaFactory 静默过滤 | §3.6 训练框架对接 |
| R4 | 端到端模拟工具评估，结果不可靠 | §3.3 评估方法 |
| R4 | 合成数据 model_id 单一，模型记忆化 | §3.5 多样性要求 |

---

*v3.0，基于 Round 1–4 实验总结*
*2026-05-14*
