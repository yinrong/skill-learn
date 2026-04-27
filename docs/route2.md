# Phase 2 详细开发计划：能力深化（第 3–9 个月）

> 基于《工业大模型后训练白皮书》第八章 Phase 2 展开
> 时间跨度：6 个月（第 3 月初 – 第 9 月末）+ 前置 Demo（1 周）+ POC（2–4 周）
> 目标：工业专项评测比通用模型提升 >30%，Agent 任务完成率 >85%
> 资源基线：32×A100（80GB），50TB 存储，8 人团队

---

## 零、前置阶段：Demo（第 0 周）+ POC（第 1–4 周）

在 Phase 2 正式启动前，先用最小投入打通完整链路、验证核心假设，再大规模投入资源。

---

### 0.1 Demo 阶段（1 周）

**目标**：用最简方案跑通"数据生成 → 微调 → 评测"完整链路，看到至少一个指标出现正向效果，建立团队信心。

**不求广度，只求跑通**：选定单一场景（SPC 异常判断），端到端走一遍。

---

#### 数据工程师（1人）— Demo 周任务

**Day 1（全天）：生成 500 条 SPC 训练样本**

目标：产出 `data/demo/spc_train_500.jsonl`，格式为 Alpaca JSON，可直接喂给 LLaMA-Factory。

步骤：

1. 克隆基础工具仓库，确认 Python 环境（Python 3.10+，numpy / pandas / matplotlib）
2. 编写 `scripts/gen_spc_demo.py`，逻辑如下：
   - 随机生成 25 个采样点的正态分布数据（μ 随机 9.5–10.5，σ 随机 0.1–0.5）
   - 按权重随机注入 Nelson 规则违规（Rule 1–8 各占约 10%，正常约 20%）
   - 随机生成规格限 USL/LSL（确保 USL > μ+2σ，LSL < μ-2σ）
   - 用规则引擎计算标准答案：违规规则编号列表 + CPK 值（保留 3 位小数）+ 建议动作
   - 将数据序列化为 CSV 字符串嵌入 `input` 字段，答案写入 `output` 字段
3. 生成 600 条（其中 500 训练、100 测试，按 `split` 字段区分），写出到同一 jsonl
4. 运行 `python scripts/validate_jsonl.py spc_train_500.jsonl` 做格式校验（字段完整、无空 output）
5. 下午 15:00 前将文件路径发给领域专家，并附上 5 条样本示例供对方快速判断

关键细节：
- `output` 字段必须包含"推理过程"（先判断每条 Nelson 规则，再给 CPK，最后给建议），不能只给结论——这决定了训练出的模型是否有思维链
- Nelson Rule 1（超出控制限）注入概率提高到 25%，因为这是最常见场景，要保证测试集覆盖
- 生成完毕后立即用 `python -c "import json; [json.loads(l) for l in open('xxx.jsonl')]"` 做 JSON 合法性检查

**Day 2 上午：根据专家反馈修正，产出 v1.1**

- 专家会在 Day 2 09:00 前返回修正清单（见领域专家 Day 2 任务）
- 按清单修改生成脚本中的"建议动作"文本模板（最常见问题：建议语言太英文化，需改为工厂口语）
- 重新生成全量数据，更新文件，通知 ML 工程师可以开始用 v1.1 版本

**Day 3 下午：生成 500 条测试集**

- 复用相同生成脚本，但随机种子不同，确保测试集与训练集无重叠
- 产出 `data/demo/spc_test_500.jsonl`，交给评测工程师

---

#### 领域专家（1人）— Demo 周任务

**Day 2 上午（2小时）：审核 50 条样本**

接收数据工程师 Day 1 发来的文件，从 `spc_train_500.jsonl` 随机抽取 50 条（可用 `shuf -n 50`）。

审核重点（按优先级）：

1. **Nelson 规则判断是否正确**（最重要）：对每条样本，自己手算一遍判断结果，对比 `output` 字段。重点检查 Rule 2（连续 9 点同侧）和 Rule 5（连续 3 点中 2 点在 2σ 外）这两条最容易出错的规则
2. **CPK 数值是否合理**：CPK = min((USL-μ)/(3σ), (μ-LSL)/(3σ))，手算抽查 5 条，误差不超过 0.005
3. **建议动作是否符合工厂实际**：语言是否口语化、建议步骤是否可操作（例如："通知工艺工程师"而不是"escalate to process engineer"）
4. **角色适配**：默认 output 定位为"装备工程师"，检查术语深度是否合适

产出：在共享文档中填写 `demo_expert_review.xlsx`（字段：sample_id / 规则判断正确Y/N / CPK正确Y/N / 建议可操作Y/N / 备注），09:00 前发给数据工程师。

**Day 3–4：构造 20 条"黄金样本"**

在已有数据基础上，手工编写 20 条高质量 SPC 样本，专门覆盖**模型最容易出错**的场景（根据 Day 2 审核中发现的问题点）。这 20 条会直接加入训练集作为高权重样本。

格式要求：
- `output` 必须有完整推理链：`<think>逐条检查 Nelson 规则...计算 CPK...结合场景判断...</think>\n\n最终分析：...`
- 建议动作需具体到"第几步做什么"，不能只说"检查设备"

**Day 4 下午：参加结果评审会（1小时）**

- 看评测报告，判断模型输出的专业性：是否说了"行话"、建议是否靠谱
- 指出最常见的错误类型（供 POC 阶段改进数据构造策略）

---

#### ML 工程师（1人）— Demo 周任务

**Day 1（全天）：搭建训练环境**

1. 申请 GPU 资源：2×A100 80GB（Demo 阶段够用），确认 CUDA 12.1+、驱动正常
   ```bash
   nvidia-smi  # 确认 GPU 可见
   nvcc --version  # 确认 CUDA 版本
   ```
2. 安装 LLaMA-Factory（锁定版本 `v0.9.x`，避免浮动依赖问题）：
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory --branch v0.9.1
   cd LLaMA-Factory && pip install -e ".[torch,metrics]"
   ```
3. 下载 Qwen3-32B 基座权重到本地（从内网模型仓库或 ModelScope）：
   ```bash
   modelscope download --model Qwen/Qwen3-32B --local_dir /data/models/Qwen3-32B
   ```
   预计耗时 1–2 小时（权重约 65GB），下载期间同步做步骤 4
4. 冒烟测试：用 `examples/train_lora/llama3_lora_sft.yaml` 改写成 Qwen3 配置，跑 10 步验证无报错：
   ```bash
   llamafactory-cli train demo_smoke_test.yaml
   # 成功标志：看到 "{'loss': x.xx, 'learning_rate': ...}" 日志输出
   ```
5. 写好 Demo 训练配置 `configs/demo_spc_sft.yaml`（等 Day 2 数据就绪即可开跑）

关键配置参数（必须明确，不能用默认值）：
```yaml
model_name_or_path: /data/models/Qwen3-32B
dataset: spc_demo_500        # 需在 dataset_info.json 中注册
template: qwen               # Qwen3 专用模板，不能用 llama3
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_target: q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj
cutoff_len: 2048
per_device_train_batch_size: 1
gradient_accumulation_steps: 8   # 等效 batch=8
learning_rate: 1.0e-4
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
flash_attn: fa2              # 必须开，否则 32B 模型显存不够
output_dir: /data/checkpoints/demo-spc-sft-v1
save_steps: 100
logging_steps: 10
```

**Day 2–3：执行训练**

- Day 2 09:00 等数据工程师确认 v1.1 数据就绪后启动训练
- 先在数据工程师的 500 条中注册 dataset：在 `data/dataset_info.json` 添加条目，指向 `spc_train_500.jsonl`
- 启动训练，预计 Qwen3-32B + 500 条 + 1 epoch ≈ 20–30 分钟（2×A100）
- 训练完成后用 `llamafactory-cli export` 将 LoRA adapter merge 到基座，产出完整权重
- 用 vLLM 启动推理服务，验证能正常响应：
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model /data/checkpoints/demo-spc-sft-v1-merged \
    --port 8000 --max-model-len 4096
  # 测试：curl http://localhost:8000/v1/chat/completions -d '{"model":"...","messages":[...]}'
  ```
- 把服务地址发给评测工程师，Day 3 下午前完成

**Day 4：看报告，记录发现**

- 仔细看评测工程师的 `demo_eval.md`，重点关注：哪类 Nelson 规则错误最多、CPK 数值误差分布
- 在 `notes/demo_ml_findings.md` 记录：哪些超参可以调、数据哪里有问题
- 提出 POC 阶段训练配置的调整建议

**Day 5：输出 Demo 总结**

- 整理训练日志中的 loss 曲线截图
- 确认 POC 阶段的基础配置（rank 是否需要提高到 128、是否需要加 DoRA 等）

---

#### 评测工程师（1人）— Demo 周任务

**Day 1（全天）：搭建 SPC 评测脚本**

产出：`eval/spc_eval.py`，可接受模型 API 地址 + 测试集路径，输出标准化 JSON 报告。

脚本逻辑：
```python
# eval/spc_eval.py
# 输入：测试集 jsonl（每条含 input + ground_truth）
# 输出：eval_result.json

def evaluate_spc(model_url, test_file, output_file):
    results = []
    for sample in load_jsonl(test_file):
        response = call_model_api(model_url, sample["input"])

        # 评测维度 1：违规规则识别（F1）
        pred_rules = extract_rule_numbers(response)  # 正则抽取 "Rule X" 或 "规则X"
        gt_rules = sample["ground_truth"]["violations"]
        rule_f1 = f1_score(gt_rules, pred_rules)

        # 评测维度 2：CPK 数值误差
        pred_cpk = extract_cpk_value(response)       # 正则抽取数字
        gt_cpk = sample["ground_truth"]["cpk"]
        cpk_error = abs(pred_cpk - gt_cpk) if pred_cpk else None

        # 评测维度 3：是否包含推理步骤（简单检查）
        has_reasoning = "<think>" in response or "逐条" in response or "检查" in response

        results.append({...})

    # 汇总
    summary = {
        "rule_detection_f1": mean([r["rule_f1"] for r in results]),
        "cpk_mae": mean([r["cpk_error"] for r in results if r["cpk_error"]]),
        "has_reasoning_rate": mean([r["has_reasoning"] for r in results]),
        "n_samples": len(results)
    }
    json.dump({"summary": summary, "details": results}, open(output_file, "w"))
```

Day 1 还需完成：对 Qwen3-32B **基座模型**跑一遍评测，记录基线数据（存为 `results/baseline_qwen3_32b.json`）。这一步必须在 ML 工程师的 SFT 模型出来之前完成，否则无法做 before/after 对比。

关键细节：
- 正则提取 CPK 时需处理多种写法：`CPK=1.23`、`Cpk：1.23`、`过程能力指数为 1.23`
- 评测脚本需支持 `--temperature 0`（贪心解码），避免随机性干扰对比
- 跑基座时注意：基座没有 system prompt 微调，需要给一个明确的指令 prompt，和 SFT 版本保持完全一致

**Day 3 下午：跑 SFT 模型评测，产出对比报告**

- 收到 ML 工程师的推理服务地址后，立即用相同测试集跑评测
- 产出 `reports/demo_eval.md`，包含：
  - 核心指标对比表（基座 vs SFT，3 个维度）
  - 错误样本列表（Top 10 错误案例，含模型输出原文）
  - 初步结论：哪类规则模型最弱

**Day 4：参加结果评审会，记录改进方向**

- 根据领域专家和 ML 工程师的反馈，补充评测维度（例如"建议可操作性"需要人工打分）
- 确认 POC 阶段评测脚本的扩展需求：需要覆盖工艺参数推荐和故障诊断问答

---

#### 验收标准

- 训练流程无阻断性问题（能跑完 1 个 epoch）
- SPC 异常判断准确率：SFT 模型 > 基座模型（绝对提升 ≥5%，哪怕只有 5% 也算正向）
- 评测脚本可自动运行，输出 JSON 格式结果

#### 失败处理

| 问题 | 立即处理 |
|------|---------|
| 训练环境故障 | 降级到 4-bit QLoRA，或换 7B 模型验证流程 |
| 数据质量太差（专家否定 >30%） | 改用纯规则生成，跳过 LLM 生成步骤 |
| 无正向效果 | 检查数据格式/prompt 模板，不进入 POC 前先修复 |

---

### 0.2 POC 阶段（2–4 周）

**目标**：在 Demo 基础上，扩展到 3 个核心任务场景、验证完整后训练方法栈（SFT + DPO + RL 初探），输出决策依据：是否值得投入 Phase 2 全量资源。

**时间灵活**：若 Demo 效果很好，POC 用 2 周；若问题多，用满 4 周。

#### POC 验证范围

| 验证维度 | 具体内容 | 成功标准 |
|---------|---------|---------|
| **场景覆盖** | SPC 判断 + 工艺参数推荐 + 故障诊断问答 | 3 个场景均有正向提升 |
| **数据生成可行性** | 验证自动生成管道（Qwen3-235B 教师模型）产出质量 | 专家抽检通过率 >80% |
| **SFT 效果** | 5K 条数据，Qwen3-32B LoRA SFT | 工业专项 >基座 +15% |
| **DPO 可行性** | 500 对偏好数据，验证 DPO 是否进一步提升 | 满意度评分 +0.2 |
| **RL 初探** | 1K 条可验证题，GRPO 跑 1000 步 | reward_mean 稳定上升，无崩溃 |
| **VLM 可行性** | 1K 张 SPC 图片，验证 Qwen2.5-VL fine-tune 流程 | SPC 图片分析 F1 >0.70 |
| **部署可行性** | SGLang 部署，压测并发 | P95 延迟 <1s（8 并发） |

#### 人员配置（POC 期间）

| 角色 | 人数 | 说明 |
|------|------|------|
| ML 工程师 A | 1 | SFT + DPO 训练主责 |
| ML 工程师 B | 1 | RL 环境 + VLM 流程 + 部署验证主责 |
| 数据工程师 | 1 | 搭建自动生成管道，产出 5K SFT + 500 偏好对 + 1K RL 题 |
| 领域专家 | 1 | 质量审核 + 偏好标注（专职 Week 1–2） |
| 评测工程师 | 1 | 自动评测 CI + 每次 checkpoint 自动触发 + 竞品对比 |

---

#### 数据工程师（1人）— POC 详细步骤

**Week 1：搭建 3 场景数据生成管道，产出 5K SFT 样本**

目标数量分配：SPC 1.5K（沿用 Demo 生成器扩量）+ 工艺参数推荐 2K + 故障诊断问答 1.5K

*工艺参数推荐数据生成（2K 条）*：

1. 整理"工序 × 参数维度"矩阵，明确每个工序的参数名称、单位、合理范围，存为 `config/process_param_specs.json`：
   ```json
   {
     "SMT回流焊": {
       "peak_temp": {"min": 235, "max": 260, "unit": "°C", "critical": true},
       "time_above_liquidus": {"min": 45, "max": 90, "unit": "s"},
       "cooling_rate": {"max": 4, "unit": "°C/s"}
     },
     "注塑": {...},
     "焊接": {...}
   }
   ```
   此文件由领域专家在 Week 1 Day 1–2 填写，数据工程师负责格式化。

2. 编写 `scripts/gen_process_param.py`：
   - 随机选择工序和参数组合，构造"给定产品规格/缺陷现象 → 推荐参数"形式的问答
   - 50% 正常场景（参数在范围内，解释为什么合理）
   - 50% 异常场景（参数越界，说明后果和调整方向）
   - 每条问题用 3 种不同提问方式变体（语义等价，表述不同），提高数据多样性

3. 用 Qwen3-235B API（或内网部署版本）将结构化答案扩展为自然语言回答：
   ```bash
   python scripts/expand_to_natural_language.py \
     --input data/poc/process_param_structured_2k.jsonl \
     --model qwen3-235b \
     --output data/poc/process_param_2k.jsonl \
     --batch_size 20 \
     --max_tokens 1024
   ```
   预计耗时：2K 条 × 平均 500 tokens ≈ 1M tokens，Qwen3-235B 内网推理约 2–3 小时

*故障诊断问答数据生成（1.5K 条）*：

1. 从 MES 系统导出近 1 年维修工单（需申请数据权限，预计 3000 条原始工单）
2. 用 `scripts/parse_work_order.py` 做结构化提取（正则 + Qwen3-8B 辅助）：
   - 字段：故障描述 / 检查步骤 / 根因 / 处置方案
   - 过滤条件：故障描述 <20 字 或 根因字段为空 → 直接丢弃
3. 用 Qwen3-235B 补全"推理链"：给定故障描述 + 根因，生成中间诊断步骤
4. 按设备类型分层，确保 SMT/注塑/机器人各 ≥300 条，避免类别不均衡

**Week 2：构建 500 对 DPO 偏好数据**

偏好数据三种来源，数据工程师负责前两种（自动部分）：

1. **自动构造（数值/代码类，200 对）**：
   - 对 SPC 和工艺参数场景，用 SFT 模型（Week 1 训练完成的）采样 8 个候选
   - 自动计算数值误差，选误差最小的为 `chosen`，误差最大的为 `rejected`
   - 过滤门槛：`chosen` 的 CPK 误差必须 <0.01，否则整条丢弃（避免 chosen 本身就是错的）
   - 格式化为 TRL DPO 格式：`{"prompt": ..., "chosen": ..., "rejected": ...}`

2. **对比采样辅助（专家评分类，300 对）**：
   - 为每条 prompt 生成 4 个候选（SFT 模型 temperature=0.8），格式化为评分表格
   - 用脚本自动生成 `data/poc/preference_review_sheet.xlsx`，每行一个候选，专家在"选最优"列填 A/B/C/D
   - 专家填完后，自动转换为偏好对（最优 vs 最差）

**Week 3：生成 1K 条可验证 RL 训练题**

1. SPC 可验证题（500 条）：复用 Demo 的 `gen_spc_demo.py`，扩量并增加难度分级（easy/medium/hard 按 Nelson 规则复杂度分）
2. 工业代码生成题（500 条）：
   - 编写 100 道题目模板（SPC Python 实现 + MES SQL 查询各 50 道）
   - 每道题写好对应的单元测试（`test_*.py`），测试文件作为 reward 验证器
   - 用 Qwen3-235B 生成参考答案，并跑一遍测试验证参考答案本身是正确的
   - 用模板变量替换生成 5 个变体，共 500 条

**Week 4（若需要）：数据复盘与 Phase 2 数据方案调整**

- 统计各类数据的生成效率（条/小时）和专家通过率，更新 Phase 2 数据规模的时间估算
- 整理"哪类数据最难生成"的问题清单，提出解决方案

---

#### 领域专家（1人）— POC 详细步骤

**Week 1 Day 1–2：填写工艺参数规格表**

填写 `config/process_param_specs.json`（由数据工程师提供空模板），覆盖 5 个工序：
- SMT 回流焊：峰值温度、液相线以上时间、冷却速率、预热时间
- 手工焊接：烙铁温度、焊接时间、助焊剂用量
- 注塑：料筒温度（前/中/后段）、注射压力、保压时间、冷却时间
- 冲压：冲裁力、压边力、润滑间隙
- 涂装：粘度、膜厚、固化温度/时间

关键细节：每个参数必须填"critical"字段（布尔值），标注参数偏差是否直接影响产品质量——数据生成脚本会对 critical=true 的参数生成更多"越界后果"场景。

**Week 1 Day 3–5：审核 SFT 数据（抽检 10%，约 500 条）**

重点审核工艺参数推荐数据（这是新场景，Demo 没有验证过）：

审核标准表（每条填写）：
```
参数值在合理范围内？（Y/N）
推荐理由是否引用了正确的工艺原理？（Y/N/部分）
"越界后果"描述是否准确？（Y/N）
输出语言风格是否适合装备工程师？（Y/N）
```

若通过率 <80%：立即通知数据工程师，暂停生成剩余批次，先修复生成模板。

**Week 2：构建 300 对专家偏好标注**

每天处理 75 对（全天 6 小时，每对约 5 分钟）：

标注流程（每对）：
1. 读 prompt（约 30 秒）
2. 分别阅读 4 个候选回答，在草稿纸上记录每个的优缺点（约 3 分钟）
3. 在 Excel 表格中选最优（chosen）和最差（rejected）

选择依据（按优先级）：
- P1：数值是否正确（工艺参数是否在规格范围内）
- P2：推理过程是否清晰（能看出为什么这样建议）
- P3：建议是否可操作（能直接执行）
- P4：语言风格是否符合目标角色

注意：若 4 个候选都很差（比如都有数值错误），在备注列填"全部不合格"，该条不纳入训练集。

**Week 3：对 SFT 模型和 DPO 模型做深度人工评测（各 50 条）**

数据工程师会提供评测题目，领域专家针对两个模型的输出做专业性评分（1–5 分）：

- 5分：专业准确，建议可直接执行
- 4分：基本正确，细节需确认
- 3分：方向正确但有明显遗漏
- 2分：有关键错误（比如参数范围错误）
- 1分：完全错误或不相关

重点对比：DPO 模型是否比 SFT 模型在"建议的可操作性"和"语言风格适配"上有提升。

---

#### ML 工程师 A（1人）— POC 详细步骤（SFT + DPO 主责）

**Week 1：扩量 SFT 训练**

*Day 1–2：数据准备与配置*

1. 等数据工程师产出 5K 数据（预计 Week 1 Day 3），先用 Demo 的 500 条做一次 3-epoch 训练，验证更长训练是否有提升（作为对比实验）
2. 在 `data/dataset_info.json` 注册 3 个新数据集：
   ```json
   "poc_sft_5k": {
     "file_name": "poc/sft_5k_merged.jsonl",
     "formatting": "alpaca",
     "columns": {"prompt": "instruction", "query": "input", "response": "output"}
   }
   ```
3. 更新训练配置 `configs/poc_sft_v1.yaml`（关键变更）：
   ```yaml
   lora_rank: 128          # Demo 用 64，POC 提升到 128
   num_train_epochs: 3     # Demo 用 1，POC 用 3
   cutoff_len: 4096        # 故障诊断输出较长，需要更长上下文
   ```

*Day 3–5：执行训练，每 epoch 评测一次*

- 5K 条 × 3 epochs × Qwen3-32B ≈ 3–4 小时（4×A100）
- 每个 epoch 结束后，通知评测工程师跑一次自动评测，记录 epoch 1/2/3 的指标变化
- 观察：loss 在 epoch 3 是否还在下降（若 epoch 2 之后 loss 平台，说明 3K 数据已经够用，无需等 5K）

**Week 2：DPO 训练**

*Day 1（准备）*

1. 等领域专家完成 300 对 + 数据工程师自动构造 200 对，合并为 `data/poc/dpo_pairs_500.jsonl`
2. 验证偏好数据质量：检查 chosen 和 rejected 的平均长度差异（若 chosen 比 rejected 长很多，可能存在"长度偏置"问题，需要过滤）
3. 配置 `configs/poc_dpo_v1.yaml`：
   ```yaml
   stage: dpo
   model_name_or_path: /data/checkpoints/poc-sft-v1    # 在 SFT 模型基础上做 DPO
   beta: 0.1
   pref_loss: sigmoid
   lora_rank: 64            # DPO 阶段 rank 可以低一些
   learning_rate: 5.0e-5    # 比 SFT 低一个数量级
   num_train_epochs: 2
   ```

*Day 2–3（训练）*

- 500 对 × 2 epochs ≈ 1 小时（4×A100），训练完成后 merge adapter
- 同时跑一个 **KTO 对比实验**（用 binary feedback 数据，看是否比 DPO 效果更好）：
  ```yaml
  pref_loss: kto            # 只需改这一行，其余配置相同
  ```
- 对比 DPO vs KTO vs SFT，评测工程师输出三方对比报告

*Day 4–5（分析）*

- 如果 DPO 比 SFT 没有提升（或退步），检查原因：
  - 偏好数据质量？（看"全部不合格"标注的比例）
  - beta 值太高导致模型退化？（尝试 beta=0.05）
  - 偏好对数量不足？（500 对对于 32B 模型可能偏少）
- 记录结论到 `notes/dpo_analysis.md`

**Week 3：辅助多模态流程验证（与 ML 工程师 B 协作）**

- 准备 Qwen2.5-VL-7B 的训练配置，供 ML 工程师 B 使用：
  ```yaml
  model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
  finetuning_type: lora
  lora_rank: 64
  visual_inputs: true      # 必须开启
  template: qwen2_vl       # 专用模板
  ```
- 协助排查 VLM 训练中的显存/格式问题

**Week 4（若需要）：超参调优实验**

系统地跑消融实验，为 Phase 2 全量训练找最优超参：
- lora_rank: 64 vs 128 vs 256
- learning_rate: 1e-4 vs 5e-5 vs 2e-5
- 用 5K 数据跑，每组 1 epoch，对比评测指标

---

#### ML 工程师 B（1人）— POC 详细步骤（RL + VLM + 部署主责）

**Week 1：搭建 veRL 环境，验证 GRPO 可运行**

*Day 1–2：veRL 安装与配置*

1. 安装 veRL（Volcano Engine RL）：
   ```bash
   git clone https://github.com/volcengine/verl --branch v0.3.x
   cd verl && pip install -e .
   # 同时安装 vLLM（veRL 用 vLLM 做 rollout）
   pip install vllm==0.6.x
   ```
2. 验证 vLLM 可以加载 Qwen3-32B：
   ```bash
   python -c "from vllm import LLM; llm = LLM('/data/models/Qwen3-32B', max_model_len=4096); print('OK')"
   ```
   预期显存占用：Qwen3-32B FP16 ≈ 64GB，需要 2×A100 80GB 做 rollout
3. 用 veRL 官方的 `gsm8k` 示例跑通 GRPO 全流程（只跑 50 步），确认无报错：
   ```bash
   python examples/grpo_trainer.py --config examples/config/gsm8k_grpo.yaml --total_training_steps 50
   ```

*Day 3–5：适配工业 SPC 任务*

1. 编写工业 reward 函数 `verl/rewards/industrial_spc_reward.py`：
   ```python
   class SPCReward:
       def compute(self, responses: List[str], ground_truths: List[dict]) -> List[float]:
           rewards = []
           for resp, gt in zip(responses, ground_truths):
               accuracy = self._check_nelson_rules(resp, gt["violations"])
               format_score = self._check_format(resp)
               length_penalty = self._check_length(resp)
               rewards.append(accuracy + format_score - length_penalty)
           return rewards

       def _check_nelson_rules(self, response, gt_violations):
           pred = extract_rule_numbers(response)
           if not pred and not gt_violations:
               return 1.0  # 正常状态，正确识别
           return f1_score(gt_violations, pred)  # 违规场景用 F1
   ```
2. 配置 `configs/poc_grpo_spc.yaml`：
   ```yaml
   actor_rollout_ref:
     model_path: /data/checkpoints/poc-sft-v1  # 在 SFT 模型上做 RL
     rollout:
       n: 8                # 每个 prompt 采样 8 个
       temperature: 0.8
       max_new_tokens: 1024
   critic:
     enable: false         # GRPO 不需要 critic
   algorithm:
     kl_ctrl:
       kl_coef: 0.01
     clip_ratio: 0.2
   trainer:
     total_training_steps: 1000
     save_steps: 200
   ```

**Week 2：运行 GRPO 1000 步，监控 reward 曲线**

*Day 1（启动）*

```bash
torchrun --nproc_per_node=8 verl/trainer/main_ppo.py \
  --config configs/poc_grpo_spc.yaml \
  --data.train_files data/poc/rl_spc_1k.jsonl \
  2>&1 | tee logs/grpo_poc_run1.log
```

关键监控指标（每 100 步记录）：
- `reward/mean`：目标是稳定上升，哪怕很慢
- `kl_divergence`：不应超过 0.3（超过说明模型变化太快，降低 learning_rate）
- `response_length/mean`：若持续增长说明模型在"堆字数"刷 format_reward，需要加强 length_penalty

*Day 2–4（问题排查预案）*

| 问题现象 | 原因 | 处理 |
|---------|------|------|
| reward 一直为 0 | reward 函数 bug 或正则提取失败 | 打印 10 条 response，手动检查提取逻辑 |
| KL 爆炸（>1.0） | learning_rate 太高 | 降低 10 倍，重启 |
| loss NaN | bf16 溢出 | 加 `--gradient_clip 1.0`，或换 fp32 |
| GPU OOM | batch_size 太大 | 减小 `rollout.n` 从 8 到 4 |

*Day 5：输出 RL 初探总结*

- reward 曲线截图（从 wandb 或 tensorboard 导出）
- 结论：reward 是否在 1000 步内出现上升趋势
- 推荐 Phase 2 的 GRPO 超参范围

**Week 2 同步：SGLang 部署验证（与 GRPO 并行，利用等待 GPU 的空隙）**

1. 安装 SGLang：
   ```bash
   pip install sglang[all]==0.3.x
   ```
2. 用 SGLang 启动 SFT 模型服务：
   ```bash
   python -m sglang.launch_server \
     --model-path /data/checkpoints/poc-sft-v1-merged \
     --port 30000 \
     --tp 2 \            # tensor parallel，2 张 A100
     --mem-fraction-static 0.85
   ```
3. 用 `locust` 做并发压测：
   ```python
   # locustfile.py
   from locust import HttpUser, task
   class InferenceUser(HttpUser):
       @task
       def predict(self):
           self.client.post("/generate", json={
               "text": "分析以下 SPC 数据：...",
               "sampling_params": {"max_new_tokens": 512, "temperature": 0}
           })
   ```
   ```bash
   locust -f locustfile.py --headless -u 8 -r 2 --run-time 60s \
     --host http://localhost:30000
   ```
4. 记录 P50/P95/P99 延迟，对比 vLLM（Demo 阶段用的）和 SGLang 的差异

**Week 3：VLM 多模态流程验证**

*Day 1–2：准备 1K 张 SPC 控制图图片*

复用 Demo 阶段数据工程师的生成脚本，但改为输出 PNG 图片而非 CSV 文本：
```python
# scripts/gen_spc_images.py
import matplotlib.pyplot as plt

for i in range(1000):
    scenario, data, violations = generate_spc_scenario()
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_control_chart(ax, data, ucl, lcl, cl)
    plt.savefig(f"data/poc/spc_images/{i:04d}.png", dpi=100, bbox_inches='tight')
    plt.close()
    save_label(i, violations, cpk)  # 保存对应标注
```

同时构造多模态对话格式（LLaMA-Factory 的 VLM 格式）：
```json
{
  "messages": [
    {"role": "user", "content": "<image>请分析这张SPC控制图，指出违规规则和CPK值"},
    {"role": "assistant", "content": "<think>观察控制图...Rule 2 违规...CPK=1.23...</think>\n\n分析结果：..."}
  ],
  "images": ["data/poc/spc_images/0001.png"]
}
```

*Day 3–5：Qwen2.5-VL-7B LoRA SFT*

```bash
llamafactory-cli train configs/poc_vlm_sft.yaml
# 配置关键点：
# - visual_inputs: true
# - template: qwen2_vl（不是 qwen）
# - 1K 图片 × 1 epoch ≈ 45 分钟（4×A100）
```

训练完成后，用 20 张 holdout 图片手动测试模型输出质量：
- 检查模型是否能正确指出违规规则
- 检查是否能读出大致正确的 CPK 值（误差 ±0.1 即可）
- 若完全无法识别（F1=0），检查图片格式、模板设置是否正确

---

#### 评测工程师（1人）— POC 详细步骤

**Week 1：扩展评测脚本，覆盖 3 个场景，建立所有基线**

*Day 1–2：扩展评测脚本*

在 Demo `spc_eval.py` 基础上，新增两个评测模块：

`eval/process_param_eval.py`（工艺参数推荐评测）：
```python
def evaluate_process_param(model_url, test_file, spec_file):
    specs = load_spec(spec_file)  # 加载 process_param_specs.json
    for sample in load_jsonl(test_file):
        response = call_model(model_url, sample["input"])

        # 评测维度 1：参数值是否在规格范围内
        pred_params = extract_parameters(response)  # 正则提取"XX°C"、"XX MPa"等
        in_spec = check_all_params_in_spec(pred_params, specs[sample["process"]])

        # 评测维度 2：是否引用了工艺原理（关键词检查）
        has_reasoning = any(kw in response for kw in
                           ["润湿性", "金属间化合物", "热传导", "粘度", "收缩率"])

        # 评测维度 3：专家评分占位（由领域专家手动填写 Excel 后合并）
```

`eval/fault_diag_eval.py`（故障诊断评测）：
```python
# 评测维度：Top-3 根因命中率
# 从 response 中提取"根因"关键词，检查是否命中 ground_truth 的 top-3 根因
def compute_top3_hit_rate(response, gt_causes):
    extracted = extract_root_causes(response)
    hits = sum(1 for cause in gt_causes[:3] if any(
        similar(cause, pred) > 0.7 for pred in extracted  # 语义相似度
    ))
    return hits / min(3, len(gt_causes))
```

*Day 3–5：跑所有基线（基座 Qwen3-32B vs GPT-5.4 API vs 内部 MiMo-v2-pro）*

```bash
# 并行跑三个基座的评测（用不同 --model-url）
python eval/run_all.py \
  --models qwen3-32b gpt-5.4 mimo-v2-pro \
  --test-sets data/poc/spc_test_500.jsonl \
             data/poc/process_param_test_200.jsonl \
             data/poc/fault_diag_test_200.jsonl \
  --output results/poc_baseline.json
```

注意：GPT-5.4 API 调用需要计费预算审批（预计 200 条 × 平均 500 tokens ≈ 0.1M tokens ≈ $5），提前走审批流程。

**Week 2：输出 SFT vs DPO vs 基座三方对比报告，搭建自动评测 CI**

*Day 1–3：SFT 模型评测*

收到 ML 工程师 A 的 SFT checkpoint（Week 1 训练结果），立即跑全量评测（3 个场景 × 900 条）。

产出 `reports/poc_w2_sft_vs_baseline.md`：
```markdown
## POC Week 2：SFT vs 基座对比报告

| 评测场景 | 指标 | Qwen3-32B 基座 | GPT-5.4 | SFT v1（5K） | 提升幅度 |
|---------|------|--------------|---------|------------|---------|
| SPC 违规检测 | F1 | 0.52 | 0.72 | 0.68 | +16pp vs 基座 |
| ...     |     |      |      |     |     |

## 关键发现
- Rule 2（连续9点同侧）SFT 提升最显著（+28pp）
- 工艺参数推荐的"越界后果"描述仍不如 GPT-5.4，需要更多专家数据
- 故障诊断 Top-1 命中率提升有限（+5pp），原因分析：...
```

*Day 4–5：搭建自动评测 CI*

```bash
# .github/workflows/eval_on_checkpoint.yml（或内网 CI 等价）
# 触发条件：新 checkpoint 目录出现时自动触发
# 运行：eval/run_all.py → 结果写入 results/ → 与上一个 checkpoint 对比 → 推送钉钉通知
```

关键：CI 必须包含"退步预警"——若任一指标比上一个 checkpoint 下降 >3%，钉钉发红色告警。

**Week 3：RL 过程监控 Dashboard + VLM 评测**

*Day 1（RL 监控）*

配置 wandb（或内网 tensorboard）自动记录 ML 工程师 B 的 GRPO 训练指标：
- `reward/mean`、`reward/std`（奖励均值和方差）
- `kl_divergence`（KL 散度，超过 0.3 飘红）
- `response_length/mean`（响应长度均值，持续增长说明有问题）
- 每 200 步截图，存入 `reports/rl_curves/`

*Day 3–5（VLM 评测）*

对 VLM 模型（Qwen2.5-VL-7B fine-tuned）跑专项评测：
```python
# eval/vlm_spc_eval.py
# 输入：图片路径 + ground_truth 标注
# 关键：需要用支持多模态的推理框架（SGLang 的 VLM 模式）
python eval/vlm_spc_eval.py \
  --model-url http://localhost:30001 \  # VLM 专用端口
  --test-dir data/poc/spc_images_holdout_200/ \
  --labels data/poc/spc_image_labels.json \
  --output results/vlm_eval.json
```

**Week 4（若需要）：POC 总结报告**

产出 `reports/poc_final_report.md`，包含：
1. 各验证维度结论（逐条对照验证范围表格）
2. 各模型版本的完整评测数据表
3. 发现的 Top 5 技术风险和对应缓解方案
4. **Phase 2 是否继续的明确建议**（通过 / 暂停 / 有条件通过）

---

#### POC 资源消耗

| 项目 | 估算 |
|------|------|
| GPU | 8×A100 × 4 周 ≈ 224 GPU·天 |
| 存储 | ~1TB（数据 + checkpoint） |
| 人力 | 5 人 × 4 周 ≈ 100 人天 |
| 外部 API（教师模型） | Qwen3-235B 推理约 5M tokens ≈ $50；GPT-5.4 评测对比 ≈ $5 |

#### POC 通过标准（进入 Phase 2 的门槛）

| 指标 | 最低通过线 | 说明 |
|------|---------|------|
| SFT 工业专项提升 | ≥15%（相对基座） | 与 Phase 1 目标对齐 |
| DPO 有效性 | 满意度评分提升 ≥0.1 | 证明偏好优化路线可行 |
| RL reward 收敛 | 1000 步内 reward_mean 上升 | 无需达标，只需不崩溃 |
| VLM 流程跑通 | SPC 图片 F1 >0 | 哪怕 0.5 也算流程通 |
| 部署延迟 | P95 <2s（8 并发） | 确认 SGLang 可用 |

若有 2 项未达标，暂停 Phase 2，先修复技术障碍。

---

## 一、阶段概览

Demo + POC 完成后，已验证完整链路可行，进入 Phase 2 全量投入。

Phase 2 在 Phase 1（10K SFT 初版）基础上做四件事：

| 工作包 | 时间 | 核心产出 |
|--------|------|----------|
| **W1** 数据扩建（SFT 10K → 100K） | M3–M6 | 高质量工业 SFT 数据集 v2 |
| **W2** DPO 偏好优化 | M5–M7 | 偏好对齐模型，幻觉↓，专业性↑ |
| **W3** RL-CoT 推理增强 | M6–M8 | 工业推理能力强化模型 |
| **W4** VLM 工业视觉微调 | M4–M9 | 支持图片理解的多模态工业模型 |

各工作包有依赖但可并行推进，具体见第五章甘特图。

---

## 二、团队角色与职责

| 角色 | 人数 | 工作包归属 | 核心职责 |
|------|------|-----------|----------|
| **ML 工程师（训练）** | 2 | W1/W2/W3/W4 | 训练流程、框架搭建、超参调优、实验管理 |
| **数据工程师** | 1 | W1 | 数据管道、格式化、去重、质量过滤 |
| **领域专家（工艺/装备）** | 2 | W1/W2 | 标注审核、偏好对话构建、RL 验证规则制定 |
| **标注团队（外包/内部）** | 2 | W1/W2 | SFT 样本标注、偏好对标注 |
| **评测工程师** | 1 | 全阶段 | 评测框架维护、自动评测 CI、竞品对比报告 |

> 合计 8 人，与白皮书 Phase 2 资源需求一致。

---

## 三、W1：数据扩建（M3–M6）

### 3.1 目标

将 SFT 数据集从 Phase 1 的 10K 条扩展至 **100K 条**，覆盖全部六类任务，质量达到"专家可直接使用"标准。

### 3.2 数据类别与数量分配

| 数据类别 | Phase 1 | Phase 2 新增 | Phase 2 总量 | 质量等级 |
|---------|---------|------------|------------|---------|
| 工业问答对（多角色） | 5K | 25K | 30K | L2（专家审核） |
| 装备 Agent 轨迹 | 2K | 18K | 20K | L3（仿真+人工） |
| SPC/CPK 分析示例 | 1K | 9K | 10K | L1（规则生成+校验） |
| 工艺参数推荐 | 1K | 14K | 15K | L2（历史数据+专家） |
| 故障诊断案例 | 1K | 19K | 20K | L2（工单结构化） |
| 工业代码生成 | 0 | 15K | 15K | L1（自动测试） |
| **合计** | **10K** | **100K** | **110K** | — |

> 质量等级：L1=自动验证可信，L2=需专家抽检，L3=需全量人工审核

### 3.3 各类数据生成方法

#### 3.3.1 工业问答对（30K）—— 多角色输出

**生成流程**：
```
原始语料（SOP/手册/标准/故障报告）
    │
    ▼ Step 1：文档切片（chunk_size=1500 tokens，overlap=200）
    │         工具：LangChain TextSplitter
    ▼ Step 2：问题种子生成
    │         使用 Qwen3-235B 对每个 chunk 生成 5 类问题种子：
    │           - 操作类（"如何..."）
    │           - 故障类（"当...出现时..."）
    │           - 标准类（"...的规范要求是..."）
    │           - 比较类（"...和...的区别是..."）
    │           - 数值类（"...的参数范围是..."）
    ▼ Step 3：四角色回答生成
    │         同一问题生成 4 个角色版本（工人/工程师/管理者/算法工程师）
    │         System Prompt 模板固定，注入角色约束
    ▼ Step 4：自动质量过滤
    │         使用 Qwen3-32B 打分（1–5），过滤 <3 分的样本
    │         过滤规则：含幻觉数值、角色不符、回答<50字
    ▼ Step 5：领域专家抽检（10% 随机采样）
              专家打分 <3 的批次整批退回重生成
```

**人力分配**：数据工程师 1 人负责 Step 1–4 自动化；领域专家 2 人负责 Step 5 审核（每人每天审 200 条，共需约 15 工作日）。

**原始语料来源**：
- 小米内部 SOP 文档（30%，需脱敏）：去除型号/产品序列号，替换为占位符 `[设备型号]`
- 公开工业标准（40%）：ISA-88、IATF-16949、GB/T 19001 等
- 合成语料（30%）：用 DeepSeek-R1 生成工业推理对话链，作为高质量种子

#### 3.3.2 装备 Agent 轨迹（20K）—— 仿真环境采集

**仿真环境架构**：
```
数字孪生工厂（基于 OpenAI Gym 接口）
  ├── 设备模拟器：SMT贴片机 / 注塑机 / 工业机器人（各独立状态机）
  ├── 传感器数据生成器：振动/温度/压力时序，可注入故障模式
  ├── MES/ERP 接口 Mock：SQLite 模拟，支持 Function Call
  └── 评分器：任务完成检验（规则判断）
```

**数据采集流程**：
```
Step 1：任务脚本生成
  - 4 类任务 × 难度3级 × 设备5类 = 60 种任务模板
  - 每模板生成 50–100 个随机参数实例

Step 2：专家示范采集（Expert Demo，2K 条）
  - 领域专家操作仿真环境，记录完整工具调用轨迹
  - 用于 BC（Behavior Cloning）冷启动

Step 3：模型自动 Rollout（18K 条）
  - 用 Phase 1 SFT 模型在仿真环境跑 rollout
  - 筛选：任务完成率 >80% 的轨迹保留
  - 失败轨迹：修正后标注为负样本（供 DPO 使用）

Step 4：轨迹格式化
  格式：system_prompt + [user_turn + assistant_tool_call + tool_result] × N + final_answer
  工具调用格式与 Qwen3 MCP 规范对齐
```

**工具集定义**（统一注册到 Function Call schema）：
```python
tools = [
    "query_mes_production_order",   # 查询 MES 工单
    "query_sensor_realtime",        # 查询实时传感器
    "query_spc_chart",              # 查询 SPC 控制图
    "adjust_device_parameter",      # 调整设备参数（写操作，仿真环境）
    "query_maintenance_history",    # 查询维修历史
    "calculate_cpk",                # 计算 CPK（调用 Python 函数）
    "query_bom",                    # 查询 BOM 物料清单
    "alert_line_manager",           # 向线长发送告警
]
```

#### 3.3.3 SPC/CPK 分析示例（10K）—— 规则生成

**全自动生成，无需人工标注**：

```python
# 生成逻辑伪代码
for _ in range(10000):
    # 随机生成受控/失控数据
    scenario = random.choice([
        "normal",          # 正常状态
        "nelson_rule_1",   # 点超出控制限
        "nelson_rule_2",   # 连续9点在均值同侧
        "nelson_rule_3",   # 连续6点单调上升/下降
        "nelson_rule_4",   # 连续14点交替升降
        "nelson_rule_5",   # 连续3点中2点在2σ外
        "nelson_rule_6",   # 连续5点中4点在1σ外
        "nelson_rule_7",   # 连续15点在1σ内
        "nelson_rule_8",   # 连续8点在1σ外（两侧）
        "high_cpk",        # CPK > 1.67（优秀）
        "low_cpk",         # CPK < 1.0（不合格）
        "off_center",      # 均值偏移
    ])
    data = generate_spc_data(scenario, n_points=25)
    usl, lsl = random_spec_limits(data)

    # 规则计算答案
    answer = {
        "violations": detect_nelson_rules(data),
        "cpk": calculate_cpk(data, usl, lsl),
        "action": lookup_action_table(violations, cpk)
    }

    # 用 Qwen3-32B 将答案扩展为自然语言回答
    response = llm.generate(data=data, answer=answer,
                             prompt="生成专业的 SPC 分析报告")
```

**验证**：所有样本经规则引擎二次验证，数值误差 >0.01 自动丢弃。

#### 3.3.4 工艺参数推荐（15K）—— 历史数据+专家

**数据来源**：
- 小米工厂历史工单（脱敏后）：50%，提取"工序描述 → 参数记录 → 质量结果"三元组
- 专家构造典型案例：30%，领域专家手写 300 道种子题，再用 Qwen3-235B 变体扩充 10×
- 公开工艺手册（SMT IPC 标准等）提取：20%

**标注流程**：
```
原始工单 → 数据工程师结构化提取（Python + regex）
         → 领域专家补全"推荐理由"字段（最耗时步骤）
         → Qwen3-235B 生成多样化提问方式（同一知识点5种问法）
         → 自动去重（MinHash，相似度阈值 0.85）
```

**预计人工时间**：2 名领域专家 × 30 工作日 × 100 条/天 = 6K 条专家构造，其余自动扩充。

#### 3.3.5 故障诊断案例（20K）—— 工单结构化

**从维修工单到 SFT 样本的流水线**：
```
Step 1：工单采集
  来源：小米工厂 MES 维修工单（近 3 年，脱敏）
  字段：设备ID（脱敏）、故障描述、检查步骤、根因、处置方案、耗时

Step 2：结构化提取（数据工程师，Python + Qwen3-8B 辅助）
  - 抽取：故障现象 / 诊断步骤 / 根因 / 维修动作
  - 过滤：描述 <20字、根因字段为空 的工单丢弃

Step 3：多步推理链补全（Qwen3-235B）
  Prompt：给定故障现象和最终根因，补全中间推理步骤（FMEA 格式）

Step 4：多故障耦合构造（10% 占比，约 2K 条）
  随机组合 2 个单故障场景，生成"多故障同时出现"的复杂案例
  答案：列出所有根因并按优先级排序

Step 5：专家审核（5% 抽样）
  重点审核：根因是否正确、维修步骤是否安全可操作
```

#### 3.3.6 工业代码生成（15K）—— 自动测试验证

**四个子任务独立流水线**：

| 子任务 | 数量 | 生成方法 | 验证方法 |
|--------|------|----------|----------|
| SPC Nelson 规则 Python | 3K | Qwen3-235B 生成 + 变体扩充 | pytest 单元测试（测试用例预制） |
| MES SQL 查询生成 | 5K | 模板化 + 随机化（JOIN/WHERE 组合） | SQLite 执行验证 |
| PLC ST 语言逻辑 | 3K | 专家写种子题 × Qwen3 扩充 | 语法解析器验证 |
| OPC-UA 数据采集脚本 | 2K | 模板化生成 | asyncua mock 服务器验证 |
| 通用工业 Python 脚本 | 2K | 问题描述 → 代码生成 | exec() 沙箱运行 |

**代码质量门槛**：可执行率 100%（语法错误直接丢弃），功能正确率 >90%（单元测试通过）。

### 3.4 数据管道架构

```
原始语料池（小米内部 + 公开数据）
        │
        ▼
    预处理层（去重/脱敏/格式化）
        │
        ▼
    生成层（Qwen3-235B / DeepSeek-R1 / 规则引擎）
        │
        ▼
    质量过滤层（Qwen3-32B 打分 + 规则验证）
        │
        ▼
    专家审核层（10% 抽检，不合格批次退回）
        │
        ▼
    最终数据集（Parquet 格式，version 控制）
        │
        ▼
    训练集/验证集/测试集切分（8:1:1）
```

**工具栈**：Python + HuggingFace datasets + DVC（数据版本管理）+ MinHash 去重

---

## 四、W2：DPO 偏好优化（M5–M7）

### 4.1 目标

在 W1 SFT 模型基础上，进一步降低幻觉率、提升工业专业性和安全边界，通过 DPO 让模型学会"哪种回答更好"。

### 4.2 偏好数据构建（20K 对）

**偏好对格式**：`(prompt, chosen, rejected)`

#### 4.2.1 数据来源策略

| 来源 | 数量 | 构建方式 |
|------|------|----------|
| **对比采样**（主要来源） | 12K | 同一 prompt，SFT 模型生成 8 个候选，专家选最优/最差 |
| **专家修正**（高质量） | 4K | 通用模型（GPT-5.4 API）生成 rejected，专家改写为 chosen |
| **自动构造**（批量） | 4K | 自动规则判断对错（数值推理类/代码类） |

#### 4.2.2 对比采样流程

```
Step 1：prompt 池准备
  从 W1 验证集中抽取 3K 条 prompt（覆盖全部 6 类任务）

Step 2：候选生成
  每个 prompt 用 SFT 模型（temperature=0.8）采样 8 个回答
  + GPT-5.4 API 生成 1 个作为"外部参照"（共 9 候选）

Step 3：专家排序（最耗时）
  每位领域专家每天排 50 个 prompt（每个 prompt 选最优+最差）
  2 名专家 × 50天 = 5K 条有效排序
  → 每 prompt 提取 (best, worst) 一对，共 5K 对
  → 扩充：(best, rank_2)、(best, rank_3) 共提取 12K 对

Step 4：一致性校验
  两位专家对同一 batch 结果，Cohen's Kappa > 0.7 为合格
  低于阈值的 batch 重新标注
```

#### 4.2.3 自动构造（数值/代码类）

```python
# 数值推理类（SPC/CPK/工艺参数）
# 自动判断：数值误差 >5% 为 rejected，精确为 chosen
for prompt in numerical_prompts:
    responses = sample_model(prompt, n=8)
    ground_truth = calculate_ground_truth(prompt)
    chosen = min(responses, key=lambda r: |extract_number(r) - ground_truth|)
    rejected = max(responses, key=lambda r: |extract_number(r) - ground_truth|)
    if |extract_number(chosen) - ground_truth| < 0.01:  # 门槛
        save_preference_pair(prompt, chosen, rejected)

# 代码类：执行结果决定 chosen/rejected
for prompt in code_prompts:
    responses = sample_model(prompt, n=8)
    results = [run_in_sandbox(r) for r in responses]
    passing = [r for r, res in zip(responses, results) if res.passed]
    failing = [r for r, res in zip(responses, results) if not res.passed]
    if passing and failing:
        save_preference_pair(prompt, passing[0], failing[0])
```

### 4.3 训练配置

**框架**：TRL（HuggingFace）DPO Trainer
**基底**：W1 SFT 模型（Qwen3-32B + LoRA）

```yaml
# DPO 训练配置
model: qwen3-32b-sft-v2
method: dpo
beta: 0.1                    # KL 散度惩罚系数（工业场景保守）
loss_type: sigmoid            # 标准 DPO loss

lora:
  r: 64
  alpha: 128
  target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  learning_rate: 5e-5
  batch_size: 32              # per device
  gradient_accumulation: 4
  num_epochs: 3
  warmup_ratio: 0.1
  lr_scheduler: cosine

data:
  train: preference_pairs_train_18k.parquet
  eval:  preference_pairs_eval_2k.parquet
  max_length: 4096
  max_prompt_length: 2048
```

**补充方法（KTO）**：对于标注团队收集的 binary feedback（工人点赞/踩），使用 KTO 增量优化，无需成对标注，降低标注成本。

### 4.4 DPO 评测指标

| 指标 | 测量方法 | Phase 2 目标 |
|------|----------|-------------|
| 幻觉率（数值错误） | 自动数值验证 | <5%（Phase 1 SFT 约 12%） |
| 工厂问答满意度 | 领域专家评分 | ≥4.0/5 |
| 角色适配度 | 人工评分（50 条） | >80% 正确适配 |
| 安全拒绝率 | 危险操作 prompt 测试集 | >95% 正确拒绝 |
| GPQA/通用能力保留 | 标准评测集 | 不低于 SFT 版本 -2% |

---

## 五、W3：RL-CoT 推理增强（M6–M8）

### 5.1 目标

在 DPO 模型基础上，用 GRPO + 可验证工业推理题强化工业推理链能力，目标：SPC 判断准确率 ≥85%，工艺参数推理 ≥80%。

### 5.2 RL 训练数据（33K 条可验证题）

| 任务类型 | 数量 | 验证方式 | 负责人 |
|---------|------|----------|--------|
| SPC 异常判断 | 5K | Nelson 规则引擎（100% 自动） | 数据工程师 |
| CPK 计算 | 3K | 数值精确匹配（±0.001） | 数据工程师 |
| 工艺参数范围判断 | 5K | 专家规则库（领域专家维护） | 领域专家 |
| 工业数学推导 | 10K | SymPy 符号计算验证 | ML 工程师 |
| 工业代码生成 | 10K | 单元测试执行（pytest） | ML 工程师 |

**SPC 数据生成**（全自动）：
```python
# 覆盖 Nelson 8 条规则 × 5 种数据分布 × 过程能力水平
# 输出格式：CSV 数据 + 规格限 → 文字分析报告（含违规规则 + CPK + 建议）
```

**工艺参数规则库**（领域专家维护）：
```json
{
  "SMT回流焊": {
    "peak_temperature": {"min": 235, "max": 260, "unit": "°C"},
    "time_above_liquidus": {"min": 45, "max": 90, "unit": "s"},
    "cooling_rate": {"max": 4, "unit": "°C/s"}
  },
  "注塑": {
    "barrel_temperature": {"min": 200, "max": 300, "unit": "°C"},
    ...
  }
}
```

### 5.3 奖励函数实现

```python
def compute_reward(prompt, response, ground_truth, task_type):
    accuracy = compute_accuracy(response, ground_truth, task_type)
    format_score = compute_format(response)
    length_penalty = compute_length_penalty(response)

    return accuracy + format_score - length_penalty

def compute_accuracy(response, ground_truth, task_type):
    if task_type == "spc_detection":
        pred_violations = extract_violations(response)
        return f1_score(pred_violations, ground_truth["violations"])

    elif task_type == "numerical":
        pred_val = extract_number(response)
        if pred_val is None: return 0.0
        if abs(pred_val - ground_truth) < 0.01: return 1.0
        if abs(pred_val - ground_truth) < 0.1: return 0.5
        return 0.0

    elif task_type == "code":
        test_result = run_tests_in_sandbox(response)
        return test_result.pass_rate  # 0.0–1.0

    elif task_type == "process_param":
        return 1.0 if is_within_spec(response, ground_truth) else 0.0

def compute_format(response):
    score = 0.0
    if has_reasoning_steps(response): score += 0.2
    if has_standard_citation(response): score += 0.1
    if has_units_on_numbers(response): score += 0.1
    return score

def compute_length_penalty(response):
    tokens = count_tokens(response)
    if tokens > 2000 and not has_substantial_content(response):
        return 0.1
    return 0.0
```

### 5.4 GRPO 训练配置

**框架**：veRL（支持 vLLM 加速 rollout）

```yaml
algorithm: grpo
group_size: 8              # 每个 prompt 采样 8 个回答组成 group
clip_ratio: 0.2
kl_coef: 0.01              # 小 KL 约束，允许较大幅度更新

rollout:
  engine: vllm
  temperature: 0.8
  max_new_tokens: 2048
  gpu_memory_utilization: 0.6

training:
  base_model: qwen3-32b-dpo-v1
  finetune_mode: full       # RL 阶段必须全量微调
  learning_rate: 1e-6
  batch_size: 256           # 总 batch（含 group rollout）
  num_steps: 5000
  save_every: 500

compute:
  rollout_gpus: 16 × A100   # vLLM rollout
  train_gpus: 16 × A100     # 参数更新
```

**关键技术（MiMo v2-pro 经验）**：
- **测试难度分级奖励**：easy 题答对奖励 ×0.5，hard 题答对奖励 ×2.0，避免模型只刷简单题
- **简单问题重采样**：若当前 batch 所有回答都对/都错，跳过该 batch（无梯度信号）
- **上下文窗口渐进扩展**：训练前 2000 步用 1K context，后续扩展至 4K→8K

### 5.5 RL 训练监控指标

| 指标 | 监控频率 | 预警阈值 |
|------|----------|----------|
| reward_mean | 每 100 步 | 连续 500 步不增长 → 调整 LR |
| KL 散度 | 每 100 步 | >0.5 → 降低 kl_coef |
| GPQA 保留率 | 每 500 步 | 下降 >3% → 停止训练 |
| 幻觉率（数值） | 每 500 步 | >8% → 检查数据质量 |
| rollout 成功率 | 实时 | <50% → 检查生成配置 |

---

## 六、W4：VLM 工业视觉微调（M4–M9）

### 6.1 目标

基于 Qwen2.5-VL-72B，构建支持工业视觉场景的多模态工业模型，覆盖 P0（图片理解）和 P1（数据图表理解）优先级。

### 6.2 工业视觉数据采集与标注（125K 张）

| 数据类型 | 数量目标 | 采集方式 | 标注方式 | 负责人 |
|---------|---------|---------|---------|--------|
| 产品外观缺陷图像 | 50K 张 | 小米工厂产线相机（自动） | 缺陷类型+位置（标注团队） | 数据工程师+标注团队 |
| 设备状态照片 | 20K 张 | 现场工程师拍摄+历史存档 | 状态描述（领域专家） | 领域专家 |
| SPC 控制图截图 | 10K 张 | 自动生成（matplotlib） | 规则自动标注 | 数据工程师 |
| P&ID 管道仪表图 | 5K 张 | 设备厂商文档 PDF 提取 | 元素识别（标注团队） | 标注团队 |
| 仪表/屏幕读数 | 20K 张 | 现场拍摄+公开数据集 | 数值标注（半自动OCR+人工校正） | 标注团队 |
| 工业数据图表 | 15K 张 | MES 系统截图+自动生成 | 趋势描述（自动+人工） | 数据工程师 |
| 产线视频截帧 | 5K 张 | 产线监控视频抽帧 | 操作步骤（领域专家） | 领域专家 |

**数据脱敏规则**（统一执行）：
- 图像中的产品型号/序列号：模糊处理或覆盖
- 操作员面部：打码
- 内部编码/IP 地址：替换为占位符

**SPC 控制图自动生成**（10K 张，零人工成本）：
```python
import matplotlib.pyplot as plt

for i in range(10000):
    scenario, data, violations = generate_spc_scenario()
    fig = plot_control_chart(data, ucl, lcl, cl)
    add_violation_annotations(fig, violations)

    label = {
        "violations": violations,
        "cpk": calculate_cpk(data),
        "recommendation": lookup_recommendation(violations)
    }
    save_with_label(fig, label, f"spc_{i:05d}.png")
```

### 6.3 VLM 微调流程

**分两阶段微调**：

**阶段 A（M4–M6）：视觉指令微调**
```yaml
base_model: Qwen2.5-VL-72B
method: LoRA SFT
lora:
  r: 64
  target: visual_projection + llm_layers  # 同时微调视觉投影层和 LLM
training_data:
  - spc_chart_qa_10k       # SPC 图表问答
  - defect_detection_50k   # 缺陷检测（图+文）
  - meter_reading_20k      # 仪表读数
  - industrial_chart_15k   # 工业图表分析
total: 95K 条多模态样本
epochs: 2
learning_rate: 2e-5
```

**阶段 B（M7–M9）：工业视觉推理增强（RL-V）**
```yaml
method: GRPO（视觉版本）
reward_functions:
  - defect_classification_accuracy   # F1
  - meter_reading_error              # 误差 <5%
  - spc_violation_detection          # F1
  - pid_element_recognition          # 准确率
training_data: 可验证视觉推理题 30K 条
```

### 6.4 多模态评测

| 评测任务 | 样本数 | 评分指标 | 对比基线 |
|---------|--------|---------|---------|
| 外观缺陷分类 | 300 张 | F1 Score | Gemini 3.1 Pro API |
| 仪表读数识别 | 200 张 | MAE（误差率） | GPT-5.4 Vision |
| SPC 控制图分析 | 200 张 | 违规识别 F1 | 文本版工业模型 |
| P&ID 流程理解 | 100 张 | 元素识别准确率 | Qwen2.5-VL-72B（未微调） |

---

## 七、甘特图

```
月份    M3      M4      M5      M6      M7      M8      M9
       ├───────┼───────┼───────┼───────┼───────┼───────┤

W1 数据扩建
  数据管道搭建  ████
  问答对生成           ████████████
  Agent 轨迹           ████████
  SPC/代码生成  ████████████████
  工艺/故障            ████████████████
  质量审核             ████████████████
  数据集封版                           ██

W2 DPO 偏好优化
  偏好数据采集                  ████████
  DPO 训练                             ████████
  DPO 评测                                      ████

W3 RL-CoT
  RL 数据准备                          ████
  GRPO 训练                                    ████████
  RL 评测                                               ████

W4 VLM 视觉微调
  视觉数据采集         ████████████████
  阶段 A 微调                   ████████████
  阶段 B RL-V                                  ████████
  视觉评测                                              ████

评测（持续进行）
  Layer 1+2 自动评测   ████████████████████████████████
  Layer 3 Agent 仿真评测                        ████████
  竞品对比报告                         ██       ██      ██
```

---

## 八、里程碑与交付物

| 里程碑 | 时间 | 交付物 | 验收标准 |
|--------|------|--------|---------|
| **M1** 数据集 v2 封版 | M6 末 | 100K SFT 数据集（Parquet + DVC） | 专家抽检通过率 >90% |
| **M2** DPO 模型 | M7 末 | Qwen3-32B-DPO-Industrial-v1 权重 | 工厂问答满意度 ≥4.0/5 |
| **M3** RL 推理模型 | M8 末 | Qwen3-32B-RL-Industrial-v1 权重 | SPC 准确率 ≥82%，工艺参数推理 ≥78% |
| **M4** VLM 初版 | M8 末 | Qwen2.5-VL-Industrial-v1 权重 | 缺陷检测 F1 ≥0.80，仪表读数误差 <7% |
| **M5** Phase 2 综合评测报告 | M9 末 | 全量评测报告（Layer 1+2+3） | 工业专项提升 >30%，Agent 完成率 >85% |

---

## 九、资源消耗估算

### 9.1 算力

| 工作包 | GPU 需求 | 时长 | GPU·天 |
|--------|---------|------|--------|
| W1 数据生成（Qwen3-235B 推理） | 8×A100 | 30 天 | 240 |
| W2 DPO 训练（Qwen3-32B） | 8×A100 | 7 天 | 56 |
| W3 GRPO 训练（全量，Qwen3-32B） | 32×A100 | 14 天 | 448 |
| W4 VLM 阶段 A（Qwen2.5-VL-72B LoRA） | 16×A100 | 10 天 | 160 |
| W4 VLM 阶段 B（GRPO） | 16×A100 | 10 天 | 160 |
| 评测运行（持续） | 4×A100 | 60 天 | 240 |
| **合计** | — | — | **~1304 GPU·天** |

> 白皮书估算 32×A100 × 4周 = 896 GPU·天，本计划略高（含数据生成算力），在合理范围内。如需压缩，优先削减 W1 数据生成的 Qwen3-235B 用量，改用 Qwen3-32B。

### 9.2 存储

| 数据类别 | 估算大小 |
|---------|---------|
| 原始语料 | 5TB |
| 生成 SFT 数据（100K 条） | 50GB |
| 偏好对数据（20K 对） | 20GB |
| RL 训练数据（33K 条） | 30GB |
| 视觉数据（125K 张图） | 30TB |
| 模型权重（各版本） | 15TB |
| **合计** | **~50TB** |

### 9.3 人力工作日估算

| 角色 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | 合计（人天） |
|------|----|----|----|----|----|----|-----|------------|
| ML 工程师 × 2 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 280 |
| 数据工程师 × 1 | 20 | 20 | 20 | 20 | 10 | 10 | 10 | 110 |
| 领域专家 × 2 | 10 | 20 | 40 | 40 | 30 | 20 | 10 | 170 |
| 标注团队 × 2 | 20 | 40 | 40 | 40 | 20 | 10 | 10 | 180 |
| 评测工程师 × 1 | 10 | 10 | 20 | 20 | 20 | 20 | 20 | 120 |

---

## 十、风险与预案

| 风险 | 概率 | 影响 | 预案 |
|------|------|------|------|
| 工厂数据脱敏耗时超预期 | 高 | W1 延期 | M3 立即启动脱敏审批流程；准备公开数据集替代方案 |
| 领域专家标注产能不足 | 中 | W2 偏好数据减少 | 提前启动自动偏好构造（数值/代码类），减少专家依赖 |
| GRPO 训练不稳定（奖励崩溃） | 中 | W3 延期 | 备用方案：RLOO 算法替代；降低 rollout 温度 |
| Qwen2.5-VL-72B LoRA 显存不足 | 低 | W4 调整 | 降级至 Qwen2.5-VL-7B；或改用 QLoRA（4-bit） |
| 通用能力（GPQA）退化 >5% | 中 | 需重新训练 | 每 500 步自动评测通用能力，触发预警立即回滚 |
| 小米工厂视觉数据采集审批延迟 | 高 | W4 数据减少 | 优先用公开数据集（MVTec AD）+ 合成数据补充 |

---

## 附录：关键配置速查

### A. 训练命令（LLaMA-Factory SFT）

```bash
llamafactory-cli train \
  --stage sft \
  --model_name_or_path Qwen/Qwen3-32B \
  --dataset industrial_sft_100k \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 128 \
  --lora_alpha 256 \
  --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
  --output_dir ./output/qwen3-32b-sft-v2 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --bf16 true \
  --flash_attn fa2 \
  --save_steps 500 \
  --logging_steps 50
```

### B. 数据格式标准（Alpaca 格式）

```json
{
  "instruction": "以装备工程师角色分析以下 SPC 控制图数据...",
  "input": "过去 25 个采样点数据：[...], UCL=10.5, LCL=9.5, CL=10.0",
  "output": "<think>\n分析 Nelson 规则...\n</think>\n\n根据 SPC 控制图分析...",
  "system": "你是一名专业的装备工程师，擅长统计过程控制分析。",
  "history": []
}
```

### C. 评测自动化 CI（每次模型检查点触发）

```bash
#!/bin/bash
# eval_checkpoint.sh
MODEL=$1

# Layer 1: 通用能力
python eval/run_gpqa.py --model $MODEL --output results/gpqa_${MODEL}.json
python eval/run_livecode.py --model $MODEL --output results/lcode_${MODEL}.json

# Layer 2: 工业专项
python eval/run_spc.py --model $MODEL --output results/spc_${MODEL}.json
python eval/run_process_param.py --model $MODEL --output results/pp_${MODEL}.json
python eval/run_fault_diag.py --model $MODEL --output results/fd_${MODEL}.json

# 对比基线
python eval/compare_baseline.py --current $MODEL --baseline qwen3-32b-base \
  --report results/report_${MODEL}.md

# 预警检查
python eval/check_regression.py --report results/report_${MODEL}.md \
  --threshold_gpqa -0.03 --threshold_spc +0.15
```
