---
name: phase2.1 关键技术决策
description: 实现过程中的关键设计选择，含原因和边界条件
type: project
---

# 关键技术决策

## 1. sigma 使用 sample stdev 而非 process sigma

**决策**：`check_nelson_rules(data, CL, statistics.stdev(data))` — 用样本标准差，CL 用输入字段

**Why**：route2.1.md §2.1 示例中 UCL = CL + 3s（s 为样本 stdev），若用注入时的 process sigma，
quality_gate 重算时 sigma 不一致，通过率只有74%。修正后通过率提升到98%。

**How to apply**：generator、quality_gate、spc_eval 三处都用 `statistics.stdev(data)`。
注意：data 存储时已 round(2)，round 误差导致约1~2% 样本 stdev 略有差异，这是可接受的。

## 2. 处置建议使用 ① 字符，避免 Python 三元运算符优先级陷阱

**决策**：`[f"{_NUM[idx]} {item}" for idx, item in enumerate(items)]`

**Why**：最初用 `f"① " if idx==1 else ... + item` 写法，Python 三元运算符优先级低于 +，
导致 idx==1 时只输出 "① " 而 item 内容丢失。改为列表索引方式。

**How to apply**：格式化处置建议时统一用 `_NUM = ["①","②","③","④","⑤"]` 索引。

## 3. 规则注入后用引擎重算（以引擎为准，不是注入列表）

**决策**：注入是"hint"，ground_truth = 引擎重算结果

**Why**：注入某规则的同时可能意外触发其他规则（例如 rule2 注入使9点在同侧，
若相邻点也在同侧则 rule6 也会触发），用注入列表作 ground_truth 会产生误标注。

**How to apply**：任何修改 data 的操作后都必须重跑 `check_nelson_rules()`。

## 4. formatter 模板优先，LLM 润色可选

**决策**：默认模板生成，设置 ANTHROPIC_AUTH_TOKEN 后可选 LLM 润色

**Why**：数据生成需要批量高速运行（500条），LLM 调用慢且有成本。
模板生成确保格式一致性（必含 x̄/s/CPU/CPL/CPK 四中间值），LLM 只做润色不改结构。

**How to apply**：生成数据时默认不传 --use_llm。最终训练集可用 --use_llm 润色20%样本。

## 5. vLLM 部署使用 enable_thinking（Qwen3 系列）

**决策**：`--enable-reasoning --reasoning-parser deepseek_r1`

**Why**：Qwen3 原生支持 `<think>` 推理模式，与训练数据格式一致。
不开启则模型输出格式与训练数据不匹配，影响规则检测。

**How to apply**：deploy_vllm.py 默认 enable_thinking=True，基座评测也要用相同参数。

## 6. LoRA rank=128 而非默认 rank=8/16

**Why**：SPC 需要学8条规则+公式+推理链格式，属于"新知识"注入，
rank=64容量不足，rank=256在500条数据上过拟合。rank=128是经验最优。

## 7. 项目文件存放在 /home/yinrong/phase2.1/（持久化）

**Why**：/home/yinrong/ 是持久化存储，/tmp 在 session 结束后可能丢失。
phase2.1/ 目录初始为 root 所有（drwxr-xr-x），需要用户先执行 `chmod 777`。
