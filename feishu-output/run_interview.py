#!/usr/bin/env python3
"""上传面试问答文档到飞书。运行：
FEISHU_APP_ID=... FEISHU_APP_SECRET=... python3 feishu-output/run_interview.py
"""
import sys, os, json
SKILL_DIR = os.path.expanduser("~/.claude/skills/feishu-doc")
sys.path.insert(0, os.path.join(SKILL_DIR, "scripts"))
from feishu_api import FeishuAuth, process_blocks

with open(os.path.join(SKILL_DIR, "config.json")) as f:
    config = json.load(f)

blocks = [
    {"type": "document_title", "text": "技能内化项目面试问答"},
    {"type": "text", "elements": [
        {"text": "项目：SPC Nelson 规则 LLM 内化（Route 2.1.1，Round 1–3）", "bold": True},
    ]},
    {"type": "text", "elements": [{"text": "最后更新：2026-05-11"}]},
    {"type": "divider"},

    # Q1
    {"type": "heading", "level": 2, "text": "Q1：这个项目做了什么，解决了什么问题？"},
    {"type": "text", "elements": [
        {"text": "背景：", "bold": True},
        {"text": "公司有 40+ 篇超长技能文档（SPC/质检/工艺规则），每次推理都要把文档塞进上下文，Token 成本极高，且有文档泄露风险。"},
    ]},
    {"type": "text", "elements": [
        {"text": "方案：", "bold": True},
        {"text": '通过监督微调（SFT）将技能文档"烧入"模型权重，使模型推理时无需任何文档，直接从权重中调用技能。'},
    ]},
    {"type": "text", "elements": [
        {"text": "验证任务：", "bold": True},
        {"text": "SPC Nelson 规则检测。给定 25 个工序数据点，判断哪些 Nelson 规则被触发（8 条规则的多标签分类），并计算过程能力指数 CPK。"},
    ]},
    {"type": "text", "elements": [
        {"text": "核心结论：", "bold": True},
        {"text": "SFT 14B 模型（无技能文档推理）在规则检测 F1 上超越 Claude Sonnet + 有技能文档（0.420 vs 0.266），CPK 提取率从 62% 提升至 100%。SFT 的核心优势不在规则知识量，而在格式一致性——输出格式对齐评测标准，可程序化解析。"},
    ]},
    {"type": "divider"},

    # Q2
    {"type": "heading", "level": 2, "text": "Q2：训练数据是怎么生成的？整个流水线是什么样子的？"},
    {"type": "text", "elements": [{"text": "两阶段流水线：", "bold": True}]},
    {"type": "heading", "level": 3, "text": "阶段 1：输入场景生成器"},
    {"type": "text", "elements": [{"text": "用随机种子程序化生成多样化输入，覆盖规则触发的各种组合。训练集和测试集严格使用不同种子池。"}]},
    {"type": "code", "language": "python", "content": "rng = random.Random(seed)\nmu    = rng.uniform(45, 55)       # 随机过程均值\nsigma = rng.uniform(2, 5)         # 随机标准差\ndata  = [rng.gauss(mu, sigma) for _ in range(25)]  # 25 个数据点\n# 程序注入目标规则触发条件（如让第 8–16 点偏向均值同侧 → 触发 rule2）"},
    {"type": "heading", "level": 3, "text": "阶段 2：教师模型调用（with_skill 格式）"},
    {"type": "code", "language": "plain", "content": "System: [完整技能文档，~1200 tokens，含 Nelson 8 条规则定义 + CPK 公式]\nUser:   [阶段 1 生成的数据场景]\nAssistant: <think>\n  逐点分析数据，依次检查 rule1–rule8 触发条件，\n  计算均值/标准差/连续点数……\n</think>\n{\"violations\": [\"rule2\", \"rule5\"], \"cpk\": 1.23, \"report\": \"...\"}"},
    {"type": "heading", "level": 3, "text": "阶段 3：质量验证"},
    {"type": "text", "elements": [{"text": "每条样本用规则引擎（Python 确定性计算）验证答案正确性，不合格直接丢弃（约 5-10%）。"}]},
    {"type": "heading", "level": 3, "text": "阶段 4：格式转换（关键）"},
    {"type": "code", "language": "python", "content": "def convert_to_noskill(example):\n    messages[0][\"content\"] = \"你是一名 SPC 工程师。\"  # 移除技能文档\n    return example"},
    {"type": "text", "elements": [
        {"text": "模型必须从 <think> 推理链中学习规则，不能依赖 system 中的文档。"},
        {"text": "with_skill 格式训练的模型推理时无文档 F1 接近 0。", "bold": True},
    ]},
    {"type": "divider"},

    # Q3
    {"type": "heading", "level": 2, "text": "Q3：如何微调？LoRA 参数怎么理解？"},
    {"type": "table",
     "headers": ["参数", "值", "含义"],
     "rows": [
         ["lora_rank", "128", "低秩矩阵的秩 r，控制可训练参数量和表达能力"],
         ["lora_alpha", "256", "缩放系数，实际更新幅度 = alpha/rank = 2.0"],
         ["lora_target", "all", "对所有线性层挂载：Q/K/V/O + FFN up/gate/down"],
         ["cutoff_len", "5120", "最关键配置，必须 ≥ 最大输出长度"],
         ["learning_rate", "1e-4", "1100条样本规模的优化值，低于此（5e-5）F1下降"],
     ]},
    {"type": "text", "elements": [
        {"text": "LoRA 数学原理：", "bold": True},
        {"text": "原始权重 W 冻结，添加 ΔW = (alpha/rank) × B·A。B 是 d×r 矩阵，A 是 r×k 矩阵，r=128 << min(d,k)，可训练参数量约为全参的 1.7%。"},
    ]},
    {"type": "divider"},

    # Q4
    {"type": "heading", "level": 2, "text": "Q4：cutoff_len 是什么？为什么这么重要？"},
    {"type": "text", "elements": [
        {"text": "静默训练失败最常见的来源。", "bold": True},
        {"text": "截断不报错，loss 看起来也在下降，但输出尾部的规则/步骤从未被训练。"},
    ]},
    {"type": "table",
     "headers": ["设置", "后果"],
     "rows": [
         ["cutoff_len=4096（Round 1 的错误设置）", "SPC 输出均值 4191 tokens，94% 样本被截断，rule8 召回=0.00"],
         ["cutoff_len=5120（Round 2 修复）", "0% 截断，rule8 恢复正常，F1 从 0.358 → 0.430"],
     ]},
    {"type": "code", "language": "python", "content": "# 上训练前必做\nlengths = [len(tokenizer(ex[\"output\"])[\"input_ids\"]) for ex in data]\nprint(f\"max={max(lengths)}, p95={np.percentile(lengths,95):.0f}\")\n# cutoff_len >= max(lengths)"},
    {"type": "divider"},

    # Q5
    {"type": "heading", "level": 2, "text": "Q5：如何评测？指标是什么？"},
    {"type": "text", "elements": [
        {"text": "始终无技能文档评测：", "bold": True},
        {"text": 'system_prompt = "你是一名 SPC 工程师。"（不含任何规则）'},
    ]},
    {"type": "table",
     "headers": ["指标", "含义", "Round 3 最优"],
     "rows": [
         ["rule_detection_F1", "规则预测集合 macro F1（主指标）", "0.420"],
         ["per_rule_recall", "每条规则单独召回率", "rule7 最低 0.333"],
         ["CPK_found_rate", "CPK 值可提取比例", "1.000"],
         ["CPK_MAE", "CPK 数值误差", "<0.05"],
     ]},
    {"type": "text", "elements": [{"text": "F1 计算：多标签集合匹配，TP=预测正确的规则数，FP=多报，FN=漏报。Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2PR/(P+R)。"}]},
    {"type": "divider"},

    # Q6
    {"type": "heading", "level": 2, "text": "Q6：GRPO 是怎么做的？样本什么样子？怎么打分？"},
    {"type": "heading", "level": 3, "text": "训练样本格式"},
    {"type": "text", "elements": [{"text": "GRPO 输入是纯问题（无答案），不携带技能文档："}]},
    {"type": "code", "language": "plain", "content": 'System: "你是一名 SPC 工程师。"   ← 无技能文档，测量内化程度\nUser:   "请分析以下采样数据，判断是否触发 Nelson 规则，计算 CPK：\n         数据：[45.2, 46.8, 47.3, ...]\n         USL=55.0, LSL=45.0"'},
    {"type": "text", "elements": [{"text": "模型为每条 prompt 采样 G=4 个候选输出，每个候选是完整的推理链+答案："}]},
    {"type": "code", "language": "plain", "content": '候选 1: <think>逐点分析 rule1...</think> {"violations": ["rule1"], "cpk": 1.23}\n候选 2: <think>分析后认为无异常...</think> {"violations": [], "cpk": 1.20}\n候选 3: ...\n候选 4: ...'},
    {"type": "heading", "level": 3, "text": "打分（Reward 函数）"},
    {"type": "text", "elements": [{"text": "每个候选用程序化 Reward 函数打分，最高约 1.15 分："}]},
    {"type": "code", "language": "python", "content": "def reward_fn(prompts, completions, ground_truth=None, **kwargs):\n    for completion, gt in zip(completions, ground_truth):\n        # 1. 主奖励：规则检测 F1（0~1.0）\n        pred_violations = extract_violations(completion)  # 正则提取 rule1~rule8\n        rule_f1 = compute_f1(pred_violations, gt[\"violations\"])\n\n        # 2. CPK 精度奖励（0~0.1）\n        pred_cpk = extract_cpk(completion)\n        if pred_cpk and abs(pred_cpk - gt[\"cpk\"]) < 0.1:\n            cpk_bonus = 0.1\n        elif pred_cpk:\n            cpk_bonus = 0.05  # cpk 存在但不精确\n\n        # 3. 格式奖励（0~0.05）\n        format_bonus = 0.05 if re.search(r'\\brule[1-8]\\b', completion) else 0.0\n\n        total = rule_f1 + cpk_bonus + format_bonus  # 最高约 1.15"},
    {"type": "heading", "level": 3, "text": "策略梯度更新"},
    {"type": "code", "language": "plain", "content": "advantage_i = reward_i - mean(rewards_for_this_prompt)\n# 同一 prompt 的 4 个候选互相比较：高于均值的候选被强化，低于均值的被抑制\n# KL 惩罚（beta=0.01）防止偏离参考模型太远"},
    {"type": "heading", "level": 3, "text": "原始实验失败原因（四个 bug）"},
    {"type": "table",
     "headers": ["Bug", "现象", "原因", "修复"],
     "rows": [
         ["max_completion_length=2048", "clipped_ratio=1.0，reward=0，零学习", "SPC 答案均值 4191 tokens，远超限制，全截断", "改为 5000（覆盖 p99+10%）"],
         ["使用 LlamaFactory 框架", "进程无法启动", "LlamaFactory 0.9.3 依赖损坏，无法 import", "改用 trl 1.3.0 GRPOTrainer"],
         ["输入含技能文档（with_skill）", "测量的是有文档性能，非内化", "prepare_grpo_dataset.py 未清除 system", "加载时强制替换为 no_skill system prompt"],
         ["1460 样本 × 8 候选", "单步 280s，预计 115h", "生成量过大：8×5500 tokens×1460 prompts", "分层采样 171 样本 + num_generations=4，降至约 8h"],
     ]},
    {"type": "divider"},

    # Q7
    {"type": "heading", "level": 2, "text": "Q7：有哪些关键发现和反直觉结论？"},
    {"type": "ordered_list", "items": [
        "数据质量 > 数据数量：去截断（-49条，F1+9.7%）比加 640 条新数据（F1 几乎不变）更有效",
        '上采样对反直觉规则无效：rule7（"过于稳定=异常"）3× 上采样无改善；正负对比样本对才有效',
        "量化损失远超预期：int4/int8 在 SPC 精确数值任务上损失 7-10%（一般文本任务 <2%）",
        "技能文档在 Claude 中反而降低 F1：有文档时 Claude 输出冗长报告，JSON 解析失败 38% vs 14%，净 F1 更低",
        "提取器偏差掩盖真实性能：用 SFT 专用 regex 提取器测 Claude，0.110；换 JSON 提取器，0.424；SFT 优势在格式可靠性而非规则知识量",
    ]},
    {"type": "divider"},

    # Q8
    {"type": "heading", "level": 2, "text": "Q8：模型选型和量化怎么做决策的？"},
    {"type": "table",
     "headers": ["配置", "F1", "推理显存", "综合得分"],
     "rows": [
         ["14B bf16 LoRA（推荐，精度优先）", "0.420", "96GB", "0.093"],
         ["14B QLoRA int4（显存优先）", "0.394（-6.2%）", "~20GB", "0.427"],
         ["14B QLoRA int8", "0.391（-6.9%）", "~20GB", "0.428"],
         ["32B QLoRA int4", "0.379（-9.8%）", "~45GB", "0.085"],
     ]},
    {"type": "text", "elements": [
        {"text": "结论：", "bold": True},
        {"text": "32B int4 不推荐——更大参数量被量化损失完全抵消，F1 反而低于 14B bf16。若显存受限，接受 6% 损失用 14B int4。"},
    ]},
    {"type": "divider"},

    # Q9
    {"type": "heading", "level": 2, "text": "Q9：退化率是什么？SFT 对通用能力有多大影响？"},
    {"type": "text", "elements": [{"text": "退化率 = SFT 后通用基准平均得分下降幅度（gsm8k/hellaswag/mmlu 等 6 项）。"}]},
    {"type": "table",
     "headers": ["实验", "F1", "退化率", "主要来源"],
     "rows": [
         ["R3-AB-v2（14B bf16）", "0.420", "3.08%", "gsm8k -18.7%（数学推理冲击最大）"],
         ["R3-D1（14B int4）", "0.394", "2.33%", "gsm8k -14.5%"],
         ["R3-D2（32B int4）", "0.379", "0.45%", "极小，均匀分布，大模型通用能力更鲁棒"],
     ]},
    {"type": "divider"},

    # Q10
    {"type": "heading", "level": 2, "text": "Q10：如果 F1 还是达不到目标（0.50），下一步怎么做？"},
    {"type": "heading", "level": 3, "text": "P0（直接提升 F1）"},
    {"type": "ordered_list", "items": [
        "rule7 对比样本对：正例（15点全在±1σ内→触发）+ 负例（第15点略超出→不触发）成对训练。上采样无效，需要对比学习。",
        "数据质量审核：人工审查 R3-AB-v2，对截断样本补充完整推理链。",
        "GRPO 迭代（Round 4）：在修正四个 bug 后的 R3-C 结果基础上继续强化。",
    ]},
    {"type": "heading", "level": 3, "text": "P1（补齐维度）"},
    {"type": "ordered_list", "items": [
        "72B 大模型：Qwen3-72B，8 卡 ZeRO-3，验证参数量上界。",
        "反直觉规则知识注入：推理链中显式写入'连续15点全在1sigma内概率约0.23%，统计异常'，让模型从原理层理解 rule7。",
    ]},
    {"type": "divider"},
    {"type": "text", "elements": [
        {"text": "基于 Route 2.1.1 Round 1–3 全部实验（20+ 次训练）| 2026-05-11"},
    ]},
]

auth = FeishuAuth()
results = process_blocks(
    auth,
    blocks,
    default_owner=config.get("default_owner") or config.get("default_owner_email") or None,
)
print(json.dumps(results, ensure_ascii=False, indent=2))
