#!/usr/bin/env python3
"""审查 blocks 列表无误后，运行：python3 feishu-output/run.py"""
import sys, os, json
SKILL_DIR = os.path.expanduser("~/.claude/skills/feishu-doc")
sys.path.insert(0, os.path.join(SKILL_DIR, "scripts"))
from feishu_api import FeishuAuth, process_blocks

with open(os.path.join(SKILL_DIR, "config.json")) as f:
    config = json.load(f)

# ============ 文档内容（审查这部分） ============
blocks = [
    {"type": "document_title", "text": "技能内化方法论 v2：通用框架"},
    {"type": "text", "elements": [
        {"text": "适用范围：将任意结构化专业技能编入语言模型权重", "bold": True},
    ]},
    {"type": "text", "elements": [{"text": "版本：v2.0（2026-05-09）"}]},
    {"type": "divider"},

    # 一、核心概念
    {"type": "heading", "level": 2, "text": "一、核心概念"},
    {"type": "heading", "level": 3, "text": '什么是"技能内化"'},
    {"type": "table",
     "headers": ["术语", "定义"],
     "rows": [
         ["技能（Skill）", "一套结构化的程序性知识——规则、判断树、操作流程——使智能体能执行特定领域任务"],
         ["内化（Internalization）", "模型推理时不依赖外部技能文档，直接从权重中调用技能"],
         ["教师数据（Teacher Data）", "由强模型（如 Claude）结合技能文档生成的高质量推理链样本"],
     ]},
    {"type": "heading", "level": 3, "text": "与其他方案的本质区别"},
    {"type": "table",
     "headers": ["方案", "推理时技能文档", "适用场景", "局限"],
     "rows": [
         ["RAG / 上下文注入", "每次携带", "文档频繁更新", "Token 成本高、有泄露风险"],
         ["工具调用", "按需查询", "需要实时数据", "依赖外部基础设施"],
         ["技能内化（SFT）", "不需要", "规则稳定、高频调用", "技能变更需重训"],
         ["GRPO 强化", "不需要", "有明确奖励信号", "需设计 reward 函数"],
     ]},
    {"type": "divider"},

    # 二、技能类型分类
    {"type": "heading", "level": 2, "text": "二、技能类型分类"},
    {"type": "text", "elements": [{"text": "不同技能类型决定了训练数据格式设计策略："}]},
    {"type": "table",
     "headers": ["类型", "特征", "示例", "内化策略", "内化上限"],
     "rows": [
         ["A：纯规则/算法型", "输入→确定性输出，无需外部信息", "SPC 规则、ICD 编码、合规检查", "完全内化", "接近 100%"],
         ["B：知识密集型", "依赖大量领域事实，规则稳定", "法律条文适用、医学诊断", "规则内化 + RAG 补 facts", "中等"],
         ["C：工具调用型", "核心逻辑可内化，部分步骤需工具", "数据分析（读 CSV）、代码执行", '内化"何时/如何用工具"的框架', "高（框架），低（结果依赖）"],
         ["D：实时数据型", "结果依赖当前状态，不可预先学习", "股价查询、实时库存", "仅内化查询策略和结果解读", "低（必须保留工具调用）"],
     ]},
    {"type": "quote", "text": '原则：技能中"哪部分是规则/逻辑"可以内化；"哪部分是实时数据"必须保留工具调用。'},
    {"type": "divider"},

    # 三、通用内化流程
    {"type": "heading", "level": 2, "text": "三、通用内化流程"},
    {
        "type": "board",
        "title": "技能内化通用流程",
        "plantuml": "@startuml\n[技能文档设计] --> [教师数据生成]\n[教师数据生成] --> [格式转换（关键）]\n[格式转换（关键）] --> [微调训练]\n[微调训练] --> [评测验证]\n[评测验证] --> [迭代优化]\n[迭代优化] --> [教师数据生成] : 未达标则迭代\n@enduml",
    },
    {"type": "divider"},

    # 四、第一步：技能文档设计
    {"type": "heading", "level": 2, "text": "四、第一步：技能文档设计"},
    {"type": "text", "elements": [{"text": "技能文档是教师数据生成的唯一依据，仅在训练阶段使用："}]},
    {"type": "table",
     "headers": ["必要元素", "说明"],
     "rows": [
         ["规则枚举", "所有决策规则，精确、完整，无歧义"],
         ["触发条件", '每条规则的输入条件（精确定义，避免"通常"等模糊词）'],
         ["应用步骤", "输入→判断→输出的完整流程"],
         ["输出格式", "正确答案的样例（字段名、格式、枚举值）"],
     ]},
    {"type": "heading", "level": 3, "text": "类型 A（纯规则型）文档模板"},
    {"type": "code", "language": "plain", "content": "===== 技能：[技能名称] =====\n一、规则列表\n  rule_id | 触发条件（精确定义） | 含义\n  ...\n\n二、判断步骤\n  Step 1: 计算基础参数\n  Step 2: 逐条检查规则\n  Step 3: 输出格式：{\"rules\": [...], \"result\": ...}\n\n三、边界说明（重要）\n  - 规则 X 与规则 Y 的区别：...\n  - 恰好在边界的处理方式：...\n========================="},
    {"type": "heading", "level": 3, "text": "类型 C（工具调用型）文档模板"},
    {"type": "code", "language": "plain", "content": "===== 技能：[技能名称] =====\n一、可用工具\n  - tool_name: 调用时机、入参格式、出参解读\n\n二、决策框架\n  满足条件 A → 调用 tool_1 → 根据结果判断 ...\n  满足条件 B → 直接推理（不需要工具）\n\n三、输出格式\n  ...\n========================="},
    {"type": "quote", "text": "技能文档验证：批量生成训练数据前，先用 3–5 个样本做试调用，确认教师模型能够严格按文档格式输出。文档中的歧义或格式不清晰会在此阶段暴露。"},
    {"type": "divider"},

    # 五、第二步：教师数据生成
    {"type": "heading", "level": 2, "text": "五、第二步：教师数据生成"},
    {"type": "text", "elements": [{"text": "教师数据生成是两阶段流水线：先用输入场景生成器制造多样化输入，再用教师模型调用技能文档生成推理链。"}]},
    {"type": "heading", "level": 3, "text": "阶段 1：输入场景生成器"},
    {"type": "text", "elements": [{"text": "用随机种子控制多样性——每个种子生成一个独立的输入场景。生成器的输出会被格式化为 User 消息，连同技能文档一起送进教师模型。不同技能类型的生成策略和 User 消息结构："}]},
    {"type": "heading", "level": 3, "text": "类型 A：纯规则/算法型"},
    {"type": "text", "elements": [{"text": "数据完全由代码程序化生成，不依赖任何外部资源："}]},
    {"type": "code", "language": "python", "content": "# 生成过程（代码）\nmu, sigma = rng.uniform(45, 55), rng.uniform(2, 5)\ndata = [rng.gauss(mu, sigma) for _ in range(25)]\n# 程序注入目标规则触发条件（如让第8-16点同侧）\n\n# 最终 User 消息\n请分析以下数据点，判断是否触发规则：\n数据：[45.2, 46.8, 47.3, ...]\nUSL=55.0, LSL=45.0"},
    {"type": "heading", "level": 3, "text": "类型 B：知识密集型"},
    {"type": "text", "elements": [{"text": "从已有语料库（合同库、案例库、知识库）采样真实文本片段，无需生成。若无现成语料库，可用 LLM 预先批量生成多样化场景文本存入语料库后再采样："}]},
    {"type": "code", "language": "python", "content": "# 从语料库随机抽取一段合同文本\ntext = corpus.sample(rng)\n\n# 最终 User 消息\n请审查以下合同条款是否合规：\n[条款文本原文，来自语料库]"},
    {"type": "heading", "level": 3, "text": "类型 C：工具调用型"},
    {"type": "text", "elements": [{"text": "合成用户问题 + 预构造工具响应。工具响应可以是真实调用结果，也可以是程序生成的模拟数据："}]},
    {"type": "code", "language": "python", "content": "# 生成一个需要工具的问题场景\nquestion = f\"分析 {rng.choice(datasets)} 中 Q{rng.randint(1,4)} 的趋势\"\ntool_response = generate_mock_csv_data(rng)   # 预构造工具返回\n\n# 最终 User 消息（含预设工具交互）\n分析 sales_2025.csv 中 Q3 的销售趋势\n[TOOL_RESULT 在 Assistant 侧展开]"},
    {"type": "heading", "level": 3, "text": "类型 D：实时数据型"},
    {"type": "text", "elements": [{"text": "从历史数据库中采样快照，将实时数据\"固化\"为训练样本："}]},
    {"type": "code", "language": "python", "content": "# 从历史记录中采样一条快照\nsnapshot = history_db.sample(rng)\n\n# 最终 User 消息\n当前库存快照（2025-03-15 14:30）：\nSKU-123：剩余 15 件，安全库存 50 件，补货周期 7 天\n请判断是否需要触发补货流程。"},
    {"type": "text", "elements": [
        {"text": "训练集与测试集"},
        {"text": "必须使用不同种子池", "bold": True},
        {"text": "，避免数据泄露。"},
    ]},
    {"type": "heading", "level": 3, "text": "阶段 2：教师模型调用（通用训练格式 with_skill）"},
    {"type": "text", "elements": [{"text": "将阶段 1 的场景 + 完整技能文档组合为 prompt，调用强模型（Claude Sonnet/Opus）生成带推理链的训练样本："}]},
    {"type": "code", "language": "plain", "content": "System: <技能文档——所有规则，完整细节>\nUser:   <任务场景 + 输入数据（来自阶段 1）>\nAssistant: <think>\n  [Step 1: 解析输入]\n  [Step 2: 逐条应用规则/调用工具]\n  [Step 3: 综合判断]\n</think>\n<结构化输出，与评测格式严格一致>"},
    {"type": "heading", "level": 3, "text": "阶段 3：质量验证"},
    {"type": "text", "elements": [
        {"text": "生成后必须用规则引擎或 oracle 验证答案正确性，不合格样本直接丢弃，不入训练库。"},
    ]},
    {"type": "table",
     "headers": ["参数", "推荐值", "说明"],
     "rows": [
         ["样本量 N", "200–1000", "从 200 开始，不足再扩"],
         ["并发数", "3–5", "API 限速与效率的平衡点"],
         ["随机种子", "与测试集隔离", "避免数据泄露"],
     ]},
    {"type": "heading", "level": 3, "text": "工具调用型技能的特殊处理"},
    {"type": "code", "language": "plain", "content": "<think>\n  分析输入，判断需要调用 [tool_name]\n  入参：{\"param\": \"value\"}\n</think>\n[TOOL_CALL]: tool_name({\"param\": \"value\"})\n[TOOL_RESULT]: {\"output\": ...}\n<think>\n  根据工具结果继续推理...\n</think>\n<最终输出>"},
    {"type": "text", "elements": [
        {"text": "内化后，模型学会的是："},
        {"text": "何时调用、传什么参数、如何解读结果", "bold": True},
        {"text": "，而非工具本身的逻辑。"},
    ]},
    {"type": "divider"},

    # 六、第三步：格式转换
    {"type": "heading", "level": 2, "text": "六、第三步：格式转换（关键步骤）"},
    {"type": "text", "elements": [
        {"text": "将系统提示中的技能文档移除", "bold": True},
        {"text": "，仅保留通用任务描述："},
    ]},
    {"type": "code", "language": "python", "content": "def convert_to_noskill(example):\n    messages = example[\"messages\"]\n    messages[0][\"content\"] = \"\"   # 移除技能文档，保留空字符串或通用描述\n    return example"},
    {"type": "heading", "level": 3, "text": "为什么这步是关键"},
    {"type": "table",
     "headers": ["训练格式", "推理时行为", "无技能文档 F1"],
     "rows": [
         ["with_skill（含文档训练）", "依赖文档，无文档则崩溃", "0.00–0.05"],
         ["no_skill（无文档训练）", "从权重调用技能", "0.35–0.43"],
         ["混合 50/50", "部分内化，不稳定", "0.20–0.30"],
     ]},
    {"type": "heading", "level": 3, "text": "各类型技能的格式转换策略"},
    {"type": "table",
     "headers": ["技能类型", "转换策略"],
     "rows": [
         ["纯规则型", "直接移除技能文档"],
         ["知识密集型", "移除规则文档，保留通用领域描述"],
         ["工具调用型", "移除技能文档，但保留工具声明（函数签名保留，规则文档移除）"],
         ["实时数据型", "保留工具声明 + 数据解读指引"],
     ]},
    {"type": "divider"},

    # 七、第四步：微调配置
    {"type": "heading", "level": 2, "text": "七、第四步：微调配置"},
    {"type": "heading", "level": 3, "text": "LoRA 核心参数"},
    {"type": "table",
     "headers": ["参数", "推荐值", "含义"],
     "rows": [
         ["lora_rank", "64–128", "低秩矩阵的秩 r，决定可训练参数量"],
         ["lora_alpha", "2× rank", "缩放系数，实际缩放倍数 = alpha/rank"],
         ["lora_target", "all", "对全部线性层（QKV、FFN）挂载 LoRA"],
         ["bf16", "true", "bfloat16 精度，H20/A100 标准选择"],
     ]},
    {"type": "heading", "level": 3, "text": "最重要的配置：cutoff_len"},
    {"type": "text", "elements": [
        {"text": "静默训练失败最常见的来源。", "bold": True},
    ]},
    {"type": "code", "language": "python", "content": "# 上训练前必做：检查输出长度分布\nfrom transformers import AutoTokenizer\nimport numpy as np\ntokenizer = AutoTokenizer.from_pretrained(model_path)\nlengths = [len(tokenizer(ex[\"output\"])[\"input_ids\"]) for ex in data]\nprint(f\"max={max(lengths)}, p95={np.percentile(lengths, 95):.0f}\")\n# 设置 cutoff_len >= max(lengths)"},
    {"type": "table",
     "headers": ["cutoff_len 设置", "后果"],
     "rows": [
         ["低于输出最大长度", "推理链被截断，输出末尾内容从未被训练"],
         ["等于 p95", "5% 样本截断，长推理链规则/步骤丢失"],
         ["≥ p100", "正常，所有样本完整训练"],
     ]},
    {"type": "heading", "level": 3, "text": "推荐超参数（通用起点）"},
    {"type": "code", "language": "yaml", "content": "num_train_epochs: 3–5\nper_device_train_batch_size: 2\ngradient_accumulation_steps: 4\nlearning_rate: 1.0e-4\nlr_scheduler_type: cosine\nwarmup_ratio: 0.05\ncutoff_len: max_output_tokens × 1.2  # 留 20% 余量"},
    {"type": "divider"},

    # 八、第五步：评测协议
    {"type": "heading", "level": 2, "text": "八、第五步：评测协议"},
    {"type": "heading", "level": 3, "text": "测试集生成"},
    {"type": "text", "elements": [{"text": "与训练数据使用同一套输入场景生成器，但使用独立的随机种子池（两者不重叠）。测试集规模建议 ≥ 200 条，对应 F1 置信区间约 ±0.01。"}]},
    {"type": "code", "language": "python", "content": "# 训练集种子：0–999，测试集种子：10000–10199（不重叠）\ntrain_data = generate(seeds=range(0, 1000))\ntest_data  = generate(seeds=range(10000, 10200))"},
    {"type": "heading", "level": 3, "text": "始终无技能文档评测"},
    {"type": "code", "language": "python", "content": "# 始终无技能文档评测\nsystem_prompt = \"\"   # 或通用任务描述，不含任何规则"},
    {"type": "heading", "level": 3, "text": "评测指标设计原则"},
    {"type": "table",
     "headers": ["维度", "通用指标", "含义"],
     "rows": [
         ["任务准确性", "F1 / Exact Match / 准确率", "根据任务类型选择"],
         ["格式可用性", "结构化输出提取率", "答案能否被程序解析"],
         ["稳定性", "多次采样方差", "生产环境可靠性"],
         ["通用能力保留", "退化率（6项基准均值）", "SFT 对通用能力的影响"],
     ]},
    {"type": "heading", "level": 3, "text": "指标选型建议"},
    {"type": "table",
     "headers": ["任务类型", "主指标", "辅助指标"],
     "rows": [
         ["多标签分类（如规则检测）", "macro F1", "per-class recall、Exact Match"],
         ["生成+提取（如报告+数值）", "F1 + 提取率", "数值 MAE"],
         ["工具调用正确性", "工具选择准确率", "参数正确率"],
         ["开放式推理", "人工评分 / LLM-as-Judge", "ROUGE / BERTScore"],
     ]},
    {"type": "divider"},

    # 九、第六步：分析与迭代
    {"type": "heading", "level": 2, "text": "九、第六步：分析与迭代"},
    {"type": "heading", "level": 3, "text": "通用诊断表"},
    {"type": "table",
     "headers": ["症状", "可能原因", "解决方案"],
     "rows": [
         ["所有指标接近 0", "训练失败 / cutoff_len 严重截断", "检查日志，增大 cutoff_len"],
         ["格式提取率低", "多样式训练数据混入", "格式验证过滤，统一输出格式"],
         ["无技能 F1 << 有技能 F1", "训练时仍含技能文档", "验证 no_skill 格式转换"],
         ["特定规则/步骤始终为 0", "该内容在训练输出末尾被截断", "增大 cutoff_len，或过滤截断样本"],
         ["增加数据量无提升", "数据质量问题（截断、格式污染）", "清洗数据优先于扩充数据"],
         ["工具调用乱报", "训练数据工具调用格式不一致", "统一工具调用格式，增加负样本"],
     ]},
    {"type": "heading", "level": 3, "text": "数据策略优先级"},
    {"type": "ordered_list", "items": [
        "数据质量（清洗截断 / 验证格式）",
        "针对性边界样本（弱规则 / 边界条件）",
        "数据数量（简单扩充）",
    ]},
    {"type": "divider"},

    # 十、常见陷阱
    {"type": "heading", "level": 2, "text": "十、常见陷阱"},
    {"type": "ordered_list", "items": [
        "cutoff_len 过小 → 输出末尾内容从未被学习",
        "技能文档留在训练 system prompt → 模型依赖文档，推理时崩溃",
        "截断样本未过滤 → 不完整推理链污染训练集，比无此样本更差",
        "多格式训练数据未验证 → 风格漂移，输出无法被解析",
        "工具调用型技能：移除了工具声明 → 模型无法知道可用工具",
        "推理时 max_model_len < cutoff_len → 生成被截断，答案残缺",
        "GRPO max_completion_length < 答案 p95 长度 → 全截断，零奖励，零学习",
        "训练步数过多 → 过拟合，泛化能力下降",
    ]},
    {"type": "divider"},

    # 附录 A：SFT 原理
    {"type": "heading", "level": 2, "text": "附录 A：监督微调（SFT）原理"},
    {"type": "heading", "level": 3, "text": "A.1 训练过程"},
    {"type": "code", "language": "plain", "content": "输入（input_ids）: [system + user + assistant_prefix]\n标签（labels）:   [-100, -100, ..., token_1, token_2, ..., token_n]\n                   ↑ 输入部分不计算 loss    ↑ 只对输出部分计算 loss\n\nLoss = -Σ log P(token_i | context)   # 交叉熵"},
    {"type": "heading", "level": 3, "text": "A.2 LoRA 低秩适配结构"},
    {
        "type": "board",
        "title": "LoRA 结构示意",
        "plantuml": "@startuml\n[输入 x] --> [原始权重 W（冻结）]\n[输入 x] --> [矩阵 A（r×k，可训练）]\n[矩阵 A（r×k，可训练）] --> [矩阵 B（d×r，可训练）]\n[矩阵 B（d×r，可训练）] --> [缩放 alpha/rank]\n[原始权重 W（冻结）] --> [输出合并]\n[缩放 alpha/rank] --> [输出合并]\n[输出合并] --> [最终输出 y]\n@enduml",
    },
    {"type": "table",
     "headers": ["参数", "作用"],
     "rows": [
         ["rank（r）", "控制表达能力（参数量），越大越强但越慢"],
         ["alpha/rank", "控制更新幅度，与 rank 解耦，独立调节"],
     ]},
    {"type": "heading", "level": 3, "text": "A.3 LoRA 挂载位置（lora_target: all）"},
    {"type": "table",
     "headers": ["模块", "层名称"],
     "rows": [
         ["注意力机制", "q_proj / k_proj / v_proj / o_proj"],
         ["前馈网络（FFN）", "up_proj / gate_proj / down_proj"],
     ]},
    {"type": "divider"},

    # 附录 B：GRPO 原理
    {"type": "heading", "level": 2, "text": "附录 B：GRPO 强化训练原理"},
    {"type": "heading", "level": 3, "text": "B.1 动机"},
    {"type": "text", "elements": [
        {"text": "SFT 的上限由教师数据质量决定。GRPO 直接优化任务指标（reward），理论上可突破教师数据上界。"},
    ]},
    {"type": "heading", "level": 3, "text": "B.2 完整训练循环"},
    {
        "type": "board",
        "title": "GRPO 训练循环",
        "plantuml": "@startuml\nstart\nrepeat\n  :输入 prompt（问题，无答案）;\n  :当前模型采样 N 个候选输出;\n  note right: num_generations=8\n  :Reward 函数对每个输出打分（0~1）;\n  :计算相对优势\\nadvantage_i = reward_i - mean(rewards);\n  :策略梯度更新\\n强化高分输出，抑制低分输出;\n  note right: KL 惩罚防止偏离参考模型\nrepeat while (未收敛?) is (继续)\n->收敛;\nstop\n@enduml",
    },
    {"type": "heading", "level": 3, "text": "B.3 模型输入 / 输出"},
    {"type": "table",
     "headers": ["阶段", "内容"],
     "rows": [
         ["输入（prompt）", "System: 空（无技能文档）\\nUser: 任务场景 + 输入数据"],
         ["输出（N个候选）", "每个候选 = 独立推理链 <think>...</think> + 结构化答案"],
     ]},
    {"type": "heading", "level": 3, "text": "B.4 Reward 函数设计"},
    {"type": "table",
     "headers": ["奖励组件", "计算方式", "示例（SPC 任务）"],
     "rows": [
         ["主奖励", "与 Ground Truth 的 F1/准确率", "rule_detection_f1（0~1）"],
         ["格式奖励", "输出格式是否可解析", "含规范格式标识符 +0.05"],
         ["辅助奖励", "任务特定的数值准确性", "|pred_cpk - gt_cpk| < 0.1 则 +0.1"],
     ]},
    {"type": "code", "language": "python", "content": "def reward_fn(output, ground_truth):\n    pred = extract_answer(output)          # 提取结构化答案\n    f1 = compute_f1(pred, ground_truth)    # 与标准答案对比\n    format_bonus = 0.05 if is_parseable(output) else 0\n    cpk_bonus = 0.1 if abs(pred.cpk - ground_truth.cpk) < 0.1 else 0\n    return f1 + format_bonus + cpk_bonus"},
    {"type": "heading", "level": 3, "text": "B.5 关键约束：max_completion_length"},
    {"type": "table",
     "headers": ["参数设置", "后果"],
     "rows": [
         ["max_completion_length < 答案 p95 长度", "所有输出被截断 → clipped_ratio=1.0 → reward=0 → 模型无法更新"],
     ]},
    {"type": "code", "language": "python", "content": "p95 = np.percentile(output_lengths, 95)\nmax_completion_length = int(p95 * 1.1)   # 留 10% 余量"},
    {"type": "divider"},

    # 附录 C：评测指标体系
    {"type": "heading", "level": 2, "text": "附录 C：评测指标体系"},
    {"type": "heading", "level": 3, "text": "C.1 多标签分类（如规则检测）"},
    {"type": "code", "language": "plain", "content": "Ground Truth: {rule2, rule5}\n预测结果:     {rule2, rule7}\n\nTP（正确报警）= 1   → rule2\nFP（误报）    = 1   → rule7（多报了）\nFN（漏报）    = 1   → rule5（漏掉了）\n\nPrecision = TP / (TP+FP) = 1/2 = 0.50  （报的里面有多少是对的）\nRecall    = TP / (TP+FN) = 1/2 = 0.50  （真实异常里找到了多少）\nF1        = 2×P×R / (P+R) = 0.50"},
    {"type": "heading", "level": 3, "text": "C.2 Exact Match（完全正确率）"},
    {"type": "text", "elements": [
        {"text": "预测集合与 Ground Truth 完全一致才算对，多标签任务的严格指标："},
    ]},
    {"type": "code", "language": "plain", "content": "{rule2, rule5} vs {rule2, rule7} → 0（不完全相同）\n{rule2, rule5} vs {rule2, rule5} → 1（完全相同）\n\nExact Match = 完全正确样本数 / 总样本数"},
    {"type": "quote", "text": '与 F1 的关系：F1 高但 EM 低，说明模型经常"部分对"（找到一些但不全）；EM 高则 F1 一定高。'},
    {"type": "heading", "level": 3, "text": "C.3 退化率"},
    {"type": "code", "language": "plain", "content": "退化率 = Σ (基准任务得分下降) / 基准任务数\n常用基准：gsm8k（数学）、hellaswag（常识）、mmlu（知识）"},
    {"type": "divider"},

    # 附录 D：SPC 案例实验数据
    {"type": "heading", "level": 2, "text": "附录 D：SPC 案例实验数据（参考）"},
    {"type": "text", "elements": [{"text": "本项目以 SPC Nelson 规则检测（类型 A：纯规则/算法型）为验证案例。"}]},
    {"type": "heading", "level": 3, "text": "实验结论摘要"},
    {"type": "table",
     "headers": ["指标", "Claude Sonnet + 技能文档", "SFT 14B（无技能文档）"],
     "rows": [
         ["规则检测 F1", "0.266（含不可用样本）", "0.420"],
         ["精确率 Precision", "0.300", "0.422"],
         ["召回率 Recall", "0.250", "0.479"],
         ["完全正确率 Exact Match", "0.360", "0.295"],
         ["CPK 提取率", "62%（38% 不可用）", "100%"],
     ]},
    {"type": "quote", "text": "Exact Match 低（29.5%）不代表任务模糊——Nelson 规则是确定性规则，理论上界 100%。低 EM 源于 8 条规则同时全对的乘法效应（每条 85% 正确 → 全对概率约 27%）。"},
    {"type": "heading", "level": 3, "text": "关键发现"},
    {"type": "table",
     "headers": ["发现", "结论"],
     "rows": [
         ["数据质量 > 数据数量", "去截断（-49条）比扩充（+640条）对 F1 提升更大"],
         ["cutoff_len 是静默杀手", "4096→5120，rule8 召回从 0.00 恢复正常"],
         ["量化损失超预期", "int4/int8 在精确数值任务上损失 7–10%，远超一般文本任务"],
         ["GRPO 需要足够的生成长度", "max_completion_length < p95 → 全截断，零奖励，零学习"],
         ["反直觉规则需对比样本", "上采样对 rule7 无效，正负对比样本才能突破边界"],
     ]},
    {"type": "divider"},
    {"type": "text", "elements": [
        {"text": "v2.0 通用框架，基于 Route 2.1.1 Round 1–3 实验验证 | 2026-05-09"},
    ]},
]
# ============ 内容结束 ============

auth = FeishuAuth()
results = process_blocks(
    auth,
    blocks,
    default_owner=config.get("default_owner") or config.get("default_owner_email") or None,
)
print(json.dumps(results, ensure_ascii=False, indent=2))
