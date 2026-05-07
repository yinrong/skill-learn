"""SPC 训练数据生成器。

用法：
    python tools/spc/generator.py --n 500 --output data/demo/train_N4.jsonl --seed 4
    python tools/spc/generator.py --validate data/demo/train_N4.jsonl
"""
from __future__ import annotations
import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from tools.spc.rules import check_nelson_rules, compute_cpk_detailed, find_rule_locations
from tools.spc.formatter import format_output


# ── 全局注入权重 ───────────────────────────────────────────────────────────────
# 单条规则注入权重（和不足100%，剩余为"正常样本"）
RULE_WEIGHTS_DEFAULT = {
    "rule1": 0.22,
    "rule2": 0.10,
    "rule3": 0.10,
    "rule4": 0.08,
    "rule5": 0.08,
    "rule6": 0.08,
    "rule7": 0.07,
    "rule8": 0.07,
    # 正常样本：剩余 0.20
}

# 均衡权重：每条规则约 11.5%，正常样本 8%
RULE_WEIGHTS_BALANCED = {
    "rule1": 0.115,
    "rule2": 0.115,
    "rule3": 0.115,
    "rule4": 0.115,
    "rule5": 0.115,
    "rule6": 0.115,
    "rule7": 0.115,
    "rule8": 0.115,
    # 正常样本：剩余 0.08
}

# rule2下调权重：rule2=0.03，其余规则约13.4%，正常样本约6.4%（消除rule2主导现象）
RULE_WEIGHTS_RULE2_DOWN = {
    "rule1": 0.134,
    "rule2": 0.030,
    "rule3": 0.134,
    "rule4": 0.134,
    "rule5": 0.134,
    "rule6": 0.134,
    "rule7": 0.134,
    "rule8": 0.134,
    # 正常样本：剩余 0.062
}

# 运行时使用的权重（可被 CLI --balanced_weights 覆盖）
RULE_WEIGHTS = dict(RULE_WEIGHTS_DEFAULT)
NORMAL_WEIGHT = 1.0 - sum(RULE_WEIGHTS.values())   # ≈ 0.20
# 二次叠加概率：15% 的样本额外注入一条规则（可被 CLI --double_rule_prob 覆盖）
DOUBLE_RULE_PROB = 0.15


# ── SPC 分析技能文档（Skill 内容）────────────────────────────────────────────
SPC_SKILL_DOC = """===== SPC 分析技能手册 =====

一、Nelson 8条控制图规则

规则名称使用英文标识（rule1～rule8），判断时必须引用标识名称。

| 规则ID | 触发条件（精确定义） | 含义 |
|--------|-------------------|------|
| rule1 | 任意**1点**超出 UCL 或 LCL（UCL=均值+3σ，LCL=均值-3σ） | 突发异常，须立即响应 |
| rule2 | 连续**9点**全部在均值同一侧（全>均值 或 全<均值） | 均值系统性漂移 |
| rule3 | 连续**6点**严格单调递增 或 严格单调递减 | 趋势性漂移（磨损、老化） |
| rule4 | 连续**14点**严格交替升降（zigzag，每相邻两点方向不同） | 周期性干扰 |
| rule5 | 连续**3点**中有**2点**在同侧2σ～3σ区间（超2σ但未超3σ） | 过程能力早期恶化信号 |
| rule6 | 连续**5点**中有**4点**在同侧1σ～3σ区间（超1σ但未超3σ） | 均值偏移预警 |
| rule7 | 连续**15点**全部在均值±1σ以内（绝对偏差<σ） | 测量分辨率不足或数据被篡改 |
| rule8 | 连续**8点**全部在均值±1σ以外，且两侧都有（±均有） | 双峰分布或混线混料 |

重要区分：
- UCL/LCL 是控制限（= 均值±3σ，由数据本身计算），Nelson 规则使用控制限
- USL/LSL 是规格限（由工程规范给定），用于计算 CPK
- 1σ带 = [均值-σ, 均值+σ]；2σ带边界 = 均值±2σ；3σ带边界 = UCL/LCL

二、区域分类（用于 rule5/rule6/rule7/rule8 判断）

  区域A（超出控制限，|偏差|>3σ）→ rule1
  区域B（2σ<|偏差|≤3σ）         → rule5 的"外侧2点"
  区域C（1σ<|偏差|≤2σ）         → rule6 的"外侧4点"
  区域D（|偏差|≤1σ，即±1σ内）  → rule7；±1σ外才算 rule8 的"外侧"

三、CPK 计算步骤

  1. 计算 x̄（样本均值）和 s（样本标准差）
  2. UCL = x̄ + 3s，LCL = x̄ - 3s（控制限，用于 Nelson 规则判断）
  3. CPU = (USL - x̄) / (3s)
  4. CPL = (x̄ - LSL) / (3s)
  5. CPK = min(CPU, CPL)

  | CPK       | 等级   | 建议处置       |
  |-----------|--------|----------------|
  | ≥ 1.67    | 优秀   | 可降低检验频次  |
  | 1.33~1.67 | 良好   | 维持现有策略    |
  | 1.00~1.33 | 一般   | 加强监控        |
  | < 1.00    | 不合格 | 立即排查原因    |

四、输出格式要求

  1. 在 <think> 块内逐条检查 rule1～rule8，触发规则必须引用具体点位和数值
  2. 列出 x̄、s、UCL、LCL、CPU、CPL、CPK 的计算过程
  3. 正文"异常判断"段必须用 rule1/rule2/.../rule8 标识名称列出违规规则
  4. 给出至少3条工厂可直接执行的处置建议
=========================="""

# ── 标准 SPC Instruction ──────────────────────────────────────────────────────
SPC_INSTRUCTION = (
    "请分析下列采样数据，依次检查所有 8 条 Nelson 规则是否违规，"
    "计算 CPK，并给出工厂可直接执行的处置建议。"
)

# system prompt：有 Skill 版本（训练时使用）
SPC_SYSTEM_WITH_SKILL = "你是一名 SPC 工程师。\n\n" + SPC_SKILL_DOC

# system prompt：无 Skill 版本（基座评测和 SFT 内化效果评测时使用）
SPC_SYSTEM_NO_SKILL = "你是一名 SPC 工程师。"

# system prompt：无 Skill 但含格式提示版本（E1-format 实验用，训练+评测配对使用）
SPC_SYSTEM_NO_SKILL_WITH_FORMAT = (
    "你是一名 SPC 工程师。"
    "分析中，违规规则必须使用英文标识符 rule1～rule8（小写，不用中文序号）。"
)


# ── 数据注入函数 ───────────────────────────────────────────────────────────────
def _inject_rule1(data: list[float], mu: float, sigma: float, rng: random.Random) -> list[float]:
    """注入 rule1：随机1点改为超出 UCL 或 LCL。"""
    d = data[:]
    pos = rng.randint(0, len(d) - 1)
    direction = rng.choice([1, -1])
    magnitude = rng.uniform(3.2, 4.5)
    d[pos] = mu + direction * magnitude * sigma
    return d


def _inject_rule2(data: list[float], mu: float, sigma: float, rng: random.Random) -> list[float]:
    """注入 rule2：连续9点在均值同侧。"""
    d = data[:]
    n = len(d)
    start = rng.randint(0, n - 9)
    direction = rng.choice([1, -1])
    for i in range(start, start + 9):
        if direction == 1 and d[i] <= mu:
            d[i] = mu + abs(d[i] - mu) * 0.3 + rng.uniform(0.05, 0.3) * sigma
        elif direction == -1 and d[i] >= mu:
            d[i] = mu - abs(d[i] - mu) * 0.3 - rng.uniform(0.05, 0.3) * sigma
    return d


def _inject_rule3(data: list[float], mu: float, sigma: float, rng: random.Random) -> list[float]:
    """注入 rule3：连续6点单调递增或递减。"""
    d = data[:]
    n = len(d)
    start = rng.randint(0, n - 6)
    direction = rng.choice([1, -1])
    base = rng.uniform(mu - 0.5 * sigma, mu + 0.5 * sigma)
    step = rng.uniform(0.1, 0.25) * sigma
    for k in range(6):
        d[start + k] = base + direction * k * step
    return d


def _inject_rule4(data: list[float], mu: float, sigma: float, rng: random.Random) -> list[float]:
    """注入 rule4：连续14点交替升降。"""
    d = data[:]
    n = len(d)
    start = rng.randint(0, n - 14)
    center = rng.uniform(mu - 0.3 * sigma, mu + 0.3 * sigma)
    amp = rng.uniform(0.4, 0.9) * sigma
    for k in range(14):
        sign = 1 if k % 2 == 0 else -1
        d[start + k] = center + sign * amp
    return d


def _inject_rule5(data: list[float], mu: float, sigma: float, rng: random.Random) -> list[float]:
    """注入 rule5：连续3点中2点在同侧2σ外。"""
    d = data[:]
    n = len(d)
    start = rng.randint(0, n - 3)
    direction = rng.choice([1, -1])
    extreme_mag = rng.uniform(2.2, 2.8)
    positions = rng.sample(range(3), 2)
    for k in positions:
        d[start + k] = mu + direction * extreme_mag * sigma
    return d


def _inject_rule6(data: list[float], mu: float, sigma: float, rng: random.Random) -> list[float]:
    """注入 rule6：连续5点中4点在同侧1σ外。"""
    d = data[:]
    n = len(d)
    start = rng.randint(0, n - 5)
    direction = rng.choice([1, -1])
    extreme_mag = rng.uniform(1.2, 1.8)
    positions = rng.sample(range(5), 4)
    for k in positions:
        d[start + k] = mu + direction * extreme_mag * sigma
    return d


def _inject_rule7(data: list[float], mu: float, sigma: float, rng: random.Random) -> list[float]:
    """注入 rule7：连续15点在1σ以内（方差过小）。"""
    d = data[:]
    n = len(d)
    start = rng.randint(0, n - 15)
    tight_sigma = sigma * rng.uniform(0.2, 0.5)
    for k in range(15):
        d[start + k] = mu + rng.gauss(0, tight_sigma)
        # 确保严格在±σ以内
        d[start + k] = max(mu - 0.95 * sigma, min(mu + 0.95 * sigma, d[start + k]))
    return d


def _inject_rule8(data: list[float], mu: float, sigma: float, rng: random.Random) -> list[float]:
    """注入 rule8：连续8点在1σ外且两侧均有。"""
    d = data[:]
    n = len(d)
    start = rng.randint(0, n - 8)
    mag = rng.uniform(1.3, 2.2)
    for k in range(8):
        sign = 1 if k % 2 == 0 else -1
        d[start + k] = mu + sign * mag * sigma
    return d


INJECTORS = {
    "rule1": _inject_rule1,
    "rule2": _inject_rule2,
    "rule3": _inject_rule3,
    "rule4": _inject_rule4,
    "rule5": _inject_rule5,
    "rule6": _inject_rule6,
    "rule7": _inject_rule7,
    "rule8": _inject_rule8,
}


# ── 单样本生成 ────────────────────────────────────────────────────────────────
def generate_sample(
    rng: random.Random,
    use_llm: bool = False,
    include_skill: bool = True,
    sample_style: str = "auto",
    format_hint: bool = False,
) -> dict:
    """生成一条训练样本。

    Args:
        include_skill:  True → system 含 Skill 文档（训练集默认）
        sample_style:   输出风格
            "auto"         - 随机选取风格（训练集推荐，增加多样性）
            "detailed"     - 逐条列举所有规则
            "brief"        - 只列触发规则
            "calc_first"   - 先算UCL/LCL再检规则
            "no_skill"     - 无Skill文档样式（think块内嵌规则定义）
    """
    # 1. 随机过程参数
    mu = rng.uniform(9.5, 10.5)
    sigma = rng.uniform(0.15, 0.45)
    usl_margin = rng.uniform(4.0, 7.0) * sigma
    lsl_margin = rng.uniform(4.0, 7.0) * sigma
    usl = round(mu + usl_margin, 2)
    lsl = round(mu - lsl_margin, 2)

    # 2. 基础25点正态数据
    base_data = [rng.gauss(mu, sigma) for _ in range(25)]

    # 3. 决定注入哪些规则
    ruleset = list(RULE_WEIGHTS.keys())
    weights = [RULE_WEIGHTS[r] for r in ruleset] + [NORMAL_WEIGHT]
    choices = ruleset + ["normal"]
    primary = rng.choices(choices, weights=weights, k=1)[0]

    target_rules = []
    if primary != "normal":
        target_rules.append(primary)
        # 15% 概率叠加第二条规则（和第一条不同）
        if rng.random() < DOUBLE_RULE_PROB:
            remaining = [r for r in ruleset if r != primary]
            if remaining:
                target_rules.append(rng.choice(remaining))

    # 4. 注入
    data = base_data[:]
    for rule in target_rules:
        data = INJECTORS[rule](data, mu, sigma, rng)

    # 5. 用样本 s 作为 sigma（与示例一致：UCL = CL + 3s）
    sample_s = statistics.stdev(data)

    # 5a. 用规则引擎重新检测（center=mu(CL), sigma=sample_s）
    violations = check_nelson_rules(data, mu, sample_s)

    # 6. 计算 CPK（用样本 x̄ 和 s）
    cpk_detail = compute_cpk_detailed(data, usl, lsl)

    # 7. 获取点位信息
    locations = find_rule_locations(data, mu, sample_s)

    # 8. 决定输出风格
    # auto 模式：60% detailed, 20% brief, 20% calc_first（均含Skill文档）
    if sample_style == "auto":
        r = rng.random()
        if r < 0.60:
            actual_style = "detailed"
        elif r < 0.80:
            actual_style = "brief"
        else:
            actual_style = "calc_first"
    else:
        actual_style = sample_style

    # 9. 生成推理链输出（传入 rng 保证每条样本短语随机不同）
    formatter_fn = __import__("tools.spc.formatter", fromlist=["format_output_with_llm"]).format_output_with_llm
    if use_llm:
        output_text = formatter_fn(data, mu, sample_s, usl, lsl, violations, cpk_detail, locations, style=actual_style, rng=rng)
    else:
        output_text = format_output(data, mu, sample_s, usl, lsl, violations, cpk_detail, locations, style=actual_style, rng=rng)

    # 10. 构造 Alpaca JSON 格式
    data_rounded = [round(x, 2) for x in data]
    input_text = (
        f"采样点数据（共25点）：{data_rounded}\n"
        f"USL={usl}，LSL={lsl}，CL={round(mu, 3)}"
    )

    # no_skill 风格的样本：不含 Skill 文档，但输出仍使用 rule1/rule2 标识
    if actual_style == "no_skill":
        system = SPC_SYSTEM_NO_SKILL_WITH_FORMAT if format_hint else SPC_SYSTEM_NO_SKILL
    else:
        system = SPC_SYSTEM_WITH_SKILL if include_skill else (
            SPC_SYSTEM_NO_SKILL_WITH_FORMAT if format_hint else SPC_SYSTEM_NO_SKILL
        )

    cpk_val = cpk_detail.get("cpk")
    sample = {
        "system": system,
        "instruction": SPC_INSTRUCTION,
        "input": input_text,
        "output": output_text,
        "ground_truth": {
            "violations": violations,
            "cpk": cpk_val,
        },
    }
    return sample


# ── 批量生成 ──────────────────────────────────────────────────────────────────
def generate_dataset(
    n: int,
    output_path: str,
    seed: int,
    use_llm: bool = False,
    include_skill: bool = True,
    mixed: bool = False,
    no_skill_ratio: float = 0.25,
    format_hint: bool = False,
    balanced_weights: bool = False,
    downweight_rule2: bool = False,
    double_rule_prob: float = DOUBLE_RULE_PROB,
) -> None:
    """生成 n 条样本并写入 JSONL 文件。

    Args:
        mixed:            True → 混合模式：(1-no_skill_ratio) 比例含Skill且auto风格，
                                         no_skill_ratio 比例无Skill且no_skill风格
        no_skill_ratio:   mixed 模式中无Skill样本的比例（默认25%）
        format_hint:      True → 无Skill样本在 system prompt 中添加格式提示
        balanced_weights: True → 所有规则等权重（均衡注入）
        double_rule_prob: 二次注入概率（默认0.15）
    """
    global RULE_WEIGHTS, NORMAL_WEIGHT, DOUBLE_RULE_PROB
    # 根据参数设置注入权重
    if downweight_rule2:
        RULE_WEIGHTS = dict(RULE_WEIGHTS_RULE2_DOWN)
        NORMAL_WEIGHT = 1.0 - sum(RULE_WEIGHTS.values())
    elif balanced_weights:
        RULE_WEIGHTS = dict(RULE_WEIGHTS_BALANCED)
        NORMAL_WEIGHT = 1.0 - sum(RULE_WEIGHTS.values())
    else:
        RULE_WEIGHTS = dict(RULE_WEIGHTS_DEFAULT)
        NORMAL_WEIGHT = 1.0 - sum(RULE_WEIGHTS.values())
    DOUBLE_RULE_PROB = double_rule_prob

    rng = random.Random(seed)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for i in range(n):
            if mixed:
                if rng.random() < no_skill_ratio:
                    sample = generate_sample(rng, use_llm=use_llm, include_skill=False,
                                             sample_style="no_skill", format_hint=format_hint)
                else:
                    sample = generate_sample(rng, use_llm=use_llm, include_skill=True,
                                             sample_style="auto", format_hint=format_hint)
            else:
                sample = generate_sample(rng, use_llm=use_llm, include_skill=include_skill,
                                         sample_style="auto" if include_skill else "detailed",
                                         format_hint=format_hint)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  生成进度：{i + 1}/{n}", flush=True)

    skill_label = "混合" if mixed else ("含Skill" if include_skill else "无Skill")
    weight_label = "rule2下调" if downweight_rule2 else ("均衡权重" if balanced_weights else "默认权重")
    print(f"✓ 已生成 {n} 条样本（{skill_label}, {weight_label}, double={double_rule_prob}）→ {output_path}")


# ── 数据集验证 ────────────────────────────────────────────────────────────────
def validate_dataset(path: str) -> None:
    """验证 JSONL 文件：格式合法 + 规则覆盖分布。"""
    samples = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"✗ 第{lineno}行 JSON 解析失败：{e}")
                return

            required = ["system", "instruction", "input", "output", "ground_truth"]
            for field in required:
                if field not in obj:
                    print(f"✗ 第{lineno}行缺少字段：{field}")
                    return
            samples.append(obj)

    # 统计规则分布
    rule_counts = {f"rule{i}": 0 for i in range(1, 9)}
    normal_count = 0
    cpk_found = 0

    for s in samples:
        gt = s.get("ground_truth", {})
        viols = gt.get("violations", [])
        if not viols:
            normal_count += 1
        for v in viols:
            if v in rule_counts:
                rule_counts[v] += 1
        if gt.get("cpk") is not None:
            cpk_found += 1

    total = len(samples)
    skill_count = sum(1 for s in samples if "Nelson" in s.get("system", ""))
    print(f"\n=== 数据集验证报告：{path} ===")
    print(f"总样本数：{total}")
    print(f"JSON 格式：全部合法")
    print(f"含Skill文档：{skill_count}（{skill_count / total * 100:.1f}%）{'✓ 训练集' if skill_count == total else ('✓ 测试集' if skill_count == 0 else '⚠ 混合')}")
    print(f"正常样本（无违规）：{normal_count}（{normal_count / total * 100:.1f}%）")
    print(f"CPK 已计算：{cpk_found}（{cpk_found / total * 100:.1f}%）")
    print("\n各规则触发频次：")
    for rule, cnt in sorted(rule_counts.items()):
        bar = "█" * int(cnt / total * 40)
        print(f"  {rule}: {cnt:4d}（{cnt / total * 100:5.1f}%）{bar}")

    # 检验：每条规则至少有样本
    missing = [r for r, c in rule_counts.items() if c == 0]
    if missing:
        print(f"\n⚠ 以下规则无样本覆盖：{missing}（可能需要调整注入权重）")
    else:
        print("\n✓ 所有 8 条规则均有覆盖")

    if normal_count / total < 0.10 or normal_count / total > 0.35:
        print(f"⚠ 正常样本比例 {normal_count / total * 100:.1f}% 偏离预期（10%~35%）")
    else:
        print(f"✓ 正常样本比例正常（{normal_count / total * 100:.1f}%）")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="SPC 训练数据生成器")
    parser.add_argument("--n", type=int, default=100, help="生成样本数")
    parser.add_argument("--output", type=str, default="data/demo/train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_llm", action="store_true", help="调用 LLM 润色输出")
    parser.add_argument("--validate", type=str, help="验证已有 JSONL 文件（跳过生成）")
    skill_group = parser.add_mutually_exclusive_group()
    skill_group.add_argument("--include_skill", action="store_true", default=True,
                             help="system 字段包含 Skill 文档（训练集默认）")
    skill_group.add_argument("--no_skill", action="store_true",
                             help="system 字段不含 Skill 文档（测试集用）")
    parser.add_argument("--mixed", action="store_true",
                        help="混合模式：75%%含Skill(auto风格) + 25%%无Skill(no_skill风格)")
    parser.add_argument("--no_skill_ratio", type=float, default=0.25,
                        help="mixed 模式中无Skill样本比例（默认0.25）")
    parser.add_argument("--format_hint", action="store_true",
                        help="无Skill样本system prompt添加英文format提示")
    parser.add_argument("--balanced_weights", action="store_true",
                        help="所有规则等权重注入（均衡，消除rule1主导）")
    parser.add_argument("--downweight_rule2", action="store_true",
                        help="下调rule2权重（rule2=0.03，其余~13.4%，消除rule2主导现象）")
    parser.add_argument("--double_rule_prob", type=float, default=0.15,
                        help="双规则叠加注入概率（默认0.15，可提高到0.30）")
    parser.add_argument("--append", type=str, default=None,
                        help="追加指定JSONL文件内容（如 textbook）到输出末尾")
    args = parser.parse_args()

    if args.validate:
        validate_dataset(args.validate)
    else:
        include_skill = not args.no_skill
        generate_dataset(
            args.n, args.output, args.seed,
            use_llm=args.use_llm,
            include_skill=include_skill,
            mixed=args.mixed,
            no_skill_ratio=args.no_skill_ratio,
            format_hint=args.format_hint,
            balanced_weights=args.balanced_weights,
            downweight_rule2=args.downweight_rule2,
            double_rule_prob=args.double_rule_prob,
        )
        # 追加附加文件
        if args.append:
            with open(args.output, "a", encoding="utf-8") as fout:
                with open(args.append, "r", encoding="utf-8") as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line)
            n_appended = sum(1 for _ in open(args.append, encoding="utf-8") if _.strip())
            print(f"✓ 追加 {n_appended} 条样本（来自 {args.append}）")


if __name__ == "__main__":
    main()
