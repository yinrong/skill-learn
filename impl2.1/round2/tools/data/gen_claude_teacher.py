"""Claude 教师数据生成器。

使用 Claude-sonnet-4.6 生成高质量 SPC 分析样本，再用规则引擎验证正确性。
通过验证的样本（F1>阈值）保存为训练数据。

用法：
    python tools/data/gen_claude_teacher.py \\
        --n 500 \\
        --output data/demo/train_claude_teacher.jsonl \\
        --verify \\
        --min_f1 0.8
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from tools.spc.rules import check_nelson_rules, compute_cpk_detailed
from tools.eval.extractor import extract_violations, extract_cpk
from tools.spc.generator import (
    SPC_SKILL_DOC, SPC_INSTRUCTION, SPC_SYSTEM_WITH_SKILL,
    RULE_WEIGHTS_DEFAULT, NORMAL_WEIGHT, DOUBLE_RULE_PROB,
    INJECTORS,
)

# ── 环境变量 ──────────────────────────────────────────────────────────────────
API_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
API_KEY  = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", "")
MODEL    = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-6")

# ── 教师 Prompt ───────────────────────────────────────────────────────────────
TEACHER_SYSTEM = """你是一名资深 SPC（统计过程控制）工程师，精通 Nelson 8条控制图规则和 CPK 分析。

{skill_doc}

分析时请严格遵循以下输出格式（不使用<think>标签）：

**【第一步：统计计算】**
计算 x̄（样本均值）、s（样本标准差）、UCL=x̄+3s、LCL=x̄-3s、CPU=(USL-x̄)/(3s)、CPL=(x̄-LSL)/(3s)、CPK=min(CPU,CPL)

**【第二步：逐条检查 Nelson 规则】**
rule1：[触发/未触发] — [说明]
rule2：[触发/未触发] — [说明]
rule3：[触发/未触发] — [说明]
rule4：[触发/未触发] — [说明]
rule5：[触发/未触发] — [说明]
rule6：[触发/未触发] — [说明]
rule7：[触发/未触发] — [说明]
rule8：[触发/未触发] — [说明]

**【结论：异常判断】**
[如有违规：检出X条违规—rule1（含义）；rule2（含义）等（英文标识符）]
[如无异常：全部8条规则均未触发，过程受控]

**【过程能力】**
CPK=[数值]，[等级]（[建议]）

**【处置建议】**
1. [具体可执行建议]
2. [具体可执行建议]
3. [具体可执行建议]
""".format(skill_doc=SPC_SKILL_DOC)

TEACHER_USER_TMPL = """{instruction}

{input_text}

请按指定格式输出完整分析（计算步骤→逐条检查规则→结论→CPK→处置建议）。违规规则必须使用英文标识符 rule1/rule2/.../rule8，不用中文。"""


# ── 数据点生成（与 generator.py 逻辑一致，但独立实现避免循环导入）──────────────
def _generate_raw_data(rng: random.Random):
    """生成随机 SPC 数据点，返回 (data, mu, sigma, usl, lsl, target_rules)。"""
    mu = rng.uniform(9.5, 10.5)
    sigma = rng.uniform(0.15, 0.45)
    usl_margin = rng.uniform(4.0, 7.0) * sigma
    lsl_margin = rng.uniform(4.0, 7.0) * sigma
    usl = round(mu + usl_margin, 2)
    lsl = round(mu - lsl_margin, 2)

    base_data = [rng.gauss(mu, sigma) for _ in range(25)]

    ruleset = list(RULE_WEIGHTS_DEFAULT.keys())
    weights = [RULE_WEIGHTS_DEFAULT[r] for r in ruleset] + [NORMAL_WEIGHT]
    choices = ruleset + ["normal"]
    primary = rng.choices(choices, weights=weights, k=1)[0]

    target_rules = []
    if primary != "normal":
        target_rules.append(primary)
        if rng.random() < DOUBLE_RULE_PROB:
            remaining = [r for r in ruleset if r != primary]
            if remaining:
                target_rules.append(rng.choice(remaining))

    data = base_data[:]
    for rule in target_rules:
        data = INJECTORS[rule](data, mu, sigma, rng)

    return data, mu, sigma, usl, lsl, target_rules


# ── Claude API 调用 ──────────────────────────────────────────────────────────
def call_claude(system: str, user: str, max_tokens: int = 4096, retries: int = 3) -> str:
    """调用 Claude Messages API，返回文本。"""
    import urllib.request

    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
        "system": system,
    }).encode()

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{API_BASE.rstrip('/')}/v1/messages",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": API_KEY,
                    "anthropic-version": "2023-06-01",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            # 提取文本内容
            text = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            return text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Claude API 调用失败（{retries} 次重试）: {e}")
    return ""


# ── 验证函数 ──────────────────────────────────────────────────────────────────
def compute_f1(pred: list[str], gt: list[str]) -> float:
    pred_set = set(pred)
    gt_set = set(gt)
    if not pred_set and not gt_set:
        return 1.0  # 都是正常样本，完全正确
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def compute_recall(pred: list[str], gt: list[str]) -> float:
    """计算 recall：GT 中有多少被 Claude 正确检出。"""
    gt_set = set(gt)
    if not gt_set:
        # 正常样本：如果 Claude 也说无异常则 recall=1.0；如果 Claude 说有异常 recall=0.5（允许宽松）
        return 1.0 if not pred else 0.5
    tp = len(set(pred) & gt_set)
    return tp / len(gt_set)


# ── 主生成函数 ────────────────────────────────────────────────────────────────
def generate_one(i: int, rng: random.Random, min_f1: float, verify: bool) -> dict | None:
    """生成并验证一条样本，返回 sample dict 或 None（验证未过）。"""
    data, mu, sigma, usl, lsl, target_rules = _generate_raw_data(rng)
    sample_s = statistics.stdev(data)

    # Ground truth
    gt_violations = check_nelson_rules(data, mu, sample_s)
    cpk_detail = compute_cpk_detailed(data, usl, lsl)
    gt_cpk = cpk_detail.get("cpk")

    data_rounded = [round(x, 2) for x in data]
    input_text = (
        f"采样点数据（共25点）：{data_rounded}\n"
        f"USL={usl}，LSL={lsl}，CL={round(mu, 3)}"
    )

    user_prompt = TEACHER_USER_TMPL.format(
        instruction=SPC_INSTRUCTION,
        input_text=input_text,
    )

    try:
        response = call_claude(TEACHER_SYSTEM, user_prompt)
    except Exception as e:
        print(f"  [样本{i}] API错误：{e}", flush=True)
        return None

    # 提取 violations 时扫描全文（模型不使用 <think> 块）
    pred_violations = extract_violations(response)

    if verify:
        recall = compute_recall(pred_violations, gt_violations)
        f1 = compute_f1(pred_violations, gt_violations)
        # 使用 recall 作为主要过滤器（确保 Claude 不漏报 GT 违规）
        # 允许 Claude 多报（precision 宽松），但不允许漏报
        if recall < min_f1:
            print(f"  [样本{i}] 验证未过：recall={recall:.2f}<{min_f1}，f1={f1:.2f}，gt={gt_violations}，pred={pred_violations}", flush=True)
            return None

    sample = {
        "system": SPC_SYSTEM_WITH_SKILL,
        "instruction": SPC_INSTRUCTION,
        "input": input_text,
        "output": response,
        "ground_truth": {
            "violations": gt_violations,
            "cpk": gt_cpk,
        },
        "meta": {
            "source": "claude_teacher",
            "model": MODEL,
            "target_rules_injected": target_rules,
        },
    }
    return sample


def generate_dataset(
    n: int,
    output_path: str,
    seed: int = 42,
    verify: bool = True,
    min_f1: float = 0.8,
    max_attempts: int = None,
    concurrency: int = 4,
) -> None:
    """生成 n 条通过验证的样本。

    Args:
        max_attempts: 最多尝试次数（默认 n×3）
    """
    if not API_KEY:
        raise RuntimeError(
            "未设置 API Key。请设置环境变量 ANTHROPIC_AUTH_TOKEN 或 ANTHROPIC_API_KEY。"
        )
    if max_attempts is None:
        max_attempts = n * 3

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    collected = 0
    attempted = 0
    skipped = 0
    rng = random.Random(seed)

    rule_coverage = {f"rule{i}": 0 for i in range(1, 9)}
    normal_count = 0

    print(f"开始生成 Claude 教师数据：目标={n}，验证阈值={min_f1 if verify else '不验证'}，并发={concurrency}")
    print(f"模型：{MODEL}，API：{API_BASE}")

    with open(out, "w", encoding="utf-8") as fout:
        # 使用线程池并发生成
        batch_size = concurrency * 2
        while collected < n and attempted < max_attempts:
            # 预先为每个 batch 任务分配种子
            batch_rngs = [random.Random(rng.randint(0, 2**31)) for _ in range(batch_size)]
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(generate_one, attempted + j, batch_rngs[j], min_f1, verify): j
                    for j in range(batch_size)
                    if attempted + j < max_attempts and collected < n
                }
                for future in as_completed(futures):
                    j = futures[future]
                    attempted += 1
                    try:
                        sample = future.result()
                    except Exception as e:
                        print(f"  [任务{j}] 异常：{e}", flush=True)
                        sample = None

                    if sample is not None:
                        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        fout.flush()
                        collected += 1
                        # 统计规则覆盖
                        for v in sample["ground_truth"]["violations"]:
                            if v in rule_coverage:
                                rule_coverage[v] += 1
                        if not sample["ground_truth"]["violations"]:
                            normal_count += 1
                        print(f"  ✓ [{collected}/{n}] 样本通过 (尝试{attempted}次)", flush=True)
                    else:
                        skipped += 1

                    if collected >= n:
                        break

    print(f"\n=== 生成完成 ===")
    print(f"  目标：{n}，收集：{collected}，失败/跳过：{skipped}，总尝试：{attempted}")
    print(f"  保留率：{collected/attempted*100:.1f}%")
    print(f"  正常样本：{normal_count}")
    print(f"\n规则覆盖分布：")
    for rule, cnt in sorted(rule_coverage.items()):
        bar = "█" * int(cnt / max(1, collected) * 40)
        print(f"  {rule}: {cnt:4d}（{cnt/max(1,collected)*100:5.1f}%）{bar}")

    missing = [r for r, c in rule_coverage.items() if c == 0]
    if missing:
        print(f"\n⚠ 以下规则无覆盖：{missing}（建议增加样本量）")
    else:
        print(f"\n✓ 所有 8 条规则均有覆盖")
    print(f"\n输出文件：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude 教师数据生成器")
    parser.add_argument("--n", type=int, default=500, help="目标样本数")
    parser.add_argument("--output", type=str, default="data/demo/train_claude_teacher.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true", default=True,
                        help="用规则引擎验证 Claude 输出（默认开启）")
    parser.add_argument("--no_verify", action="store_true",
                        help="跳过验证（所有样本均保留）")
    parser.add_argument("--min_f1", type=float, default=0.8,
                        help="验证通过阈值（默认0.8，即GT中80%%规则被正确预测）")
    parser.add_argument("--max_attempts", type=int, default=None,
                        help="最大尝试次数（默认 n×3）")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="并发 API 调用数（默认4，根据 API 限速调整）")
    args = parser.parse_args()

    generate_dataset(
        n=args.n,
        output_path=args.output,
        seed=args.seed,
        verify=not args.no_verify,
        min_f1=args.min_f1,
        max_attempts=args.max_attempts,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
