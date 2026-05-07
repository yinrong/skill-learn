"""扩充 ns 数据池 + 多角色 ws 数据生成器（Direction B）。

扩充策略：
  - 扩充 ns 池：在 v4 seed 策略基础上生成 300 条新 ns 样本（共 300+300=600）
  - 多角色 ws 数据：4 种角色各生成 50 条（共 200 条额外 ws）
    角色：资深SPC工程师（基线）、新手工程师、质量经理、统计分析师

用法：
    cd /home/yinrong/post-train/impl2.1
    python round3/tools/data/gen_expanded_ns.py \\
        --mode ns --n 300 --output round3/data/ns_v5.jsonl --seed 401 --concurrency 6
    python round3/tools/data/gen_expanded_ns.py \\
        --mode multirole_ws --n_per_role 50 --output round3/data/multirole_ws.jsonl --seed 501
"""
from __future__ import annotations
import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3] / "common"
sys.path.insert(0, str(_ROOT))

from tools.spc.rules import check_nelson_rules, compute_cpk_detailed
from tools.spc.generator import (
    SPC_SKILL_DOC, SPC_INSTRUCTION, SPC_SYSTEM_WITH_SKILL, SPC_SYSTEM_NO_SKILL,
    RULE_WEIGHTS_DEFAULT, NORMAL_WEIGHT, DOUBLE_RULE_PROB, INJECTORS,
)
from tools.eval.extractor import extract_violations, extract_cpk

API_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
API_KEY  = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", "")
MODEL    = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-6")

# ── 多角色系统提示 ─────────────────────────────────────────────────────────────

ROLES = {
    "senior_engineer": {
        "system_prefix": "你是一名资深 SPC 工程师，有15年生产线质量管理经验。",
        "style_hint": "精确、结构化，直接给出规则判断和处置建议",
    },
    "junior_engineer": {
        "system_prefix": "你是一名 SPC 新手工程师，正在学习Nelson控制图分析。分析时请逐步推理，明确写出每个判断依据。",
        "style_hint": "逐步推理，显式写出判断步骤和依据",
    },
    "quality_manager": {
        "system_prefix": "你是一名工厂质量经理，负责统计过程控制的管理决策。请从质量管理视角分析数据，说明异常对生产的影响和优先级。",
        "style_hint": "管理视角，说明异常影响和处置优先级",
    },
    "statistician": {
        "system_prefix": "你是一名统计分析师，专注于制造业过程能力分析。请详细展示 σ 边界计算、区域划分和每条规则的精确统计依据。",
        "style_hint": "强调数学计算过程，展示区域边界和统计公式",
    },
}

OUTPUT_FORMAT_TEMPLATE = """
分析时请遵循以下输出格式（不使用<think>标签）：

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

违规规则必须使用英文标识符 rule1/rule2/.../rule8，不用中文。
"""

TEACHER_USER_TMPL = """{instruction}

{input_text}

请按指定格式输出完整分析（计算步骤→逐条检查规则→结论→CPK→处置建议）。违规规则必须使用英文标识符 rule1/rule2/.../rule8，不用中文。"""


def build_system_for_role(role_key: str) -> str:
    role = ROLES[role_key]
    return (
        role["system_prefix"]
        + "\n\n" + SPC_SKILL_DOC
        + "\n" + OUTPUT_FORMAT_TEMPLATE
    )


def build_system_no_skill() -> str:
    return "你是一名 SPC 工程师。"


# ── 数据点生成 ─────────────────────────────────────────────────────────────────

def generate_raw_data(rng: random.Random):
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


# ── API ─────────────────────────────────────────────────────────────────────

def call_claude(system: str, user: str, max_tokens: int = 8192, retries: int = 3) -> str:
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
            text = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            return text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Claude API 失败：{e}")
    return ""


def compute_recall(pred: list[str], gt: list[str]) -> float:
    gt_set = set(gt)
    if not gt_set:
        return 1.0 if not pred else 0.5
    return len(set(pred) & gt_set) / len(gt_set)


# ── 单样本生成 ────────────────────────────────────────────────────────────────

def generate_one_ns(idx: int, rng: random.Random, min_recall: float) -> dict | None:
    """生成一条 no-skill 样本（系统提示不含 Skill 文档）。"""
    data, mu, sigma, usl, lsl, target_rules = generate_raw_data(rng)
    sample_s = statistics.stdev(data)
    gt_violations = check_nelson_rules(data, mu, sample_s)
    cpk_detail = compute_cpk_detailed(data, usl, lsl)
    gt_cpk = cpk_detail.get("cpk")

    data_rounded = [round(x, 2) for x in data]
    input_text = (
        f"采样点数据（共25点）：{data_rounded}\n"
        f"USL={usl}，LSL={lsl}，CL={round(mu, 3)}"
    )
    user_prompt = TEACHER_USER_TMPL.format(instruction=SPC_INSTRUCTION, input_text=input_text)
    # ns 样本用带 skill 的系统提示生成（保证质量），再转 ns 格式
    system = build_system_for_role("senior_engineer")
    try:
        response = call_claude(system, user_prompt)
    except Exception as e:
        print(f"  [NS样本{idx}] API错误：{e}", flush=True)
        return None

    pred_violations = extract_violations(response)
    if compute_recall(pred_violations, gt_violations) < min_recall:
        return None

    return {
        "system": build_system_no_skill(),   # ns 格式：无 Skill 文档
        "instruction": SPC_INSTRUCTION,
        "input": input_text,
        "output": response,
        "ground_truth": {"violations": gt_violations, "cpk": gt_cpk},
        "meta": {
            "source": "expanded_ns_v5",
            "model": MODEL,
            "target_rules_injected": target_rules,
        },
    }


def generate_one_role_ws(
    idx: int, role_key: str, rng: random.Random, min_recall: float
) -> dict | None:
    """生成一条多角色 ws 样本。"""
    data, mu, sigma, usl, lsl, target_rules = generate_raw_data(rng)
    sample_s = statistics.stdev(data)
    gt_violations = check_nelson_rules(data, mu, sample_s)
    cpk_detail = compute_cpk_detailed(data, usl, lsl)
    gt_cpk = cpk_detail.get("cpk")

    data_rounded = [round(x, 2) for x in data]
    input_text = (
        f"采样点数据（共25点）：{data_rounded}\n"
        f"USL={usl}，LSL={lsl}，CL={round(mu, 3)}"
    )
    user_prompt = TEACHER_USER_TMPL.format(instruction=SPC_INSTRUCTION, input_text=input_text)
    system = build_system_for_role(role_key)

    try:
        response = call_claude(system, user_prompt)
    except Exception as e:
        print(f"  [角色{role_key}样本{idx}] API错误：{e}", flush=True)
        return None

    pred_violations = extract_violations(response)
    if compute_recall(pred_violations, gt_violations) < min_recall:
        return None

    return {
        "system": SPC_SYSTEM_WITH_SKILL,
        "instruction": SPC_INSTRUCTION,
        "input": input_text,
        "output": response,
        "ground_truth": {"violations": gt_violations, "cpk": gt_cpk},
        "meta": {
            "source": f"multirole_ws_{role_key}",
            "role": role_key,
            "model": MODEL,
            "target_rules_injected": target_rules,
        },
    }


# ── 批量生成 ──────────────────────────────────────────────────────────────────

def generate_batch(
    generator_fn,
    n: int,
    rng: random.Random,
    min_recall: float,
    concurrency: int,
    label: str = "",
) -> list[dict]:
    collected: list[dict] = []
    attempted = 0
    max_attempts = n * 5

    print(f"\n{'─'*50}")
    print(f"[{label}] 目标={n}，最大尝试={max_attempts}，并发={concurrency}")

    while len(collected) < n and attempted < max_attempts:
        batch = min(concurrency * 2, max_attempts - attempted)
        batch_rngs = [random.Random(rng.randint(0, 2**31)) for _ in range(batch)]
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(generator_fn, attempted + j, batch_rngs[j], min_recall): j
                for j in range(batch)
            }
            for future in as_completed(futures):
                attempted += 1
                sample = None
                try:
                    sample = future.result()
                except Exception as e:
                    print(f"  异常：{e}", flush=True)
                if sample is not None:
                    collected.append(sample)
                    print(f"  ✓ [{len(collected)}/{n}] 通过（尝试{attempted}）", flush=True)
                if len(collected) >= n:
                    break

    print(f"[{label}] 完成：{len(collected)}/{n}")
    return collected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ns", "multirole_ws"], required=True)
    parser.add_argument("--n", type=int, default=300, help="ns 模式：目标样本数")
    parser.add_argument("--n_per_role", type=int, default=50, help="multirole_ws 模式：每角色样本数")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--min_recall", type=float, default=0.8)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_samples: list[dict] = []

    if args.mode == "ns":
        samples = generate_batch(
            lambda idx, r, rec: generate_one_ns(idx, r, rec),
            args.n, rng, args.min_recall, args.concurrency,
            label="expanded_ns_v5",
        )
        all_samples.extend(samples)

    elif args.mode == "multirole_ws":
        for role_key in ROLES:
            role_samples = generate_batch(
                lambda idx, r, rec, rk=role_key: generate_one_role_ws(idx, rk, r, rec),
                args.n_per_role, rng, args.min_recall, args.concurrency,
                label=f"multirole_ws_{role_key}",
            )
            all_samples.extend(role_samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n✓ 输出：{args.output}（{len(all_samples)} 条）")


if __name__ == "__main__":
    main()
