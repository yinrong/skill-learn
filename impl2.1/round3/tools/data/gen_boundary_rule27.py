"""Rule2 / Rule7 边界样本生成器（Direction A）。

生成策略：
  - rule2 边界：刚好 9 点同侧（不是 8/10），末尾点贴近均值线
  - rule2 近似正常：连续 7~8 点同侧（不触发），外观与 rule2 相似
  - rule7 边界：刚好 15 点在 ±1σ 内（末尾点 |z| ≈ 0.85~0.99）
  - rule7 近似正常：连续 13~14 点在 ±1σ 内（不触发）

用法：
    cd /home/yinrong/post-train/impl2.1
    python round3/tools/data/gen_boundary_rule27.py \\
        --n_rule2 90 --n_rule7 90 \\
        --output_ws  round3/data/boundary_ws.jsonl \\
        --output_ns  round3/data/boundary_ns.jsonl \\
        --seed 301 --concurrency 6
"""
from __future__ import annotations
import argparse
import json
import math
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
from tools.spc.generator import SPC_SKILL_DOC, SPC_INSTRUCTION, SPC_SYSTEM_WITH_SKILL
from tools.eval.extractor import extract_violations, extract_cpk

API_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
API_KEY  = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", "")
MODEL    = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-6")

NO_SKILL_SYSTEM = "你是一名 SPC 工程师。"

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


# ── 边界样本注入器 ─────────────────────────────────────────────────────────────
# 注意：注入器尽量可靠，但不能保证 100% 成功（base_data 可能有既存违规）。
# generate_one() 会用规则引擎验证并丢弃失败样本，成功率约 50~80%。

def _fresh_normal_data(mu: float, sigma: float, n: int,
                       rng: random.Random) -> list[float]:
    """生成纯高斯数据，确保无规则违规（多次重试）。"""
    from tools.spc.rules import check_nelson_rules
    import statistics as _stat
    for _ in range(20):
        d = [rng.gauss(mu, sigma) for _ in range(n)]
        s = _stat.stdev(d)
        if s > 0 and not check_nelson_rules(d, mu, s):
            return d
    # 如果重试后仍有违规，返回标准正态数据（保守策略）
    return [mu + rng.uniform(-0.8 * sigma, 0.8 * sigma) for _ in range(n)]


def _inject_rule2_boundary(data: list[float], mu: float, sigma: float,
                           rng: random.Random) -> tuple[list[float], bool]:
    """注入 rule2 边界：恰好 9 点同侧，末尾点贴近均值（|z| 约 0.03~0.20）。"""
    # 从干净基础数据开始，避免既存 rule2 干扰
    d = _fresh_normal_data(mu, sigma, len(data), rng)
    n = len(d)
    start = rng.randint(0, n - 10)  # 留出 1 点缓冲，用于打断前序
    direction = rng.choice([1, -1])

    # 确保 start-1 位置在对侧（防止借用前面的 run 凑 9+）
    if start > 0:
        d[start - 1] = mu - direction * rng.uniform(0.15, 0.6) * sigma

    # 前 8 点明确在同侧
    for i in range(start, start + 8):
        d[i] = mu + direction * rng.uniform(0.08, 0.55) * sigma

    # 第 9 点（边界点）贴近均值同侧
    d[start + 8] = mu + direction * rng.uniform(0.02, 0.18) * sigma

    # 确保 start+9 位置（如存在）在对侧（防止 run 延伸触发更长 run）
    if start + 9 < n:
        d[start + 9] = mu - direction * rng.uniform(0.15, 0.6) * sigma

    return d, True


def _inject_rule2_near_miss(data: list[float], mu: float, sigma: float,
                             rng: random.Random) -> tuple[list[float], bool]:
    """注入 rule2 近似：恰好 7~8 点同侧（不触发），前后各有对侧点。"""
    d = _fresh_normal_data(mu, sigma, len(data), rng)
    n = len(d)
    run_len = rng.choice([7, 8])
    # 留出前后各 1 点缓冲，防止借用相邻点
    start = rng.randint(1, n - run_len - 1)
    direction = rng.choice([1, -1])

    # 前一位置强制对侧
    d[start - 1] = mu - direction * rng.uniform(0.15, 0.6) * sigma

    # run_len 点在同侧
    for i in range(start, start + run_len):
        d[i] = mu + direction * rng.uniform(0.05, 0.45) * sigma

    # 后一位置强制对侧（打断 run）
    if start + run_len < n:
        d[start + run_len] = mu - direction * rng.uniform(0.15, 0.6) * sigma

    return d, False


def _inject_rule7_boundary(data: list[float], mu: float, sigma: float,
                           rng: random.Random) -> tuple[list[float], bool]:
    """注入 rule7 边界：恰好 15 点在 ±sample_s 内，末尾点贴近 ±sample_s 边界。
    基于注入后的 sample_s 而非真实 sigma 设置边界，避免 sample_s 收缩问题。"""
    import statistics as _stat

    d = _fresh_normal_data(mu, sigma, len(data), rng)
    n = len(d)
    start = rng.randint(0, n - 16)  # 留出 1 点尾部缓冲
    tight_sigma = sigma * rng.uniform(0.25, 0.42)

    # 14 点：明确在 ±0.7σ 内（保守范围，使 sample_s 不会被压缩太多）
    for k in range(14):
        d[start + k] = mu + rng.gauss(0, tight_sigma)
        d[start + k] = max(mu - 0.70 * sigma, min(mu + 0.70 * sigma, d[start + k]))

    # 先不设第 15 点，计算当前 sample_s
    current_s = _stat.stdev(d)

    # 第 15 点：基于当前 sample_s，取 0.75~0.97 倍（确保在 ±sample_s 内）
    edge_z = rng.uniform(0.75, 0.97) * current_s
    d[start + 14] = mu + rng.choice([1, -1]) * edge_z

    # 确保 start+15（如存在）越过 ±sample_s（防止 run 继续）
    if start + 15 < n:
        final_s = _stat.stdev(d)
        direction = rng.choice([1, -1])
        d[start + 15] = mu + direction * rng.uniform(1.1, 2.0) * final_s

    return d, True


def _inject_rule7_near_miss(data: list[float], mu: float, sigma: float,
                            rng: random.Random) -> tuple[list[float], bool]:
    """注入 rule7 近似：13~14 点在 ±1σ 内（不触发），后接越界点。"""
    d = _fresh_normal_data(mu, sigma, len(data), rng)
    n = len(d)
    run_len = rng.choice([13, 14])
    start = rng.randint(0, n - run_len - 1)
    tight_sigma = sigma * rng.uniform(0.25, 0.45)

    for k in range(run_len):
        d[start + k] = mu + rng.gauss(0, tight_sigma)
        d[start + k] = max(mu - 0.92 * sigma, min(mu + 0.92 * sigma, d[start + k]))

    # 紧接的点越过 ±1σ（打断可能触发 rule7 的窗口）
    if start + run_len < n:
        direction = rng.choice([1, -1])
        d[start + run_len] = mu + direction * rng.uniform(1.1, 1.9) * sigma

    return d, False


# ── API 调用 ──────────────────────────────────────────────────────────────────

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
                raise RuntimeError(f"Claude API 失败（{retries} 次重试）: {e}")
    return ""


def compute_recall(pred: list[str], gt: list[str]) -> float:
    gt_set = set(gt)
    if not gt_set:
        return 1.0 if not pred else 0.5
    return len(set(pred) & gt_set) / len(gt_set)


# ── 单样本生成 ────────────────────────────────────────────────────────────────

_INJECTION_MODES = {
    "rule2_boundary":  (_inject_rule2_boundary,  "rule2"),
    "rule2_near_miss": (_inject_rule2_near_miss,  None),
    "rule7_boundary":  (_inject_rule7_boundary,  "rule7"),
    "rule7_near_miss": (_inject_rule7_near_miss,  None),
}


def generate_one(idx: int, mode: str, rng: random.Random, min_recall: float) -> dict | None:
    """生成一条边界/近似样本，验证通过则返回。"""
    injector_fn, expected_rule = _INJECTION_MODES[mode]

    mu = rng.uniform(9.5, 10.5)
    sigma = rng.uniform(0.15, 0.45)
    usl_margin = rng.uniform(4.0, 7.0) * sigma
    lsl_margin = rng.uniform(4.0, 7.0) * sigma
    usl = round(mu + usl_margin, 2)
    lsl = round(mu - lsl_margin, 2)

    base_data = [rng.gauss(mu, sigma) for _ in range(25)]
    data, _ = injector_fn(base_data, mu, sigma, rng)
    data_rounded = [round(x, 3) for x in data]

    # 用规则引擎验证实际 violations
    sample_s = statistics.stdev(data)
    gt_violations = check_nelson_rules(data, mu, sample_s)
    cpk_detail = compute_cpk_detailed(data, usl, lsl)
    gt_cpk = cpk_detail.get("cpk")

    # 验证注入是否符合预期（near_miss 不应触发目标规则）
    target_rule = f"rule{mode.split('_')[0][-1]}" if "rule2" in mode else (
        "rule7" if "rule7" in mode else None
    )
    # 用 target rule 字符来区分
    if "rule2" in mode:
        tr = "rule2"
    else:
        tr = "rule7"

    if "near_miss" in mode:
        if tr in gt_violations:
            return None  # 应该不触发但触发了，丢弃
    else:
        if tr not in gt_violations:
            return None  # 应该触发但没触发，丢弃

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
        print(f"  [样本{idx}] API错误：{e}", flush=True)
        return None

    pred_violations = extract_violations(response)
    recall = compute_recall(pred_violations, gt_violations)
    if recall < min_recall:
        print(
            f"  [样本{idx}] 验证未过：recall={recall:.2f}，"
            f"gt={gt_violations}，pred={pred_violations}",
            flush=True,
        )
        return None

    return {
        "system": SPC_SYSTEM_WITH_SKILL,
        "instruction": SPC_INSTRUCTION,
        "input": input_text,
        "output": response,
        "ground_truth": {"violations": gt_violations, "cpk": gt_cpk},
        "meta": {
            "source": "boundary_rule27",
            "mode": mode,
            "model": MODEL,
            "actual_violations": gt_violations,
        },
    }


def generate_for_mode(
    mode: str,
    n: int,
    rng: random.Random,
    min_recall: float,
    concurrency: int,
) -> list[dict]:
    """生成指定模式的 n 条样本。"""
    collected: list[dict] = []
    attempted = 0
    max_attempts = n * 6

    print(f"\n{'─'*50}")
    print(f"[{mode}] 目标={n}，最大尝试={max_attempts}，并发={concurrency}")

    while len(collected) < n and attempted < max_attempts:
        batch = min(concurrency * 2, max_attempts - attempted, n - len(collected) + concurrency)
        batch_rngs = [random.Random(rng.randint(0, 2**31)) for _ in range(batch)]
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(generate_one, attempted + j, mode, batch_rngs[j], min_recall): j
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

    print(f"[{mode}] 完成：{len(collected)}/{n}，总尝试：{attempted}")
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="rule2/rule7 边界样本生成器")
    parser.add_argument("--n_rule2", type=int, default=90, help="rule2 目标样本数（边界+近似各半）")
    parser.add_argument("--n_rule7", type=int, default=90, help="rule7 目标样本数（边界+近似各半）")
    parser.add_argument("--output_ws", default="round3/data/boundary_ws.jsonl")
    parser.add_argument("--output_ns", default="round3/data/boundary_ns.jsonl")
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--min_recall", type=float, default=0.8)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    n_r2_boundary = args.n_rule2 // 2
    n_r2_near = args.n_rule2 - n_r2_boundary
    n_r7_boundary = args.n_rule7 // 2
    n_r7_near = args.n_rule7 - n_r7_boundary

    all_samples: list[dict] = []
    for mode, n in [
        ("rule2_boundary",  n_r2_boundary),
        ("rule2_near_miss", n_r2_near),
        ("rule7_boundary",  n_r7_boundary),
        ("rule7_near_miss", n_r7_near),
    ]:
        samples = generate_for_mode(mode, n, rng, args.min_recall, args.concurrency)
        all_samples.extend(samples)

    # 写入 ws 格式
    Path(args.output_ws).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_ws, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n✓ WS 数据写入：{args.output_ws}（{len(all_samples)} 条）")

    # 转换为 ns 格式
    Path(args.output_ns).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_ns, "w", encoding="utf-8") as f:
        for s in all_samples:
            ns = dict(s)
            ns["system"] = NO_SKILL_SYSTEM
            f.write(json.dumps(ns, ensure_ascii=False) + "\n")
    print(f"✓ NS 数据写入：{args.output_ns}（{len(all_samples)} 条）")

    # 统计
    from collections import Counter
    mode_counts = Counter(s["meta"]["mode"] for s in all_samples)
    print("\n分布：")
    for mode, cnt in sorted(mode_counts.items()):
        print(f"  {mode}: {cnt}")


if __name__ == "__main__":
    main()
