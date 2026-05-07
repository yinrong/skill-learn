#!/usr/bin/env python3
"""
Agent-Beta: Adaptive Batch 10 Planner & Launcher
Waits for B8+B9 results, analyzes them, updates configs if needed,
then trains and evals all 8 B10 experiments.

Run: nohup python3 adaptive_planner_b10.py > history-route2.1.1/logs/b10_planner.log 2>&1 &
"""
from __future__ import annotations
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HISTDIR = Path("history-route2.1.1")
MODEL_8B  = "/home/yinrong/models/Qwen3-8B"
MODEL_14B = "/home/yinrong/models/Qwen3-14B"
MODEL_32B = "/home/yinrong/models/Qwen3-32B"

# B10 experiment definitions (post-update with TTT/UUU/VVV)
B10_EXPS = {
    "expNNN": {"gpu": 0, "n": 200, "model": MODEL_32B, "steps": 125},
    "expOOO": {"gpu": 1, "n": 200, "model": MODEL_14B, "steps": 125},
    "expPPP": {"gpu": 2, "n": 400, "model": MODEL_14B, "steps": 150},
    "expQQQ": {"gpu": 3, "n": 200, "model": MODEL_8B,  "steps": 125},
    "expSSS": {"gpu": 4, "n": 200, "model": MODEL_14B, "steps": 125},
    "expTTT": {"gpu": 5, "n": 500, "model": MODEL_14B, "steps": 125},  # 250ws+250ns
    "expUUU": {"gpu": 6, "n": 500, "model": MODEL_32B, "steps": 125},  # 32B 250ws+250ns
    "expVVV": {"gpu": 7, "n": 200, "model": MODEL_14B, "steps": 125},
}

B9_EXPS = ["expIII", "expJJJ", "expKKK", "expLLL", "expMMM"]
B8_EXPS = ["expGGG", "expHHH"]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def results_ready(exps: list[str]) -> bool:
    return all((HISTDIR / f"results/{e}.json").exists() for e in exps)


def load_result(exp: str) -> dict | None:
    p = HISTDIR / f"results/{exp}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())["summary"]
        except Exception:
            pass
    return None


def training_done(exp: str) -> bool:
    log_p = HISTDIR / f"logs/{exp}_train.log"
    return log_p.exists() and "train_runtime" in log_p.read_text()


def get_step(exp: str) -> str:
    log_p = HISTDIR / f"logs/{exp}_train.log"
    if not log_p.exists():
        return "?"
    for line in reversed(log_p.read_text().split("\n")):
        m = re.search(r"(\d+/\d+) \[", line)
        if m:
            return m.group(1)
    return "0/?"


def analyze_b9(results: dict) -> dict:
    """Analyze B9 results and return any config adjustments for B10."""
    adjustments = {}
    log("=== B9 Analysis ===")
    for exp, s in results.items():
        r8 = s.get("per_rule_recall", {}).get("rule8", 0)
        log(f"  {exp}: F1={s['rule_detection_f1']:.3f}  r8={r8:.2f}")

    # Check if lower LR (expKKK) beats standard (expMMM)
    kkk = results.get("expKKK")
    mmm = results.get("expMMM")
    if kkk and mmm:
        if kkk["rule_detection_f1"] > mmm["rule_detection_f1"] + 0.01:
            log("  → LR=5e-5 beats 1e-4 by >0.01: updating expOOO to use LR=5e-5 (already planned)")
            # expOOO already uses LR=5e-5, no change needed
        else:
            log(f"  → LR comparison: KKK={kkk['rule_detection_f1']:.3f} vs MMM={mmm['rule_detection_f1']:.3f}")

    # Check if v3 pool (expIII) beats v1 pool (expMMM/KKK)
    iii = results.get("expIII")
    if iii and mmm:
        if iii["rule_detection_f1"] > mmm["rule_detection_f1"] + 0.01:
            log("  → v3 pool significantly beats v1: expSSS (v3) likely strong in B10")
        else:
            log(f"  → v3 vs v1: III={iii['rule_detection_f1']:.3f} vs MMM={mmm['rule_detection_f1']:.3f}")

    # Identify winner
    best_exp = max(results, key=lambda x: results[x]["rule_detection_f1"])
    best_f1 = results[best_exp]["rule_detection_f1"]
    log(f"  Best B9: {best_exp} F1={best_f1:.3f}")

    return adjustments


def analyze_b8(results: dict) -> dict:
    log("=== B8 Analysis (32B) ===")
    for exp, s in results.items():
        r8 = s.get("per_rule_recall", {}).get("rule8", 0)
        log(f"  {exp}: F1={s['rule_detection_f1']:.3f}  r8={r8:.2f}")

    # Compare 32B vs best 14B
    best_14b = 0.385  # expXX
    for exp, s in results.items():
        f1 = s["rule_detection_f1"]
        if f1 > best_14b:
            log(f"  → 32B ({exp}) beats best 14B! F1={f1:.3f} > {best_14b:.3f}")
        else:
            log(f"  → 32B ({exp}) F1={f1:.3f} vs best 14B {best_14b:.3f}")
    return {}


def kill_vllm():
    log("Killing vLLM servers...")
    try:
        subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"],
                       capture_output=True)
    except Exception:
        pass
    time.sleep(10)


def start_training(exp: str, cfg: dict):
    log_p = HISTDIR / f"logs/{exp}_train.log"
    proc = subprocess.Popen(
        ["llamafactory-cli", "train", str(HISTDIR / f"configs/{exp}.yaml")],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["gpu"]),
             "DISABLE_VERSION_CHECK": "1"},
        stdout=open(log_p, "w"),
        stderr=subprocess.STDOUT,
    )
    log(f"  {exp}: training started PID={proc.pid} GPU={cfg['gpu']}")
    return proc


def start_eval(exp: str, cfg: dict):
    log_p = HISTDIR / f"logs/{exp}_eval_full.log"
    proc = subprocess.Popen(
        ["bash", "eval_and_report.sh", exp, cfg["model"],
         str(cfg["gpu"]), str(cfg["n"]), "5000"],
        stdout=open(log_p, "w"),
        stderr=subprocess.STDOUT,
    )
    log(f"  {exp}: eval started PID={proc.pid} GPU={cfg['gpu']}")
    return proc


# ── Wait phase ──────────────────────────────────────────────────────────────
log("Agent-Beta (Adaptive B10 Planner) started.")
log(f"Waiting for B8 results ({B8_EXPS}) + B9 results ({B9_EXPS})...")

while True:
    b8_done = results_ready(B8_EXPS)
    b9_done = results_ready(B9_EXPS)
    if b8_done and b9_done:
        log("All B8 + B9 results ready. Starting B10.")
        break

    missing_b8 = [e for e in B8_EXPS if not (HISTDIR / f"results/{e}.json").exists()]
    missing_b9 = [e for e in B9_EXPS if not (HISTDIR / f"results/{e}.json").exists()]
    log(f"Waiting... B8 missing={missing_b8}, B9 missing={missing_b9}")
    for exp_list in [B8_EXPS, B9_EXPS]:
        for exp in exp_list:
            r = load_result(exp)
            if r:
                log(f"  ✓ {exp}: F1={r['rule_detection_f1']:.3f}")
            else:
                if training_done(exp):
                    log(f"  🔄 {exp}: training done, eval in progress...")
                else:
                    step = get_step(exp)
                    log(f"  ⏳ {exp}: step {step}")
    time.sleep(120)

# ── Analysis phase ───────────────────────────────────────────────────────────
b9_results = {e: load_result(e) for e in B9_EXPS if load_result(e)}
b8_results = {e: load_result(e) for e in B8_EXPS if load_result(e)}
analyze_b9(b9_results)
analyze_b8(b8_results)

# Print all-time best
log("=== All-time Top 10 ===")
all_results = []
for p in sorted((HISTDIR / "results").glob("exp*.json")):
    try:
        s = json.loads(p.read_text())["summary"]
        all_results.append((s["rule_detection_f1"], p.stem, s.get("per_rule_recall", {}).get("rule8", 0)))
    except Exception:
        pass
for f1, exp, r8 in sorted(all_results, reverse=True)[:10]:
    log(f"  {exp}: F1={f1:.3f}  r8={r8:.2f}")

# Run prepare_batch10.sh to create data + configs
log("Running prepare_batch10.sh...")
result = subprocess.run(["bash", "prepare_batch10.sh"], capture_output=True, text=True)
log(result.stdout)
if result.returncode != 0:
    log(f"WARNING: prepare_batch10.sh returned {result.returncode}: {result.stderr}")

# Kill vLLM
kill_vllm()

# ── Training phase ────────────────────────────────────────────────────────────
log("Starting Batch 10 training (8 GPUs)...")
for exp, cfg in B10_EXPS.items():
    start_training(exp, cfg)
    time.sleep(5)

# ── Monitor phase ─────────────────────────────────────────────────────────────
log("Monitoring Batch 10 training + eval...")
train_done = {exp: False for exp in B10_EXPS}
eval_started = {exp: False for exp in B10_EXPS}

while True:
    for exp, cfg in B10_EXPS.items():
        if train_done[exp]:
            continue
        if training_done(exp):
            train_done[exp] = True
            result_p = HISTDIR / f"results/{exp}.json"
            if result_p.exists():
                f1 = json.loads(result_p.read_text())["summary"]["rule_detection_f1"]
                log(f"🎯 {exp}: eval already done F1={f1:.3f}")
                eval_started[exp] = True
                continue
            if not eval_started[exp]:
                eval_started[exp] = True
                start_eval(exp, cfg)
        else:
            step = get_step(exp)
            log(f"⏳ {exp} (GPU {cfg['gpu']}): step {step}")

    # Check eval completions
    for exp, cfg in B10_EXPS.items():
        if eval_started[exp]:
            result_p = HISTDIR / f"results/{exp}.json"
            if result_p.exists():
                try:
                    s = json.loads(result_p.read_text())["summary"]
                    r8 = s.get("per_rule_recall", {}).get("rule8", 0)
                    log(f"🏁 {exp}: F1={s['rule_detection_f1']:.3f}  r8={r8:.2f}")
                except Exception:
                    pass

    if all(train_done.values()):
        pending = sum(1 for e in B10_EXPS
                      if eval_started[e] and not (HISTDIR / f"results/{e}.json").exists())
        if pending == 0:
            log("✅ Batch 10 complete!")
            break

    time.sleep(120)

# ── Final summary ─────────────────────────────────────────────────────────────
log("=== Batch 10 Final Results ===")
for exp in B10_EXPS:
    s = load_result(exp)
    if s:
        r8 = s.get("per_rule_recall", {}).get("rule8", 0)
        log(f"  {exp}: F1={s['rule_detection_f1']:.3f}  cpk={s.get('cpk_found_rate', 0):.2f}  r8={r8:.2f}")
log("Agent-Beta B10 planner exiting.")
