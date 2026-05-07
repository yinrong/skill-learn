#!/usr/bin/env python3
"""Monitor pre-launched Batch 9 experiments.
Pre-launch GPU mapping: expIII→5, expJJJ→6, expKKK→2, expLLL→3, expMMM→4
Waits for each to finish training, then runs eval on same GPU.
Also waits for Batch 8 evals to finish before running B9 evals (avoid vLLM conflicts on GPU 0,1).
Run: nohup python3 monitor_batch9_prelaunched.py > history-route2.1.1/logs/batch9_monitor.log 2>&1 &
"""
from __future__ import annotations
import json
import os
import subprocess
import time
from pathlib import Path

HISTDIR = Path("history-route2.1.1")
MODEL_14B = "/home/yinrong/models/Qwen3-14B"
MODEL_8B = "/home/yinrong/models/Qwen3-8B"

EXPS = {
    "expIII": {"gpu": 5, "n": 200, "model": MODEL_14B},
    "expJJJ": {"gpu": 6, "n": 400, "model": MODEL_14B},
    "expKKK": {"gpu": 2, "n": 200, "model": MODEL_14B},
    "expLLL": {"gpu": 3, "n": 200, "model": MODEL_8B},
    "expMMM": {"gpu": 4, "n": 200, "model": MODEL_14B},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def b8_evals_done():
    return all((HISTDIR / f"results/{e}.json").exists() for e in ["expGGG", "expHHH"])


def training_done(exp):
    log_path = HISTDIR / f"logs/{exp}_train.log"
    if not log_path.exists():
        return False
    return "train_runtime" in log_path.read_text()


def start_eval(exp, cfg):
    log(f"✅ {exp}: training complete, starting eval (GPU {cfg['gpu']})...")
    proc = subprocess.Popen(
        ["bash", "eval_and_report.sh", exp, cfg["model"],
         str(cfg["gpu"]), str(cfg["n"]), "5000"],
        stdout=open(HISTDIR / f"logs/{exp}_eval_full.log", "w"),
        stderr=subprocess.STDOUT,
    )
    log(f"📊 {exp}: eval started PID={proc.pid}")
    return proc


done = {exp: False for exp in EXPS}
eval_started = {exp: False for exp in EXPS}

log("Batch 9 pre-launched monitor started.")
log(f"Experiments: {list(EXPS.keys())}")

while True:
    # Check if Batch 8 evals are done (needed to free GPU 0,1 vLLM processes)
    b8_done = b8_evals_done()
    if not b8_done:
        log("⏳ Waiting for Batch 8 evals (expGGG, expHHH)...")

    for exp, cfg in EXPS.items():
        if done[exp]:
            continue
        result_path = HISTDIR / f"results/{exp}.json"
        if result_path.exists():
            done[exp] = True
            f1 = json.loads(result_path.read_text())["summary"]["rule_detection_f1"]
            log(f"🎯 {exp}: F1={f1}")
            continue
        if training_done(exp):
            if not eval_started[exp]:
                # Wait for B8 evals to finish before starting B9 evals
                # (to avoid vLLM conflicts when GPU 0/1 are still busy)
                if not b8_done:
                    log(f"⌛ {exp}: training done, waiting for B8 evals before starting eval...")
                else:
                    eval_started[exp] = True
                    start_eval(exp, cfg)
        else:
            log_path = HISTDIR / f"logs/{exp}_train.log"
            step = "?"
            if log_path.exists():
                lines = log_path.read_text().split("\n")
                for line in reversed(lines):
                    if "/125 [" in line or "/150 [" in line or "/75 [" in line:
                        import re
                        m = re.search(r'(\d+/\d+) \[', line)
                        if m:
                            step = m.group(1)
                        break
            log(f"⏳ {exp} (GPU {cfg['gpu']}): step {step}")

    # Check all done
    if all(done.values()):
        log("✅ All Batch 9 experiments complete!")
        break

    # Re-check B8 and start pending evals
    if b8_done:
        for exp, cfg in EXPS.items():
            if training_done(exp) and not eval_started[exp] and not done[exp]:
                result_path = HISTDIR / f"results/{exp}.json"
                if not result_path.exists():
                    eval_started[exp] = True
                    start_eval(exp, cfg)

    time.sleep(120)

log("Monitor exiting.")
