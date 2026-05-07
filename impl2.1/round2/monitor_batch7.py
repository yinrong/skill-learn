#!/usr/bin/env python3
"""Monitor Batch 7 training on GPUs 2,3,5,6 and auto-launch evals."""
import json, os, re, subprocess, sys, time
from pathlib import Path

HISTDIR = Path("history-route2.1.1")
MODEL = "/home/yinrong/models/Qwen3-14B"

# Batch 7 launched on GPUs 2,3,5,6 (not 0,1,2,3 as in the launcher default)
EXPS = {
    "expZZ":  {"gpu": 2, "n": 200, "model": MODEL, "total_steps": 75},
    "expAAA": {"gpu": 5, "n": 200, "model": MODEL, "total_steps": 125},
    "expBBB": {"gpu": 3, "n": 400, "model": MODEL, "total_steps": 100},
    "expCCC": {"gpu": 6, "n": 400, "model": MODEL, "total_steps": 150},
}

done = {}
eval_launched = {}
eval_done = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


os.chdir(Path(__file__).parent)
log(f"Batch 7 monitor starting for: {list(EXPS.keys())}")

while True:
    for exp, cfg in EXPS.items():
        if eval_done.get(exp):
            continue

        train_log = HISTDIR / "logs" / f"{exp}_train.log"
        result_path = HISTDIR / "results" / f"{exp}.json"

        if not done.get(exp):
            if train_log.exists() and "train_runtime" in train_log.read_text():
                done[exp] = True
                log(f"✅ {exp}: training complete")

        if done.get(exp) and not eval_launched.get(exp):
            if result_path.exists():
                try:
                    d = json.loads(result_path.read_text())
                    f1 = d["summary"]["rule_detection_f1"]
                    log(f"⏭  {exp}: result exists (F1={f1:.3f}), skipping eval")
                    eval_launched[exp] = eval_done[exp] = True
                except Exception:
                    pass
                continue
            eval_launched[exp] = True
            log(f"📊 {exp}: launching eval on GPU {cfg['gpu']}...")
            proc = subprocess.Popen(
                ["bash", "eval_and_report.sh", exp, cfg["model"],
                 str(cfg["gpu"]), str(cfg["n"]), "5000"],
                stdout=open(HISTDIR / "logs" / f"{exp}_eval_full.log", "w"),
                stderr=subprocess.STDOUT,
            )
            log(f"   eval PID={proc.pid}")

        if eval_launched.get(exp) and not eval_done.get(exp):
            if result_path.exists():
                try:
                    d = json.loads(result_path.read_text())
                    f1 = d["summary"]["rule_detection_f1"]
                    cpk = d["summary"]["cpk_found_rate"]
                    pr = d["summary"].get("per_rule_recall", {})
                    log(f"🎯 {exp}: F1={f1:.3f} cpk={cpk:.3f} per_rule={pr}")
                    eval_done[exp] = True
                except Exception:
                    pass

        if not done.get(exp) and train_log.exists():
            text = train_log.read_text().splitlines()
            for line in reversed(text):
                m = re.search(rf"(\d+)/{cfg['total_steps']}\s+\[", line)
                if m:
                    log(f"⏳ {exp}: {m.group(1)}/{cfg['total_steps']}")
                    break

    if all(eval_done.get(exp) for exp in EXPS):
        log("✅ Batch 7 complete!")
        print("\n=== Batch 7 Results ===")
        for exp in EXPS:
            rp = HISTDIR / "results" / f"{exp}.json"
            if rp.exists():
                d = json.loads(rp.read_text())
                s = d["summary"]
                print(f"  {exp}: F1={s['rule_detection_f1']:.3f} cpk={s['cpk_found_rate']:.3f}")
                print(f"    per_rule: {s.get('per_rule_recall', {})}")
        sys.exit(0)

    time.sleep(60)
