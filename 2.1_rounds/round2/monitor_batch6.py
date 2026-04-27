#!/usr/bin/env python3
"""Monitor Batch 6 training and auto-launch evals when each job finishes."""
import json, os, subprocess, sys, time
from pathlib import Path

HISTDIR = Path("history-route2.1.1")
MODEL = "/home/yinrong/models/Qwen3-14B"

EXPS = {
    "expWW":  {"gpu": 0, "n": 500,  "model": MODEL},
    "expXX":  {"gpu": 1, "n": 500,  "model": MODEL},
    "expVV":  {"gpu": 4, "n": 800,  "model": MODEL},
    "expYY2": {"gpu": 7, "n": 600,  "model": MODEL},
}

done = {}
eval_launched = {}
eval_done = {}

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

while True:
    for exp, cfg in EXPS.items():
        if eval_done.get(exp):
            continue

        train_log = HISTDIR / "logs" / f"{exp}_train.log"
        result_path = HISTDIR / "results" / f"{exp}.json"

        # Check training complete
        if not done.get(exp):
            if train_log.exists() and "train_runtime" in train_log.read_text():
                done[exp] = True
                log(f"✅ {exp}: training complete")

        # Launch eval if training done and not yet launched
        if done.get(exp) and not eval_launched.get(exp):
            if result_path.exists():
                eval_done[exp] = True
                log(f"⏭  {exp}: result already exists, skipping eval")
                continue
            eval_launched[exp] = True
            log(f"📊 {exp}: launching eval on GPU {cfg['gpu']}...")
            subprocess.Popen(
                ["bash", "eval_and_report.sh", exp, cfg["model"],
                 str(cfg["gpu"]), str(cfg["n"]), "5000"],
                stdout=open(HISTDIR / "logs" / f"{exp}_eval_full.log", "w"),
                stderr=subprocess.STDOUT,
            )

        # Check eval done
        if eval_launched.get(exp) and not eval_done.get(exp):
            if result_path.exists():
                try:
                    d = json.loads(result_path.read_text())
                    f1 = d["summary"]["rule_detection_f1"]
                    cpk = d["summary"]["cpk_found_rate"]
                    log(f"🎯 {exp}: F1={f1:.3f} cpk={cpk:.3f}")
                    eval_done[exp] = True
                except Exception:
                    pass

        # Progress
        if not done.get(exp) and train_log.exists():
            lines = train_log.read_text().splitlines()
            step_lines = [l for l in lines if "/189" in l or "/200" in l or "/225" in l or "/189" in l]
            if step_lines:
                last = step_lines[-1].strip()
                # Extract step fraction
                import re
                m = re.search(r"(\d+)/(\d+) \[", last)
                if m:
                    log(f"⏳ {exp}: {m.group(1)}/{m.group(2)}")

    # Check all done
    if all(eval_done.get(exp) for exp in EXPS):
        log("✅ All Batch 6 complete!")
        # Print summary
        print("\n=== Batch 6 Results ===")
        for exp in EXPS:
            rp = HISTDIR / "results" / f"{exp}.json"
            if rp.exists():
                d = json.loads(rp.read_text())
                s = d["summary"]
                print(f"  {exp}: F1={s['rule_detection_f1']:.3f} cpk={s['cpk_found_rate']:.3f}")
                print(f"    per_rule: {s['per_rule_recall']}")
        sys.exit(0)

    time.sleep(60)
