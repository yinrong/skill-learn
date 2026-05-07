#!/usr/bin/env python3
"""Universal batch monitor: watch training logs, auto-launch evals when done.

Usage:
    python3 monitor_batch.py --batch 7
    python3 monitor_batch.py --exps expZZ expAAA expBBB expCCC --gpu 0 1 2 3 --n 200 200 400 400 --model 14B

The script is designed to be launched by batch launchers AFTER training starts.
It monitors training, launches eval_and_report.sh, and reports final F1.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HISTDIR = Path("history-route2.1.1")
MODEL_14B = "/home/yinrong/models/Qwen3-14B"
MODEL_32B = "/home/yinrong/models/Qwen3-32B"
MODEL_8B  = "/home/yinrong/models/Qwen3-8B"

# Per-batch experiment definitions
BATCH_DEFS = {
    7: {
        "exps": ["expZZ", "expAAA", "expBBB", "expCCC"],
        "gpu":  [0, 1, 2, 3],
        "n":    [200, 200, 400, 400],
        "model": [MODEL_14B, MODEL_14B, MODEL_14B, MODEL_14B],
    },
    8: {
        "exps": ["expGGG", "expHHH"],
        "gpu":  [0, 4],
        "n":    [200, 200],  # will be updated based on best B7
        "model": [MODEL_32B, MODEL_32B],
    },
    9: {
        "exps": ["expIII", "expJJJ", "expKKK", "expLLL", "expMMM"],
        "gpu":  [5, 6, 2, 3, 4],  # pre-launched: III→5, JJJ→6, KKK→2, LLL→3, MMM→4
        "n":    [200, 400, 200, 200, 200],
        "model": [MODEL_14B, MODEL_14B, MODEL_14B, MODEL_8B, MODEL_14B],
    },
    10: {
        # Expanded to 8 GPUs; TTT/UUU/VVV added as critical new experiments
        "exps": ["expNNN", "expOOO", "expPPP", "expQQQ", "expSSS", "expTTT", "expUUU", "expVVV"],
        "gpu":  [0, 1, 2, 3, 4, 5, 6, 7],
        "n":    [200, 200, 400, 200, 200, 500, 500, 200],
        "model": [MODEL_32B, MODEL_14B, MODEL_14B, MODEL_8B, MODEL_14B, MODEL_14B, MODEL_32B, MODEL_14B],
    },
    11: {
        # Designed adaptively by adaptive_planner_b11.py; placeholder for watchdog
        "exps": ["expWWW", "expXXX", "expYYY", "expZZZ"],
        "gpu":  [0, 1, 2, 3],
        "n":    [500, 500, 500, 200],
        "model": [MODEL_14B, MODEL_14B, MODEL_14B, MODEL_32B],
    },
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_monitor(exps_cfg: list[dict], max_tokens: int = 5000):
    """Monitor a list of {exp, gpu, n, model} dicts."""
    done = {}
    eval_launched = {}
    eval_done = {}

    while True:
        all_finished = True

        for cfg in exps_cfg:
            exp = cfg["exp"]

            if eval_done.get(exp):
                continue

            all_finished = False

            train_log = HISTDIR / "logs" / f"{exp}_train.log"
            result_path = HISTDIR / "results" / f"{exp}.json"

            # Check training complete
            if not done.get(exp):
                if train_log.exists():
                    text = train_log.read_text()
                    if "train_runtime" in text:
                        done[exp] = True
                        log(f"✅ {exp}: training complete")

            # Launch eval
            if done.get(exp) and not eval_launched.get(exp):
                if result_path.exists():
                    try:
                        d = json.loads(result_path.read_text())
                        f1 = d["summary"]["rule_detection_f1"]
                        log(f"⏭  {exp}: result already exists (F1={f1:.3f}), skipping eval")
                        eval_launched[exp] = eval_done[exp] = True
                    except Exception:
                        pass
                    continue

                eval_launched[exp] = True
                log(f"📊 {exp}: launching eval on GPU {cfg['gpu']}...")
                proc = subprocess.Popen(
                    ["bash", "eval_and_report.sh", exp, cfg["model"],
                     str(cfg["gpu"]), str(cfg["n"]), str(max_tokens)],
                    stdout=open(HISTDIR / "logs" / f"{exp}_eval_full.log", "w"),
                    stderr=subprocess.STDOUT,
                )
                log(f"   eval PID={proc.pid}")

            # Check eval done
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

            # Progress
            if not done.get(exp) and train_log.exists():
                import re
                lines = train_log.read_text().splitlines()
                for line in reversed(lines):
                    m = re.search(r"(\d+)/(\d+)\s+\[", line)
                    if m:
                        log(f"⏳ {exp}: {m.group(1)}/{m.group(2)}")
                        break

        if all(eval_done.get(cfg["exp"]) for cfg in exps_cfg):
            log("✅ All experiments complete!")
            break

        time.sleep(60)

    # Summary
    print("\n=== Results Summary ===")
    for cfg in exps_cfg:
        exp = cfg["exp"]
        rp = HISTDIR / "results" / f"{exp}.json"
        if rp.exists():
            try:
                d = json.loads(rp.read_text())
                s = d["summary"]
                print(f"  {exp}: F1={s['rule_detection_f1']:.3f} cpk={s['cpk_found_rate']:.3f}")
                print(f"    per_rule: {s.get('per_rule_recall', {})}")
            except Exception as e:
                print(f"  {exp}: error reading result ({e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, help="Batch number (7-11)")
    parser.add_argument("--exps", nargs="+", help="Experiment names")
    parser.add_argument("--gpu", nargs="+", type=int, help="GPU indices")
    parser.add_argument("--n", nargs="+", type=int, help="N_TRAIN values")
    parser.add_argument("--model", nargs="+", help="Model paths or shortcuts (14B/32B/8B)")
    parser.add_argument("--max_tokens", type=int, default=5000)
    args = parser.parse_args()

    if args.batch:
        defn = BATCH_DEFS[args.batch]
        exps_cfg = [
            {"exp": exp, "gpu": gpu, "n": n, "model": mdl}
            for exp, gpu, n, mdl in zip(
                defn["exps"], defn["gpu"], defn["n"], defn["model"]
            )
        ]
    elif args.exps:
        model_map = {"14B": MODEL_14B, "32B": MODEL_32B, "8B": MODEL_8B}
        models = [model_map.get(m, m) for m in args.model]
        exps_cfg = [
            {"exp": exp, "gpu": gpu, "n": n, "model": mdl}
            for exp, gpu, n, mdl in zip(args.exps, args.gpu, args.n, models)
        ]
    else:
        parser.error("Specify --batch or --exps/--gpu/--n/--model")

    os.chdir(Path(__file__).parent)
    log(f"Starting monitor for: {[c['exp'] for c in exps_cfg]}")
    run_monitor(exps_cfg, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
