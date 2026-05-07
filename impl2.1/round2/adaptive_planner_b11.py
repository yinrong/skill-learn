#!/usr/bin/env python3
"""
Agent-Beta B11: Adaptive Batch 11 Planner
Analyzes B10 results, dynamically designs B11 experiments, trains and evals.
B11 focuses on: best recipe × scale, step count refinement, final combination.

Run: nohup python3 adaptive_planner_b11.py > history-route2.1.1/logs/b11_planner.log 2>&1 &
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import time
from pathlib import Path

HISTDIR = Path("history-route2.1.1")
MODEL_8B  = "/home/yinrong/models/Qwen3-8B"
MODEL_14B = "/home/yinrong/models/Qwen3-14B"
MODEL_32B = "/home/yinrong/models/Qwen3-32B"

B10_EXPS = ["expNNN", "expOOO", "expPPP", "expQQQ", "expSSS", "expTTT", "expUUU", "expVVV"]
V4_NS = HISTDIR / "data/train_claude_teacher_v4_noskill.jsonl"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_result(exp: str):
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


def write_config(exp: str, model: str, dataset: str, epochs: int,
                 lr: float = 1e-4, cutoff: int = 5120,
                 batch_size: int = 2, grad_acc: int = 4) -> None:
    cfg = f"""model_name_or_path: {model}
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_{exp}
cutoff_len: {cutoff}
num_train_epochs: {epochs}
per_device_train_batch_size: {batch_size}
gradient_accumulation_steps: {grad_acc}
learning_rate: {lr:.1e}
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
flash_attn: sdpa
logging_steps: 5
save_strategy: epoch
save_total_limit: 1
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/{exp}
do_train: true
report_to: none
"""
    (HISTDIR / f"configs/{exp}.yaml").write_text(cfg)


def register_dataset(exp: str, file_path: str) -> None:
    info = json.loads(Path("data/dataset_info.json").read_text())
    key = f"spc_r5_{exp}"
    if key not in info:
        info[key] = {
            "file_name": file_path,
            "formatting": "alpaca",
            "columns": {"system": "system", "prompt": "instruction",
                        "query": "input", "response": "output"},
        }
        Path("data/dataset_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2))
        log(f"  Registered {key}")


def kill_vllm():
    log("Killing vLLM servers...")
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"],
                   capture_output=True)
    time.sleep(10)


def start_training(exp: str, gpu: int) -> subprocess.Popen:
    log_p = HISTDIR / f"logs/{exp}_train.log"
    proc = subprocess.Popen(
        ["llamafactory-cli", "train", str(HISTDIR / f"configs/{exp}.yaml")],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
             "DISABLE_VERSION_CHECK": "1"},
        stdout=open(log_p, "w"),
        stderr=subprocess.STDOUT,
    )
    log(f"  {exp}: training started PID={proc.pid} GPU={gpu}")
    return proc


def start_eval(exp: str, model: str, gpu: int, n: int):
    log_p = HISTDIR / f"logs/{exp}_eval_full.log"
    proc = subprocess.Popen(
        ["bash", "eval_and_report.sh", exp, model, str(gpu), str(n), "5000"],
        stdout=open(log_p, "w"),
        stderr=subprocess.STDOUT,
    )
    log(f"  {exp}: eval started PID={proc.pid} GPU={gpu}")
    return proc


# ── Wait for B10 ─────────────────────────────────────────────────────────────
log("Agent-Beta B11 Planner started. Waiting for B10 results...")

while True:
    missing = [e for e in B10_EXPS if not (HISTDIR / f"results/{e}.json").exists()]
    if not missing:
        log("All B10 results ready.")
        break
    log(f"Waiting... B10 missing={missing}")
    for exp in B10_EXPS:
        s = load_result(exp)
        if s:
            r8 = s.get("per_rule_recall", {}).get("rule8", 0)
            log(f"  ✓ {exp}: F1={s['rule_detection_f1']:.3f}  r8={r8:.2f}")
        elif training_done(exp):
            log(f"  🔄 {exp}: training done, eval in progress...")
        else:
            log(f"  ⏳ {exp}: step {get_step(exp)}")
    time.sleep(120)

# ── B10 Analysis & B11 Design ─────────────────────────────────────────────────
log("=== B10 Results Analysis ===")
b10_results = {e: load_result(e) for e in B10_EXPS if load_result(e)}

# All-time sorted
all_results = []
for p in sorted((HISTDIR / "results").glob("exp*.json")):
    try:
        s = json.loads(p.read_text())["summary"]
        all_results.append((s["rule_detection_f1"], p.stem,
                            s.get("per_rule_recall", {}).get("rule8", 0)))
    except Exception:
        pass
all_results.sort(reverse=True)
log("All-time Top 10:")
for f1, exp, r8 in all_results[:10]:
    log(f"  {exp}: F1={f1:.3f}  r8={r8:.2f}")

# Find best B10 experiment
best_b10 = max(b10_results, key=lambda x: b10_results[x]["rule_detection_f1"])
best_b10_f1 = b10_results[best_b10]["rule_detection_f1"]
log(f"Best B10: {best_b10} F1={best_b10_f1:.3f}")

# Infer best data strategy
ttt_f1 = b10_results.get("expTTT", {}).get("rule_detection_f1", 0)
nnn_f1 = b10_results.get("expNNN", {}).get("rule_detection_f1", 0)
uuu_f1 = b10_results.get("expUUU", {}).get("rule_detection_f1", 0)
ppp_f1 = b10_results.get("expPPP", {}).get("rule_detection_f1", 0)
vvv_f1 = b10_results.get("expVVV", {}).get("rule_detection_f1", 0)
sss_f1 = b10_results.get("expSSS", {}).get("rule_detection_f1", 0)

ws_ns_wins = ttt_f1 > max(nnn_f1, ppp_f1, sss_f1) + 0.005
pool32b_wins = uuu_f1 > ttt_f1 + 0.01
log(f"ws+ns strategy (TTT) wins: {ws_ns_wins} ({ttt_f1:.3f})")
log(f"32B advantage (UUU vs TTT): {pool32b_wins} ({uuu_f1:.3f} vs {ttt_f1:.3f})")

# ── Design B11 adaptively ────────────────────────────────────────────────────
log("Designing B11 experiments based on B10 findings...")

B11_EXPS = {}  # exp -> {gpu, n, model, data_builder}

# Always: test best B10 winner with more/fewer steps
if ws_ns_wins:
    # Best strategy: ws+ns_v2, test step variants
    best_ws_ns_data = str(HISTDIR / "data/train_expTTT.jsonl")  # 250ws+250ns_v2

    # expWWW: 3ep but with different seed pool (250ws + 250ns_v3)
    v3_ns = HISTDIR / "data/train_claude_teacher_v3_noskill.jsonl"
    v1_ws = HISTDIR / "data/train_claude_teacher.jsonl"
    if v3_ns.exists():
        wwwdata = HISTDIR / "data/train_expWWW.jsonl"
        lines = Path(v1_ws).read_text().split("\n")[:250]
        lines += Path(v3_ns).read_text().split("\n")[:250]
        wwwdata.write_text("\n".join(l for l in lines if l.strip()))
        register_dataset("expWWW", str(wwwdata))
        write_config("expWWW", MODEL_14B, "expWWW", epochs=3)
        B11_EXPS["expWWW"] = {"gpu": 0, "n": 500, "model": MODEL_14B}
        log("  expWWW: 250ws + 250ns_v3, 3ep (GPU 0)")

    # expXXX: ws+ns_v2 with MORE steps (5ep = 208 steps) — overfit test
    xxxdata = HISTDIR / "data/train_expXXX.jsonl"
    import shutil; shutil.copy(best_ws_ns_data, str(xxxdata))
    register_dataset("expXXX", str(xxxdata))
    write_config("expXXX", MODEL_14B, "expXXX", epochs=5)
    B11_EXPS["expXXX"] = {"gpu": 1, "n": 500, "model": MODEL_14B}
    log("  expXXX: 250ws + 250ns_v2, 5ep = 208 steps (GPU 1) — overfit test")

    # expYYY: ws+ns with v4 pool (250ws + 250ns_v4)
    v4_ns = HISTDIR / "data/train_claude_teacher_v4_noskill.jsonl"
    if v4_ns.exists() and len(v4_ns.read_text().split("\n")) >= 250:
        yyydata = HISTDIR / "data/train_expYYY.jsonl"
        lines = Path(v1_ws).read_text().split("\n")[:250]
        lines += Path(v4_ns).read_text().split("\n")[:250]
        yyydata.write_text("\n".join(l for l in lines if l.strip()))
        register_dataset("expYYY", str(yyydata))
        write_config("expYYY", MODEL_14B, "expYYY", epochs=3)
        B11_EXPS["expYYY"] = {"gpu": 2, "n": 500, "model": MODEL_14B}
        log("  expYYY: 250ws + 250ns_v4, 3ep (GPU 2)")
    else:
        # fallback: ws+ns_v2 with lower LR
        import shutil; shutil.copy(best_ws_ns_data, str(HISTDIR / "data/train_expYYY.jsonl"))
        register_dataset("expYYY", str(HISTDIR / "data/train_expYYY.jsonl"))
        write_config("expYYY", MODEL_14B, "expYYY", epochs=3, lr=5e-5)
        B11_EXPS["expYYY"] = {"gpu": 2, "n": 500, "model": MODEL_14B}
        log("  expYYY: 250ws + 250ns_v2, 3ep, LR=5e-5 (GPU 2)")
else:
    # Best strategy is pure ns — test variants
    v1ns = HISTDIR / "data/train_expRR.jsonl"

    # expWWW: pure ns_v2, fewer steps (3ep=75 steps)
    v2_ns = HISTDIR / "data/train_claude_teacher_v2_noskill.jsonl"
    wwwdata = HISTDIR / "data/train_expWWW.jsonl"
    import shutil; shutil.copy(str(v2_ns if v2_ns.exists() else v1ns), str(wwwdata))
    register_dataset("expWWW", str(wwwdata))
    write_config("expWWW", MODEL_14B, "expWWW", epochs=3)
    B11_EXPS["expWWW"] = {"gpu": 0, "n": 200, "model": MODEL_14B}
    log("  expWWW: ns_v2, 3ep (GPU 0)")

    # expXXX: ns_v1+v2 400 samples, 5ep = 250 steps (more data + more steps)
    xxxdata = HISTDIR / "data/train_expXXX.jsonl"
    lines = v1ns.read_text().split("\n")[:200]
    if v2_ns.exists():
        lines += v2_ns.read_text().split("\n")[:200]
    xxxdata.write_text("\n".join(l for l in lines if l.strip()))
    register_dataset("expXXX", str(xxxdata))
    write_config("expXXX", MODEL_14B, "expXXX", epochs=5)
    B11_EXPS["expXXX"] = {"gpu": 1, "n": 400, "model": MODEL_14B}
    log("  expXXX: 400(v1+v2), 5ep = 250 steps (GPU 1)")

    # expYYY: ns_v1+v3+v4 600 samples, 2ep = 150 steps (max diversity)
    yyydata = HISTDIR / "data/train_expYYY.jsonl"
    lines = v1ns.read_text().split("\n")[:200]
    for pool in [HISTDIR / "data/train_claude_teacher_v3_noskill.jsonl",
                 HISTDIR / "data/train_claude_teacher_v4_noskill.jsonl"]:
        if pool.exists():
            lines += pool.read_text().split("\n")[:200]
    yyydata.write_text("\n".join(l for l in lines if l.strip()))
    register_dataset("expYYY", str(yyydata))
    write_config("expYYY", MODEL_14B, "expYYY", epochs=2)
    B11_EXPS["expYYY"] = {"gpu": 2, "n": len([l for l in lines if l.strip()]), "model": MODEL_14B}
    log(f"  expYYY: {B11_EXPS['expYYY']['n']} ns samples (v1+v3+v4), 2ep (GPU 2)")

# Always: 32B with best configuration
if pool32b_wins:
    # 32B wins: test 32B with cross-pool diversity
    v2_ns = HISTDIR / "data/train_claude_teacher_v2_noskill.jsonl"
    v1_ws = HISTDIR / "data/train_claude_teacher.jsonl"
    zzzdata = HISTDIR / "data/train_expZZZ.jsonl"
    lines = v1_ws.read_text().split("\n")[:250]
    if v2_ns.exists():
        lines += v2_ns.read_text().split("\n")[:200]
    else:
        lines += v1_ws.read_text().split("\n")[250:450]
    zzzdata.write_text("\n".join(l for l in lines if l.strip()))
    register_dataset("expZZZ", str(zzzdata))
    write_config("expZZZ", MODEL_32B, "expZZZ", epochs=3,
                 batch_size=1, grad_acc=8)
    B11_EXPS["expZZZ"] = {"gpu": 3, "n": len([l for l in lines if l.strip()]), "model": MODEL_32B}
    log(f"  expZZZ: 32B, {B11_EXPS['expZZZ']['n']} samples (ws+ns diversity), 3ep (GPU 3)")
else:
    # 32B not clearly better: test 32B with best ns recipe
    v1ns = HISTDIR / "data/train_expRR.jsonl"
    zzzdata = HISTDIR / "data/train_expZZZ.jsonl"
    import shutil; shutil.copy(str(v1ns), str(zzzdata))
    register_dataset("expZZZ", str(zzzdata))
    write_config("expZZZ", MODEL_32B, "expZZZ", epochs=5, batch_size=1, grad_acc=8)
    B11_EXPS["expZZZ"] = {"gpu": 3, "n": 200, "model": MODEL_32B}
    log("  expZZZ: 32B + best ns recipe (200 v1, 5ep) (GPU 3)")

# Kill vLLM
kill_vllm()

# ── Training phase ─────────────────────────────────────────────────────────────
log(f"Starting Batch 11 training ({len(B11_EXPS)} experiments)...")
for exp, cfg in B11_EXPS.items():
    start_training(exp, cfg["gpu"])
    time.sleep(5)

# ── Monitor phase ──────────────────────────────────────────────────────────────
train_done = {exp: False for exp in B11_EXPS}
eval_started = {exp: False for exp in B11_EXPS}

while True:
    for exp, cfg in B11_EXPS.items():
        if train_done[exp]:
            continue
        log_p = HISTDIR / f"logs/{exp}_train.log"
        if log_p.exists() and "train_runtime" in log_p.read_text():
            train_done[exp] = True
            result_p = HISTDIR / f"results/{exp}.json"
            if not result_p.exists() and not eval_started[exp]:
                eval_started[exp] = True
                start_eval(exp, cfg["model"], cfg["gpu"], cfg["n"])
        else:
            step = get_step(exp)
            log(f"⏳ {exp} (GPU {cfg['gpu']}): step {step}")

    for exp, cfg in B11_EXPS.items():
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
        pending = sum(1 for e in B11_EXPS
                      if eval_started[e] and not (HISTDIR / f"results/{e}.json").exists())
        if pending == 0:
            break
    time.sleep(120)

log("=== Batch 11 Final Results ===")
for exp in B11_EXPS:
    s = load_result(exp)
    if s:
        r8 = s.get("per_rule_recall", {}).get("rule8", 0)
        log(f"  {exp}: F1={s['rule_detection_f1']:.3f}  cpk={s.get('cpk_found_rate', 0):.2f}  r8={r8:.2f}")

# Final all-time best
log("=== FINAL All-time Top 10 ===")
all_results = []
for p in sorted((HISTDIR / "results").glob("exp*.json")):
    try:
        s = json.loads(p.read_text())["summary"]
        all_results.append((s["rule_detection_f1"], p.stem,
                            s.get("per_rule_recall", {}).get("rule8", 0)))
    except Exception:
        pass
for f1, exp, r8 in sorted(all_results, reverse=True)[:10]:
    log(f"  {exp}: F1={f1:.3f}  r8={r8:.2f}")
log("Agent-Beta B11 planner exiting.")
