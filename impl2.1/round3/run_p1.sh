#!/usr/bin/env bash
# run_p1.sh — P1 实验：等待 R3-AB-v2 完成，然后依次运行 GRPO(R3-C) + QLoRA(R3-D1, R3-D2)
set -euo pipefail
cd /home/yinrong/post-train/impl2.1

ROOT="$(pwd)"
R3="$ROOT/round3"
COMMON="$ROOT/common"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
MODEL_32B="/home/yinrong/models/Qwen3-32B"
TEST_DATA="$COMMON/data/test.jsonl"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

print_result() {
  local name="$1"
  local result="$2"
  if [[ -f "$result" ]]; then
    python3 -c "
import json
d = json.load(open('$result'))
s = d['summary']
print('=== $name 评测结果 ===')
print(f'  rule_f1={s[\"rule_detection_f1\"]:.3f}')
print('  per_rule_recall:', {k: round(v,3) if v is not None else None for k,v in s['per_rule_recall'].items()})
cpk = s['cpk_mae']
cpk_str = f'{cpk:.3f}' if cpk is not None else 'N/A'
print(f'  cpk_mae={cpk_str}, inference={s[\"inference_time_ms_mean\"]/1000:.1f}s')
"
  fi
}

wait_for_vllm() {
  local port="$1"
  log "等待 vLLM 就绪（port $port）..."
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
      log "vLLM $port 就绪（${i}×5s）"
      return 0
    fi
    sleep 5
  done
  log "ERROR: vLLM $port 超时未就绪"
  return 1
}

run_eval() {
  local name="$1"
  local merged="$2"
  local port="$3"
  local result="$4"

  log "[$name] 启动 vLLM（端口 $port）..."
  python "$COMMON/tools/train/deploy_vllm.py" \
    --model "$merged" --port "$port" \
    2>&1 | tee "$R3/logs/vllm_${name}.log" &
  local vllm_pid=$!

  wait_for_vllm "$port"

  log "[$name] 开始评测..."
  python "$COMMON/tools/eval/spc_eval.py" \
    --model_url "http://localhost:$port" \
    --model_name "$name" \
    --test "$TEST_DATA" \
    --output "$result" \
    --max_tokens 4096 \
    --concurrency 4 \
    2>&1 | tee "$R3/logs/eval_${name}.log"
  log "[$name] 评测完成：$result"

  kill $vllm_pid 2>/dev/null || true
  kill $(ps aux | grep "[v]llm.*$port" | awk '{print $2}') 2>/dev/null || true
  sleep 10
}

# ── 等待 R3-AB-v2 评测完成 ────────────────────────────────────────────────────
RESULT_V2="$R3/results/R3-AB-v2.json"
log "等待 R3-AB-v2 评测完成..."
while true; do
  if [[ -f "$RESULT_V2" ]]; then
    log "R3-AB-v2 评测完成！"
    break
  fi
  sleep 30
done
print_result "R3-AB-v2" "$RESULT_V2"

# ── R3-C：GRPO ────────────────────────────────────────────────────────────────
CKPT_C="$R3/checkpoints/R3-C"
MERGED_C="$R3/checkpoints/R3-C-merged"
RESULT_C="$R3/results/R3-C.json"

log "===== 开始 R3-C GRPO 训练 ====="
if [[ ! -d "$CKPT_C" ]]; then
  log "[R3-C] 开始 GRPO 训练..."
  DISABLE_VERSION_CHECK=1 llamafactory-cli train "$R3/configs/R3-C-grpo.yaml" \
    2>&1 | tee "$R3/logs/train_R3-C.log" || {
    log "[R3-C] GRPO 训练失败，跳过（继续 QLoRA 实验）"
    touch "$CKPT_C/.failed"
  }
else
  log "[R3-C] 已有 checkpoint，跳过训练"
fi

if [[ -d "$CKPT_C" ]] && [[ ! -f "$CKPT_C/.failed" ]]; then
  if [[ ! -d "$MERGED_C" ]]; then
    log "[R3-C] 合并 adapter..."
    python "$COMMON/tools/train/merge_adapter.py" \
      --base "/home/yinrong/post-train/impl2.1/round2/history-route2.1.1/checkpoints/expYYY-merged" \
      --adapter "$CKPT_C" \
      --output "$MERGED_C" \
      --template qwen3 \
      2>&1 | tee "$R3/logs/merge_R3-C.log"
    log "[R3-C] 合并完成：$MERGED_C"
  fi
  run_eval "R3-C" "$MERGED_C" 8035 "$RESULT_C"
  print_result "R3-C" "$RESULT_C"
fi

# ── R3-D1：QLoRA 14B ─────────────────────────────────────────────────────────
CKPT_D1="$R3/checkpoints/R3-D1-qlora-14b"
MERGED_D1="$R3/checkpoints/R3-D1-qlora-14b-merged"
RESULT_D1="$R3/results/R3-D1.json"

log "===== 开始 R3-D1 QLoRA 14B 训练 ====="
if [[ ! -d "$CKPT_D1" ]]; then
  log "[R3-D1] 开始 QLoRA 训练..."
  DISABLE_VERSION_CHECK=1 llamafactory-cli train "$R3/configs/R3-D1-qlora-14b.yaml" \
    2>&1 | tee "$R3/logs/train_R3-D1.log"
  log "[R3-D1] 训练完成"
else
  log "[R3-D1] 已有 checkpoint，跳过训练"
fi

if [[ ! -d "$MERGED_D1" ]]; then
  log "[R3-D1] 合并 adapter..."
  python "$COMMON/tools/train/merge_adapter.py" \
    --base "$MODEL_14B" \
    --adapter "$CKPT_D1" \
    --output "$MERGED_D1" \
    --template qwen3 \
    2>&1 | tee "$R3/logs/merge_R3-D1.log"
  log "[R3-D1] 合并完成：$MERGED_D1"
fi

run_eval "R3-D1" "$MERGED_D1" 8036 "$RESULT_D1"
print_result "R3-D1" "$RESULT_D1"

# ── R3-D2：QLoRA 32B ─────────────────────────────────────────────────────────
CKPT_D2="$R3/checkpoints/R3-D2-qlora-32b"
MERGED_D2="$R3/checkpoints/R3-D2-qlora-32b-merged"
RESULT_D2="$R3/results/R3-D2.json"

log "===== 开始 R3-D2 QLoRA 32B 训练 ====="
if [[ ! -d "$CKPT_D2" ]]; then
  log "[R3-D2] 开始 QLoRA 训练..."
  DISABLE_VERSION_CHECK=1 llamafactory-cli train "$R3/configs/R3-D2-qlora-32b.yaml" \
    2>&1 | tee "$R3/logs/train_R3-D2.log"
  log "[R3-D2] 训练完成"
else
  log "[R3-D2] 已有 checkpoint，跳过训练"
fi

if [[ ! -d "$MERGED_D2" ]]; then
  log "[R3-D2] 合并 adapter..."
  python "$COMMON/tools/train/merge_adapter.py" \
    --base "$MODEL_32B" \
    --adapter "$CKPT_D2" \
    --output "$MERGED_D2" \
    --template qwen3 \
    2>&1 | tee "$R3/logs/merge_R3-D2.log"
  log "[R3-D2] 合并完成：$MERGED_D2"
fi

run_eval "R3-D2" "$MERGED_D2" 8037 "$RESULT_D2"
print_result "R3-D2" "$RESULT_D2"

# ── 最终汇总 ──────────────────────────────────────────────────────────────────
log "======================================================"
log "P1 实验全部完成，结果汇总："
for exp in R3-AB R3-AB-v2 R3-C R3-D1 R3-D2; do
  rf="$R3/results/${exp}.json"
  if [[ -f "$rf" ]]; then
    python3 -c "
import json
d = json.load(open('$rf'))
s = d['summary']
cpk = s['cpk_mae']
cpk_str = f'{cpk:.3f}' if cpk is not None else 'N/A'
print(f'  $exp: F1={s[\"rule_detection_f1\"]:.3f}, CPK_MAE={cpk_str}')
" 2>/dev/null || echo "  $exp: 结果解析失败"
  else
    echo "  $exp: 结果未找到"
  fi
done
log "======================================================"
