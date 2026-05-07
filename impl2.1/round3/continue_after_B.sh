#!/usr/bin/env bash
# continue_after_B.sh — 等待 R3-B 完成，然后 merge + eval + R3-AB 训练
set -euo pipefail
cd /home/yinrong/post-train/impl2.1

ROOT="$(pwd)"
R3="$ROOT/round3"
COMMON="$ROOT/common"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
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

# ── 等待 R3-B 训练完成 ────────────────────────────────────────────────────────
log "等待 R3-B 训练完成..."
while true; do
  if grep -q "train_runtime" "$R3/logs/train_R3-B.log" 2>/dev/null; then
    log "R3-B 训练完成！"
    break
  fi
  if ! ps aux | grep -q "[l]lamafactory-cli train round3/configs/R3-B"; then
    log "WARNING: R3-B llamafactory 进程消失"
    tail -5 "$R3/logs/train_R3-B.log" 2>/dev/null || true
    break
  fi
  sleep 30
done

# ── 合并 R3-B adapter ─────────────────────────────────────────────────────────
CKPT_B="$R3/checkpoints/R3-B"
MERGED_B="$R3/checkpoints/R3-B-merged"

if [[ ! -d "$MERGED_B" ]]; then
  log "[R3-B] 合并 adapter..."
  python "$COMMON/tools/train/merge_adapter.py" \
    --base "$MODEL_14B" \
    --adapter "$CKPT_B" \
    --output "$MERGED_B" \
    --template qwen3 \
    2>&1 | tee "$R3/logs/merge_R3-B.log"
  log "[R3-B] 合并完成：$MERGED_B"
else
  log "[R3-B] 已有 merged 模型，跳过"
fi

# ── 评测 R3-B ────────────────────────────────────────────────────────────────
RESULT_B="$R3/results/R3-B.json"

log "[R3-B] 启动 vLLM（端口 8032）..."
python "$COMMON/tools/train/deploy_vllm.py" \
  --model "$MERGED_B" --port 8032 \
  2>&1 | tee "$R3/logs/vllm_R3-B.log" &
VLLM_PID=$!

wait_for_vllm 8032

log "[R3-B] 开始评测..."
python "$COMMON/tools/eval/spc_eval.py" \
  --model_url "http://localhost:8032" \
  --model_name "R3-B" \
  --test "$TEST_DATA" \
  --output "$RESULT_B" \
  --max_tokens 4096 \
  --concurrency 4 \
  2>&1 | tee "$R3/logs/eval_R3-B.log"
log "[R3-B] 评测完成：$RESULT_B"

kill $VLLM_PID 2>/dev/null || true
kill $(ps aux | grep "[v]llm.*8032" | awk '{print $2}') 2>/dev/null || true
sleep 5

print_result "R3-B" "$RESULT_B"

# ── 启动 R3-AB 训练 ───────────────────────────────────────────────────────────
CKPT_AB="$R3/checkpoints/R3-AB"
MERGED_AB="$R3/checkpoints/R3-AB-merged"
RESULT_AB="$R3/results/R3-AB.json"

log "===== 开始 R3-AB 训练（A+B 组合，1460 条）====="
if [[ ! -d "$CKPT_AB" ]]; then
  log "[R3-AB] 开始训练..."
  DISABLE_VERSION_CHECK=1 llamafactory-cli train "$R3/configs/R3-AB.yaml" \
    2>&1 | tee "$R3/logs/train_R3-AB.log"
  log "[R3-AB] 训练完成"
else
  log "[R3-AB] 已有 checkpoint，跳过训练"
fi

# ── 合并 + 评测 R3-AB ─────────────────────────────────────────────────────────
if [[ ! -d "$MERGED_AB" ]]; then
  log "[R3-AB] 合并 adapter..."
  python "$COMMON/tools/train/merge_adapter.py" \
    --base "$MODEL_14B" \
    --adapter "$CKPT_AB" \
    --output "$MERGED_AB" \
    --template qwen3 \
    2>&1 | tee "$R3/logs/merge_R3-AB.log"
  log "[R3-AB] 合并完成：$MERGED_AB"
fi

log "[R3-AB] 启动 vLLM（端口 8033）..."
python "$COMMON/tools/train/deploy_vllm.py" \
  --model "$MERGED_AB" --port 8033 \
  2>&1 | tee "$R3/logs/vllm_R3-AB.log" &
VLLM_PID=$!

wait_for_vllm 8033

log "[R3-AB] 开始评测..."
python "$COMMON/tools/eval/spc_eval.py" \
  --model_url "http://localhost:8033" \
  --model_name "R3-AB" \
  --test "$TEST_DATA" \
  --output "$RESULT_AB" \
  --max_tokens 4096 \
  --concurrency 4 \
  2>&1 | tee "$R3/logs/eval_R3-AB.log"
log "[R3-AB] 评测完成：$RESULT_AB"

kill $VLLM_PID 2>/dev/null || true
kill $(ps aux | grep "[v]llm.*8033" | awk '{print $2}') 2>/dev/null || true

print_result "R3-AB" "$RESULT_AB"

log "======================================================"
log "✓ R3-B + R3-AB 完成，结果在 $R3/results/"
log "======================================================"
