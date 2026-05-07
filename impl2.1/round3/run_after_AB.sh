#!/usr/bin/env bash
# run_after_AB.sh — 等待 R3-AB 评测完成，然后训练 R3-AB-v2（清洗数据版）
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

# ── 等待 R3-AB 评测完成 ───────────────────────────────────────────────────────
RESULT_AB="$R3/results/R3-AB.json"
log "等待 R3-AB 评测完成..."
while true; do
  if [[ -f "$RESULT_AB" ]]; then
    log "R3-AB 评测完成！"
    break
  fi
  # 检查 continue_after_B.sh 是否还在运行
  if ! ps aux | grep -q "[c]ontinue_after_B.sh"; then
    log "WARNING: continue_after_B.sh 进程消失"
    if [[ -f "$RESULT_AB" ]]; then
      break
    fi
    log "R3-AB 结果也未出现，等待更长时间..."
  fi
  sleep 30
done

print_result "R3-AB" "$RESULT_AB"

# ── 等待 vLLM 端口 8033 释放 ──────────────────────────────────────────────────
log "等待 vLLM 8033 完全停止..."
for i in $(seq 1 30); do
  if ! curl -sf "http://localhost:8033/health" >/dev/null 2>&1; then
    log "vLLM 8033 已停止"
    break
  fi
  sleep 5
done
sleep 10

# ── 训练 R3-AB-v2 ─────────────────────────────────────────────────────────────
CKPT_V2="$R3/checkpoints/R3-AB-v2"
MERGED_V2="$R3/checkpoints/R3-AB-v2-merged"
RESULT_V2="$R3/results/R3-AB-v2.json"

log "===== 开始 R3-AB-v2 训练（1131 条清洗数据）====="
if [[ ! -d "$CKPT_V2" ]]; then
  log "[R3-AB-v2] 开始训练..."
  DISABLE_VERSION_CHECK=1 llamafactory-cli train "$R3/configs/R3-AB-v2.yaml" \
    2>&1 | tee "$R3/logs/train_R3-AB-v2.log"
  log "[R3-AB-v2] 训练完成"
else
  log "[R3-AB-v2] 已有 checkpoint，跳过训练"
fi

# ── 合并 R3-AB-v2 ─────────────────────────────────────────────────────────────
if [[ ! -d "$MERGED_V2" ]]; then
  log "[R3-AB-v2] 合并 adapter..."
  python "$COMMON/tools/train/merge_adapter.py" \
    --base "$MODEL_14B" \
    --adapter "$CKPT_V2" \
    --output "$MERGED_V2" \
    --template qwen3 \
    2>&1 | tee "$R3/logs/merge_R3-AB-v2.log"
  log "[R3-AB-v2] 合并完成：$MERGED_V2"
else
  log "[R3-AB-v2] 已有 merged 模型，跳过"
fi

# ── 评测 R3-AB-v2 ─────────────────────────────────────────────────────────────
log "[R3-AB-v2] 启动 vLLM（端口 8034）..."
python "$COMMON/tools/train/deploy_vllm.py" \
  --model "$MERGED_V2" --port 8034 \
  2>&1 | tee "$R3/logs/vllm_R3-AB-v2.log" &
VLLM_PID=$!

wait_for_vllm 8034

log "[R3-AB-v2] 开始评测..."
python "$COMMON/tools/eval/spc_eval.py" \
  --model_url "http://localhost:8034" \
  --model_name "R3-AB-v2" \
  --test "$TEST_DATA" \
  --output "$RESULT_V2" \
  --max_tokens 4096 \
  --concurrency 4 \
  2>&1 | tee "$R3/logs/eval_R3-AB-v2.log"
log "[R3-AB-v2] 评测完成：$RESULT_V2"

kill $VLLM_PID 2>/dev/null || true
kill $(ps aux | grep "[v]llm.*8034" | awk '{print $2}') 2>/dev/null || true

print_result "R3-AB-v2" "$RESULT_V2"

log "======================================================"
log "✓ R3-AB-v2 完成，结果在 $R3/results/R3-AB-v2.json"
log "======================================================"
