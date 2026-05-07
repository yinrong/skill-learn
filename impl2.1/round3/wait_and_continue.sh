#!/usr/bin/env bash
# wait_and_continue.sh — 等待 R3-A 训练完成，然后执行 merge + eval，再启动 R3-B
set -euo pipefail
cd /home/yinrong/post-train/impl2.1

ROOT="$(pwd)"
R3="$ROOT/round3"
COMMON="$ROOT/common"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
TEST_DATA="$COMMON/data/test.jsonl"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 等待 R3-A 训练完成 ────────────────────────────────────────────────────────
log "等待 R3-A 训练完成..."
while true; do
  if grep -q "train_runtime" "$R3/logs/train_R3-A.log" 2>/dev/null; then
    log "R3-A 训练完成！"
    break
  fi
  # 检查训练进程是否还在
  if ! ps aux | grep -q "[l]lamafactory-cli train round3/configs/R3-A"; then
    log "WARNING: llamafactory 进程消失，检查日志..."
    tail -5 "$R3/logs/train_R3-A.log"
    break
  fi
  sleep 30
done

# ── 合并 R3-A adapter ─────────────────────────────────────────────────────────
CKPT="$R3/checkpoints/R3-A"
MERGED="$R3/checkpoints/R3-A-merged"

if [[ ! -d "$MERGED" ]]; then
  log "[R3-A] 合并 adapter..."
  python "$COMMON/tools/train/merge_adapter.py" \
    --base "$MODEL_14B" \
    --adapter "$CKPT" \
    --output "$MERGED" \
    --template qwen3 \
    2>&1 | tee "$R3/logs/merge_R3-A.log"
  log "[R3-A] 合并完成：$MERGED"
else
  log "[R3-A] 已有 merged 模型，跳过"
fi

# ── 评测 R3-A ────────────────────────────────────────────────────────────────
RESULT="$R3/results/R3-A.json"

log "[R3-A] 启动 vLLM（端口 8031）..."
python "$COMMON/tools/train/deploy_vllm.py" \
  --model "$MERGED" --port 8031 \
  2>&1 | tee "$R3/logs/vllm_R3-A.log" &
VLLM_PID=$!
# 等待 vLLM 真正就绪（最多300秒）
log "[R3-A] 等待 vLLM 就绪..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8031/health >/dev/null 2>&1; then
    log "[R3-A] vLLM 就绪（${i}×5s）"
    break
  fi
  sleep 5
done

log "[R3-A] 开始评测..."
python "$COMMON/tools/eval/spc_eval.py" \
  --model_url "http://localhost:8031" \
  --model_name "R3-A" \
  --test "$TEST_DATA" \
  --output "$RESULT" \
  --max_tokens 4096 \
  --concurrency 4 \
  2>&1 | tee "$R3/logs/eval_R3-A.log"
log "[R3-A] 评测完成：$RESULT"

kill $VLLM_PID 2>/dev/null || true
python "$COMMON/tools/train/deploy_vllm.py" --kill --port 8031 2>/dev/null || true
sleep 5

# 打印结果
if [[ -f "$RESULT" ]]; then
  python3 -c "
import json
d = json.load(open('$RESULT'))
s = d['summary']
print('=== R3-A 评测结果 ===')
print(f'  rule_f1={s[\"rule_detection_f1\"]:.3f}')
print(f'  per_rule_recall:', {k: round(v,3) for k,v in s[\"per_rule_recall\"].items()})
cpk = s['cpk_mae']
cpk_str = f'{cpk:.3f}' if cpk is not None else 'N/A'
print(f'  cpk_mae={cpk_str}, inference={s[\"inference_time_ms_mean\"]/1000:.1f}s')
"
fi

# ── 启动 R3-B 训练 ─────────────────────────────────────────────────────────────
log "===== 开始 R3-B 训练 ====="
CKPT_B="$R3/checkpoints/R3-B"
if [[ ! -d "$CKPT_B" ]]; then
  log "[R3-B] 开始训练..."
  DISABLE_VERSION_CHECK=1 llamafactory-cli train "$R3/configs/R3-B.yaml" \
    2>&1 | tee "$R3/logs/train_R3-B.log"
  log "[R3-B] 训练完成"
else
  log "[R3-B] 已有 checkpoint，跳过训练"
fi

# ── 合并 R3-B adapter ─────────────────────────────────────────────────────────
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
fi

# ── 评测 R3-B ────────────────────────────────────────────────────────────────
RESULT_B="$R3/results/R3-B.json"

log "[R3-B] 启动 vLLM（端口 8032）..."
python "$COMMON/tools/train/deploy_vllm.py" \
  --model "$MERGED_B" --port 8032 \
  2>&1 | tee "$R3/logs/vllm_R3-B.log" &
VLLM_PID=$!
log "[R3-B] 等待 vLLM 就绪..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8032/health >/dev/null 2>&1; then
    log "[R3-B] vLLM 就绪（${i}×5s）"
    break
  fi
  sleep 5
done

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
python "$COMMON/tools/train/deploy_vllm.py" --kill --port 8032 2>/dev/null || true

# 打印 R3-B 结果
if [[ -f "$RESULT_B" ]]; then
  python3 -c "
import json
d = json.load(open('$RESULT_B'))
s = d['summary']
print('=== R3-B 评测结果 ===')
print(f'  rule_f1={s[\"rule_detection_f1\"]:.3f}')
print(f'  per_rule_recall:', {k: round(v,3) for k,v in s[\"per_rule_recall\"].items()})
cpk = s['cpk_mae']
cpk_str = f'{cpk:.3f}' if cpk is not None else 'N/A'
print(f'  cpk_mae={cpk_str}, inference={s[\"inference_time_ms_mean\"]/1000:.1f}s')
"
fi

log "======================================================"
log "✓ R3-A + R3-B 完成，结果在 $R3/results/"
log "======================================================"
