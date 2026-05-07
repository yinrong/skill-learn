#!/usr/bin/env bash
set -euo pipefail
cd /home/yinrong/post-train/impl2.1
ROOT="$(pwd)"; R3="$ROOT/round3"; COMMON="$ROOT/common"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_vllm() {
  for i in $(seq 1 60); do
    curl -sf "http://localhost:$1/health" >/dev/null 2>&1 && log "vLLM $1 就绪" && return 0
    sleep 5
  done; return 1
}

# ── 训练 ──────────────────────────────────────────────────────────────────────
CKPT="$R3/checkpoints/R3-AB-v3"
MERGED="$R3/checkpoints/R3-AB-v3-merged"
RESULT="$R3/results/R3-AB-v3.json"

if [[ ! -d "$CKPT" ]]; then
  log "[R3-AB-v3] 开始训练（1771 条清洗数据）..."
  DISABLE_VERSION_CHECK=1 llamafactory-cli train "$R3/configs/R3-AB-v3.yaml" \
    2>&1 | tee "$R3/logs/train_R3-AB-v3.log"
  log "[R3-AB-v3] 训练完成"
else
  log "[R3-AB-v3] 已有 checkpoint，跳过"
fi

# ── 合并 ──────────────────────────────────────────────────────────────────────
if [[ ! -d "$MERGED" ]]; then
  log "[R3-AB-v3] 合并 adapter..."
  python "$COMMON/tools/train/merge_adapter.py" \
    --base /home/yinrong/models/Qwen3-14B \
    --adapter "$CKPT" --output "$MERGED" --template qwen3 \
    2>&1 | tee "$R3/logs/merge_R3-AB-v3.log"
fi

# ── 评测 ──────────────────────────────────────────────────────────────────────
log "[R3-AB-v3] 启动 vLLM（端口 8038）..."
python "$COMMON/tools/train/deploy_vllm.py" \
  --model "$MERGED" --port 8038 \
  2>&1 | tee "$R3/logs/vllm_R3-AB-v3.log" &
VLLM_PID=$!
wait_for_vllm 8038

python "$COMMON/tools/eval/spc_eval.py" \
  --model_url http://localhost:8038 --model_name R3-AB-v3 \
  --test "$COMMON/data/test.jsonl" --output "$RESULT" \
  --max_tokens 4096 --concurrency 4 \
  2>&1 | tee "$R3/logs/eval_R3-AB-v3.log"

kill $VLLM_PID 2>/dev/null || true
kill $(ps aux | grep "[v]llm.*8038" | awk '{print $2}') 2>/dev/null || true

python3 -c "
import json
d=json.load(open('$RESULT'))
s=d['summary']
cpk=s['cpk_mae']
cpk_str=f'{cpk:.3f}' if cpk is not None else 'N/A'
print('=== R3-AB-v3 结果 ===')
print(f'  F1={s[\"rule_detection_f1\"]:.3f}, CPK_MAE={cpk_str}')
print('  per_rule_recall:', {k: round(v,3) if v is not None else None for k,v in s['per_rule_recall'].items()})
"
log "完成：$RESULT"
