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

# ── 等待 v3 训练结束（GPU 释放）────────────────────────────────────────────────
log "等待 R3-AB-v3 训练完成（GPU 释放）..."
while ps aux | grep -q "[t]orchrun.*v3\|[l]auncher.*v3"; do sleep 30; done
log "等待 GPU 显存释放..."
for i in $(seq 1 30); do
  free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  if [[ "$free_mem" -gt 60000 ]]; then log "GPU 空闲（${free_mem}MiB）"; break; fi
  sleep 10
done
sleep 15

# ── 训练 D2（单进程 device_map=auto，8 卡分摊）──────────────────────────────────
CKPT="$R3/checkpoints/R3-D2-qlora-32b"
MERGED="$R3/checkpoints/R3-D2-qlora-32b-merged"
RESULT="$R3/results/R3-D2.json"

if [[ ! -d "$CKPT" ]]; then
  log "[R3-D2] 开始训练（单进程 8 卡 device_map=auto）..."
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 \
    python /usr/local/lib/python3.10/dist-packages/llamafactory/launcher.py \
    "$R3/configs/R3-D2-qlora-32b.yaml" \
    2>&1 | tee "$R3/logs/train_R3-D2.log"
  log "[R3-D2] 训练完成"
else
  log "[R3-D2] 已有 checkpoint，跳过"
fi

if [[ ! -d "$MERGED" ]]; then
  log "[R3-D2] 合并 adapter..."
  python "$COMMON/tools/train/merge_adapter.py" \
    --base /home/yinrong/models/Qwen3-32B \
    --adapter "$CKPT" --output "$MERGED" --template qwen3 \
    2>&1 | tee "$R3/logs/merge_R3-D2.log"
fi

log "[R3-D2] 启动 vLLM（端口 8037）..."
python "$COMMON/tools/train/deploy_vllm.py" \
  --model "$MERGED" --port 8037 \
  2>&1 | tee "$R3/logs/vllm_R3-D2.log" &
VLLM_PID=$!
wait_for_vllm 8037

python "$COMMON/tools/eval/spc_eval.py" \
  --model_url http://localhost:8037 --model_name R3-D2 \
  --test "$COMMON/data/test.jsonl" --output "$RESULT" \
  --max_tokens 4096 --concurrency 4 \
  2>&1 | tee "$R3/logs/eval_R3-D2.log"

kill $VLLM_PID 2>/dev/null || true
kill $(ps aux | grep "[v]llm.*8037" | awk '{print $2}') 2>/dev/null || true

python3 -c "
import json; d=json.load(open('$RESULT'))
s=d['summary']; cpk=s['cpk_mae']
cpk_str=f'{cpk:.3f}' if cpk is not None else 'N/A'
print(f'R3-D2: F1={s[\"rule_detection_f1\"]:.3f}, CPK_MAE={cpk_str}')
print('per_rule_recall:', {k:round(v,3) if v is not None else None for k,v in s['per_rule_recall'].items()})
"
log "完成"
