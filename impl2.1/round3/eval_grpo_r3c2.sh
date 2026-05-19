#!/bin/bash
# R3-C2 GRPO 训练完成后：合并 adapter + 评测
# 运行：cd /home/yinrong/post-train/impl2.1 && bash round3/eval_grpo_r3c2.sh
set -e

ROOT=/home/yinrong/post-train/impl2.1
COMMON=$ROOT/common
R3=$ROOT/round3
BASE_MODEL=$ROOT/round2/history-route2.1.1/checkpoints/expYYY-merged
CKPT=$R3/checkpoints/R3-C2
MERGED=$R3/checkpoints/R3-C2-merged
RESULT=$R3/results/R3-C2.json
TEST_DATA=$COMMON/data/test.jsonl

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 1. 合并 LoRA adapter
if [[ ! -d "$MERGED" ]]; then
    log "合并 GRPO LoRA adapter..."
    python "$COMMON/tools/train/merge_adapter.py" \
        --base "$BASE_MODEL" \
        --adapter "$CKPT" \
        --output "$MERGED" \
        --template qwen3 \
        2>&1 | tee "$R3/logs/merge_R3-C2.log"
    log "合并完成 → $MERGED"
else
    log "已存在 merged checkpoint，跳过合并"
fi

# 2. 启动 vLLM
log "启动 vLLM 服务（port 8034）..."
python "$COMMON/tools/train/deploy_vllm.py" \
    --model "$MERGED" \
    --port 8034 \
    --max_len 5120 \
    2>&1 > "$R3/logs/vllm_R3-C2.log" &
VLLM_PID=$!

# 等待 vLLM 就绪
for i in $(seq 1 60); do
    if curl -sf "http://localhost:8034/health" >/dev/null 2>&1; then
        log "vLLM 就绪"
        break
    fi
    sleep 5
done

# 3. 评测
log "开始评测 R3-C2..."
python "$COMMON/tools/eval/spc_eval.py" \
    --model_url "http://localhost:8034" \
    --model_name "R3-C2" \
    --test "$TEST_DATA" \
    --output "$RESULT" \
    2>&1 | tee "$R3/logs/eval_R3-C2.log"

kill $VLLM_PID 2>/dev/null || true

log "评测完成 → $RESULT"
python3 -c "
import json
r = json.load(open('$RESULT'))
print(f'F1={r[\"f1\"]:.3f}  P={r[\"precision\"]:.3f}  R={r[\"recall\"]:.3f}  EM={r[\"exact_match\"]:.3f}  CPK={r[\"cpk_found_rate\"]:.3f}')
"
