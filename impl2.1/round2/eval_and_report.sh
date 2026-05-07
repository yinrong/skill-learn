#!/bin/bash
# 评测脚本：合并 LoRA adapter + 部署 vLLM + 评测 + 停止 vLLM
# 用法：bash eval_and_report.sh <EXP_NAME> <BASE_MODEL> <GPU> <N_TRAIN> [MAX_TOKENS]
# 例如：bash eval_and_report.sh expA /home/yinrong/models/Qwen3-14B 0 251 3500

set -e
cd "$(dirname "$0")"

EXP=$1
BASE_MODEL=$2
GPU=$3
N_TRAIN=${4:-251}
MAX_TOKENS=${5:-3500}

HISTDIR="$(pwd)/history-route2.1.1"
ADAPTER="$HISTDIR/checkpoints/$EXP"
MERGED="$HISTDIR/checkpoints/${EXP}-merged"
RESULT="$HISTDIR/results/${EXP}.json"
PORT=$((8100 + GPU))

# Check if training is complete
TRAINER_LOG="$HISTDIR/logs/${EXP}_train.log"
# Note: HISTDIR is absolute (set above with pwd)
if ! grep -q "Training completed" "$TRAINER_LOG" 2>/dev/null && \
   ! grep -q "train_runtime" "$TRAINER_LOG" 2>/dev/null; then
    echo "⚠ $EXP: 训练可能未完成，继续尝试..."
fi

# Skip if already evaluated
if [ -f "$RESULT" ]; then
    F1=$(python3 -c "import json; d=json.load(open('$RESULT')); print(d['summary']['rule_detection_f1'])" 2>/dev/null || echo "?")
    echo "⏭ $EXP: 结果已存在 (F1=$F1)"
    exit 0
fi

echo "=== 评测 $EXP (GPU $GPU, port $PORT) ==="

# Step 1: Merge adapter
if [ ! -d "$MERGED" ] || [ -z "$(ls -A $MERGED 2>/dev/null)" ]; then
    echo "🔗 合并 LoRA adapter..."
    LOG_MERGE="$HISTDIR/logs/${EXP}_merge.log"
    DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=$GPU python tools/train/merge_adapter.py \
        --base "$BASE_MODEL" \
        --adapter "$ADAPTER" \
        --output "$MERGED" \
        --template qwen3 > "$LOG_MERGE" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ 合并失败：查看 $LOG_MERGE"
        cat "$LOG_MERGE" | tail -10
        exit 1
    fi
    echo "  ✓ 合并完成 → $MERGED"
else
    echo "  ✓ 已有合并模型，跳过"
fi

# Step 2: Deploy vLLM
echo "🚀 启动 vLLM (GPU $GPU, port $PORT)..."
LOG_VLLM="$HISTDIR/logs/${EXP}_vllm.log"
DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=$GPU \
    python tools/train/deploy_vllm.py --model "$MERGED" --port "$PORT" \
    > "$LOG_VLLM" 2>&1 &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "  等待 vLLM 启动..."
for i in $(seq 1 60); do
    sleep 5
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "  ✓ vLLM 就绪 (${i}×5s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  ❌ vLLM 进程意外终止"
        tail -20 "$LOG_VLLM"
        exit 1
    fi
done

# Step 3: Run eval
echo "📊 开始评测（200条测试集）..."
LOG_EVAL="$HISTDIR/logs/${EXP}_eval.log"
DISABLE_VERSION_CHECK=1 python tools/eval/spc_eval.py \
    --model_url "http://localhost:$PORT" \
    --model_name "$EXP" \
    --test "data/demo/test.jsonl" \
    --output "$RESULT" \
    --n_train_samples "$N_TRAIN" \
    --max_tokens "$MAX_TOKENS" \
    --concurrency 1 > "$LOG_EVAL" 2>&1

# Step 4: Stop vLLM (kill both wrapper and actual server)
kill $VLLM_PID 2>/dev/null || true
# Also kill the actual vLLM server process by port
lsof -ti tcp:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 2
echo "  ✓ vLLM 已停止"

# Print results
echo ""
echo "=== $EXP 评测结果 ==="
python3 -c "
import json
d = json.load(open('$RESULT'))
s = d['summary']
print(f'  rule_detection_f1: {s[\"rule_detection_f1\"]}')
print(f'  cpk_found_rate:    {s[\"cpk_found_rate\"]}')
print(f'  per_rule_recall:')
for rule, val in s['per_rule_recall'].items():
    bar = '█' * int((val or 0) * 20)
    print(f'    {rule}: {val}  {bar}')
"
