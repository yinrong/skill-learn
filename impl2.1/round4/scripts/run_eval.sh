#!/bin/bash
# Round 4 评估流程
# 先合并 adapter → 部署微调模型 → 运行评估 → 汇总结果
#
# 用法：bash round4/scripts/run_eval.sh [GPU_ID]
# 例：  bash round4/scripts/run_eval.sh 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKDIR"

GPU_ID="${1:-2}"
PORT=8035
MERGED_DIR="round4/checkpoints/R4-base-merged"
TEST_FILE="round4/data/test_ns.jsonl"
GRPO_FILE="round4/data/grpo_raw.jsonl"
RESULT_DIR="round4/results"

echo "========================================"
echo "Round 4 Evaluation Pipeline"
echo "GPU: $GPU_ID | Port: $PORT"
echo "========================================"

# Step 1: Merge adapter
echo ""
echo "[1/4] Merging LoRA adapter..."
bash round4/scripts/merge_adapter.sh \
    round4/checkpoints/R4-base \
    "$MERGED_DIR"

# Step 2: Deploy vLLM in background
echo ""
echo "[2/4] Starting vLLM service on port $PORT..."
CUDA_VISIBLE_DEVICES="$GPU_ID" python -m vllm.entrypoints.openai.api_server \
    --model "$WORKDIR/$MERGED_DIR" \
    --served-model-name "R4-base-merged" \
    --port "$PORT" \
    --max-model-len 6144 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    2>&1 | tee "round4/logs/vllm_R4-base.log" &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM to start..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "vLLM ready!"
        break
    fi
    sleep 5
    echo "  ... $((i*5))s"
done

# Step 3: Run tool_call_eval
echo ""
echo "[3/4] Running tool_call_eval (step prediction F1)..."
python round4/eval/tool_call_eval.py \
    --test_file "$TEST_FILE" \
    --output "$RESULT_DIR/R4-base-tool-f1.json" \
    --model_url "http://localhost:$PORT" \
    --model_name "R4-base-merged" \
    --verbose

# Step 4: Run LLM judge (subset)
echo ""
echo "[4/4] Running LLM judge (answer quality, 30 samples)..."
python round4/eval/llm_judge.py \
    --grpo_file "$GRPO_FILE" \
    --output "$RESULT_DIR/R4-base-judge.json" \
    --model_url "http://localhost:$PORT" \
    --model_name "R4-base-merged" \
    --max_samples 30

# Cleanup vLLM
kill "$VLLM_PID" 2>/dev/null || true

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results:"
echo "  Tool-call F1  : $RESULT_DIR/R4-base-tool-f1.json"
echo "  LLM Judge     : $RESULT_DIR/R4-base-judge.json"
echo ""
echo "Quick summary:"
python3 -c "
import json
try:
    tf = json.load(open('$RESULT_DIR/R4-base-tool-f1.json'))
    print('  tool_call_f1  :', tf['summary']['combined_f1'])
    print('  tool_name_acc :', tf['summary']['name_acc'])
except: pass
try:
    jf = json.load(open('$RESULT_DIR/R4-base-judge.json'))
    print('  judge_overall :', jf['summary']['overall'])
except: pass
"
echo "========================================"
