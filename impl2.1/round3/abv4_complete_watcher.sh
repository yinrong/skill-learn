#!/bin/bash
# R3-AB-v4 完成监控 + 自动 merge + eval → R3-AB-v4.json
# 训练在 GPU 0，merge 用 CPU，eval 用 GPU 0（训练完后释放）
cd /home/yinrong/post-train/impl2.1
LOG="round3/logs/abv4_watcher.log"
ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]"; }
exec > >(tee -a "$LOG") 2>&1

echo "$(ts) AB-v4 watcher started. Watching for train_R3-AB-v4.log completion..."

ADAPTER_DIR="round3/checkpoints/R3-AB-v4"
MERGED_DIR="round3/checkpoints/R3-AB-v4-merged"
RESULT_FILE="round3/results/R3-AB-v4.json"
EVAL_PORT=8044
EVAL_GPU=0

if [ -f "$RESULT_FILE" ]; then
    echo "$(ts) $RESULT_FILE already exists. Done."
    exit 0
fi

# Step 1: Wait for training to complete
while true; do
    if grep -q "train_runtime" round3/logs/train_R3-AB-v4.log 2>/dev/null; then
        echo "$(ts) R3-AB-v4 training completed."
        break
    fi
    STEP=$(tail -1 "${ADAPTER_DIR}/trainer_log.jsonl" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['current_steps']}/{d['total_steps']} ({d['percentage']:.1f}%)\")" 2>/dev/null || echo "?")
    echo "$(ts) R3-AB-v4 still training: $STEP"
    sleep 120
done

# Step 2: PEFT merge (CPU)
if [ -d "$MERGED_DIR" ] && ls "$MERGED_DIR"/*.safetensors 2>/dev/null | head -1 | grep -q .; then
    echo "$(ts) Merged dir already exists, skipping merge."
else
    echo "$(ts) Starting PEFT merge (CPU)..."
    python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = '/home/yinrong/models/Qwen3-14B'
adapter = 'round3/checkpoints/R3-AB-v4'
output = 'round3/checkpoints/R3-AB-v4-merged'

print('Loading base model...')
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cpu')
print('Loading PEFT adapter...')
model = PeftModel.from_pretrained(model, adapter)
print('Merging...')
model = model.merge_and_unload()
print('Saving...')
model.save_pretrained(output, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
tok.save_pretrained(output)
print('Merge done.')
" > round3/logs/merge_R3-AB-v4.log 2>&1

    if [ $? -ne 0 ]; then
        echo "$(ts) MERGE FAILED. See round3/logs/merge_R3-AB-v4.log"
        exit 1
    fi
    echo "$(ts) Merge complete."
fi

# Step 3: Wait for GPU to be free
echo "$(ts) Waiting for GPU $EVAL_GPU to be free (< 5000 MB used)..."
for i in $(seq 1 120); do
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $EVAL_GPU 2>/dev/null | tr -d ' ')
    if [ -n "$GPU_MEM" ] && [ "$GPU_MEM" -lt 5000 ]; then
        echo "$(ts) GPU $EVAL_GPU is free (${GPU_MEM} MB). Proceeding."
        break
    fi
    echo "$(ts) GPU $EVAL_GPU has ${GPU_MEM} MB used. Waiting..."
    sleep 30
done

# Step 4: Start vLLM
echo "$(ts) Starting vLLM on GPU $EVAL_GPU port $EVAL_PORT..."
CUDA_VISIBLE_DEVICES=$EVAL_GPU nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_DIR" \
    --port $EVAL_PORT \
    --dtype bfloat16 \
    --max-model-len 7168 \
    --served-model-name default \
    --trust-remote-code \
    >> round3/logs/vllm_R3-AB-v4.log 2>&1 &
VLLM_PID=$!
echo "$(ts) vLLM PID $VLLM_PID. Waiting for ready..."

for i in $(seq 1 60); do
    sleep 5
    if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:$EVAL_PORT/health', timeout=3)" 2>/dev/null; then
        echo "$(ts) vLLM ready after ${i}×5 seconds."
        break
    fi
done

# Step 5: Evaluate
echo "$(ts) Running spc_eval..."
PYTHONUNBUFFERED=1 python3 common/tools/eval/spc_eval.py \
    --model_url http://localhost:$EVAL_PORT \
    --model_name default \
    --test common/data/test.jsonl \
    --output "$RESULT_FILE" \
    --concurrency 4 \
    --max_tokens 4096 \
    > round3/logs/eval_R3-AB-v4.log 2>&1

EVAL_EXIT=$?
python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
d['summary']['model'] = 'R3-AB-v4'
json.dump(d, open('$RESULT_FILE', 'w'), indent=2, ensure_ascii=False)
" 2>/dev/null

kill $VLLM_PID 2>/dev/null
fuser -k ${EVAL_PORT}/tcp 2>/dev/null

if [ $EVAL_EXIT -ne 0 ]; then
    echo "$(ts) EVAL FAILED (exit=$EVAL_EXIT)."
    exit 1
fi

F1=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(d['summary']['rule_detection_f1'])" 2>/dev/null)
echo "$(ts) R3-AB-v4 eval done. F1=$F1"
echo "$(ts) $RESULT_FILE written."
