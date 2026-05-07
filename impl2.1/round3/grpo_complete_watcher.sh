#!/bin/bash
# GRPO 完成监控 + 自动 merge + eval → R3-C-grpo.json
# 修复点：
#   1. PEFT 直接合并（绕过 llamafactory-cli ImportError）
#   2. vLLM 不加 --reasoning-parser（避免 content="None" bug）
#   3. model_name=default（与 deploy_vllm.py 一致）
#   4. max_tokens=4096（SPC 响应约 4000 tokens）
#   5. GPU 选择：GRPO 使用 GPU1+2，eval 用 GPU1（GRPO 完成后自动释放）
cd /home/yinrong/post-train/impl2.1
LOG="round3/logs/grpo_watcher.log"
ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]"; }
exec > >(tee -a "$LOG") 2>&1

echo "$(ts) GRPO watcher started. Watching for train_R3-C-grpo.log completion..."

# Step 1: Wait for GRPO training to complete
while true; do
    if grep -q "train_runtime" round3/logs/train_R3-C-grpo.log 2>/dev/null; then
        echo "$(ts) GRPO training completed (train_runtime detected)."
        break
    fi
    # Check if process died without completing
    GRPO_RUNNING=$(ps aux | grep "train_grpo_trl.py" | grep -v grep | wc -l)
    if [ "$GRPO_RUNNING" -eq 0 ]; then
        echo "$(ts) WARNING: GRPO process not running. Checking if already done..."
        if [ -d "round3/checkpoints/R3-C-grpo" ] && \
           ls round3/checkpoints/R3-C-grpo/*.safetensors 2>/dev/null | head -1 | grep -q .; then
            echo "$(ts) Checkpoint exists, assuming complete."
            break
        else
            echo "$(ts) No checkpoint found and no process running. Waiting for restart..."
            sleep 120
            continue
        fi
    fi
    echo "$(ts) GRPO still training (step $(grep -oP '\d+/283' round3/logs/train_R3-C-grpo.log 2>/dev/null | tail -1 || echo '?'))..."
    sleep 300
done

# Step 2: Find checkpoint directory
ADAPTER_DIR="round3/checkpoints/R3-C-grpo"
MERGED_DIR="round3/checkpoints/R3-C-grpo-merged"
RESULT_FILE="round3/results/R3-C-grpo.json"

if [ -f "$RESULT_FILE" ]; then
    echo "$(ts) R3-C-grpo.json already exists. Done."
    exit 0
fi

echo "$(ts) Starting PEFT merge..."
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = 'round3/checkpoints/R3-AB-v2-merged'
adapter = 'round3/checkpoints/R3-C-grpo'
output = 'round3/checkpoints/R3-C-grpo-merged'

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
" > round3/logs/merge_R3-C-grpo.log 2>&1

if [ $? -ne 0 ]; then
    echo "$(ts) MERGE FAILED. See round3/logs/merge_R3-C-grpo.log"
    exit 1
fi
echo "$(ts) Merge complete."

# Step 3: Wait for GPU 1 to be free (GRPO was using GPU1+2, they'll be freed)
echo "$(ts) Waiting for GPU 1 to be free (< 5000 MB used)..."
for i in $(seq 1 120); do
    GPU1_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
    if [ -n "$GPU1_MEM" ] && [ "$GPU1_MEM" -lt 5000 ]; then
        echo "$(ts) GPU 1 is free (${GPU1_MEM} MB used). Proceeding."
        break
    fi
    echo "$(ts) GPU 1 has ${GPU1_MEM} MB used. Waiting..."
    sleep 30
done

# Step 4: Start vLLM on GPU 1 (freed by GRPO), port 8042
echo "$(ts) Starting vLLM on GPU 1 port 8042..."
CUDA_VISIBLE_DEVICES=1 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_DIR" \
    --port 8042 \
    --dtype bfloat16 \
    --max-model-len 7168 \
    --served-model-name default \
    --trust-remote-code \
    >> round3/logs/vllm_R3-C-grpo.log 2>&1 &
VLLM_PID=$!
echo "$(ts) vLLM PID $VLLM_PID. Waiting for ready..."

# Wait for health endpoint
for i in $(seq 1 60); do
    sleep 5
    if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8042/health', timeout=3)" 2>/dev/null; then
        echo "$(ts) vLLM ready after ${i}×5 seconds."
        break
    fi
done

# Step 5: Evaluate
echo "$(ts) Running spc_eval..."
PYTHONUNBUFFERED=1 python3 common/tools/eval/spc_eval.py \
    --model_url http://localhost:8042 \
    --model_name default \
    --test common/data/test.jsonl \
    --output "$RESULT_FILE" \
    --concurrency 4 \
    --max_tokens 4096 \
    > round3/logs/eval_R3-C-grpo.log 2>&1

EVAL_EXIT=$?
# Fix model field
python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
d['summary']['model'] = 'R3-C-grpo'
json.dump(d, open('$RESULT_FILE', 'w'), indent=2, ensure_ascii=False)
" 2>/dev/null

kill $VLLM_PID 2>/dev/null

if [ $EVAL_EXIT -ne 0 ]; then
    echo "$(ts) EVAL FAILED (exit=$EVAL_EXIT)."
    exit 1
fi

F1=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(d['summary']['rule_detection_f1'])" 2>/dev/null)
echo "$(ts) GRPO eval done. F1=$F1"
echo "$(ts) R3-C-grpo.json written. E1 launcher will auto-trigger."
