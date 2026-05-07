#!/bin/bash
# 等待 GRPO 训练和评测全部完成后，启动 R3-E1（32B bf16 4卡 FSDP）
# 从 impl2.1/ 目录运行：bash round3/launch_R3-E1_when_ready.sh
#
# 依赖条件：
#   1. R3-C GRPO 训练完成（round3/results/R3-C-grpo.json 存在）
#   2. GPU 0 和 GPU 3 空闲（vLLM eval 不再占用）
#   3. GPU 1 和 GPU 2 空闲（GRPO 训练不再占用）

cd /home/yinrong/post-train/impl2.1
LOG="round3/logs/monitor.log"
ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]"; }

echo "$(ts) [E1 launcher] Waiting for all GPUs to be free and GRPO eval done..."

while true; do
    # Check GRPO eval done
    if [ ! -f "round3/results/R3-C-grpo.json" ]; then
        echo "$(ts) [E1 launcher] Waiting: R3-C-grpo.json not yet present"
        sleep 300
        continue
    fi

    # Check all 4 GPUs free (< 5GB used each = effectively free)
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}')
    if [ -z "$GPU_USED" ] || [ "$GPU_USED" -gt 20000 ]; then
        echo "$(ts) [E1 launcher] Waiting: total GPU memory in use = ${GPU_USED}MB (need < 20000MB)"
        sleep 300
        continue
    fi

    # All conditions met — launch R3-E1
    echo "$(ts) [E1 launcher] All conditions met! Launching R3-E1 (32B bf16 4-GPU FSDP)..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 \
        llamafactory-cli train round3/configs/R3-E1-fsdp-32b-bf16.yaml \
        --fsdp "full_shard auto_wrap" \
        --fsdp_config round3/configs/fsdp_config.json \
        > round3/logs/train_R3-E1.log 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "$(ts) [E1 launcher] R3-E1 training FAILED (exit $EXIT_CODE)" | tee -a "$LOG"
        exit 1
    fi

    echo "$(ts) [E1 launcher] R3-E1 training done. Merging (PEFT direct, not llamafactory-cli)..."
    python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = '/home/yinrong/models/Qwen3-32B'
adapter = 'round3/checkpoints/R3-E1-fsdp-32b-bf16'
output = 'round3/checkpoints/R3-E1-fsdp-32b-bf16-merged'
print('Loading base...'); model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cpu')
print('Loading adapter...'); model = PeftModel.from_pretrained(model, adapter)
print('Merging...'); model = model.merge_and_unload()
model.save_pretrained(output, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
tok.save_pretrained(output)
print('Done.')
" > round3/logs/merge_R3-E1.log 2>&1

    if [ $? -ne 0 ]; then
        echo "$(ts) [E1 launcher] R3-E1 merge FAILED" | tee -a "$LOG"
        exit 1
    fi

    echo "$(ts) [E1 launcher] Merge done. Starting vLLM on GPU 0 port 8043 (no reasoning-parser)..."
    CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
        --model round3/checkpoints/R3-E1-fsdp-32b-bf16-merged \
        --port 8043 --dtype bfloat16 --max-model-len 7168 \
        --served-model-name default --trust-remote-code \
        >> round3/logs/vllm_R3-E1.log 2>&1 &
    VLLM_PID=$!

    # Wait for health with proper check
    for i in $(seq 1 60); do
        sleep 5
        if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8043/health', timeout=3)" 2>/dev/null; then
            echo "$(ts) [E1 launcher] vLLM ready."; break; fi
    done

    echo "$(ts) [E1 launcher] Running eval..."
    PYTHONUNBUFFERED=1 python3 common/tools/eval/spc_eval.py \
        --model_url http://localhost:8043 \
        --model_name default \
        --test common/data/test.jsonl \
        --output round3/results/R3-E1.json \
        --concurrency 4 \
        --max_tokens 4096 \
        > round3/logs/eval_R3-E1.log 2>&1

    python3 -c "import json; d=json.load(open('round3/results/R3-E1.json')); d['summary']['model']='R3-E1'; json.dump(d, open('round3/results/R3-E1.json','w'), indent=2)" 2>/dev/null
    kill $VLLM_PID 2>/dev/null
    F1=$(python3 -c "import json; d=json.load(open('round3/results/R3-E1.json')); print(d['summary']['rule_detection_f1'])" 2>/dev/null)
    echo "$(ts) [E1 launcher] R3-E1 eval done. F1=$F1" | tee -a "$LOG"
    break
done
