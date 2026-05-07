#!/bin/bash
cd /home/yinrong/post-train/impl2.1

echo "[$(date)] Monitoring lm_eval processes..."
while true; do
    LMEVAL_PIDS=$(ps aux | grep "lm_eval" | grep -v grep | awk '{print $2}')
    if [ -z "$LMEVAL_PIDS" ]; then
        echo "[$(date)] All lm_eval processes finished."
        break
    fi
    echo "[$(date)] lm_eval running, waiting 60s..."
    sleep 60
done

echo "[$(date)] Computing R3 degradation rates..."
python3 common/tools/eval/compute_degradation.py \
    --benchmark_dir common/benchmark_results \
    --sft_tags sft-14B-R3-AB-v2 sft-14B-R3-D1 sft-32B-R3-D2 \
    --base_map 14B=base-14B 32B=base-32B \
    --output common/benchmark_results/degradation_summary.json \
    2>&1 | tee round3/logs/degradation_R3.log
echo "[$(date)] Degradation computed."

echo "[$(date)] Starting GRPO training on GPU 0 (single GPU, 14B bf16 fits in 97GB)..."
# 用 torchrun 启动（单卡），num_generations=2, max_new_tokens=512 减少耗时
CUDA_VISIBLE_DEVICES=0 DISABLE_VERSION_CHECK=1 \
    python3 round3/tools/train/train_grpo_trl.py \
        --model round3/checkpoints/R3-AB-v2-merged \
        --train_data round3/data/train_R3-AB-v2.jsonl \
        --output_dir round3/checkpoints/R3-C-grpo \
        --num_train_epochs 1 \
        --per_device_batch_size 1 \
        --num_generations 2 \
        --learning_rate 5e-7 \
        --max_new_tokens 512 \
        --num_generations 2 \
        2>&1 | tee round3/logs/train_R3-C-grpo.log
GRPO_EXIT=$?
echo "[$(date)] GRPO training finished (exit=$GRPO_EXIT)."

# 等待 R3-D3 完成 + GRPO 已完成，再启动 R3-E1（需要全部 8 卡）
echo "[$(date)] Waiting for R3-D3 to finish before starting R3-E1 (8-GPU)..."
while true; do
    D3_RUNNING=$(ps aux | grep "R3-D3-qlora" | grep -v grep)
    if [ -z "$D3_RUNNING" ]; then
        echo "[$(date)] R3-D3 done. Starting R3-E1 on all 8 GPUs..."
        break
    fi
    echo "[$(date)] R3-D3 still running, waiting 5min..."
    sleep 300
done

echo "[$(date)] Starting R3-E1: 32B bf16 8-GPU FSDP..."
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 llamafactory-cli train \
    round3/configs/R3-E1-fsdp-32b-bf16.yaml \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config round3/configs/fsdp_config.json \
    2>&1 | tee round3/logs/train_R3-E1.log
echo "[$(date)] R3-E1 training completed."
