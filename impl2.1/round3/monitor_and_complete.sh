#!/bin/bash
# 监控所有进行中的实验，完成后自动执行 merge + eval + 更新报告
# 从 impl2.1 目录运行：bash round3/monitor_and_complete.sh

cd /home/yinrong/post-train/impl2.1
LOG="round3/logs/monitor.log"
exec > >(tee -a "$LOG") 2>&1

ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]"; }

merge_and_eval() {
    local NAME=$1
    local BASE=$2
    local ADAPTER=$3
    local MERGED=$4
    local PORT=$5
    local OUT="round3/results/${NAME}.json"

    echo "$(ts) === $NAME: Starting merge ==="

    # Merge
    DISABLE_VERSION_CHECK=1 llamafactory-cli export \
        --model_name_or_path "$BASE" \
        --adapter_name_or_path "$ADAPTER" \
        --template qwen3 \
        --finetuning_type lora \
        --export_dir "$MERGED" \
        --export_legacy_format false \
        > round3/logs/merge_${NAME}.log 2>&1

    if [ $? -ne 0 ]; then
        echo "$(ts) $NAME merge FAILED. See round3/logs/merge_${NAME}.log"
        return 1
    fi
    echo "$(ts) $NAME merge done."

    # Start vLLM
    echo "$(ts) $NAME: Starting vLLM on port $PORT..."
    CUDA_VISIBLE_DEVICES=$(( PORT - 8040 )) python3 common/tools/train/deploy_vllm.py \
        --model "$MERGED" --port "$PORT" \
        > round3/logs/vllm_${NAME}.log 2>&1 &
    VLLM_PID=$!

    sleep 60  # wait for vLLM to start

    # Eval
    echo "$(ts) $NAME: Running evaluation..."
    python3 common/tools/eval/spc_eval.py \
        --model_url "http://localhost:${PORT}" \
        --model_name "$NAME" \
        --test common/data/test.jsonl \
        --output "$OUT" \
        --concurrency 4 \
        > round3/logs/eval_${NAME}.log 2>&1

    EVAL_EXIT=$?
    kill $VLLM_PID 2>/dev/null

    if [ $EVAL_EXIT -ne 0 ]; then
        echo "$(ts) $NAME eval FAILED."
        return 1
    fi

    F1=$(python3 -c "import json; d=json.load(open('$OUT')); print(d['summary']['rule_detection_f1'])" 2>/dev/null)
    echo "$(ts) $NAME eval done. F1=$F1. Result: $OUT"
}

# ── R3-D3 (int8) ──────────────────────────────────────────────────────────────
echo "$(ts) Monitoring R3-D3 (int8 training)..."
while true; do
    if ! pgrep -f "R3-D3-qlora-14b-int8" > /dev/null 2>&1; then
        if [ -d "round3/checkpoints/R3-D3-qlora-14b-int8" ]; then
            echo "$(ts) R3-D3 training completed. Starting merge+eval..."
            merge_and_eval "R3-D3" \
                "/home/yinrong/models/Qwen3-14B" \
                "round3/checkpoints/R3-D3-qlora-14b-int8" \
                "round3/checkpoints/R3-D3-qlora-14b-int8-merged" \
                "8041"
            break
        fi
    fi
    sleep 300
done &
WATCH_D3=$!

# ── R3-C (GRPO) ───────────────────────────────────────────────────────────────
echo "$(ts) Monitoring R3-C (GRPO)..."
while true; do
    if [ -f "round3/logs/train_R3-C-grpo.log" ]; then
        if grep -q "Training completed\|trainer.train() done\|Model saved to" round3/logs/train_R3-C-grpo.log 2>/dev/null; then
            echo "$(ts) R3-C GRPO training completed. Starting eval..."
            MERGED="round3/checkpoints/R3-C-grpo"

            # Start vLLM
            CUDA_VISIBLE_DEVICES=0 python3 common/tools/train/deploy_vllm.py \
                --model "$MERGED" --port 8042 \
                > round3/logs/vllm_R3-C.log 2>&1 &
            sleep 60

            python3 common/tools/eval/spc_eval.py \
                --model_url "http://localhost:8042" \
                --model_name "R3-C-grpo" \
                --test common/data/test.jsonl \
                --output "round3/results/R3-C-grpo.json" \
                --concurrency 4 \
                > round3/logs/eval_R3-C-grpo.log 2>&1

            kill %% 2>/dev/null
            F1=$(python3 -c "import json; d=json.load(open('round3/results/R3-C-grpo.json')); print(d['summary']['rule_detection_f1'])" 2>/dev/null)
            echo "$(ts) R3-C eval done. F1=$F1"
            break
        fi
    fi
    sleep 120
done &
WATCH_C=$!

# ── R3-E1 (32B bf16 8-GPU) ────────────────────────────────────────────────────
echo "$(ts) Monitoring R3-E1 (32B bf16 8-GPU)..."
while true; do
    if [ -f "round3/logs/train_R3-E1.log" ]; then
        if grep -q "train_runtime\|Training completed" round3/logs/train_R3-E1.log 2>/dev/null; then
            echo "$(ts) R3-E1 training completed. Starting merge+eval..."
            merge_and_eval "R3-E1" \
                "/home/yinrong/models/Qwen3-32B" \
                "round3/checkpoints/R3-E1-fsdp-32b-bf16" \
                "round3/checkpoints/R3-E1-fsdp-32b-bf16-merged" \
                "8043"
            break
        fi
    fi
    sleep 300
done &
WATCH_E1=$!

echo "$(ts) All monitors started (PIDs: D3=$WATCH_D3, C=$WATCH_C, E1=$WATCH_E1)"
echo "$(ts) This script will run in the background and auto-complete experiments."
wait
echo "$(ts) All monitors completed."
