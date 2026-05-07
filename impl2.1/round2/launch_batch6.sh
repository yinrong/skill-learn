#!/bin/bash
# Batch 6 launch: wait for all Batch 4+5 evals to complete, then start training
# Run: nohup bash launch_batch6.sh > history-route2.1.1/logs/batch6_launch.log 2>&1 &
cd "$(dirname "$0")"
HISTDIR="history-route2.1.1"
MODEL_14B="/home/yinrong/models/Qwen3-14B"

# Wait for all Batch 4+5 eval results
WAIT_EXPS="expMM expNN expPP expQQ expRR expSS expTT expUU"

echo "[$(date)] Batch 6 launcher started. Waiting for Batch 4+5 results..."

while true; do
    missing=0
    for exp in $WAIT_EXPS; do
        if [ ! -f "$HISTDIR/results/${exp}.json" ]; then
            missing=$((missing + 1))
        fi
    done
    if [ $missing -eq 0 ]; then
        echo "[$(date)] All Batch 4+5 evals complete!"
        break
    fi
    echo "[$(date)] Waiting... $missing results still missing"
    # Show what's done
    for exp in $WAIT_EXPS; do
        if [ -f "$HISTDIR/results/${exp}.json" ]; then
            f1=$(python3 -c "import json; d=json.load(open('$HISTDIR/results/${exp}.json')); print(d['summary']['rule_detection_f1'])" 2>/dev/null || echo "?")
            echo "  ✓ $exp: F1=$f1"
        else
            echo "  ⏳ $exp: pending"
        fi
    done
    sleep 120
done

# Print Batch 4+5 results summary
echo ""
echo "=== Batch 4+5 Results Summary ==="
for exp in expMM expNN expPP expQQ expRR expSS expTT expUU; do
    result="$HISTDIR/results/${exp}.json"
    if [ -f "$result" ]; then
        f1=$(python3 -c "import json; d=json.load(open('$result')); s=d['summary']; print(f'F1={s[\"rule_detection_f1\"]} cpk={s[\"cpk_found_rate\"]} per_rule={s[\"per_rule_recall\"]}')" 2>/dev/null || echo "error")
        echo "  $exp: $f1"
    fi
done

# Kill all existing vLLM servers to free GPU memory
echo ""
echo "[$(date)] Killing all vLLM servers to free GPU memory..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 10

# Verify GPUs are free
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null
echo "[$(date)] GPUs freed."

# Start Batch 6 training on 4 GPUs simultaneously
# expWW: 500 with_skill teacher, GPU 0
# expXX: 500 mixed (250+250), GPU 1
# expVV: 800 no_skill teacher, GPU 4 (needs more VRAM due to longer sequences)
# expYY2: 600 mixed (300 ws + 300 ns_v2), GPU 7

echo "[$(date)] Starting Batch 6 training..."
echo "  expWW: 500 with_skill teacher, GPU 0"
echo "  expXX: 500 mixed teacher, GPU 1"
echo "  expVV: 800 no_skill teacher, GPU 4"
echo "  expYY2: 600 mixed teacher, GPU 7"

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 \
    llamafactory-cli train "$HISTDIR/configs/expWW.yaml" \
    > "$HISTDIR/logs/expWW_train.log" 2>&1 &
PID_WW=$!
echo "[$(date)] expWW training started (PID=$PID_WW, GPU 0)"

sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=1 \
    llamafactory-cli train "$HISTDIR/configs/expXX.yaml" \
    > "$HISTDIR/logs/expXX_train.log" 2>&1 &
PID_XX=$!
echo "[$(date)] expXX training started (PID=$PID_XX, GPU 1)"

sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=4 \
    llamafactory-cli train "$HISTDIR/configs/expVV.yaml" \
    > "$HISTDIR/logs/expVV_train.log" 2>&1 &
PID_VV=$!
echo "[$(date)] expVV training started (PID=$PID_VV, GPU 4)"

sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=7 \
    llamafactory-cli train "$HISTDIR/configs/expYY2.yaml" \
    > "$HISTDIR/logs/expYY2_train.log" 2>&1 &
PID_YY2=$!
echo "[$(date)] expYY2 training started (PID=$PID_YY2, GPU 7)"

# Now monitor Batch 6 training + eval
declare -A EXP_MODEL
declare -A EXP_GPU
declare -A EXP_N
declare -A DONE
declare -A EVAL_RUNNING

EXP_MODEL=([expWW]=$MODEL_14B [expXX]=$MODEL_14B [expVV]=$MODEL_14B [expYY2]=$MODEL_14B)
EXP_GPU=([expWW]=0 [expXX]=1 [expVV]=4 [expYY2]=7)
EXP_N=([expWW]=500 [expXX]=500 [expVV]=800 [expYY2]=600)

echo ""
echo "[$(date)] Monitoring Batch 6 training..."

while true; do
    all_done=true

    for exp in expWW expXX expVV expYY2; do
        if [ "${DONE[$exp]}" = "1" ]; then
            continue
        fi

        log="$HISTDIR/logs/${exp}_train.log"
        result="$HISTDIR/results/${exp}.json"

        if grep -q "train_runtime" "$log" 2>/dev/null; then
            DONE[$exp]="1"

            if [ -f "$result" ]; then
                echo "[$(date)] ⏭ $exp: 评测已完成"
                continue
            fi

            if [ "${EVAL_RUNNING[$exp]}" = "1" ]; then
                echo "[$(date)] 🔄 $exp: 评测进行中..."
                continue
            fi

            EVAL_RUNNING[$exp]="1"
            echo "[$(date)] ✅ $exp: 训练完成，启动评测..."

            GPU="${EXP_GPU[$exp]}"
            N="${EXP_N[$exp]}"

            # expVV uses cutoff_len=5120 (ns-only), others use cutoff_len=6144 (ws data)
            # All use max_tokens=5000 to accommodate full outputs
            MAX_TOK=5000
            bash eval_and_report.sh "$exp" "$MODEL_14B" "$GPU" "$N" "$MAX_TOK" \
                >> "$HISTDIR/logs/${exp}_eval_full.log" 2>&1 &
            echo "[$(date)] 📊 $exp: 评测已在 GPU $GPU 后台启动 (PID=$!)"
        else
            all_done=false
            step=$(grep -oE '\s+[0-9]+/[0-9]+ \[' "$log" 2>/dev/null | tail -1 | tr -d '[] ' | xargs)
            echo "[$(date)] ⏳ $exp: step=$step"
        fi
    done

    # Check eval completions
    for exp in expWW expXX expVV expYY2; do
        if [ "${DONE[$exp]}" = "1" ] && [ "${EVAL_RUNNING[$exp]}" = "1" ]; then
            result="$HISTDIR/results/${exp}.json"
            if [ -f "$result" ]; then
                f1=$(python3 -c "import json; d=json.load(open('$result')); print(d['summary']['rule_detection_f1'])" 2>/dev/null || echo "?")
                echo "[$(date)] 🎯 $exp: 评测完成 F1=$f1"
                EVAL_RUNNING[$exp]="done"
            fi
        fi
    done

    if [ "$all_done" = "true" ]; then
        pending_evals=0
        for exp in expWW expXX expVV expYY2; do
            if [ "${EVAL_RUNNING[$exp]}" = "1" ]; then
                pending_evals=$((pending_evals + 1))
            fi
        done
        if [ $pending_evals -eq 0 ]; then
            echo "[$(date)] ✅ Batch 6 全部完成！"
            break
        fi
    fi

    sleep 60
done

# Final Batch 6 summary
echo ""
echo "=== Batch 6 Results Summary ==="
for exp in expWW expXX expVV expYY2; do
    result="$HISTDIR/results/${exp}.json"
    if [ -f "$result" ]; then
        python3 -c "
import json
d = json.load(open('$result'))
s = d['summary']
print(f'  $exp: F1={s[\"rule_detection_f1\"]} cpk={s[\"cpk_found_rate\"]}')
print(f'    per_rule: {s[\"per_rule_recall\"]}')
" 2>/dev/null || echo "  $exp: error reading result"
    fi
done

echo "[$(date)] Batch 6 launcher complete."
