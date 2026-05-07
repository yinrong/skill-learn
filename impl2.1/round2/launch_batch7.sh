#!/bin/bash
# Batch 7 launch: wait for all Batch 6 evals to complete, then start training
# Run: nohup bash launch_batch7.sh > history-route2.1.1/logs/batch7_launch.log 2>&1 &
cd "$(dirname "$0")"
HISTDIR="history-route2.1.1"
MODEL_14B="/home/yinrong/models/Qwen3-14B"

# Wait for all Batch 6 eval results
WAIT_EXPS="expWW expXX expVV expYY2"

echo "[$(date)] Batch 7 launcher started. Waiting for Batch 6 results..."

while true; do
    missing=0
    for exp in $WAIT_EXPS; do
        if [ ! -f "$HISTDIR/results/${exp}.json" ]; then
            missing=$((missing + 1))
        fi
    done
    if [ $missing -eq 0 ]; then
        echo "[$(date)] All Batch 6 evals complete!"
        break
    fi
    echo "[$(date)] Waiting... $missing results still missing"
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

# Print Batch 6 results summary
echo ""
echo "=== Batch 6 Results Summary ==="
for exp in expWW expXX expVV expYY2; do
    result="$HISTDIR/results/${exp}.json"
    if [ -f "$result" ]; then
        f1=$(python3 -c "import json; d=json.load(open('$result')); s=d['summary']; print(f'F1={s[\"rule_detection_f1\"]} cpk={s[\"cpk_found_rate\"]}')" 2>/dev/null || echo "error")
        echo "  $exp: $f1"
    fi
done

# Check if Batch 7 training already done (may have been pre-launched on idle GPUs)
B7_PRETRAINED=0
b7_count=0
for exp in expZZ expAAA expBBB expCCC; do
    if grep -q "train_runtime" "$HISTDIR/logs/${exp}_train.log" 2>/dev/null; then
        b7_count=$((b7_count + 1))
    fi
done
[ $b7_count -ge 4 ] && B7_PRETRAINED=1

if [ "$B7_PRETRAINED" = "1" ]; then
    echo "[$(date)] Batch 7 training already complete (pre-launched). Waiting for all B7 evals..."
    while true; do
        missing=0
        for exp in expZZ expAAA expBBB expCCC; do
            [ ! -f "$HISTDIR/results/${exp}.json" ] && missing=$((missing + 1))
        done
        [ $missing -eq 0 ] && { echo "[$(date)] All B7 results ready."; break; }
        echo "[$(date)] Waiting for B7 evals... $missing still pending"
        sleep 120
    done
else
    # Kill all existing vLLM servers to free GPU memory
    echo ""
    echo "[$(date)] Killing all vLLM servers to free GPU memory..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 10
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null
    echo "[$(date)] GPUs freed."

    # Batch 7 experiments:
    # expZZ:  200 ns_v1 × 3ep = 75 steps  (fewer steps than expRR's 125)   → GPU 0
    # expAAA: 200 ns_v2 × 5ep = 125 steps (same steps, different seed pool) → GPU 1
    # expBBB: 400 (v1+v2) × 2ep ≈100 steps (cross-pool diversity, fewer steps) → GPU 2
    # expCCC: 400 (v1+v2) × 3ep = 150 steps (cross-pool diversity, more steps)  → GPU 3

    echo "[$(date)] Starting Batch 7 training..."
    echo "  expZZ:  200 ns_v1 x 3ep = 75 steps,   GPU 0"
    echo "  expAAA: 200 ns_v2 x 5ep = 125 steps,  GPU 1"
    echo "  expBBB: 400 mixed x 2ep = 100 steps,  GPU 2"
    echo "  expCCC: 400 mixed x 3ep = 150 steps,  GPU 3"

    DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 \
        llamafactory-cli train "$HISTDIR/configs/expZZ.yaml" \
        > "$HISTDIR/logs/expZZ_train.log" 2>&1 &
    PID_ZZ=$!
    echo "[$(date)] expZZ training started (PID=$PID_ZZ, GPU 0)"
    sleep 5

    DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=1 \
        llamafactory-cli train "$HISTDIR/configs/expAAA.yaml" \
        > "$HISTDIR/logs/expAAA_train.log" 2>&1 &
    PID_AAA=$!
    echo "[$(date)] expAAA training started (PID=$PID_AAA, GPU 1)"
    sleep 5

    DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=2 \
        llamafactory-cli train "$HISTDIR/configs/expBBB.yaml" \
        > "$HISTDIR/logs/expBBB_train.log" 2>&1 &
    PID_BBB=$!
    echo "[$(date)] expBBB training started (PID=$PID_BBB, GPU 2)"
    sleep 5

    DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=3 \
        llamafactory-cli train "$HISTDIR/configs/expCCC.yaml" \
        > "$HISTDIR/logs/expCCC_train.log" 2>&1 &
    PID_CCC=$!
fi

# If pre-trained, all results are already in — no monitoring needed
if [ "$B7_PRETRAINED" = "1" ]; then
    echo ""
    echo "=== Batch 7 Results Summary (pre-trained) ==="
    for exp in expZZ expAAA expBBB expCCC; do
        result="$HISTDIR/results/${exp}.json"
        if [ -f "$result" ]; then
            f1=$(python3 -c "import json; d=json.load(open('$result')); print(d['summary']['rule_detection_f1'])" 2>/dev/null || echo "?")
            echo "  $exp: F1=$f1"
        fi
    done
    echo "[$(date)] Batch 7 launcher complete (pre-trained path)."
    exit 0
fi

echo "[$(date)] expCCC training started (PID=$PID_CCC, GPU 3)"

# Monitor Batch 7 training + eval
declare -A EXP_GPU
declare -A EXP_N
declare -A DONE
declare -A EVAL_RUNNING

EXP_GPU=([expZZ]=0 [expAAA]=1 [expBBB]=2 [expCCC]=3)
EXP_N=([expZZ]=200 [expAAA]=200 [expBBB]=400 [expCCC]=400)

echo ""
echo "[$(date)] Monitoring Batch 7 training..."

while true; do
    all_done=true

    for exp in expZZ expAAA expBBB expCCC; do
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

            # All Batch 7 experiments use cutoff_len=5120, max_tokens=5000
            bash eval_and_report.sh "$exp" "$MODEL_14B" "$GPU" "$N" 5000 \
                >> "$HISTDIR/logs/${exp}_eval_full.log" 2>&1 &
            echo "[$(date)] 📊 $exp: 评测已在 GPU $GPU 后台启动 (PID=$!)"
        else
            all_done=false
            step=$(grep -oE '\s+[0-9]+/[0-9]+ \[' "$log" 2>/dev/null | tail -1 | tr -d '[] ' | xargs)
            echo "[$(date)] ⏳ $exp: step=$step"
        fi
    done

    for exp in expZZ expAAA expBBB expCCC; do
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
        for exp in expZZ expAAA expBBB expCCC; do
            if [ "${EVAL_RUNNING[$exp]}" = "1" ]; then
                pending_evals=$((pending_evals + 1))
            fi
        done
        if [ $pending_evals -eq 0 ]; then
            echo "[$(date)] ✅ Batch 7 全部完成！"
            break
        fi
    fi

    sleep 60
done

# Final Batch 7 summary
echo ""
echo "=== Batch 7 Results Summary ==="
for exp in expZZ expAAA expBBB expCCC; do
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

echo "[$(date)] Batch 7 launcher complete."
