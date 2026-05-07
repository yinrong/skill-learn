#!/bin/bash
# Batch 10: Apply best Batch 9 findings — expected: cutoff_len=5120 is key fix
# Experiments:
#   expNNN: 32B + cutoff_len=5120 + 200 ns_v1 × 5ep (apply cutoff fix to 32B)
#   expOOO: 14B + cutoff_len=5120 + LR=5e-5 + 200 ns_v1 × 5ep (combine MMM + KKK)
#   expPPP: 14B + cutoff_len=5120 + 400 (ns_v1+ns_v3) × 3ep = 150 steps (cross-pool + fix)
#   expQQQ: 7B + cutoff_len=5120 + 200 ns_v1 × 5ep (smallest model + fix)
#   expRRR: 14B + cutoff_len=5120 + 400 (ns_v1+ns_v3) × 5ep = 250 steps (overfit test)
# Run: nohup bash launch_batch10.sh > history-route2.1.1/logs/batch10_launch.log 2>&1 &
cd "$(dirname "$0")"
HISTDIR="history-route2.1.1"
MODEL_7B="/home/yinrong/models/Qwen3-7B"
MODEL_8B="/home/yinrong/models/Qwen3-8B"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
MODEL_32B="/home/yinrong/models/Qwen3-32B"

# Wait for Batch 9 results
WAIT_EXPS="expIII expJJJ expKKK expLLL expMMM"

echo "[$(date)] Batch 10 launcher started. Waiting for Batch 9 results..."

while true; do
    missing=0
    for exp in $WAIT_EXPS; do
        [ ! -f "$HISTDIR/results/${exp}.json" ] && missing=$((missing + 1))
    done
    if [ $missing -eq 0 ]; then
        echo "[$(date)] All Batch 9 evals complete!"
        break
    fi
    echo "[$(date)] Waiting... $missing B9 results missing"
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

# Print Batch 9 results summary
echo ""
echo "=== Batch 9 Results ==="
python3 -c "
import json, os, glob

results_dir = '$HISTDIR/results'
exps = ['expIII','expJJJ','expKKK','expLLL','expMMM']
for exp in exps:
    path = f'{results_dir}/{exp}.json'
    if os.path.exists(path):
        d = json.load(open(path))
        s = d['summary']
        print(f'  {exp}: F1={s[\"rule_detection_f1\"]:.3f} cpk={s[\"cpk_found_rate\"]:.3f}')
        for r, v in s['per_rule_recall'].items():
            bar = '█' * int((v or 0) * 15)
            print(f'    {r}: {v:.3f}  {bar}')
" 2>/dev/null

# Analyze best config from Batch 9
BEST_B9=$(python3 -c "
import json, os
exps = ['expIII','expJJJ','expKKK','expLLL','expMMM']
best_exp, best_f1 = None, 0
for exp in exps:
    path = '$HISTDIR/results/' + exp + '.json'
    if os.path.exists(path):
        f1 = json.load(open(path))['summary']['rule_detection_f1']
        if f1 > best_f1:
            best_f1 = f1
            best_exp = exp
print(best_exp or 'expMMM')
" 2>/dev/null)
echo "[$(date)] Best Batch 9 experiment: $BEST_B9"

# Check if expMMM (cutoff fix) is confirmed better than expRR
MMM_F1=$(python3 -c "import json; d=json.load(open('$HISTDIR/results/expMMM.json')); print(d['summary']['rule_detection_f1'])" 2>/dev/null || echo "0")
RRR_F1=$(python3 -c "import json; d=json.load(open('$HISTDIR/results/expRR.json')); print(d['summary']['rule_detection_f1'])" 2>/dev/null || echo "0")
echo "[$(date)] expMMM (cutoff=5120): F1=$MMM_F1 vs expRR (cutoff=4096): F1=$RRR_F1"

# Prepare Batch 10 training data
bash prepare_batch10.sh

# Kill all existing vLLM servers
echo "[$(date)] Killing vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 10

echo "[$(date)] Starting Batch 10 training..."
echo "  expNNN: 32B + cutoff=5120 + 200 ns_v1 × 5ep, GPU 0"
echo "  expOOO: 14B + cutoff=5120 + LR=5e-5 + 200 ns_v1 × 5ep, GPU 1"
echo "  expPPP: 14B + cutoff=5120 + 400 (v1+v3) × 3ep = 150 steps, GPU 2"
echo "  expQQQ: 7B + cutoff=5120 + 200 ns_v1 × 5ep, GPU 3"
echo "  expSSS: 14B + cutoff=5120 + 200 ns_v3 × 5ep, GPU 4"

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 \
    llamafactory-cli train "$HISTDIR/configs/expNNN.yaml" \
    > "$HISTDIR/logs/expNNN_train.log" 2>&1 &
echo "[$(date)] expNNN started (PID=$!, GPU 0)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=1 \
    llamafactory-cli train "$HISTDIR/configs/expOOO.yaml" \
    > "$HISTDIR/logs/expOOO_train.log" 2>&1 &
echo "[$(date)] expOOO started (PID=$!, GPU 1)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=2 \
    llamafactory-cli train "$HISTDIR/configs/expPPP.yaml" \
    > "$HISTDIR/logs/expPPP_train.log" 2>&1 &
echo "[$(date)] expPPP started (PID=$!, GPU 2)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=3 \
    llamafactory-cli train "$HISTDIR/configs/expQQQ.yaml" \
    > "$HISTDIR/logs/expQQQ_train.log" 2>&1 &
echo "[$(date)] expQQQ started (PID=$!, GPU 3)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=4 \
    llamafactory-cli train "$HISTDIR/configs/expSSS.yaml" \
    > "$HISTDIR/logs/expSSS_train.log" 2>&1 &
echo "[$(date)] expSSS started (PID=$!, GPU 4)"

# Monitor and eval
declare -A EXP_GPU EXP_N EXP_MODEL DONE EVAL_RUNNING
EXP_GPU=([expNNN]=0 [expOOO]=1 [expPPP]=2 [expQQQ]=3 [expSSS]=4)
EXP_N=([expNNN]=200 [expOOO]=200 [expPPP]=400 [expQQQ]=200 [expSSS]=200)
EXP_MODEL=([expNNN]="$MODEL_32B" [expOOO]="$MODEL_14B" [expPPP]="$MODEL_14B" [expQQQ]="$MODEL_7B" [expSSS]="$MODEL_14B")

echo "[$(date)] Monitoring Batch 10 training..."
while true; do
    all_done=true
    for exp in expNNN expOOO expPPP expQQQ expSSS; do
        if [ "${DONE[$exp]}" = "1" ]; then continue; fi
        log="$HISTDIR/logs/${exp}_train.log"
        result="$HISTDIR/results/${exp}.json"
        if grep -q "train_runtime" "$log" 2>/dev/null; then
            DONE[$exp]="1"
            if [ "${EVAL_RUNNING[$exp]}" = "1" ] || [ -f "$result" ]; then continue; fi
            EVAL_RUNNING[$exp]="1"
            echo "[$(date)] ✅ $exp: 训练完成，启动评测..."
            MAX_TOK=5000  # All B10 exps use cutoff_len=5120, need larger max_tokens
            bash eval_and_report.sh "$exp" "${EXP_MODEL[$exp]}" "${EXP_GPU[$exp]}" "${EXP_N[$exp]}" "$MAX_TOK" \
                >> "$HISTDIR/logs/${exp}_eval_full.log" 2>&1 &
        else
            all_done=false
            step=$(grep -oE '\s+[0-9]+/[0-9]+ \[' "$log" 2>/dev/null | tail -1 | tr -d '[] ')
            echo "[$(date)] ⏳ $exp: $step"
        fi
    done
    for exp in expNNN expOOO expPPP expQQQ expSSS; do
        if [ "${DONE[$exp]}" = "1" ] && [ "${EVAL_RUNNING[$exp]}" = "1" ]; then
            result="$HISTDIR/results/${exp}.json"
            if [ -f "$result" ]; then
                f1=$(python3 -c "import json; d=json.load(open('$result')); print(d['summary']['rule_detection_f1'])" 2>/dev/null)
                echo "[$(date)] 🎯 $exp: F1=$f1"
                EVAL_RUNNING[$exp]="done"
            fi
        fi
    done
    if [ "$all_done" = "true" ]; then
        pending=0
        for exp in expNNN expOOO expPPP expQQQ expSSS; do [ "${EVAL_RUNNING[$exp]}" = "1" ] && pending=$((pending + 1)); done
        [ $pending -eq 0 ] && { echo "[$(date)] ✅ Batch 10 完成!"; break; }
    fi
    sleep 60
done

echo ""
echo "=== Batch 10 Results ==="
for exp in expNNN expOOO expPPP expQQQ expSSS; do
    result="$HISTDIR/results/${exp}.json"
    if [ -f "$result" ]; then
        python3 -c "
import json
d = json.load(open('$result'))
s = d['summary']
print(f'  $exp: F1={s[\"rule_detection_f1\"]} cpk={s[\"cpk_found_rate\"]}')
print(f'    per_rule: {s[\"per_rule_recall\"]}')
" 2>/dev/null
    fi
done
echo "[$(date)] Batch 10 launcher complete."
