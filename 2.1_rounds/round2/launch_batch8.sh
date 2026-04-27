#!/bin/bash
# Batch 8 launch: 32B model experiments after Batch 7
# Run: nohup bash launch_batch8.sh > history-route2.1.1/logs/batch8_launch.log 2>&1 &
cd "$(dirname "$0")"
HISTDIR="history-route2.1.1"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
MODEL_32B="/home/yinrong/models/Qwen3-32B"

# Wait for Batch 7 results AND Batch 6 results (to avoid killing B6 evals)
WAIT_EXPS="expZZ expAAA expBBB expCCC expWW expXX expVV expYY2"

echo "[$(date)] Batch 8 launcher started. Waiting for Batch 7+6 results..."

while true; do
    missing=0
    for exp in $WAIT_EXPS; do
        if [ ! -f "$HISTDIR/results/${exp}.json" ]; then
            missing=$((missing + 1))
        fi
    done
    if [ $missing -eq 0 ]; then
        echo "[$(date)] All Batch 7 evals complete!"
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

# Print Batch 7 results
echo ""
echo "=== Batch 7 Results ==="
for exp in expZZ expAAA expBBB expCCC; do
    result="$HISTDIR/results/${exp}.json"
    if [ -f "$result" ]; then
        f1=$(python3 -c "import json; d=json.load(open('$result')); s=d['summary']; print(f'F1={s[\"rule_detection_f1\"]} cpk={s[\"cpk_found_rate\"]}')" 2>/dev/null || echo "error")
        echo "  $exp: $f1"
    fi
done

# Update expHHH config to use best Batch 7 data if needed
BEST_B7=$(python3 -c "
import json, os
exps = ['expZZ','expAAA','expBBB','expCCC']
best_exp, best_f1 = None, 0
for exp in exps:
    path = '$HISTDIR/results/' + exp + '.json'
    if os.path.exists(path):
        f1 = json.load(open(path))['summary']['rule_detection_f1']
        if f1 > best_f1:
            best_f1 = f1
            best_exp = exp
print(best_exp or 'expCCC')
" 2>/dev/null)
echo "[$(date)] Best Batch 7 experiment: $BEST_B7"

# Update expHHH config to use best Batch 7 data
BEST_B7_DATASET="spc_r5_${BEST_B7}"
BEST_B7_N=$(python3 -c "
import json
di = json.load(open('data/dataset_info.json'))
ds = di.get('$BEST_B7_DATASET', {})
fname = ds.get('file_name','')
if fname:
    import subprocess
    n = int(subprocess.check_output(['wc','-l',fname]).decode().split()[0])
    print(n)
else:
    print(400)
" 2>/dev/null || echo 400)
echo "[$(date)] Updating expHHH to use $BEST_B7_DATASET (N=$BEST_B7_N)"
python3 -c "
import re
content = open('$HISTDIR/configs/expHHH.yaml').read()
content = re.sub(r'dataset: spc_r5_\w+', 'dataset: $BEST_B7_DATASET', content)
open('$HISTDIR/configs/expHHH.yaml', 'w').write(content)
print('expHHH config updated')
" 2>/dev/null

# Kill all existing vLLM servers
echo "[$(date)] Killing vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 10

# Batch 8: Run 32B experiments on 2 GPUs (others idle)
# expGGG: 200 ns_v1 × 5ep = 125 steps, 32B model → GPU 0
# expHHH: best Batch 7 data, 32B model → GPU 1

echo "[$(date)] Starting Batch 8 training (32B model)..."
echo "  expGGG: 200 ns_v1 × 5ep = 125 steps, Qwen3-32B (GPU 0)"
echo "  expHHH: best Batch 7 data ($BEST_B7_DATASET, N=$BEST_B7_N), Qwen3-32B (GPU 1)"

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 \
    llamafactory-cli train "$HISTDIR/configs/expGGG.yaml" \
    > "$HISTDIR/logs/expGGG_train.log" 2>&1 &
PID_GGG=$!
echo "[$(date)] expGGG training started (PID=$PID_GGG, GPU 0)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=1 \
    llamafactory-cli train "$HISTDIR/configs/expHHH.yaml" \
    > "$HISTDIR/logs/expHHH_train.log" 2>&1 &
PID_HHH=$!
echo "[$(date)] expHHH training started (PID=$PID_HHH, GPU 1)"

# Monitor and eval
declare -A EXP_GPU EXP_N DONE EVAL_RUNNING
EXP_GPU=([expGGG]=0 [expHHH]=1)
EXP_N=([expGGG]=200 [expHHH]=$BEST_B7_N)

echo "[$(date)] Monitoring Batch 8 training..."

while true; do
    all_done=true
    for exp in expGGG expHHH; do
        if [ "${DONE[$exp]}" = "1" ]; then continue; fi
        log="$HISTDIR/logs/${exp}_train.log"
        result="$HISTDIR/results/${exp}.json"
        if grep -q "train_runtime" "$log" 2>/dev/null; then
            DONE[$exp]="1"
            if [ "${EVAL_RUNNING[$exp]}" = "1" ]; then continue; fi
            if [ -f "$result" ]; then continue; fi
            EVAL_RUNNING[$exp]="1"
            echo "[$(date)] ✅ $exp: 训练完成，启动评测..."
            GPU="${EXP_GPU[$exp]}"
            N="${EXP_N[$exp]}"
            # Batch 8 (32B) uses ns data, cutoff_len=4096 (configs from Batch 7 best)
            # expGGG uses spc_r5_expZZ (cutoff=4096), expHHH uses best B7 config (cutoff=5120)
            # Use max_tokens=5000 for safety
            bash eval_and_report.sh "$exp" "$MODEL_32B" "$GPU" "$N" 5000 \
                >> "$HISTDIR/logs/${exp}_eval_full.log" 2>&1 &
            echo "[$(date)] 📊 $exp: 评测在 GPU $GPU 启动 (PID=$!)"
        else
            all_done=false
            step=$(grep -oE '\s+[0-9]+/[0-9]+ \[' "$log" 2>/dev/null | tail -1 | tr -d '[] ')
            echo "[$(date)] ⏳ $exp: $step"
        fi
    done
    for exp in expGGG expHHH; do
        if [ "${DONE[$exp]}" = "1" ] && [ "${EVAL_RUNNING[$exp]}" = "1" ]; then
            if [ -f "$HISTDIR/results/${exp}.json" ]; then
                f1=$(python3 -c "import json; d=json.load(open('$HISTDIR/results/${exp}.json')); print(d['summary']['rule_detection_f1'])" 2>/dev/null || echo "?")
                echo "[$(date)] 🎯 $exp: F1=$f1"
                EVAL_RUNNING[$exp]="done"
            fi
        fi
    done
    if [ "$all_done" = "true" ]; then
        pending=0
        for exp in expGGG expHHH; do
            [ "${EVAL_RUNNING[$exp]}" = "1" ] && pending=$((pending + 1))
        done
        [ $pending -eq 0 ] && { echo "[$(date)] ✅ Batch 8 完成!"; break; }
    fi
    sleep 60
done

echo ""
echo "=== Batch 8 Results ==="
for exp in expGGG expHHH; do
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
echo "[$(date)] Batch 8 launcher complete."
