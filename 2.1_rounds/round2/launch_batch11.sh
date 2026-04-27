#!/bin/bash
# Batch 11: Refine best configuration from Batch 10
# Focus: step count sweet spot with cutoff_len=5120, data diversity, 32B final config
# Experiments:
#   expTTT: 14B + cutoff=5120 + 200 ns_v1 × 3ep = 75 steps (fewer steps test)
#   expUUU: 14B + cutoff=5120 + 200 ns_v1 × 8ep = 200 steps (overfit test)
#   expVVV: 14B + cutoff=5120 + 400 (v1+v4) × 3ep = 150 steps (new v4 seed pool)
#   expWWW: 32B + cutoff=5120 + 400 (v1+v3) × 3ep = 150 steps (best 32B + diversity)
# Dynamically adds/removes based on Batch 10 findings
# Run: nohup bash launch_batch11.sh > history-route2.1.1/logs/batch11_launch.log 2>&1 &
cd "$(dirname "$0")"
HISTDIR="history-route2.1.1"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
MODEL_32B="/home/yinrong/models/Qwen3-32B"

WAIT_EXPS="expNNN expOOO expPPP expQQQ expSSS"
V4_NS="$HISTDIR/data/train_claude_teacher_v4_noskill.jsonl"

echo "[$(date)] Batch 11 launcher started. Waiting for Batch 10 results + teacher v4..."

while true; do
    missing=0
    for exp in $WAIT_EXPS; do
        [ ! -f "$HISTDIR/results/${exp}.json" ] && missing=$((missing + 1))
    done
    v4_ready=false
    [ -f "$V4_NS" ] && [ "$(wc -l < $V4_NS)" -ge 200 ] && v4_ready=true

    if [ $missing -eq 0 ] && [ "$v4_ready" = "true" ]; then
        echo "[$(date)] All Batch 10 evals complete + teacher v4 ready!"
        break
    fi
    echo "[$(date)] Waiting... $missing B10 results missing, v4_ready=$v4_ready"
    for exp in $WAIT_EXPS; do
        if [ -f "$HISTDIR/results/${exp}.json" ]; then
            f1=$(python3 -c "import json; d=json.load(open('$HISTDIR/results/${exp}.json')); print(d['summary']['rule_detection_f1'])" 2>/dev/null || echo "?")
            echo "  ✓ $exp: F1=$f1"
        fi
    done
    sleep 120
done

# Print Batch 10 results
echo ""
echo "=== Batch 10 Results ==="
python3 -c "
import json, os
exps = ['expNNN','expOOO','expPPP','expQQQ','expSSS']
for exp in exps:
    path = '$HISTDIR/results/' + exp + '.json'
    if os.path.exists(path):
        d = json.load(open(path))
        s = d['summary']
        print(f'  {exp}: F1={s[\"rule_detection_f1\"]:.3f} cpk={s[\"cpk_found_rate\"]:.3f}')
        for r, v in s['per_rule_recall'].items():
            bar = '█' * int((v or 0) * 15)
            print(f'    {r}: {v:.3f}  {bar}')
" 2>/dev/null

# Analyze best config from all experiments
ALL_BEST=$(python3 -c "
import json, os, glob
results_dir = '$HISTDIR/results'
best = sorted([(json.load(open(p))['summary']['rule_detection_f1'], os.path.basename(p).replace('.json',''))
               for p in glob.glob(f'{results_dir}/exp*.json')
               if os.path.exists(p)], reverse=True)[:5]
for f1, exp in best:
    print(f'  {exp}: F1={f1:.3f}')
" 2>/dev/null)
echo "[$(date)] All-time top 5:"
echo "$ALL_BEST"

# Prepare Batch 11 training data
bash prepare_batch11.sh

# Kill all existing vLLM servers
echo "[$(date)] Killing vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 10

echo "[$(date)] Starting Batch 11 training..."
echo "  expTTT: 14B + cutoff=5120 + 200 ns_v1 × 3ep = 75 steps, GPU 0"
echo "  expUUU: 14B + cutoff=5120 + 200 ns_v1 × 8ep = 200 steps, GPU 1"
echo "  expVVV: 14B + cutoff=5120 + 400 (ns_v1+ns_v4) × 3ep = 150 steps, GPU 2"
echo "  expWWW: 32B + cutoff=5120 + 400 (ns_v1+ns_v3) × 3ep = 150 steps, GPU 3"

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 \
    llamafactory-cli train "$HISTDIR/configs/expTTT.yaml" \
    > "$HISTDIR/logs/expTTT_train.log" 2>&1 &
echo "[$(date)] expTTT started (PID=$!, GPU 0)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=1 \
    llamafactory-cli train "$HISTDIR/configs/expUUU.yaml" \
    > "$HISTDIR/logs/expUUU_train.log" 2>&1 &
echo "[$(date)] expUUU started (PID=$!, GPU 1)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=2 \
    llamafactory-cli train "$HISTDIR/configs/expVVV.yaml" \
    > "$HISTDIR/logs/expVVV_train.log" 2>&1 &
echo "[$(date)] expVVV started (PID=$!, GPU 2)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=3 \
    llamafactory-cli train "$HISTDIR/configs/expWWW.yaml" \
    > "$HISTDIR/logs/expWWW_train.log" 2>&1 &
echo "[$(date)] expWWW started (PID=$!, GPU 3)"

# Monitor and eval
declare -A EXP_GPU EXP_N EXP_MODEL DONE EVAL_RUNNING
EXP_GPU=([expTTT]=0 [expUUU]=1 [expVVV]=2 [expWWW]=3)
EXP_N=([expTTT]=200 [expUUU]=200 [expVVV]=400 [expWWW]=400)
EXP_MODEL=([expTTT]="$MODEL_14B" [expUUU]="$MODEL_14B" [expVVV]="$MODEL_14B" [expWWW]="$MODEL_32B")

echo "[$(date)] Monitoring Batch 11 training..."
while true; do
    all_done=true
    for exp in expTTT expUUU expVVV expWWW; do
        if [ "${DONE[$exp]}" = "1" ]; then continue; fi
        log="$HISTDIR/logs/${exp}_train.log"
        result="$HISTDIR/results/${exp}.json"
        if grep -q "train_runtime" "$log" 2>/dev/null; then
            DONE[$exp]="1"
            if [ "${EVAL_RUNNING[$exp]}" = "1" ] || [ -f "$result" ]; then continue; fi
            EVAL_RUNNING[$exp]="1"
            echo "[$(date)] ✅ $exp: 训练完成，启动评测..."
            bash eval_and_report.sh "$exp" "${EXP_MODEL[$exp]}" "${EXP_GPU[$exp]}" "${EXP_N[$exp]}" 5000 \
                >> "$HISTDIR/logs/${exp}_eval_full.log" 2>&1 &
        else
            all_done=false
            step=$(grep -oE '\s+[0-9]+/[0-9]+ \[' "$log" 2>/dev/null | tail -1 | tr -d '[] ')
            echo "[$(date)] ⏳ $exp: $step"
        fi
    done
    for exp in expTTT expUUU expVVV expWWW; do
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
        for exp in expTTT expUUU expVVV expWWW; do [ "${EVAL_RUNNING[$exp]}" = "1" ] && pending=$((pending + 1)); done
        [ $pending -eq 0 ] && { echo "[$(date)] ✅ Batch 11 完成!"; break; }
    fi
    sleep 60
done

echo ""
echo "=== Batch 11 Results ==="
for exp in expTTT expUUU expVVV expWWW; do
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
echo "[$(date)] Batch 11 launcher complete."
