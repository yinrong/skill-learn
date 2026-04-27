#!/bin/bash
# Batch 9: New seed data pool + LR variations + best 32B config refinement
# Prerequisites: teacher v3 complete + Batch 8 results
# Run: nohup bash launch_batch9.sh > history-route2.1.1/logs/batch9_launch.log 2>&1 &
cd "$(dirname "$0")"
HISTDIR="history-route2.1.1"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
MODEL_32B="/home/yinrong/models/Qwen3-32B"

# Wait for Batch 8 AND teacher v3
WAIT_EXPS="expGGG expHHH"
V3_NS="$HISTDIR/data/train_claude_teacher_v3_noskill.jsonl"

echo "[$(date)] Batch 9 launcher started. Waiting for Batch 8 results + teacher v3..."

while true; do
    missing=0
    for exp in $WAIT_EXPS; do
        [ ! -f "$HISTDIR/results/${exp}.json" ] && missing=$((missing + 1))
    done
    v3_ready=false
    [ -f "$V3_NS" ] && [ "$(wc -l < $V3_NS)" -ge 300 ] && v3_ready=true

    if [ $missing -eq 0 ] && [ "$v3_ready" = "true" ]; then
        echo "[$(date)] All Batch 8 evals complete + teacher v3 ready!"
        break
    fi
    echo "[$(date)] Waiting... $missing B8 results missing, v3_ready=$v3_ready"
    sleep 120
done

# Summarize best results so far
echo ""
echo "=== Best Results Summary ==="
python3 -c "
import json, os, glob

results_dir = '$HISTDIR/results'
results = {}
for path in glob.glob(f'{results_dir}/exp*.json'):
    exp = os.path.basename(path).replace('.json','')
    try:
        d = json.load(open(path))
        f1 = d['summary']['rule_detection_f1']
        results[exp] = f1
    except: pass

for exp, f1 in sorted(results.items(), key=lambda x: -x[1])[:10]:
    print(f'  {exp}: F1={f1:.3f}')
" 2>/dev/null

# Run prepare_batch9.sh
bash prepare_batch9.sh

# Kill all existing vLLM servers
echo "[$(date)] Killing vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 10

# Batch 9 experiments:
# expIII: 200 ns_v3 × 5ep = 125 steps (new seed pool) → GPU 0
# expJJJ: 400 (ns_v1+ns_v3) × 3ep = 150 steps (v1+v3 diversity) → GPU 1
# expKKK: 200 ns_v1 × 5ep = 125 steps, LR=5e-5 (lower LR) → GPU 2
# expLLL: 200 ns_v1 × 5ep = 125 steps, Qwen3-8B (model size comparison) → GPU 3
# expMMM: 200 ns_v1 × 5ep = 125 steps, cutoff_len=5120 (FIX: ~49% rule8 truncation in expRR) → GPU 4

echo "[$(date)] Starting Batch 9 training..."
echo "  expIII: 200 ns_v3 × 5ep = 125 steps (new seed pool), GPU 0"
echo "  expJJJ: 400 (v1+v3) × 3ep = 150 steps (cross-pool), GPU 1"
echo "  expKKK: 200 ns_v1 × 5ep, LR=5e-5 (lower LR), GPU 2"
echo "  expLLL: 200 ns_v1 × 5ep, Qwen3-8B (model size test), GPU 3"
echo "  expMMM: 200 ns_v1 × 5ep, cutoff_len=5120 (fix rule8 truncation), GPU 4"

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 \
    llamafactory-cli train "$HISTDIR/configs/expIII.yaml" \
    > "$HISTDIR/logs/expIII_train.log" 2>&1 &
echo "[$(date)] expIII started (PID=$!, GPU 0)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=1 \
    llamafactory-cli train "$HISTDIR/configs/expJJJ.yaml" \
    > "$HISTDIR/logs/expJJJ_train.log" 2>&1 &
echo "[$(date)] expJJJ started (PID=$!, GPU 1)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=3 \
    llamafactory-cli train "$HISTDIR/configs/expLLL.yaml" \
    > "$HISTDIR/logs/expLLL_train.log" 2>&1 &
echo "[$(date)] expLLL started (PID=$!, GPU 3)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=2 \
    llamafactory-cli train "$HISTDIR/configs/expKKK.yaml" \
    > "$HISTDIR/logs/expKKK_train.log" 2>&1 &
echo "[$(date)] expKKK started (PID=$!, GPU 2)"
sleep 5

DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=4 \
    llamafactory-cli train "$HISTDIR/configs/expMMM.yaml" \
    > "$HISTDIR/logs/expMMM_train.log" 2>&1 &
echo "[$(date)] expMMM started (PID=$!, GPU 4)"

# Monitor and eval
declare -A EXP_GPU EXP_N EXP_MODEL DONE EVAL_RUNNING
EXP_GPU=([expIII]=0 [expJJJ]=1 [expKKK]=2 [expLLL]=3 [expMMM]=4)
EXP_N=([expIII]=200 [expJJJ]=400 [expKKK]=200 [expLLL]=200 [expMMM]=200)
EXP_MODEL=([expIII]="$MODEL_14B" [expJJJ]="$MODEL_14B" [expKKK]="$MODEL_14B" [expLLL]="/home/yinrong/models/Qwen3-8B" [expMMM]="$MODEL_14B")

echo "[$(date)] Monitoring Batch 9 training..."
while true; do
    all_done=true
    for exp in expIII expJJJ expKKK expLLL expMMM; do
        if [ "${DONE[$exp]}" = "1" ]; then continue; fi
        log="$HISTDIR/logs/${exp}_train.log"
        result="$HISTDIR/results/${exp}.json"
        if grep -q "train_runtime" "$log" 2>/dev/null; then
            DONE[$exp]="1"
            if [ "${EVAL_RUNNING[$exp]}" = "1" ] || [ -f "$result" ]; then continue; fi
            EVAL_RUNNING[$exp]="1"
            echo "[$(date)] ✅ $exp: 训练完成，启动评测..."
            # All Batch 9 experiments now use cutoff_len=5120, max_tokens=5000
            MAX_TOK=5000
            bash eval_and_report.sh "$exp" "${EXP_MODEL[$exp]}" "${EXP_GPU[$exp]}" "${EXP_N[$exp]}" "$MAX_TOK" \
                >> "$HISTDIR/logs/${exp}_eval_full.log" 2>&1 &
        else
            all_done=false
            step=$(grep -oE '\s+[0-9]+/[0-9]+ \[' "$log" 2>/dev/null | tail -1 | tr -d '[] ')
            echo "[$(date)] ⏳ $exp: $step"
        fi
    done
    for exp in expIII expJJJ expKKK expLLL expMMM; do
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
        for exp in expIII expJJJ expKKK expLLL expMMM; do [ "${EVAL_RUNNING[$exp]}" = "1" ] && pending=$((pending + 1)); done
        [ $pending -eq 0 ] && { echo "[$(date)] ✅ Batch 9 完成!"; break; }
    fi
    sleep 60
done

echo ""
echo "=== Batch 9 Results ==="
for exp in expIII expJJJ expKKK expLLL expMMM; do
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
echo "[$(date)] Batch 9 launcher complete."
