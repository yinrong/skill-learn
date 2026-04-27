#!/bin/bash
# 监控训练进度，训练完成后自动触发评测
# 用法：bash monitor_and_eval.sh
# 在后台运行：nohup bash monitor_and_eval.sh > history-route2.1.1/logs/monitor.log 2>&1 &

cd "$(dirname "$0")"
HISTDIR="history-route2.1.1"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
MODEL_8B="/home/yinrong/models/Qwen3-8B"

# 实验配置：name -> (base_model, gpu_for_eval, n_train)
declare -A EXP_MODEL
declare -A EXP_GPU
declare -A EXP_N
EXP_MODEL=([expA]=$MODEL_14B [expB]=$MODEL_14B [expC]=$MODEL_14B [expD]=$MODEL_14B [expE]=$MODEL_8B [expF2]=$MODEL_14B [expG2]=$MODEL_14B [expX]=$MODEL_14B [expY]=$MODEL_14B [expZ]=$MODEL_14B [expW]=$MODEL_14B [expU]=$MODEL_14B [expT]=$MODEL_14B [expV]=$MODEL_14B [expS]=$MODEL_14B [expAA]=$MODEL_14B [expBB]=$MODEL_14B [expCC]=$MODEL_14B [expDD]=$MODEL_14B [expFF]=$MODEL_14B [expGG]=$MODEL_14B [expMM]=$MODEL_14B [expNN]=$MODEL_14B [expOO]=$MODEL_14B [expPP]=$MODEL_14B [expQQ]=$MODEL_14B [expRR]=$MODEL_14B [expSS]=$MODEL_14B [expTT]=$MODEL_14B [expUU]=$MODEL_14B)
EXP_GPU=([expA]=0 [expB]=1 [expC]=2 [expD]=3 [expE]=4 [expF2]=6 [expG2]=7 [expX]=5 [expY]=4 [expZ]=0 [expW]=1 [expU]=2 [expT]=6 [expV]=3 [expS]=7 [expAA]=0 [expBB]=1 [expCC]=2 [expDD]=4 [expFF]=6 [expGG]=7 [expMM]=2 [expNN]=3 [expOO]=5 [expPP]=6 [expQQ]=5 [expRR]=0 [expSS]=1 [expTT]=4 [expUU]=7)
EXP_N=([expA]=251 [expB]=251 [expC]=251 [expD]=251 [expE]=251 [expF2]=151 [expG2]=451 [expX]=751 [expY]=251 [expZ]=251 [expW]=251 [expU]=251 [expT]=451 [expV]=451 [expS]=251 [expAA]=234 [expBB]=285 [expCC]=251 [expDD]=251 [expFF]=351 [expGG]=251 [expMM]=234 [expNN]=251 [expOO]=50 [expPP]=251 [expQQ]=301 [expRR]=200 [expSS]=300 [expTT]=500 [expUU]=300)

declare -A DONE
declare -A EVAL_RUNNING

echo "[$(date)] 开始监控训练进度..."

while true; do
    all_done=true

    for exp in expA expB expC expD expE expF2 expG2 expX expY expZ expW expU expT expV expS expAA expBB expCC expDD expFF expGG expMM expNN expOO expPP expQQ expRR expSS expTT expUU; do
        if [ "${DONE[$exp]}" = "1" ]; then
            continue
        fi

        log="$HISTDIR/logs/${exp}_train.log"
        result="$HISTDIR/results/${exp}.json"

        # Check if training complete (only match train_runtime which appears at actual completion)
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

            BASE="${EXP_MODEL[$exp]}"
            GPU="${EXP_GPU[$exp]}"
            N="${EXP_N[$exp]}"

            bash eval_and_report.sh "$exp" "$BASE" "$GPU" "$N" \
                >> "$HISTDIR/logs/${exp}_eval_full.log" 2>&1 &
            echo "[$(date)] 📊 $exp: 评测已在 GPU $GPU 后台启动 (PID=$!)"

        else
            all_done=false
            # Show progress
            step=$(grep -oE '\s+[0-9]+/[0-9]+ \[' "$log" 2>/dev/null | tail -1 | tr -d '[] ' | xargs)
            echo "[$(date)] ⏳ $exp: step=$step"
        fi
    done

    # Check eval completion
    for exp in expA expB expC expD expE expF2 expG2 expX expY expZ expW expU expT expV expS expAA expBB expCC expDD expFF expGG expMM expNN expOO expPP expQQ expRR expSS expTT expUU; do
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
        # All training done, wait for evals to finish
        pending_evals=0
        for exp in expA expB expC expD expE expF2 expG2 expX expY expZ expW expU expT expV expS expAA expBB expCC expDD expFF expGG expMM expNN expOO expPP expQQ expRR expSS expTT expUU; do
            if [ "${EVAL_RUNNING[$exp]}" = "1" ]; then
                pending_evals=$((pending_evals + 1))
            fi
        done
        if [ $pending_evals -eq 0 ]; then
            echo "[$(date)] ✅ 所有训练和评测已完成！"

            # Print summary
            echo ""
            echo "=== Round 4 实验结果汇总 ==="
            for exp in expA expB expC expD expE expF2 expG2 expX expY expZ expW expU expT expV expS expAA expBB expCC expDD expFF expGG expMM expNN expOO expPP expQQ expRR expSS expTT expUU; do
                result="$HISTDIR/results/${exp}.json"
                if [ -f "$result" ]; then
                    f1=$(python3 -c "import json; d=json.load(open('$result')); print(d['summary']['rule_detection_f1'])" 2>/dev/null || echo "N/A")
                    r1=$(python3 -c "import json; d=json.load(open('$result')); print(d['summary']['per_rule_recall']['rule1'])" 2>/dev/null || echo "?")
                    r2=$(python3 -c "import json; d=json.load(open('$result')); print(d['summary']['per_rule_recall']['rule2'])" 2>/dev/null || echo "?")
                    echo "  $exp: F1=$f1  rule1=$r1  rule2=$r2"
                fi
            done
            break
        fi
    fi

    sleep 60  # check every minute
done

echo "[$(date)] 监控完成。"
