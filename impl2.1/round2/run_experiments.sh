#!/bin/bash
# route2.1.1 实验编排脚本
# 执行方式：cd /home/yinrong/impl/2.1.1 && bash run_experiments.sh [BATCH]
# BATCH=1: 数据生成+训练（Batch1，并行4个GPU）
# BATCH=eval: 合并+部署+评测（所有已完成checkpoints）
# BATCH=2: 基于Batch1结果，训练Batch2（Claude教师数据+混合）
# BATCH=grpo: GRPO强化训练

set -e
cd "$(dirname "$0")"

BASEDIR="/home/yinrong/impl/2.1.1"
HISTDIR="$BASEDIR/history-route2.1.1"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
MODEL_8B="/home/yinrong/models/Qwen3-8B"
MODEL_32B="/home/yinrong/models/Qwen3-32B"
TEXTBOOK="$BASEDIR/history-2026-04-23-1213-round3/data/train_textbook.jsonl"

mkdir -p "$HISTDIR"/{data,configs,checkpoints,logs,results,reports}

BATCH=${1:-1}

# ── 函数：注册数据集 ──────────────────────────────────────────────────────────
register() {
    local name=$1
    local file=$2
    python "$BASEDIR/tools/train/register_dataset.py" --name "$name" --file "$file" 2>/dev/null || true
}

# ── 函数：生成训练 YAML ───────────────────────────────────────────────────────
make_config() {
    local name=$1
    local model=$2
    local dataset=$3
    local epochs=$4
    local outdir=$5
    cat > "$HISTDIR/configs/${name}.yaml" << EOF
model_name_or_path: $model
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: $dataset
cutoff_len: 4096
num_train_epochs: $epochs
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
flash_attn: sdpa
logging_steps: 5
save_strategy: epoch
save_total_limit: 2
output_dir: $outdir
do_train: true
report_to: none
EOF
    echo "✓ 配置已生成：$HISTDIR/configs/${name}.yaml"
}

# ── 函数：启动训练 ────────────────────────────────────────────────────────────
train() {
    local name=$1
    local gpu=$2
    local config="$HISTDIR/configs/${name}.yaml"
    local log="$HISTDIR/logs/${name}_train.log"
    echo "🚀 启动训练 $name (GPU $gpu)..."
    CUDA_VISIBLE_DEVICES=$gpu nohup llamafactory-cli train "$config" > "$log" 2>&1 &
    echo "  PID=$!  LOG=$log"
}

# ── 函数：合并 LoRA ───────────────────────────────────────────────────────────
merge() {
    local name=$1
    local base=$2
    local adapter="$HISTDIR/checkpoints/$name"
    local output="$HISTDIR/checkpoints/${name}-merged"
    local log="$HISTDIR/logs/${name}_merge.log"
    if [ -d "$output" ] && [ "$(ls -A $output)" ]; then
        echo "  跳过 $name（已合并）"
        return
    fi
    echo "🔗 合并 $name..."
    python "$BASEDIR/tools/train/merge_adapter.py" \
        --base "$base" --adapter "$adapter" --output "$output" > "$log" 2>&1
    echo "  完成 → $output"
}

# ── 函数：评测 ────────────────────────────────────────────────────────────────
evaluate() {
    local name=$1
    local gpu=$2
    local n_train=$3
    local merged="$HISTDIR/checkpoints/${name}-merged"
    local port=$((8100 + gpu))
    local log_vllm="$HISTDIR/logs/${name}_vllm.log"
    local log_eval="$HISTDIR/logs/${name}_eval.log"
    local result="$HISTDIR/results/${name}.json"

    if [ -f "$result" ]; then
        echo "  跳过评测 $name（结果已存在）"
        return
    fi

    echo "📊 启动vLLM+评测 $name (GPU $gpu, port $port)..."
    CUDA_VISIBLE_DEVICES=$gpu python "$BASEDIR/tools/train/deploy_vllm.py" \
        --model "$merged" --port "$port" > "$log_vllm" 2>&1 &
    local vllm_pid=$!

    # 等待 vLLM 启动
    echo "  等待 vLLM 启动..."
    for i in $(seq 1 60); do
        sleep 5
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  vLLM 已就绪 (${i}×5s)"
            break
        fi
    done

    python "$BASEDIR/tools/eval/spc_eval.py" \
        --model_url "http://localhost:$port" \
        --model_name "$name" \
        --test "$BASEDIR/data/demo/test.jsonl" \
        --output "$result" \
        --n_train_samples "$n_train" \
        --concurrency 1 > "$log_eval" 2>&1

    kill $vllm_pid 2>/dev/null || true
    echo "  评测完成 → $result"
}

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH 1: 数据生成 + 并行训练
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$BATCH" = "1" ]; then
    echo "═══ BATCH 1: 生成数据并启动4路并行训练 ═══"
    echo "实验矩阵："
    echo "  Exp-A: 14B, N2, 默认权重, 6ep (基线复现)"
    echo "  Exp-B: 14B, N2, 均衡权重, 6ep (消除rule1主导)"
    echo "  Exp-C: 14B, N2, double_rule_prob=0.30, 6ep (更多多规则样本)"
    echo "  Exp-D: 14B, N2, no_skill_ratio=0.40, 5ep (600次曝光)"
    echo "  Exp-E: 8B,  N2, 均衡权重, 6ep (模型对比)"
    echo ""

    # ── 生成实验数据 ──────────────────────────────────────────────────────────
    echo "─── 生成实验数据 ───"

    # Exp-A: baseline（与round3相同条件）
    if [ ! -f "$HISTDIR/data/train_expA.jsonl" ]; then
        echo "生成 Exp-A 数据..."
        python "$BASEDIR/tools/spc/generator.py" \
            --n 200 --output "$HISTDIR/data/train_expA.jsonl" \
            --seed 100 --mixed --no_skill_ratio 0.25 \
            --append "$TEXTBOOK"
        python "$BASEDIR/tools/spc/generator.py" --validate "$HISTDIR/data/train_expA.jsonl"
    fi

    # Exp-B: balanced rule weights
    if [ ! -f "$HISTDIR/data/train_expB.jsonl" ]; then
        echo "生成 Exp-B 数据（均衡权重）..."
        python "$BASEDIR/tools/spc/generator.py" \
            --n 200 --output "$HISTDIR/data/train_expB.jsonl" \
            --seed 101 --mixed --no_skill_ratio 0.25 \
            --balanced_weights \
            --append "$TEXTBOOK"
        python "$BASEDIR/tools/spc/generator.py" --validate "$HISTDIR/data/train_expB.jsonl"
    fi

    # Exp-C: higher double_rule_prob
    if [ ! -f "$HISTDIR/data/train_expC.jsonl" ]; then
        echo "生成 Exp-C 数据（double_rule_prob=0.30）..."
        python "$BASEDIR/tools/spc/generator.py" \
            --n 200 --output "$HISTDIR/data/train_expC.jsonl" \
            --seed 102 --mixed --no_skill_ratio 0.25 \
            --double_rule_prob 0.30 \
            --append "$TEXTBOOK"
        python "$BASEDIR/tools/spc/generator.py" --validate "$HISTDIR/data/train_expC.jsonl"
    fi

    # Exp-D: higher no_skill_ratio=0.40
    if [ ! -f "$HISTDIR/data/train_expD.jsonl" ]; then
        echo "生成 Exp-D 数据（no_skill_ratio=0.40）..."
        python "$BASEDIR/tools/spc/generator.py" \
            --n 200 --output "$HISTDIR/data/train_expD.jsonl" \
            --seed 103 --mixed --no_skill_ratio 0.40 \
            --append "$TEXTBOOK"
        python "$BASEDIR/tools/spc/generator.py" --validate "$HISTDIR/data/train_expD.jsonl"
    fi

    # Exp-E: 8B model, balanced weights（与Exp-B相同数据）
    echo "Exp-E 使用 Exp-B 数据（均衡权重）"

    # ── 注册数据集 ────────────────────────────────────────────────────────────
    echo "─── 注册数据集 ───"
    register "spc_r4_expA" "$HISTDIR/data/train_expA.jsonl"
    register "spc_r4_expB" "$HISTDIR/data/train_expB.jsonl"
    register "spc_r4_expC" "$HISTDIR/data/train_expC.jsonl"
    register "spc_r4_expD" "$HISTDIR/data/train_expD.jsonl"

    # ── 创建训练配置 ──────────────────────────────────────────────────────────
    echo "─── 创建训练配置 ───"
    make_config "expA" "$MODEL_14B" "spc_r4_expA" 6 "$HISTDIR/checkpoints/expA"
    make_config "expB" "$MODEL_14B" "spc_r4_expB" 6 "$HISTDIR/checkpoints/expB"
    make_config "expC" "$MODEL_14B" "spc_r4_expC" 6 "$HISTDIR/checkpoints/expC"
    make_config "expD" "$MODEL_14B" "spc_r4_expD" 5 "$HISTDIR/checkpoints/expD"
    make_config "expE" "$MODEL_8B"  "spc_r4_expB" 6 "$HISTDIR/checkpoints/expE"

    # ── 启动并行训练 ─────────────────────────────────────────────────────────
    echo "─── 启动并行训练（5个实验，GPU 0-4）───"
    train "expA" 0
    train "expB" 1
    train "expC" 2
    train "expD" 3
    train "expE" 4

    echo ""
    echo "✅ Batch 1 训练已全部启动。等待完成（约40~50分钟）。"
    echo "监控：tail -f $HISTDIR/logs/expA_train.log"
    echo "下一步：bash run_experiments.sh eval"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# EVAL: 合并所有已完成checkpoints并评测
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$BATCH" = "eval" ]; then
    echo "═══ EVAL: 合并 + 评测 ═══"

    for exp in expA expB expC expD; do
        ckpt="$HISTDIR/checkpoints/$exp"
        if [ -d "$ckpt" ] && [ "$(ls -A $ckpt)" ]; then
            merge "$exp" "$MODEL_14B"
        fi
    done

    if [ -d "$HISTDIR/checkpoints/expE" ] && [ "$(ls -A $HISTDIR/checkpoints/expE)" ]; then
        merge "expE" "$MODEL_8B"
    fi

    # 串行评测（每个用不同GPU，串行避免显存竞争）
    for i in 0 1 2 3 4; do
        exps=("expA" "expB" "expC" "expD" "expE")
        ntrain=(251 251 251 251 251)
        exp="${exps[$i]}"
        nt="${ntrain[$i]}"
        merged="$HISTDIR/checkpoints/${exp}-merged"
        if [ -d "$merged" ]; then
            evaluate "$exp" $i "$nt"
        fi
    done

    echo ""
    echo "✅ 评测完成。汇总结果："
    for exp in expA expB expC expD expE; do
        result="$HISTDIR/results/${exp}.json"
        if [ -f "$result" ]; then
            f1=$(python3 -c "import json; d=json.load(open('$result')); print(d['summary']['rule_detection_f1'])" 2>/dev/null || echo "N/A")
            echo "  $exp: rule_f1=$f1"
        fi
    done
fi

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH 2: Claude 教师数据 + 最佳方案混合训练
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$BATCH" = "2" ]; then
    echo "═══ BATCH 2: Claude 教师数据 + 混合训练 ═══"

    # 找最佳 Batch1 实验
    BEST_EXP=""
    BEST_F1=0
    for exp in expA expB expC expD; do
        result="$HISTDIR/results/${exp}.json"
        if [ -f "$result" ]; then
            f1=$(python3 -c "import json; d=json.load(open('$result')); print(d['summary']['rule_detection_f1'])" 2>/dev/null || echo "0")
            if python3 -c "exit(0 if float('$f1') > float('$BEST_F1') else 1)" 2>/dev/null; then
                BEST_F1=$f1
                BEST_EXP=$exp
            fi
        fi
    done

    echo "Batch1 最佳实验：$BEST_EXP（F1=$BEST_F1）"

    # 生成 Claude 教师数据
    TEACHER_DATA="$HISTDIR/data/train_claude_teacher.jsonl"
    if [ ! -f "$TEACHER_DATA" ]; then
        echo "生成 Claude 教师数据（500条）..."
        python "$BASEDIR/tools/data/gen_claude_teacher.py" \
            --n 500 --output "$TEACHER_DATA" --seed 200 \
            --verify --min_f1 0.8 --concurrency 4
    else
        n=$(wc -l < "$TEACHER_DATA")
        echo "Claude 教师数据已存在（$n 条）"
    fi

    # 确定最佳数据文件
    BEST_DATA="$HISTDIR/data/train_${BEST_EXP}.jsonl"
    if [ -z "$BEST_EXP" ]; then
        BEST_DATA="$HISTDIR/data/train_expA.jsonl"
        echo "⚠ 未找到Batch1结果，使用 Exp-A 数据"
    fi

    # Exp-F: Claude教师 500条 + 模板200条混合（1:0.4比例）
    if [ ! -f "$HISTDIR/data/train_expF.jsonl" ]; then
        echo "生成 Exp-F 混合数据..."
        python "$BASEDIR/tools/data/mix_dataset.py" \
            --inputs "$TEACHER_DATA:1.0" "$BEST_DATA:0.4" \
            --output "$HISTDIR/data/train_expF.jsonl" --seed 300
        python "$BASEDIR/tools/spc/generator.py" --validate "$HISTDIR/data/train_expF.jsonl"
    fi

    # Exp-G: Claude教师 500条 + 模板500条（均等）
    if [ ! -f "$HISTDIR/data/train_expG.jsonl" ]; then
        echo "生成 Exp-G 混合数据（1:1）..."
        python "$BASEDIR/tools/data/mix_dataset.py" \
            --inputs "$TEACHER_DATA:1.0" "$BEST_DATA:1.0" \
            --output "$HISTDIR/data/train_expG.jsonl" --seed 301
    fi

    # Exp-H: 纯 Claude 教师数据（500条）
    echo "Exp-H 使用纯 Claude 教师数据"

    register "spc_r4_expF" "$HISTDIR/data/train_expF.jsonl"
    register "spc_r4_expG" "$HISTDIR/data/train_expG.jsonl"
    register "spc_r4_expH" "$TEACHER_DATA"

    # 计算epochs（目标~600 no_skill exposures）
    # Claude数据全是with_skill（500条），50%混合中no_skill=~100，6ep=600
    make_config "expF" "$MODEL_14B" "spc_r4_expF" 6 "$HISTDIR/checkpoints/expF"
    make_config "expG" "$MODEL_14B" "spc_r4_expG" 4 "$HISTDIR/checkpoints/expG"
    make_config "expH" "$MODEL_14B" "spc_r4_expH" 8 "$HISTDIR/checkpoints/expH"

    train "expF" 0
    train "expG" 1
    train "expH" 2

    echo "✅ Batch 2 训练已启动。"
    echo "下一步：bash run_experiments.sh eval"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# GRPO: 在最佳SFT模型上做强化训练
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$BATCH" = "grpo" ]; then
    echo "═══ GRPO: 强化学习训练 ═══"
    echo "⚠ GRPO 需要先确认最佳 SFT 模型路径并手动配置。"
    echo "参考配置模板：$HISTDIR/configs/grpo_template.yaml"

    cat > "$HISTDIR/configs/grpo_template.yaml" << 'EOF'
# LLaMA-Factory GRPO 配置模板
# 使用前修改 model_name_or_path 为最佳 SFT 合并后模型路径

model_name_or_path: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/BEST_EXP-merged
template: qwen3
trust_remote_code: true

finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.0
lora_target: all

# GRPO 参数
stage: grpo
reward_model: /home/yinrong/impl/2.1.1/tools/train/grpo_reward.py
grpo_num_generations: 8   # 每个 prompt 采样 8 个输出

dataset: spc_grpo_prompts  # 只含 input，无 output（由模型生成）
cutoff_len: 2048
num_train_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
lr_scheduler_type: cosine
warmup_ratio: 0.05

bf16: true
flash_attn: sdpa
logging_steps: 5
save_strategy: epoch
save_total_limit: 2
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/grpo-v1
do_train: true
report_to: none
EOF
    echo "✓ 模板已写入 $HISTDIR/configs/grpo_template.yaml"
    echo "步骤："
    echo "  1. 修改 model_name_or_path 为最佳 SFT 合并后模型"
    echo "  2. 准备 grpo prompts 数据集（只含 input，注册为 spc_grpo_prompts）"
    echo "  3. CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train \$HISTDIR/configs/grpo_template.yaml"
fi

echo "完成。"
