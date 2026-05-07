#!/usr/bin/env bash
# run_experiments.sh — Round3 实验运行器
# 必须从 impl2.1/ 目录运行：cd /home/yinrong/post-train/impl2.1 && bash round3/run_experiments.sh <target>
#
# 用法：
#   bash round3/run_experiments.sh p0      # R3-A + R3-B（P0 弱规则 + 扩数据）
#   bash round3/run_experiments.sh ab      # R3-AB（A+B 组合）
#   bash round3/run_experiments.sh a       # 仅 R3-A
#   bash round3/run_experiments.sh b       # 仅 R3-B
#   bash round3/run_experiments.sh grpo    # R3-C GRPO
#   bash round3/run_experiments.sh qlora   # R3-D1 + R3-D2 QLoRA
#   bash round3/run_experiments.sh 72b     # R3-E1 8卡 ZeRO-3 72B

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R3="$ROOT/round3"
COMMON="$ROOT/common"
TEST_DATA="$COMMON/data/test.jsonl"
TARGET="${1:-p0}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 通用：训练 + 合并 + 评测 ─────────────────────────────────────────────────

run_one_experiment() {
  local EXP="$1"           # 实验名称，如 R3-A
  local CONFIG="$2"        # 配置文件路径
  local PORT="$3"          # vLLM 推理端口
  local BASE_MODEL="$4"    # 基座模型路径（用于合并）

  local CKPT="$R3/checkpoints/$EXP"
  local MERGED="$R3/checkpoints/${EXP}-merged"
  local RESULT="$R3/results/${EXP}.json"

  log "===== $EXP ====="

  # 1. 训练
  if [[ ! -d "$CKPT" ]]; then
    log "[$EXP] 开始训练..."
    DISABLE_VERSION_CHECK=1 llamafactory-cli train "$CONFIG" 2>&1 | tee "$R3/logs/train_${EXP}.log"
    log "[$EXP] 训练完成"
  else
    log "[$EXP] 已有 checkpoint，跳过训练"
  fi

  # 2. 合并 adapter
  if [[ ! -d "$MERGED" ]]; then
    log "[$EXP] 合并 adapter..."
    python "$COMMON/tools/train/merge_adapter.py" \
      --base "$BASE_MODEL" \
      --adapter "$CKPT" \
      --output "$MERGED" \
      --template qwen3 \
      2>&1 | tee "$R3/logs/merge_${EXP}.log"
    log "[$EXP] 合并完成：$MERGED"
  else
    log "[$EXP] 已有 merged 模型，跳过合并"
  fi

  # 3. 启动 vLLM
  log "[$EXP] 启动 vLLM（端口 $PORT）..."
  python "$COMMON/tools/train/deploy_vllm.py" \
    --model "$MERGED" --port "$PORT" \
    2>&1 | tee "$R3/logs/vllm_${EXP}.log" &
  VLLM_PID=$!
  # 等待 vLLM 真正就绪（最多300秒）
  log "[$EXP] 等待 vLLM 就绪..."
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      log "[$EXP] vLLM 就绪（${i}×5s）"
      break
    fi
    sleep 5
  done

  # 4. 评测
  log "[$EXP] 开始评测..."
  python "$COMMON/tools/eval/spc_eval.py" \
    --model_url "http://localhost:$PORT" \
    --model_name "$EXP" \
    --test "$TEST_DATA" \
    --output "$RESULT" \
    --max_tokens 4096 \
    --concurrency 4 \
    2>&1 | tee "$R3/logs/eval_${EXP}.log"
  log "[$EXP] 评测完成：$RESULT"

  # 5. 停止 vLLM
  kill $VLLM_PID 2>/dev/null || true
  python "$COMMON/tools/train/deploy_vllm.py" --kill --port "$PORT" 2>/dev/null || true

  # 6. 打印结果摘要
  if [[ -f "$RESULT" ]]; then
    python3 -c "
import json
d = json.load(open('$RESULT'))
s = d['summary']
print(f'  rule_f1={s[\"rule_detection_f1\"]:.3f}')
print(f'  per_rule:', {k: round(v,3) for k,v in s[\"per_rule_recall\"].items()})
cpk = s['cpk_mae']
cpk_str = f'{cpk:.3f}' if cpk is not None else 'N/A'
print(f'  cpk_mae={cpk_str}, inference={s[\"inference_time_ms_mean\"]/1000:.1f}s')
"
  fi
}

# ── 目标分发 ─────────────────────────────────────────────────────────────────

MODEL_14B="/home/yinrong/models/Qwen3-14B"
MODEL_32B="/home/yinrong/models/Qwen3-32B"
MODEL_72B="/home/yinrong/models/Qwen3-72B"

case "$TARGET" in

  a)
    run_one_experiment "R3-A" "$R3/configs/R3-A.yaml" 8031 "$MODEL_14B"
    ;;

  b)
    run_one_experiment "R3-B" "$R3/configs/R3-B.yaml" 8032 "$MODEL_14B"
    ;;

  p0)
    # 串行运行 R3-A 和 R3-B（共用 GPU，节省显存）
    run_one_experiment "R3-A" "$R3/configs/R3-A.yaml" 8031 "$MODEL_14B"
    run_one_experiment "R3-B" "$R3/configs/R3-B.yaml" 8032 "$MODEL_14B"
    ;;

  ab)
    run_one_experiment "R3-AB" "$R3/configs/R3-AB.yaml" 8033 "$MODEL_14B"
    ;;

  grpo)
    # R3-C：GRPO 接续训练（基于 expYYY-merged）
    EXPYYY_MERGED="$ROOT/round2/history-route2.1.1/checkpoints/expYYY-merged"
    if [[ ! -d "$EXPYYY_MERGED" ]]; then
      echo "错误：expYYY-merged 不存在：$EXPYYY_MERGED"
      exit 1
    fi

    CKPT="$R3/checkpoints/R3-C"
    MERGED="$R3/checkpoints/R3-C-merged"
    RESULT="$R3/results/R3-C.json"

    if [[ ! -d "$CKPT" ]]; then
      log "[R3-C] 开始 GRPO 训练..."
      DISABLE_VERSION_CHECK=1 llamafactory-cli train "$R3/configs/R3-C-grpo.yaml" \
        2>&1 | tee "$R3/logs/train_R3-C.log"
    fi

    if [[ ! -d "$MERGED" ]]; then
      log "[R3-C] 合并 GRPO adapter..."
      python "$COMMON/tools/train/merge_adapter.py" \
        --base "$EXPYYY_MERGED" \
        --adapter "$CKPT" \
        --output "$MERGED" \
        --template qwen3 \
        2>&1 | tee "$R3/logs/merge_R3-C.log"
    fi

    log "[R3-C] 评测..."
    python "$COMMON/tools/train/deploy_vllm.py" --model "$MERGED" --port 8034 &
    VLLM_PID=$!
    sleep 30
    python "$COMMON/tools/eval/spc_eval.py" \
      --model_url "http://localhost:8034" \
      --model_name "R3-C" \
      --test "$TEST_DATA" \
      --output "$RESULT" \
      2>&1 | tee "$R3/logs/eval_R3-C.log"
    kill $VLLM_PID 2>/dev/null || true
    python "$COMMON/tools/train/deploy_vllm.py" --kill --port 8034 2>/dev/null || true
    ;;

  qlora)
    # R3-D1：QLoRA int4 14B
    run_one_experiment "R3-D1-qlora-14b" "$R3/configs/R3-D1-qlora-14b.yaml" 8035 "$MODEL_14B"
    # R3-D2：QLoRA int4 32B
    run_one_experiment "R3-D2-qlora-32b" "$R3/configs/R3-D2-qlora-32b.yaml" 8036 "$MODEL_32B"
    ;;

  72b)
    # R3-E1：8卡 ZeRO-3 72B
    if [[ ! -d "$MODEL_72B" ]]; then
      echo "错误：Qwen3-72B 未下载，请先运行："
      echo "  modelscope download --model Qwen/Qwen3-72B --local_dir $MODEL_72B"
      exit 1
    fi
    log "[R3-E1] 8卡 ZeRO-3 72B 训练..."
    FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 llamafactory-cli train "$R3/configs/R3-E1-zero3-72b.yaml" \
      --deepspeed "$R3/configs/deepspeed_zero3.json" \
      2>&1 | tee "$R3/logs/train_R3-E1.log"
    ;;

  *)
    echo "未知目标：$TARGET"
    echo "用法：bash round3/run_experiments.sh <p0|a|b|ab|grpo|qlora|72b>"
    exit 1
    ;;
esac

log "======================================================"
log "✓ 目标 '$TARGET' 完成"
log "结果目录：$R3/results/"
log "======================================================"
