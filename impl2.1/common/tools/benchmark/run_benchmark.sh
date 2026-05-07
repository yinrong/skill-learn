#!/bin/bash
# 通用能力 benchmark 评测脚本
# 用法：bash run_benchmark.sh [--dry-run]
# 输出：impl2.1/common/benchmark_results/<model_tag>/
#
# 评测任务（Open LLM Leaderboard 经典集）：
#   mmlu, gsm8k, arc_challenge, hellaswag, winogrande,
#   truthfulqa_mc1, cmmlu（中文）
#
# GPU 分配：每个模型独占一张 H20 (96GB)，6 个模型并行跑完

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../../benchmark_results"
mkdir -p "$RESULTS_DIR"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# HuggingFace 镜像（无外网时使用）
export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_OFFLINE=0

# ─── 阶段0：aria2c 并发预下载所有数据集（16连接/文件，8文件并行）──────────────
prefetch_datasets() {
  echo "[$(date '+%H:%M:%S')] aria2c 预下载数据集（16连接/文件，8文件并行）..."
  local PREFETCH_LOG="${RESULTS_DIR}/prefetch.log"
  python "${SCRIPT_DIR}/prefetch_datasets.py" \
    --endpoint "${HF_ENDPOINT}" \
    --connections 16 \
    --parallel 8 \
    >> "${PREFETCH_LOG}" 2>&1 \
    && echo "[$(date '+%H:%M:%S')] 数据集预下载完成" \
    || echo "[$(date '+%H:%M:%S')] 预下载部分失败（见 prefetch.log），继续推理"
  # 切离线模式，6 个推理进程不再访问网络
  export HF_DATASETS_OFFLINE=1
  echo "[$(date '+%H:%M:%S')] HF_DATASETS_OFFLINE=1，开始并行推理"
}

[[ $DRY_RUN -eq 0 ]] && prefetch_datasets

# ─── 模型定义 ────────────────────────────────────────────────────────────────
# 格式：TAG|MODEL_PATH|GPU_ID
MODELS=(
  "base-8B|/home/yinrong/models/Qwen3-8B|0"
  "sft-8B-expLLL|/home/yinrong/post-train/impl2.1/round2/history-route2.1.1/checkpoints/expLLL-merged|1"
  "base-14B|/home/yinrong/models/Qwen3-14B|2"
  "sft-14B-expYYY|/home/yinrong/post-train/impl2.1/round2/history-route2.1.1/checkpoints/expYYY-merged|3"
  "base-32B|/home/yinrong/models/Qwen3-32B|4"
  "sft-32B-expHHH|/home/yinrong/post-train/impl2.1/round2/history-route2.1.1/checkpoints/expHHH-merged|5"
)

# ─── 评测任务 ─────────────────────────────────────────────────────────────────
# 分为两组，以防 cmmlu 未安装时不影响主流程
TASKS_STANDARD="mmlu,gsm8k,arc_challenge,hellaswag,winogrande,truthfulqa_mc1"
TASKS_CN="cmmlu"

# ─── 单模型评测函数 ───────────────────────────────────────────────────────────
run_one() {
  local TAG="$1"
  local MODEL_PATH="$2"
  local GPU="$3"
  local OUT="${RESULTS_DIR}/${TAG}"
  mkdir -p "$OUT"

  echo "[$(date '+%H:%M:%S')] START  ${TAG}  GPU=${GPU}"

  # 使用 hf 后端（无需 ray，稳定可靠）；32B 在单张 H20 96GB 上以 bf16 加载
  local MODEL_ARGS="pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True"

  local CMD_STD=(
    lm_eval
    --model hf
    --model_args "${MODEL_ARGS}"
    --tasks "${TASKS_STANDARD}"
    --batch_size 4
    --num_fewshot 0
    --output_path "${OUT}/standard"
    --log_samples
  )

  local CMD_CN=(
    lm_eval
    --model hf
    --model_args "${MODEL_ARGS}"
    --tasks "${TASKS_CN}"
    --batch_size 4
    --num_fewshot 0
    --output_path "${OUT}/cn"
    --log_samples
  )

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] GPU=${GPU}  ${CMD_STD[*]}"
    echo "[DRY-RUN] GPU=${GPU}  ${CMD_CN[*]}"
    return
  fi

  local LOG="${OUT}/run.log"

  # 标准任务
  CUDA_VISIBLE_DEVICES="${GPU}" "${CMD_STD[@]}" >> "${LOG}" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] DONE standard  ${TAG}" >> "${LOG}" || \
    echo "[$(date '+%H:%M:%S')] FAILED standard ${TAG}" >> "${LOG}"

  # 中文任务（可选，失败不阻断）
  CUDA_VISIBLE_DEVICES="${GPU}" "${CMD_CN[@]}" >> "${LOG}" 2>&1 && \
    echo "[$(date '+%H:%M:%S')] DONE cmmlu  ${TAG}" >> "${LOG}" || \
    echo "[$(date '+%H:%M:%S')] SKIP  cmmlu (not installed or failed): ${TAG}" >> "${LOG}"

  echo "[$(date '+%H:%M:%S')] FINISH ${TAG}"
}

# ─── 并行启动 ─────────────────────────────────────────────────────────────────
PIDS=()
for ENTRY in "${MODELS[@]}"; do
  IFS='|' read -r TAG MODEL_PATH GPU <<< "$ENTRY"
  run_one "$TAG" "$MODEL_PATH" "$GPU" &
  PIDS+=($!)
done

# ─── 等待全部完成 ─────────────────────────────────────────────────────────────
ALL_OK=1
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "OK:   ${MODELS[$i]%%|*}"
  else
    echo "FAIL: ${MODELS[$i]%%|*}"
    ALL_OK=0
  fi
done

if [[ $ALL_OK -eq 1 ]]; then
  echo ""
  echo "所有模型评测完成，运行 compute_degradation.py 计算退化率"
  python "${SCRIPT_DIR}/compute_degradation.py" --results_dir "$RESULTS_DIR"
else
  echo "部分模型评测失败，请查看各 run.log"
  exit 1
fi
