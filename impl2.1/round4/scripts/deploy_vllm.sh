#!/bin/bash
# 部署合并后的模型到 vLLM 服务
#
# 用法：
#   bash round4/scripts/deploy_vllm.sh [GPU_ID] [PORT] [MODEL_DIR]
#
# 示例：
#   bash round4/scripts/deploy_vllm.sh 2 8035 round4/checkpoints/R4-base-merged
#
# 基线对比（未微调 base model）：
#   bash round4/scripts/deploy_vllm.sh 3 8036 /home/yinrong/models/Qwen3-14B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Support running from round4/ or impl2.1/
if [ -f "$SCRIPT_DIR/../CLAUDE.md" ]; then
    WORKDIR="$(cd "$SCRIPT_DIR/.." && pwd)"   # round4/
else
    WORKDIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # impl2.1/
fi
cd "$WORKDIR"

GPU_ID="${1:-2}"
PORT="${2:-8035}"
MODEL_DIR="${3:-round4/checkpoints/R4-base-merged}"

# Resolve to absolute path
if [[ "$MODEL_DIR" != /* ]]; then
    MODEL_DIR="$WORKDIR/$MODEL_DIR"
fi

echo "============================================"
echo "Deploying vLLM service"
echo "  GPU       : $GPU_ID"
echo "  Port      : $PORT"
echo "  Model     : $MODEL_DIR"
echo "============================================"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Run merge_adapter.sh first."
    exit 1
fi

LOG_FILE="round4/logs/vllm_R4-$(basename "$MODEL_DIR").log"
echo "Log: $LOG_FILE"
echo ""
echo "Starting vLLM... (press Ctrl+C to stop)"
echo ""

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$(basename "$MODEL_DIR")" \
    --port "$PORT" \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    2>&1 | tee "$LOG_FILE"
# Note: Qwen3 uses <tool_call>...</tool_call> format (hermes-compatible).
# If tool calls are not parsed correctly, try: --tool-call-parser qwen3_xml
