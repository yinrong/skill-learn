#!/bin/bash
# 合并 LoRA adapter 到 base model，生成完整权重目录
# 从 round4/ 目录运行：
#   bash scripts/merge_adapter.sh [CHECKPOINT_DIR] [OUTPUT_DIR]
#
# 示例（从 round4/）：
#   bash scripts/merge_adapter.sh \
#     checkpoints/R4-sft-v2 \
#     checkpoints/R4-sft-v2-merged

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Determine WORKDIR: support running from round4/ or impl2.1/
if [ -f "$SCRIPT_DIR/../CLAUDE.md" ]; then
    # Running from round4/scripts/ → WORKDIR is round4/
    WORKDIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    WORKDIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$WORKDIR"

CHECKPOINT="${1:-checkpoints/R4-sft-v2}"
MERGED="${2:-checkpoints/R4-sft-v2-merged}"
BASE_MODEL="/home/yinrong/models/Qwen3-14B"

echo "============================================"
echo "Merging LoRA adapter → full model"
echo "  Checkpoint : $CHECKPOINT"
echo "  Merged out : $MERGED"
echo "  Base model : $BASE_MODEL"
echo "============================================"

# Find the best checkpoint directory (last epoch or latest)
if [ -d "$CHECKPOINT/checkpoint-441" ]; then
    CKPT="$CHECKPOINT/checkpoint-441"
elif [ -d "$CHECKPOINT" ] && ls "$CHECKPOINT"/checkpoint-* 2>/dev/null | tail -1 | grep -q .; then
    CKPT=$(ls -d "$CHECKPOINT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
else
    CKPT="$CHECKPOINT"
fi
echo "Using checkpoint: $CKPT"

mkdir -p "$MERGED"

DISABLE_VERSION_CHECK=1 llamafactory-cli export \
    --model_name_or_path "$BASE_MODEL" \
    --adapter_name_or_path "$CKPT" \
    --template qwen3 \
    --finetuning_type lora \
    --export_dir "$MERGED" \
    --export_size 5 \
    --export_legacy_format false \
    --trust_remote_code true \
    2>&1 | tee logs/merge_adapter.log

echo ""
echo "Done! Merged model at: $MERGED"
echo "$(ls -lh "$MERGED"/*.safetensors 2>/dev/null | wc -l) safetensors files"
