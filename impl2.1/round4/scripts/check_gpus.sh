#!/bin/bash
# 检查 GPU 空闲状态，识别自己进程（round4 目录相关），输出可用 GPU 列表
#
# 用法：
#   bash scripts/check_gpus.sh          # 显示 GPU 状态
#   GPU=$(bash scripts/check_gpus.sh --free 2)  # 获取 2 个空闲 GPU

NEED="${1:---status}"  # --status 或 --free N
N_NEED="${2:-1}"

ROUND4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== GPU 状态 ===" >&2
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader >&2
echo "" >&2

# 找出 round4 相关进程（MY 进程）
echo "=== 我的进程（round4 相关）===" >&2
MY_PIDS=$(ps -eo pid,cmd | grep -E "$ROUND4_DIR" | grep -v grep | awk '{print $1}')
if [ -n "$MY_PIDS" ]; then
    for pid in $MY_PIDS; do
        cmd=$(ps -p $pid -o cmd --no-headers 2>/dev/null | head -1)
        gpus=$(cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n' | grep CUDA_VISIBLE_DEVICES | head -1)
        echo "  PID $pid: $gpus  $cmd" >&2
    done
else
    echo "  (无)" >&2
fi
echo "" >&2

# 找出空闲 GPU（< 2GB 使用）
echo "=== 空闲 GPU（< 2GB 使用）===" >&2
FREE_GPUS=()
while IFS=, read -r idx name used free; do
    used_mib=$(echo "$used" | tr -d ' MiB')
    if [ "$used_mib" -lt 2048 ] 2>/dev/null; then
        FREE_GPUS+=("$idx")
        echo "  GPU $idx: ${used} used, ${free} free" >&2
    fi
done < <(nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader)

if [ "$NEED" = "--status" ]; then
    echo "" >&2
    echo "空闲 GPU: ${FREE_GPUS[*]}" >&2
elif [ "$NEED" = "--free" ]; then
    # Return N free GPUs as comma-separated list
    selected=("${FREE_GPUS[@]:0:$N_NEED}")
    if [ "${#selected[@]}" -lt "$N_NEED" ]; then
        echo "WARNING: Only ${#selected[@]} free GPU(s), need $N_NEED" >&2
    fi
    IFS=, ; echo "${selected[*]}"
fi
