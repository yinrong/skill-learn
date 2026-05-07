#!/bin/bash
# 等待 GRPO eval 完成 → kill 单卡 AB-v4 → 3 卡 DDP 重启（从头）
# 预计节省: ~52min（T+7.63h → T+6.76h）
# 注意: 有效 batch 从 8→24（3x），梯度更新从 726→243 步，训练动态有差异
cd /home/yinrong/post-train/impl2.1
LOG="round3/logs/accelerate_abv4.log"
ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]"; }
exec > >(tee -a "$LOG") 2>&1

echo "$(ts) AB-v4 加速脚本启动。等待 R3-C-grpo.json..."

# Step 1: 等待 GRPO eval 完成
while [ ! -f "round3/results/R3-C-grpo.json" ]; do
    sleep 60
done
echo "$(ts) GRPO eval 完成。准备加速 R3-AB-v4..."

# Step 2: 等待 GPU 1+2 真正空闲（vLLM eval 结束后）
echo "$(ts) 等待 GPU 1+2 空闲 (< 5000MB)..."
for i in $(seq 1 60); do
    GPU1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
    GPU2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -d ' ')
    if [ "$GPU1" -lt 5000 ] && [ "$GPU2" -lt 5000 ]; then
        echo "$(ts) GPU 1+2 空闲 (${GPU1}MB / ${GPU2}MB). 继续."
        break
    fi
    echo "$(ts) GPU1=${GPU1}MB GPU2=${GPU2}MB. 等待..."
    sleep 30
done

# Step 3: 检查 AB-v4 是否已完成（若已有结果则跳过）
if [ -f "round3/results/R3-AB-v4.json" ]; then
    echo "$(ts) R3-AB-v4.json 已存在，无需重启。退出。"
    exit 0
fi

# Step 4: 找到并杀死单卡 AB-v4 进程（GPU 0 上）
AB4_GPU0_PID=$(nvidia-smi pmon -c 1 2>/dev/null | awk '$1=="0" {print $2}' | head -1)
if [ -n "$AB4_GPU0_PID" ]; then
    echo "$(ts) 杀死 GPU 0 上的 AB-v4 进程 PID=$AB4_GPU0_PID..."
    kill "$AB4_GPU0_PID" 2>/dev/null
    sleep 20
    # 确保进程终止
    kill -9 "$AB4_GPU0_PID" 2>/dev/null
    sleep 5
else
    echo "$(ts) GPU 0 上无进程，AB-v4 可能已完成。检查结果..."
    if [ -f "round3/results/R3-AB-v4.json" ]; then
        echo "$(ts) 已有结果。退出。"
        exit 0
    fi
fi

# Step 5: 杀死旧 watcher，清空残缺 checkpoint
pkill -f "abv4_complete_watcher" 2>/dev/null
sleep 3
echo "$(ts) 清空残缺单卡 checkpoint..."
rm -rf round3/checkpoints/R3-AB-v4/
mkdir -p round3/checkpoints/R3-AB-v4/

# Step 6: 3 卡 DDP 重启（CUDA 0+1+2）
echo "$(ts) 启动 3-GPU DDP 训练 (GPU 0+1+2)..."
CUDA_VISIBLE_DEVICES=0,1,2 FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 \
    nohup llamafactory-cli train round3/configs/R3-AB-v4.yaml \
    >> round3/logs/train_R3-AB-v4.log 2>&1 &
NEW_PID=$!
echo "$(ts) R3-AB-v4 3-GPU 训练已启动 PID=$NEW_PID，预计约 3h 完成。"
echo "$(ts) 注意: effective_batch=24 (3x)，optimizer steps=243 (1/3)，动态有差异。"

# Step 7: 等待进程启动稳定，然后重启 watcher
sleep 30
if kill -0 $NEW_PID 2>/dev/null; then
    echo "$(ts) 训练进程健在。启动 abv4_complete_watcher..."
    nohup bash round3/abv4_complete_watcher.sh > /dev/null 2>&1 &
    echo "$(ts) watcher PID=$!"
else
    echo "$(ts) 警告：训练进程已退出。检查 train_R3-AB-v4.log。"
fi
