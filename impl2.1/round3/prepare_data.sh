#!/usr/bin/env bash
# prepare_data.sh — Round3 数据准备管道
# 必须从 impl2.1/ 目录运行：cd /home/yinrong/post-train/impl2.1 && bash round3/prepare_data.sh
#
# 分两个阶段：
#   Phase 1（P0）：方向A边界数据 + 方向B扩充ns/多角色ws → 并行生成
#   Phase 2      ：合并数据集，注册到 LLaMA-Factory
#
# 环境变量：
#   SKIP_GEN=1     跳过数据生成（仅合并/注册，用于数据已存在时）
#   CONCURRENCY=6  API 并发数（默认6）

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R3="$ROOT/round3"
COMMON="$ROOT/common"
R2DATA="$ROOT/round2/history-route2.1.1/data"

CONCURRENCY="${CONCURRENCY:-6}"

echo "======================================================"
echo "Round3 数据准备"
echo "ROOT: $ROOT"
echo "======================================================"

# ── Phase 1：数据生成（并行） ────────────────────────────────────────────────

if [[ "${SKIP_GEN:-0}" != "1" ]]; then

  echo ""
  echo "[Phase 1] 数据生成 ..."

  # 方向A：rule2/rule7 边界样本（后台）
  echo "  [A] 启动 boundary_rule27 生成（并发=$CONCURRENCY）..."
  python "$R3/tools/data/gen_boundary_rule27.py" \
    --n_rule2 90 --n_rule7 90 \
    --output_ws "$R3/data/boundary_ws.jsonl" \
    --output_ns "$R3/data/boundary_ns.jsonl" \
    --seed 301 --concurrency "$CONCURRENCY" \
    2>&1 | tee "$R3/logs/gen_boundary.log" &
  PID_A=$!

  # 方向B-1：扩充 ns 池（后台）
  echo "  [B1] 启动 expanded_ns_v5 生成（并发=$CONCURRENCY）..."
  python "$R3/tools/data/gen_expanded_ns.py" \
    --mode ns --n 300 \
    --output "$R3/data/ns_v5.jsonl" \
    --seed 401 --concurrency "$CONCURRENCY" \
    2>&1 | tee "$R3/logs/gen_ns_v5.log" &
  PID_B1=$!

  # 方向B-2：多角色 ws 数据（后台，等 B1 启动后再启动避免 API 过载）
  sleep 5
  echo "  [B2] 启动 multirole_ws 生成（并发=$CONCURRENCY）..."
  python "$R3/tools/data/gen_expanded_ns.py" \
    --mode multirole_ws --n_per_role 50 \
    --output "$R3/data/multirole_ws.jsonl" \
    --seed 501 --concurrency "$CONCURRENCY" \
    2>&1 | tee "$R3/logs/gen_multirole_ws.log" &
  PID_B2=$!

  echo "  等待所有生成任务完成... (PID: A=$PID_A, B1=$PID_B1, B2=$PID_B2)"
  wait $PID_A  && echo "  [A] ✓ boundary 完成"
  wait $PID_B1 && echo "  [B1] ✓ ns_v5 完成"
  wait $PID_B2 && echo "  [B2] ✓ multirole_ws 完成"

else
  echo "[Phase 1] 跳过生成（SKIP_GEN=1）"
fi

# 检查文件存在
for f in "$R3/data/boundary_ws.jsonl" "$R3/data/boundary_ns.jsonl" \
         "$R3/data/ns_v5.jsonl" "$R3/data/multirole_ws.jsonl"; do
  if [[ ! -f "$f" ]]; then
    echo "错误：缺少文件 $f"
    exit 1
  fi
  echo "  ✓ $(basename $f)：$(wc -l < $f) 条"
done

# ── Phase 2：数据合并 ─────────────────────────────────────────────────────────

echo ""
echo "[Phase 2] 合并数据集 ..."

# expYYY 基础数据（ws 250 + ns 250 = 500）
BASE_WS="$R2DATA/train_claude_teacher_v4.jsonl"
BASE_NS="$R2DATA/train_claude_teacher_v4_noskill.jsonl"

# R3-A：expYYY(500) + boundary(180)
python "$R3/tools/data/merge_datasets.py" \
  --inputs "$BASE_WS" "$BASE_NS" \
            "$R3/data/boundary_ws.jsonl" "$R3/data/boundary_ns.jsonl" \
  --output "$R3/data/train_R3-A.jsonl" \
  --shuffle --seed 42
echo "  ✓ train_R3-A.jsonl：$(wc -l < $R3/data/train_R3-A.jsonl) 条"

# R3-B：expYYY(500) + ns_v5(300) + multirole_ws(200)
python "$R3/tools/data/merge_datasets.py" \
  --inputs "$BASE_WS" "$BASE_NS" \
            "$R3/data/ns_v5.jsonl" \
            "$R3/data/multirole_ws.jsonl" \
  --output "$R3/data/train_R3-B.jsonl" \
  --shuffle --seed 43
echo "  ✓ train_R3-B.jsonl：$(wc -l < $R3/data/train_R3-B.jsonl) 条"

# R3-AB：expYYY(500) + boundary(180) + ns_v5(300) + multirole_ws(200)
python "$R3/tools/data/merge_datasets.py" \
  --inputs "$BASE_WS" "$BASE_NS" \
            "$R3/data/boundary_ws.jsonl" "$R3/data/boundary_ns.jsonl" \
            "$R3/data/ns_v5.jsonl" \
            "$R3/data/multirole_ws.jsonl" \
  --output "$R3/data/train_R3-AB.jsonl" \
  --shuffle --seed 44
echo "  ✓ train_R3-AB.jsonl：$(wc -l < $R3/data/train_R3-AB.jsonl) 条"

# R3-GRPO：同 R3-AB（GRPO 使用相同训练数据）
cp "$R3/data/train_R3-AB.jsonl" "$R3/data/train_R3-grpo.jsonl"
echo "  ✓ train_R3-grpo.jsonl（GRPO 数据，同 R3-AB）"

# ── Phase 3：注册到 LLaMA-Factory ───────────────────────────────────────────

echo ""
echo "[Phase 3] 注册数据集 ..."

for name_file in \
  "spc_r3_A:$R3/data/train_R3-A.jsonl" \
  "spc_r3_B:$R3/data/train_R3-B.jsonl" \
  "spc_r3_AB:$R3/data/train_R3-AB.jsonl" \
  "spc_r3_grpo:$R3/data/train_R3-grpo.jsonl"; do
  name="${name_file%%:*}"
  file="${name_file#*:}"
  python "$COMMON/tools/train/register_dataset.py" \
    --name "$name" --file "$file" && echo "  ✓ 已注册 $name"
done

# ── 注入 dataset 名到 configs ───────────────────────────────────────────────

echo ""
echo "[Phase 4] 更新训练配置中的 dataset 字段 ..."

# 使用 sed 取消注释 dataset 行
sed -i 's/^# dataset: spc_r3_A$/dataset: spc_r3_A/' "$R3/configs/R3-A.yaml"
sed -i 's/^# dataset: spc_r3_B$/dataset: spc_r3_B/' "$R3/configs/R3-B.yaml"
sed -i 's/^# dataset: spc_r3_AB$/dataset: spc_r3_AB/' "$R3/configs/R3-AB.yaml"

echo ""
echo "======================================================"
echo "✓ Round3 数据准备完成！"
echo ""
echo "下一步运行："
echo "  bash round3/run_experiments.sh p0    # 运行 R3-A, R3-B (P0)"
echo "  bash round3/run_experiments.sh ab    # 运行 R3-AB (P0)"
echo "  bash round3/run_experiments.sh grpo  # 运行 R3-C GRPO (P1)"
echo "  bash round3/run_experiments.sh qlora # 运行 R3-D1/D2 QLoRA (P1)"
echo "======================================================"
