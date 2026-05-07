#!/usr/bin/env bash
# eval_and_report.sh — 对已有 merged 模型单独评测并生成对比报告
# 用法：cd /home/yinrong/post-train/impl2.1 && bash round3/eval_and_report.sh [exp_name]
#
# 不加参数：评测所有 *-merged 模型
# 加参数：仅评测指定实验，如 bash round3/eval_and_report.sh R3-A

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R3="$ROOT/round3"
COMMON="$ROOT/common"
TEST_DATA="$COMMON/data/test.jsonl"
FILTER="${1:-}"

eval_one() {
  local EXP="$1"
  local PORT="$2"
  local MERGED="$R3/checkpoints/${EXP}-merged"
  local RESULT="$R3/results/${EXP}.json"

  if [[ ! -d "$MERGED" ]]; then
    echo "[跳过] $EXP：merged 模型不存在"
    return
  fi

  if [[ -f "$RESULT" ]]; then
    echo "[跳过] $EXP：已有结果（删除 $RESULT 可重跑）"
    return
  fi

  echo "[评测] $EXP ..."
  python "$COMMON/tools/train/deploy_vllm.py" --model "$MERGED" --port "$PORT" &
  local VLLM_PID=$!
  sleep 30

  python "$COMMON/tools/eval/spc_eval.py" \
    --model_url "http://localhost:$PORT" \
    --model_name "$EXP" \
    --test "$TEST_DATA" \
    --output "$RESULT"

  kill $VLLM_PID 2>/dev/null || true
  python "$COMMON/tools/train/deploy_vllm.py" --kill --port "$PORT" 2>/dev/null || true
  echo "[完成] $EXP → $RESULT"
}

# 实验端口映射
declare -A PORT_MAP=(
  ["R3-A"]=8031
  ["R3-B"]=8032
  ["R3-AB"]=8033
  ["R3-C"]=8034
  ["R3-D1-qlora-14b"]=8035
  ["R3-D2-qlora-32b"]=8036
  ["R3-E1-zero3-72b"]=8037
)

if [[ -n "$FILTER" ]]; then
  PORT="${PORT_MAP[$FILTER]:-8039}"
  eval_one "$FILTER" "$PORT"
else
  for EXP in "${!PORT_MAP[@]}"; do
    eval_one "$EXP" "${PORT_MAP[$EXP]}"
  done
fi

# ── 生成对比报告 ──────────────────────────────────────────────────────────────

echo ""
echo "生成对比报告..."
python3 - << 'EOF'
import json, os, sys
from pathlib import Path

results_dir = Path(os.environ.get("R3", ".")) / "results" if "R3" not in os.environ else Path(os.environ["R3"]) / "results"
# fallback
results_dir = Path(__file__).parent / "results" if not results_dir.exists() else results_dir

# 参考 round2 expYYY baseline
r2_result = Path("/home/yinrong/post-train/impl2.1/round2/history-route2.1.1/results/expYYY.json")
experiments = {}
if r2_result.exists():
    experiments["expYYY (round2 baseline)"] = json.load(open(r2_result))["summary"]

for f in sorted(results_dir.glob("*.json")):
    d = json.load(open(f))
    experiments[f.stem] = d.get("summary", d)

if not experiments:
    print("无结果文件")
    sys.exit(0)

# 打印表格
print(f"\n{'='*90}")
print("Round3 实验对比")
print(f"{'='*90}")
header = f"{'实验':<30} {'F1':>6} {'rule2':>6} {'rule7':>6} {'rule3':>6} {'CPK_mae':>8}"
print(header)
print("-" * 90)
for name, s in experiments.items():
    per = s.get("per_rule_recall", {})
    print(
        f"{name:<30} "
        f"{s.get('rule_detection_f1',0):>6.3f} "
        f"{per.get('rule2',0):>6.3f} "
        f"{per.get('rule7',0):>6.3f} "
        f"{per.get('rule3',0):>6.3f} "
        f"{s.get('cpk_mae',0):>8.3f}"
    )
print(f"{'='*90}")
EOF

echo ""
echo "✓ 评测完成，结果在 round3/results/"
