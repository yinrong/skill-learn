#!/usr/bin/env bash
# finish_d2.sh — 等待 D2 训练完成后自动 merge + eval + 汇总
set -euo pipefail
cd /home/yinrong/post-train/impl2.1
ROOT="$(pwd)"; R3="$ROOT/round3"; COMMON="$ROOT/common"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_vllm() {
  for i in $(seq 1 60); do
    curl -sf "http://localhost:$1/health" >/dev/null 2>&1 && log "vLLM $1 就绪" && return 0
    sleep 5
  done; log "ERROR: vLLM $1 超时"; return 1
}

log "等待 D2 训练完成..."
while true; do
  if grep -q "train_runtime" "$R3/logs/train_R3-D2.log" 2>/dev/null; then
    log "D2 训练完成"; break
  fi
  if ! ps aux | grep -q "[l]auncher.*D2\|[p]ython.*R3-D2"; then
    if grep -q "Error\|error" "$R3/logs/train_R3-D2.log" 2>/dev/null; then
      log "ERROR: D2 训练失败，退出"
      tail -5 "$R3/logs/train_R3-D2.log"
      exit 1
    fi
    log "WARNING: D2 进程消失但无 train_runtime，继续等待..."
  fi
  sleep 30
done

CKPT="$R3/checkpoints/R3-D2-qlora-32b"
MERGED="$R3/checkpoints/R3-D2-qlora-32b-merged"
RESULT="$R3/results/R3-D2.json"

if [[ ! -d "$MERGED" ]]; then
  log "合并 D2 adapter..."
  python "$COMMON/tools/train/merge_adapter.py" \
    --base /home/yinrong/models/Qwen3-32B \
    --adapter "$CKPT" --output "$MERGED" --template qwen3 \
    2>&1 | tee "$R3/logs/merge_R3-D2.log"
fi

log "启动 vLLM 端口 8037..."
python "$COMMON/tools/train/deploy_vllm.py" \
  --model "$MERGED" --port 8037 \
  2>&1 | tee "$R3/logs/vllm_R3-D2.log" &
VLLM_PID=$!
wait_for_vllm 8037

python "$COMMON/tools/eval/spc_eval.py" \
  --model_url http://localhost:8037 --model_name R3-D2 \
  --test "$COMMON/data/test.jsonl" --output "$RESULT" \
  --max_tokens 4096 --concurrency 4 \
  2>&1 | tee "$R3/logs/eval_R3-D2.log"

kill $VLLM_PID 2>/dev/null || true
kill $(ps aux | grep "[v]llm.*8037" | awk '{print $2}') 2>/dev/null || true

# 更新 index.md：添加 D2 评测结果和 merge 链接
INDEX="$R3/index.md"
sed -i 's/| R3-D2 (32B QLoRA) | 🔄 训练中 | — | — | — | 待生成 |/| R3-D2 (32B QLoRA FSDP 4卡) | 见下表 | — | — | — | [results\/R3-D2.json](results\/R3-D2.json) |/' "$INDEX" 2>/dev/null || true
sed -i 's/| R3-D2 | 待生成 |$/| R3-D2 | [logs\/eval_R3-D2.log](logs\/eval_R3-D2.log) |/' "$INDEX" 2>/dev/null || true
sed -i 's/| R3-D2-qlora-32b-merged | 待生成 |/| R3-D2-qlora-32b-merged | [checkpoints\/R3-D2-qlora-32b-merged\/](checkpoints\/R3-D2-qlora-32b-merged\/) |/' "$INDEX" 2>/dev/null || true
sed -i "s/.*最后更新.*/\n*最后更新：$(date '+%Y-%m-%d %H:%M')（D2 全部完成）*/" "$INDEX" 2>/dev/null || true
log "index.md 已更新"

log "=== 最终结果汇总 ==="
python3 -c "
import json, os
results_dir = '$R3/results'
exps = ['R3-A','R3-B','R3-AB','R3-AB-v2','R3-AB-v3','R3-D1','R3-D2']
print(f'  {\"实验\":<14} {\"F1\":>6} {\"rule2\":>6} {\"rule7\":>6} {\"CPK_MAE\":>8}')
print('  ' + '-'*46)
for exp in exps:
    f = os.path.join(results_dir, exp+'.json')
    if not os.path.exists(f): continue
    d = json.load(open(f)); s = d['summary']
    cpk = s['cpk_mae']; cpk_str = f'{cpk:.3f}' if cpk is not None else 'N/A'
    r = s['per_rule_recall']
    print(f'  {exp:<14} {s[\"rule_detection_f1\"]:>6.3f} {r[\"rule2\"]:>6.3f} {r[\"rule7\"]:>6.3f} {cpk_str:>8}')
"
log "Round3 完成！结果在 $R3/results/"
