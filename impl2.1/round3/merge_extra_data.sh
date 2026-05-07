#!/usr/bin/env bash
# merge_extra_data.sh — 等待额外数据生成完成，过滤并合并到 v3 数据集
# 运行条件：ns_v5_extra, multirole_ws_extra, boundary_ws/ns_extra 全部生成完毕
set -euo pipefail
cd /home/yinrong/post-train/impl2.1

ROOT="$(pwd)"
R3="$ROOT/round3"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 等待所有额外数据生成完成 ───────────────────────────────────────────────────
log "等待额外数据生成进程完成..."
wait_for_gen() {
  local pid_keyword="$1"
  local label="$2"
  while ps aux | grep -q "$pid_keyword" | grep -v grep 2>/dev/null; do
    sleep 30
  done
  log "$label 生成完成"
}

# 等待 ns_extra 生成完成
while ps aux | grep -q "[g]en_expanded_ns.py.*ns_v5_extra" 2>/dev/null; do
  sleep 30
done
log "ns_v5_extra 生成完成"

# 等待 multirole_extra 生成完成
while ps aux | grep -q "[g]en_expanded_ns.py.*multirole_ws_extra" 2>/dev/null; do
  sleep 30
done
log "multirole_ws_extra 生成完成"

# 等待 boundary_extra 生成完成
while ps aux | grep -q "[g]en_boundary_rule27.py" 2>/dev/null; do
  sleep 30
done
log "boundary_ws/ns_extra 生成完成"

log "所有额外数据生成完毕，开始过滤..."

# ── 过滤额外数据 ──────────────────────────────────────────────────────────────
filter_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]] && [[ $(wc -l < "$src") -gt 0 ]]; then
    python round3/tools/data/filter_complete.py --input "$src" --output "$dst"
  else
    log "跳过不存在或为空的文件: $src"
  fi
}

filter_if_exists "$R3/data/ns_v5_extra.jsonl"          "$R3/data/ns_v5_extra_clean.jsonl"
filter_if_exists "$R3/data/multirole_ws_extra.jsonl"   "$R3/data/multirole_ws_extra_clean.jsonl"
filter_if_exists "$R3/data/boundary_ws_extra.jsonl"    "$R3/data/boundary_ws_extra_clean.jsonl"
filter_if_exists "$R3/data/boundary_ns_extra.jsonl"    "$R3/data/boundary_ns_extra_clean.jsonl"

# ── 统计清洗后数量 ─────────────────────────────────────────────────────────────
total_extra=0
for f in ns_v5_extra_clean multirole_ws_extra_clean boundary_ws_extra_clean boundary_ns_extra_clean; do
  if [[ -f "$R3/data/${f}.jsonl" ]]; then
    cnt=$(wc -l < "$R3/data/${f}.jsonl")
    log "  $f: $cnt 条"
    total_extra=$((total_extra + cnt))
  fi
done
log "额外清洗数据总计: $total_extra 条"

# ── 合并为 v3 数据集 ──────────────────────────────────────────────────────────
log "合并 v3 数据集..."

python3 - <<'EOF'
import json
from pathlib import Path

R3 = Path("round3/data")

# v3 = v2 + extra clean
v2_file = R3 / "train_R3-AB-v2.jsonl"
extra_files = [
    R3 / "ns_v5_extra_clean.jsonl",
    R3 / "multirole_ws_extra_clean.jsonl",
    R3 / "boundary_ws_extra_clean.jsonl",
    R3 / "boundary_ns_extra_clean.jsonl",
]

samples = []
if v2_file.exists():
    for line in v2_file.read_text().splitlines():
        if line.strip():
            samples.append(line)
    print(f"  v2 base: {len(samples)} samples")

for ef in extra_files:
    if ef.exists():
        extra = [l for l in ef.read_text().splitlines() if l.strip()]
        samples.extend(extra)
        print(f"  + {ef.name}: {len(extra)} samples")

import random
random.seed(42)
random.shuffle(samples)

out = R3 / "train_R3-AB-v3.jsonl"
out.write_text("\n".join(samples) + "\n")
print(f"\ntrain_R3-AB-v3.jsonl: {len(samples)} samples total")
EOF

# ── 注册 v3 到 dataset_info.json ─────────────────────────────────────────────
python3 - <<'EOF'
import json
from pathlib import Path

info_file = Path("round3/data/dataset_info.json")
info = json.loads(info_file.read_text())

if "spc_r3_AB_v3" not in info:
    info["spc_r3_AB_v3"] = {
        "file_name": "/home/yinrong/post-train/impl2.1/round3/data/train_R3-AB-v3.jsonl",
        "formatting": "alpaca",
        "columns": {
            "system": "system",
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
    info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print("  spc_r3_AB_v3 registered in dataset_info.json")
else:
    print("  spc_r3_AB_v3 already registered")
EOF

log "v3 数据集准备完毕：round3/data/train_R3-AB-v3.jsonl"
log "如需训练 v3，运行："
log "  DISABLE_VERSION_CHECK=1 llamafactory-cli train round3/configs/R3-AB-v3.yaml"
