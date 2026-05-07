#!/bin/bash
# Run after Batch 9 results are in
# Creates Batch 10 training data and configs (all use cutoff_len=5120)
cd "$(dirname "$0")"
HISTDIR="history-route2.1.1"
MODEL_7B="/home/yinrong/models/Qwen3-7B"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
MODEL_32B="/home/yinrong/models/Qwen3-32B"

V1_NS="$HISTDIR/data/train_expRR.jsonl"   # 200 ns_v1 samples (gold standard)
V3_NS="$HISTDIR/data/train_claude_teacher_v3_noskill.jsonl"

if [ ! -f "$V1_NS" ]; then
    echo "ERROR: train_expRR.jsonl (ns_v1 data) not found."
    exit 1
fi
if [ ! -f "$V3_NS" ] || [ "$(wc -l < $V3_NS)" -lt 200 ]; then
    echo "WARNING: teacher v3 no_skill not ready or < 200 samples. expPPP may use v1+v1 instead."
    # Fall back to v1+v1 for expPPP if v3 not ready
    cat "$V1_NS" "$V1_NS" > "$HISTDIR/data/train_expPPP.jsonl"
    echo "expPPP: Using v1+v1 fallback (400 samples)"
else
    cat "$V1_NS" <(head -200 "$V3_NS") > "$HISTDIR/data/train_expPPP.jsonl"
    echo "expPPP: $(wc -l < $HISTDIR/data/train_expPPP.jsonl) lines (v1+v3)"
fi

echo "Creating Batch 10 training data..."

# expNNN: 32B + cutoff=5120 + 200 ns_v1 × 5ep
cp "$V1_NS" "$HISTDIR/data/train_expNNN.jsonl"
echo "expNNN: $(wc -l < $HISTDIR/data/train_expNNN.jsonl) lines (ns_v1, 32B+cutoff5120)"

# expOOO: 14B + cutoff=5120 + LR=5e-5 + 200 ns_v1 × 5ep
cp "$V1_NS" "$HISTDIR/data/train_expOOO.jsonl"
echo "expOOO: $(wc -l < $HISTDIR/data/train_expOOO.jsonl) lines (ns_v1, 14B+lr5e-5+cutoff5120)"

# expPPP: created above (v1+v3 or v1+v1)

# expQQQ: 7B + cutoff=5120 + 200 ns_v1 × 5ep
cp "$V1_NS" "$HISTDIR/data/train_expQQQ.jsonl"
echo "expQQQ: $(wc -l < $HISTDIR/data/train_expQQQ.jsonl) lines (ns_v1, 7B+cutoff5120)"

# expSSS: 14B + cutoff=5120 + 200 ns_v3 × 5ep
if [ -f "$V3_NS" ] && [ "$(wc -l < $V3_NS)" -ge 200 ]; then
    head -200 "$V3_NS" > "$HISTDIR/data/train_expSSS.jsonl"
    echo "expSSS: $(wc -l < $HISTDIR/data/train_expSSS.jsonl) lines (ns_v3, 14B+cutoff5120)"
else
    echo "WARNING: v3_noskill not ready, expSSS using ns_v1 as fallback"
    cp "$V1_NS" "$HISTDIR/data/train_expSSS.jsonl"
fi

# Create configs
cat > "$HISTDIR/configs/expNNN.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-32B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expNNN
cutoff_len: 5120
num_train_epochs: 5
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
flash_attn: sdpa
logging_steps: 5
save_strategy: epoch
save_total_limit: 1
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expNNN
do_train: true
report_to: none
EOF

cat > "$HISTDIR/configs/expOOO.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-14B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expOOO
cutoff_len: 5120
num_train_epochs: 5
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
flash_attn: sdpa
logging_steps: 5
save_strategy: epoch
save_total_limit: 1
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expOOO
do_train: true
report_to: none
EOF

cat > "$HISTDIR/configs/expPPP.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-14B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expPPP
cutoff_len: 5120
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
flash_attn: sdpa
logging_steps: 5
save_strategy: epoch
save_total_limit: 1
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expPPP
do_train: true
report_to: none
EOF

cat > "$HISTDIR/configs/expSSS.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-14B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expSSS
cutoff_len: 5120
num_train_epochs: 5
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
flash_attn: sdpa
logging_steps: 5
save_strategy: epoch
save_total_limit: 1
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expSSS
do_train: true
report_to: none
EOF

cat > "$HISTDIR/configs/expQQQ.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-7B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expQQQ
cutoff_len: 5120
num_train_epochs: 5
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
flash_attn: sdpa
logging_steps: 5
save_strategy: epoch
save_total_limit: 1
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expQQQ
do_train: true
report_to: none
EOF

# Register datasets
python3 -c "
import json
info = json.load(open('data/dataset_info.json'))
base_template = {'formatting': 'alpaca', 'columns': {'system': 'system', 'prompt': 'instruction', 'query': 'input', 'response': 'output'}}
new_ds = {
    'spc_r5_expNNN': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expNNN.jsonl', **base_template},
    'spc_r5_expOOO': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expOOO.jsonl', **base_template},
    'spc_r5_expPPP': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expPPP.jsonl', **base_template},
    'spc_r5_expQQQ': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expQQQ.jsonl', **base_template},
    'spc_r5_expSSS': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expSSS.jsonl', **base_template},
}
for k, v in new_ds.items():
    if k not in info:
        info[k] = v
        print(f'Added {k}')
    else:
        print(f'Already exists: {k}')
json.dump(info, open('data/dataset_info.json', 'w'), ensure_ascii=False, indent=2)
print('Done')
"

echo "Batch 10 data and configs ready."
echo "  expNNN: 32B + cutoff=5120 (apply MMM fix to 32B)"
echo "  expOOO: 14B + cutoff=5120 + LR=5e-5 (combine MMM + KKK)"
echo "  expPPP: 14B + cutoff=5120 + 400 (v1+v3) × 3ep=150 steps (cross-pool + fix)"
echo "  expQQQ: 7B + cutoff=5120 (smallest model + fix)"
echo "  expSSS: 14B + cutoff=5120 + ns_v3 seed pool (v3 + cutoff fix)"

# === NEW B10 EXTENSIONS (GPUs 5,6,7) ===
# Critical missing: ws+ns_v2 at cutoff=5120 (the "expXX fix" with proper seed pool)
V1_WS="$HISTDIR/data/train_claude_teacher.jsonl"      # 500 ws samples (v1)
V2_NS="$HISTDIR/data/train_claude_teacher_v2_noskill.jsonl"  # 300 ns samples (v2)
V4_NS="$HISTDIR/data/train_claude_teacher_v4_noskill.jsonl"  # 300 ns samples (v4)

# expTTT: 250 ws_v1 + 250 ns_v2 + cutoff=5120, 3ep = 125 steps (GPU 5)
# KEY: ws+ns combo with proper ns pool that has rule8 → "expXX fix"
if [ -f "$V2_NS" ] && [ "$(wc -l < $V2_NS)" -ge 250 ]; then
    head -250 "$V1_WS" > "$HISTDIR/data/train_expTTT.jsonl"
    head -250 "$V2_NS" >> "$HISTDIR/data/train_expTTT.jsonl"
    echo "expTTT: $(wc -l < $HISTDIR/data/train_expTTT.jsonl) lines (250ws_v1 + 250ns_v2, cutoff5120)"
else
    echo "WARNING: v2_ns not ready, expTTT using expXX data as fallback"
    cp "$HISTDIR/data/train_expXX.jsonl" "$HISTDIR/data/train_expTTT.jsonl"
fi

# expUUU: 32B + 250 ws_v1 + 250 ns_v2 + cutoff=5120, 3ep = 125 steps (GPU 6)
# KEY: 32B version of expTTT — if 32B model is significantly better, this is the best config
cp "$HISTDIR/data/train_expTTT.jsonl" "$HISTDIR/data/train_expUUU.jsonl"
echo "expUUU: $(wc -l < $HISTDIR/data/train_expUUU.jsonl) lines (250ws_v1+250ns_v2, 32B)"

# expVVV: 14B + 200 ns_v4 + cutoff=5120, 5ep = 125 steps (GPU 7)
# Tests v4 seed pool (yet-unused, different diversity than v1/v2/v3)
if [ -f "$V4_NS" ] && [ "$(wc -l < $V4_NS)" -ge 200 ]; then
    head -200 "$V4_NS" > "$HISTDIR/data/train_expVVV.jsonl"
    echo "expVVV: $(wc -l < $HISTDIR/data/train_expVVV.jsonl) lines (ns_v4, cutoff5120)"
else
    echo "WARNING: v4_ns not ready, expVVV using ns_v1 as fallback"
    cp "$V1_NS" "$HISTDIR/data/train_expVVV.jsonl"
fi

# Configs
cat > "$HISTDIR/configs/expTTT.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-14B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expTTT
cutoff_len: 5120
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
flash_attn: sdpa
logging_steps: 5
save_strategy: epoch
save_total_limit: 1
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expTTT
do_train: true
report_to: none
EOF

cat > "$HISTDIR/configs/expUUU.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-32B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expUUU
cutoff_len: 5120
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
flash_attn: sdpa
logging_steps: 5
save_strategy: epoch
save_total_limit: 1
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expUUU
do_train: true
report_to: none
EOF

cat > "$HISTDIR/configs/expVVV.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-14B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expVVV
cutoff_len: 5120
num_train_epochs: 5
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
flash_attn: sdpa
logging_steps: 5
save_strategy: epoch
save_total_limit: 1
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expVVV
do_train: true
report_to: none
EOF

# Register new datasets
python3 -c "
import json
info = json.load(open('data/dataset_info.json'))
base_template = {'formatting': 'alpaca', 'columns': {'system': 'system', 'prompt': 'instruction', 'query': 'input', 'response': 'output'}}
new_ds = {
    'spc_r5_expTTT': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expTTT.jsonl', **base_template},
    'spc_r5_expUUU': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expUUU.jsonl', **base_template},
    'spc_r5_expVVV': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expVVV.jsonl', **base_template},
}
for k, v in new_ds.items():
    if k not in info:
        info[k] = v
        print(f'Added {k}')
    else:
        print(f'Already exists: {k}')
json.dump(info, open('data/dataset_info.json', 'w'), ensure_ascii=False, indent=2)
"
echo ""
echo "=== Batch 10 FULL plan (8 GPUs) ==="
echo "GPU 0: expNNN — 32B + 200ns_v1 + cutoff=5120 (32B pure ns baseline)"
echo "GPU 1: expOOO — 14B + 200ns_v1 + cutoff=5120 + LR=5e-5"
echo "GPU 2: expPPP — 14B + 400(v1+v3) + cutoff=5120, 150 steps"
echo "GPU 3: expQQQ — 8B + 200ns_v1 + cutoff=5120"
echo "GPU 4: expSSS — 14B + 200ns_v3 + cutoff=5120"
echo "GPU 5: expTTT — 14B + 250ws+250ns_v2 + cutoff=5120 ← KEY: expXX fix"
echo "GPU 6: expUUU — 32B + 250ws+250ns_v2 + cutoff=5120 ← KEY: 32B expXX fix"
echo "GPU 7: expVVV — 14B + 200ns_v4 + cutoff=5120"
