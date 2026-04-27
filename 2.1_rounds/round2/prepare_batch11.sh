#!/bin/bash
# Prepare Batch 11 training data and configs
# Prerequisites: Batch 10 results + teacher v4 no_skill ready
cd "$(dirname "$0")"
HISTDIR="history-route2.1.1"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
MODEL_32B="/home/yinrong/models/Qwen3-32B"

V1_NS="$HISTDIR/data/train_expRR.jsonl"   # 200 ns_v1 (gold standard)
V3_NS="$HISTDIR/data/train_claude_teacher_v3_noskill.jsonl"
V4_NS="$HISTDIR/data/train_claude_teacher_v4_noskill.jsonl"

echo "Creating Batch 11 training data..."

# expTTT: 14B + cutoff=5120 + 200 ns_v1 × 3ep = 75 steps (fewer steps test)
cp "$V1_NS" "$HISTDIR/data/train_expTTT.jsonl"
echo "expTTT: $(wc -l < $HISTDIR/data/train_expTTT.jsonl) lines (ns_v1, 75 steps test)"

# expUUU: 14B + cutoff=5120 + 200 ns_v1 × 8ep = 200 steps (overfit test)
cp "$V1_NS" "$HISTDIR/data/train_expUUU.jsonl"
echo "expUUU: $(wc -l < $HISTDIR/data/train_expUUU.jsonl) lines (ns_v1, 200 steps overfit test)"

# expVVV: 14B + cutoff=5120 + 400 (ns_v1+ns_v4) × 3ep = 150 steps (v1+v4 diversity)
if [ -f "$V4_NS" ] && [ "$(wc -l < $V4_NS)" -ge 200 ]; then
    cat "$V1_NS" <(head -200 "$V4_NS") > "$HISTDIR/data/train_expVVV.jsonl"
    echo "expVVV: $(wc -l < $HISTDIR/data/train_expVVV.jsonl) lines (ns_v1+ns_v4)"
else
    echo "WARNING: v4 not ready, expVVV using v1+v3 fallback"
    cat "$V1_NS" <(head -200 "$V3_NS") > "$HISTDIR/data/train_expVVV.jsonl"
fi

# expWWW: 32B + cutoff=5120 + 400 (ns_v1+ns_v3) × 3ep = 150 steps (32B + diversity)
cat "$V1_NS" <(head -200 "$V3_NS") > "$HISTDIR/data/train_expWWW.jsonl"
echo "expWWW: $(wc -l < $HISTDIR/data/train_expWWW.jsonl) lines (ns_v1+ns_v3, 32B)"

# Create configs
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
model_name_or_path: /home/yinrong/models/Qwen3-14B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expUUU
cutoff_len: 5120
num_train_epochs: 8
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
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expVVV
do_train: true
report_to: none
EOF

cat > "$HISTDIR/configs/expWWW.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-32B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expWWW
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
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expWWW
do_train: true
report_to: none
EOF

# Register datasets
python3 -c "
import json
info = json.load(open('data/dataset_info.json'))
base_template = {'formatting': 'alpaca', 'columns': {'system': 'system', 'prompt': 'instruction', 'query': 'input', 'response': 'output'}}
new_ds = {
    'spc_r5_expTTT': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expTTT.jsonl', **base_template},
    'spc_r5_expUUU': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expUUU.jsonl', **base_template},
    'spc_r5_expVVV': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expVVV.jsonl', **base_template},
    'spc_r5_expWWW': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expWWW.jsonl', **base_template},
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

echo "Batch 11 data and configs ready."
echo "  expTTT: 14B + cutoff=5120 + 200 ns_v1 × 3ep = 75 steps (fewer steps)"
echo "  expUUU: 14B + cutoff=5120 + 200 ns_v1 × 8ep = 200 steps (overfit test)"
echo "  expVVV: 14B + cutoff=5120 + 400 (ns_v1+ns_v4) × 3ep = 150 steps (v1+v4 diversity)"
echo "  expWWW: 32B + cutoff=5120 + 400 (ns_v1+ns_v3) × 3ep = 150 steps (32B + diversity)"
