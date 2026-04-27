#!/bin/bash
# Run after teacher v3 is complete and Batch 8 results are in
# Creates Batch 9 training data and configs
cd "$(dirname "$0")"
HISTDIR="history-route2.1.1"
MODEL_14B="/home/yinrong/models/Qwen3-14B"
MODEL_32B="/home/yinrong/models/Qwen3-32B"

V3_NS="$HISTDIR/data/train_claude_teacher_v3_noskill.jsonl"
if [ ! -f "$V3_NS" ]; then
    echo "ERROR: teacher v3 no_skill not ready. Run convert_teacher_noskill.py first."
    exit 1
fi

echo "Creating Batch 9 training data..."

# expIII: 200 ns_v3 × 5ep = 125 steps (new seed pool, same config as expRR)
head -200 "$V3_NS" > "$HISTDIR/data/train_expIII.jsonl"
echo "expIII: $(wc -l < $HISTDIR/data/train_expIII.jsonl) lines (ns_v3 x5ep=125 steps)"

# expJJJ: 200 ns_v1 + 200 ns_v3 = 400 total × 3ep = 150 steps
cat "$HISTDIR/data/train_expRR.jsonl" <(head -200 "$V3_NS") > "$HISTDIR/data/train_expJJJ.jsonl"
echo "expJJJ: $(wc -l < $HISTDIR/data/train_expJJJ.jsonl) lines (v1+v3 x3ep=150 steps)"

# expKKK: Best 14B config with LR=5e-5 (lower learning rate)
# Uses same data as expRR (200 ns_v1)
cp "$HISTDIR/data/train_expRR.jsonl" "$HISTDIR/data/train_expKKK.jsonl"
echo "expKKK: $(wc -l < $HISTDIR/data/train_expKKK.jsonl) lines (expRR data, lower LR)"

# expLLL: 8B model (Qwen3-8B) with same expRR config (model size comparison)
cp "$HISTDIR/data/train_expRR.jsonl" "$HISTDIR/data/train_expLLL.jsonl"
echo "expLLL: $(wc -l < $HISTDIR/data/train_expLLL.jsonl) lines (expRR data, 8B model)"

# Create configs
cat > "$HISTDIR/configs/expIII.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-14B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expIII
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
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expIII
do_train: true
report_to: none
EOF

cat > "$HISTDIR/configs/expJJJ.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-14B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expJJJ
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
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expJJJ
do_train: true
report_to: none
EOF

cat > "$HISTDIR/configs/expKKK.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-14B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expKKK
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
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expKKK
do_train: true
report_to: none
EOF

# expLLL config: Qwen3-8B model, same expRR data
cat > "$HISTDIR/configs/expLLL.yaml" << 'EOF'
model_name_or_path: /home/yinrong/models/Qwen3-8B
template: qwen3
trust_remote_code: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.0
lora_target: all
dataset: spc_r5_expLLL
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
output_dir: /home/yinrong/impl/2.1.1/history-route2.1.1/checkpoints/expLLL
do_train: true
report_to: none
EOF

# Register datasets
python3 -c "
import json
info = json.load(open('data/dataset_info.json'))
base_template = {'formatting': 'alpaca', 'columns': {'system': 'system', 'prompt': 'instruction', 'query': 'input', 'response': 'output'}}
new_ds = {
    'spc_r5_expIII': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expIII.jsonl', **base_template},
    'spc_r5_expJJJ': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expJJJ.jsonl', **base_template},
    'spc_r5_expKKK': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expKKK.jsonl', **base_template},
    'spc_r5_expLLL': {'file_name': '/home/yinrong/impl/2.1.1/history-route2.1.1/data/train_expLLL.jsonl', **base_template},
}
for k, v in new_ds.items():
    if k not in info:
        info[k] = v
        print(f'Added {k}')
json.dump(info, open('data/dataset_info.json', 'w'), ensure_ascii=False, indent=2)
"

echo "Batch 9 data and configs ready."
echo "To start Batch 9: nohup bash launch_batch9.sh > logs/batch9_launch.log 2>&1 &"
