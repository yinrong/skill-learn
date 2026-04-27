---
library_name: peft
license: other
base_model: /home/yinrong/models/Qwen3-32B
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: expHHH
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# expHHH

This model is a fine-tuned version of [/home/yinrong/models/Qwen3-32B](https://huggingface.co//home/yinrong/models/Qwen3-32B) on the spc_r5_expCCC dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 3

### Training results



### Framework versions

- PEFT 0.15.2
- Transformers 4.57.6
- Pytorch 2.10.0+cu126
- Datasets 3.6.0
- Tokenizers 0.22.2