"""
R3-C：GRPO 强化训练 (trl >= 1.0.0)
基于 R3-AB-v2-merged checkpoint，用 rule_detection_f1 作为奖励函数

运行命令（2~4卡）：
  CUDA_VISIBLE_DEVICES=0,1 python round3/tools/train/train_grpo_trl.py \
      --model /home/yinrong/post-train/impl2.1/round3/checkpoints/R3-AB-v2-merged \
      --train_data round3/data/train_R3-AB-v2.jsonl \
      --output_dir round3/checkpoints/R3-C-grpo \
      --num_train_epochs 1 \
      --per_device_batch_size 1 \
      --num_generations 4

从根目录运行：cd /home/yinrong/post-train/impl2.1
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

from common.tools.eval.extractor import extract_violations, extract_cpk


def rule_f1_reward(
    completions: list[list[dict]],
    ground_truth: list[dict],
    **kwargs,
) -> list[float]:
    """
    GRPO 奖励函数：对每个 completion 计算 rule_detection_f1 + cpk_bonus。

    trl 1.0+ 接口规范：
        completions: list[list[dict]]  — 每个 completion 是 [{"role":"assistant","content":"..."}]
        ground_truth: list[dict]  — 来自数据集列 ground_truth，每个是 {"violations":[...], "cpk":...}

    返回: list[float]  — 每个 completion 的奖励值
    """
    rewards = []
    for gt, comp in zip(ground_truth, completions):
        gt_violations = set(gt.get("violations", []))
        gt_cpk = gt.get("cpk")
        # comp is a list with one message dict
        text = comp[0]["content"] if isinstance(comp, list) else comp

        pred_violations = set(extract_violations(text))
        pred_cpk = extract_cpk(text)

        tp = len(gt_violations & pred_violations)
        fp = len(pred_violations - gt_violations)
        fn = len(gt_violations - pred_violations)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        cpk_bonus = 0.0
        if pred_cpk is not None and gt_cpk is not None:
            cpk_bonus = 0.05 if abs(pred_cpk - gt_cpk) < 0.1 else 0.0

        rewards.append(f1 + cpk_bonus)
    return rewards


def load_train_dataset(train_path: str, tokenizer) -> Dataset:
    """将 SFT 格式 jsonl 转换为 GRPO 训练格式（只保留 prompt，不含 answer）。"""
    records = []
    with open(train_path) as f:
        for line in f:
            d = json.loads(line)
            system = d.get("system", "你是一名 SPC 工程师。")
            instruction = d.get("instruction", "")
            input_text = d.get("input", "")
            gt = d.get("ground_truth", {})

            prompt_text = f"{instruction}\n\n{input_text}" if input_text else instruction

            # Tokenize into chat format for Qwen3
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt_text},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            records.append({
                "prompt": prompt,
                "ground_truth": gt,  # passed to reward function
            })

    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base or SFT merged checkpoint")
    parser.add_argument("--train_data", required=True, help="JSONL training file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Group size G in GRPO (completions per prompt)")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # PEFT/LoRA 配置：仅训练 LoRA 参数（~500MB），大幅降低显存需求
    # 使用 PEFT 时，GRPOTrainer 自动以冻结参数作为参考策略（无需独立参考模型副本）
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_train_dataset(args.train_data, tokenizer)
    print(f"Dataset size: {len(dataset)}")

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        gradient_accumulation_steps=4,
        bf16=True,
        report_to="none",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        dataloader_num_workers=0,
        temperature=0.9,
        beta=0.04,  # KL coefficient (trl 1.x 的参数名)
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=rule_f1_reward,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting GRPO training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
