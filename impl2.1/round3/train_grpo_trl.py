#!/usr/bin/env python3
"""R3-C GRPO 训练脚本（trl 1.3.0）

修正了原 LlamaFactory 方案的两个问题：
1. max_new_tokens=2048 → max_completion_length=5500（覆盖 p99+20%）
2. LlamaFactory 0.9.3 import 损坏 → 改用 trl GRPOTrainer

运行方式（4卡）：
    cd /home/yinrong/post-train/impl2.1
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 round3/train_grpo_trl.py
"""
import sys
import os
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common"))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer

from tools.eval.extractor import extract_violations, extract_cpk

# ── 路径 ──────────────────────────────────────────────────────────────
BASE_MODEL   = ROOT / "round2/history-route2.1.1/checkpoints/expYYY-merged"
DATA_PATH    = ROOT / "round3/data/train_R3-grpo-200.jsonl"   # 171样本分层子集，加速调试
OUTPUT_DIR   = ROOT / "round3/checkpoints/R3-C2"
LOG_DIR      = ROOT / "round3/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_COMPLETION_LENGTH = 5000   # 覆盖 p99(4591)，比 5500 省约 10% 生成时间
NUM_GENERATIONS       = 2      # 单卡模式：generation_batch_size=2 需整除 num_generations

# ── 奖励函数 ──────────────────────────────────────────────────────────
def compute_f1(pred: list, gt: list) -> float:
    pred_set, gt_set = set(pred), set(gt)
    if not pred_set and not gt_set:
        return 1.0
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def reward_fn(prompts, completions, ground_truth=None, **kwargs):
    """trl GRPOTrainer 调用接口：(prompts, completions, **dataset_columns)

    ground_truth 字段由 dataset 自动透传（list[dict]，每项含 violations/cpk）。

    R3-C2 reward 设计：
      - 主奖励：rule detection F1（0~1）
      - 空预测惩罚：violations=[] 但 GT 有违规 → -0.3（封堵"始终空预测"捷径）
      - CPK 精度奖励：|pred_cpk - gt_cpk| < 0.1 → +0.1
      - beta=0.05（防止过度偏离 SFT 基础）

    注：移除了 end_think 检查。Qwen3 在此 prompt 格式下固定输出 <think>\n\n</think>
    （0字符推理链），真正的推理写在 </think> 之后的答案区域——提取器仍能正确
    解析 violations 和 cpk，reward 能正常反映任务质量。
    """
    if ground_truth is None:
        return [0.0] * len(completions)

    rewards = []
    for completion, gt in zip(completions, ground_truth):
        if isinstance(gt, str):
            gt = json.loads(gt)
        gt_violations = gt.get("violations", [])
        gt_cpk = gt.get("cpk")

        pred_violations = extract_violations(completion)
        pred_cpk        = extract_cpk(completion)

        rule_f1 = compute_f1(pred_violations, gt_violations)

        # 空预测惩罚：封堵"始终预测空违规"捷径
        if len(pred_violations) == 0 and len(gt_violations) > 0:
            rule_f1 -= 0.3

        cpk_bonus = 0.0
        if pred_cpk is not None and gt_cpk is not None:
            cpk_bonus = 0.1 if abs(pred_cpk - gt_cpk) < 0.1 else 0.05

        rewards.append(round(max(-0.3, rule_f1 + cpk_bonus), 4))

    # 诊断打印：每批次输出 reward 摘要
    print(f"[REWARD] n={len(rewards)} mean={sum(rewards)/len(rewards):.4f} "
          f"vals={rewards}", flush=True)
    return rewards


# ── 数据集 ────────────────────────────────────────────────────────────
NO_SKILL_SYSTEM = "你是一名 SPC 工程师。"   # GRPO 训练不携带技能文档，测量内化程度


def load_dataset(tokenizer):
    records = [json.loads(l) for l in open(DATA_PATH)]
    prompts, ground_truths = [], []
    for r in records:
        # 关键：移除技能文档，使用 no_skill system prompt
        # （若携带文档，模型在 GRPO 阶段仍依赖文档，不测量内化）
        system  = NO_SKILL_SYSTEM
        user    = r.get("instruction", "") + "\n" + r.get("input", "")
        messages = [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user.strip()},
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt_str)
        # ground_truth 需序列化为字符串（trl 会把数据集列按原样传入 kwargs）
        ground_truths.append(json.dumps(r.get("ground_truth", {}), ensure_ascii=False))

    return Dataset.from_dict({"prompt": prompts, "ground_truth": ground_truths})


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    print(f"[R3-C2 GRPO] BASE_MODEL: {BASE_MODEL}")
    print(f"[R3-C2 GRPO] max_completion_length: {MAX_COMPLETION_LENGTH}")
    print(f"[R3-C2 GRPO] num_generations: {NUM_GENERATIONS}")
    print(f"[R3-C2 GRPO] reward fix: no-think=0, empty-pred-penalty=-0.3, beta=0.05")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), trust_remote_code=True)
    tokenizer.padding_side = "left"   # GRPO 生成需要 left-padding

    # Dataset
    dataset = load_dataset(tokenizer)
    print(f"[R3-C GRPO] 训练样本数: {len(dataset)}")

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # LoRA 配置（传给 GRPOTrainer，不手动 get_peft_model）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules="all-linear",
    )

    # GRPOConfig（trl 1.3.0：参数名 args，非 config）
    grpo_args = GRPOConfig(
        # ── 核心修复 ──────────────────────────────────
        # 原始失败原因 1：max_completion_length=2048 远小于输出均值 4191
        # → 所有候选被截断 → reward=0 → 策略无法更新
        max_completion_length=MAX_COMPLETION_LENGTH,   # 5500 覆盖 p99+20%
        num_generations=NUM_GENERATIONS,               # G=8 候选/prompt
        # ── 原始失败原因 2：输入携带技能文档（with_skill 格式）──
        # → 模型依赖文档，GRPO 阶段不测量内化；已在 load_dataset 中修复
        # ──────────────────────────────────────────────

        # 训练超参
        num_train_epochs=3,            # 样本少，多跑几轮
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,

        # KL 惩罚系数（R3-C2 提高至 0.05，防止偏离 SFT 基础太远）
        beta=0.05,

        # 日志 & 保存
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,

        # 精度（DeepSpeed 不可用，标准 torchrun DDP）
        bf16=True,
        dataloader_num_workers=2,

        output_dir=str(OUTPUT_DIR),
        report_to="none",
        use_vllm=False,   # vLLM 0.19.1 与 trl 1.3.0 有兼容警告，用标准生成
    )

    # Trainer（trl 1.3.0：参数名 args 非 config；peft_config 自动挂 LoRA）
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        peft_config=lora_config,
    )

    print("[R3-C2 GRPO] 开始训练...")
    trainer.train()

    print(f"[R3-C2 GRPO] 训练完成，保存至 {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
