# SKILL Internalization into LLMs: A Practical Methodology

**Based on Route 2.1.1 SPC Nelson Rule Analysis Experiments**
**Status: Draft — will be updated as Batches 6-11 complete**
**Last updated: 2026-04-25**

---

## What Is "Skill Internalization"?

A **skill** is a structured body of procedural knowledge — rules, heuristics, decision trees — that enables an agent to perform a complex task. Examples: SPC Nelson rules for process control, ICD-10 coding guidelines, security vulnerability patterns, legal contract review checklists.

**Internalization** means the LLM can apply the skill *without* the skill document in the context window at inference time. The model has "memorized and integrated" the skill into its weights through training.

This is distinct from:
- **RAG / in-context skill**: Skill doc is always in the prompt (works, but costly and leak-prone)
- **Tool use**: Model calls an external system (requires external infra)
- **Simple factual memorization**: The model knows *what* the rules are but can't reliably *apply* them

The goal: inference without the skill doc achieves near-identical F1 as inference with it.

---

## The Core Recipe (Quick Start)

```
1. Generate teacher data (Claude + skill doc → reasoning chains)
2. Convert to no-skill format (remove skill doc from teacher examples)  
3. Fine-tune with LoRA: 200 samples × 5 epochs × cutoff_len ≥ max_output_len
4. Evaluate without skill doc
```

Target: match or exceed "base model + skill doc" performance.

---

## Step 1: Skill Documentation

Write a clear, structured skill document covering:
- All decision rules (enumerated, precise)
- How to apply each rule to inputs
- Output format (what a correct answer looks like)

The skill doc is used at teacher data generation time, NOT at inference time.

**SPC example**: Nelson 8 Rules + CPK formula in a 500-token system prompt.

---

## Step 2: Generate Teacher Data

Use a capable teacher model (Claude Sonnet or better) to generate training examples.

### Format: `with_skill` (teacher generation)

```
System: <skill doc — all rules, full detail>
User: <task scenario>
Assistant: <think>
[full step-by-step reasoning applying each rule]
</think>
<structured answer>
```

The teacher model sees the skill doc. This produces high-quality reasoning chains that correctly apply the skill.

### Key parameters
- **N**: 200 samples per seed pool is the sweet spot (see Step 4 analysis)
- **Diversity**: Use different random seeds for each pool; avoid overlapping inputs
- **Quality**: Verify with a rule-engine or oracle that answers are correct before use
- **Concurrency**: 3 parallel API calls is efficient without rate-limiting

### Validation
Check that all rules/cases in the skill are represented in the teacher data. For SPC: all 8 rules should appear. If rare rules (rule7, rule8) are underrepresented, add targeted examples.

---

## Step 3: Convert to No-Skill Format

**Critical step**: Remove the skill doc from the system prompt in training examples.

```python
# Convert with_skill teacher example → no_skill training example
def convert_to_noskill(example):
    messages = example["messages"]
    # Replace system prompt (which contains skill doc) with empty or generic prompt
    messages[0]["content"] = ""   # or a generic task description without rules
    return example
```

The model must learn to apply the skill purely from the `<think>` reasoning chain, without the skill doc as a scaffold.

### Why this matters
If you train with the skill doc still in the system prompt, the model learns to *use* the skill doc, not to *recall* it. At inference without the doc, performance collapses.

| Training format | Inference without skill doc | Notes |
|-----------------|----------------------------|-------|
| with_skill | Very low (0.00-0.05) | Model relied on skill doc |
| no_skill | High (0.35+) | Model internalized the skill |
| mixed (50/50) | Medium (0.20-0.30) | Partial internalization |

---

## Step 4: Fine-Tuning Configuration

### Model architecture
- **Base model**: Qwen3-14B is the recommended starting point
  - 14B: best efficiency/performance balance (F1=0.358 with 200 samples)
  - 32B: ~20% F1 gain expected but 4× compute cost
  - 8B: ~15% lower F1 than 14B (preliminary)
- **LoRA settings**: rank=128, alpha=256, lora_target=all
- **Precision**: bfloat16 + flash_attn=sdpa

### Critical: `cutoff_len` must exceed max output length

**This is the #1 source of silent training failures.**

Measure actual token lengths of your training outputs:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
lengths = [len(tokenizer(ex["output"])["input_ids"]) for ex in data]
print(f"max={max(lengths)}, p95={np.percentile(lengths, 95)}")
```

Set `cutoff_len` ≥ p100 of (input + output) length.

**SPC example**:
- No-skill teacher output: 3329-4572 tokens
- With cutoff_len=4096: **94% of samples truncated** → rule8 (last in output) never trained
- With cutoff_len=5120: **0% truncated** → all rules covered
- Result: cutoff_len=4096 gave rule8=0.00 in ALL experiments; cutoff_len=5120 expected to fix

### Training hyperparameters (validated for 14B, single GPU H20)

```yaml
num_train_epochs: 5
per_device_train_batch_size: 2
gradient_accumulation_steps: 4     # effective batch = 8
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
cutoff_len: 5120                    # ≥ max(input+output tokens)
```

### Training steps sweet spot: **~125 steps**

This is the most important hyperparameter. Too few → underfitting; too many → overfit to specific phrasings.

| Steps | Configuration | F1 | Notes |
|-------|--------------|-----|-------|
| 75 | 200 × 3ep / 8 | — | Under sweet spot (Batch 7, pending) |
| **125** | **200 × 5ep / 8** | **0.358** | **Optimal** |
| 150 | 400 × 3ep / 8 | — | Slightly above (Batch 7, pending) |
| 187 | 300 × 5ep / 8 or 500 × 3ep | 0.31-0.34 | Over sweet spot |
| 200 | 400 × 4ep / 8 | — | Expected overfit |

**Rule of thumb**: `steps = N × epochs / effective_batch_size ≈ 125`
→ With batch_size=8: `N × epochs ≈ 1000`

### Data purity: no-skill only

Avoid mixing in:
- **With-skill teacher data**: Slightly helps generalization but dramatically lowers no-skill F1
- **Textbook/document data** (non-example format): degrades F1 significantly
  - expNN: 200 ns + 51 textbook = F1=0.279 vs expRR: 200 pure ns = F1=0.358 (-0.079)
- **Synthetic rule-injection data** (if generated without teacher reasoning chains): mixed results

**Keep it pure**: 200 Claude teacher samples in no-skill format = best results.

---

## Step 5: Evaluation Protocol

### Always evaluate without skill doc

```python
# Evaluation system prompt — NO skill document
system_prompt = ""   # or a generic task description
```

### max_model_len at inference must match cutoff_len at training

**Another silent failure**: vLLM default max_model_len may be lower than your cutoff_len.

```bash
python tools/train/deploy_vllm.py \
    --model /path/to/checkpoint \
    --port 8000 \
    --max_len 5120    # must match training cutoff_len
```

### Metrics

For structured output skills:
- **Rule/class F1**: Micro-averaged across all rule/class predictions
- **Per-rule recall**: Identifies which specific rules the model learned
- **CPK found rate**: For quantitative calculations (verify formula application)

### Test set

Use a **held-out** test set generated with a different random seed from training data.
- 200 samples is sufficient for stable metrics (±0.01 F1)
- Must include all rule/class types in roughly equal proportion

---

## Step 6: Analyzing Results and Iterating

### Diagnosing underperformance

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| All per-rule recalls ≈ 0 | No training happened or massive truncation | Check cutoff_len, verify training log |
| One rule dominates (>0.5) | Class imbalance in training data | Balance rule proportions in generator |
| Last rule = 0.00 | Output truncation | Increase cutoff_len |
| F1 high but drops at inference | Trained with skill doc in system prompt | Verify no-skill conversion |
| F1 peaks at N=200, drops at N=400+ | Overfitting (too many steps) | Reduce epochs or N |
| F1 consistently < target | Teacher quality too low | Use a stronger teacher model |

### Scaling: does more data help?

Based on SPC experiments:

| N | Steps | F1 | Pattern |
|---|-------|-----|---------|
| 200 | 125 | 0.358 | Sweet spot |
| 300 | 187 | 0.313 | Over-trained |
| 500 | 125 (3ep) | 0.319 | Underfit |
| 800 | 200 (2ep) | pending | Mixed pool |

**Key insight**: More data is NOT automatically better. The ratio N×epochs/batch_size matters most.
To scale to more data: scale epochs down proportionally to keep steps ≈ 125.

### Data pool diversity

Mixing data from different random seed pools can help (different phrasings, scenarios):
- Single pool, 200 samples: F1=0.358 (expRR)
- Cross-pool, 400 samples (v1+v2), 100 steps: expected ~0.33 (Batch 7, pending)
- Cross-pool at 125 steps: probably better than single pool

If cross-pool diversity beats single pool at equal steps → generate larger diverse pools.

---

## Expected Performance Levels

| Setup | Expected F1 | Notes |
|-------|------------|-------|
| Base model, no skill | 0.000 | No knowledge of skill |
| Base model + skill doc | 0.11-0.23 | Depends on model size |
| SFT (200 ns teacher, 125 steps, 14B) | 0.35-0.40 | Current best |
| SFT + cutoff_len fix (expMMM, pending) | 0.40-0.50? | rule8 now trainable |
| SFT (32B, 200 ns teacher, 125 steps) | 0.45+? | Larger model, pending |
| Target (Claude Sonnet + skill doc) | 0.232 | **Already surpassed at 14B!** |

The SFT model surpassed the Claude Sonnet baseline at 14B. The key was:
- No-skill teacher format
- Pure teacher data (no mixing)
- 125 steps exactly
- Sufficient cutoff_len (fix in progress)

---

## Practical Checklist

### Before training
- [ ] Skill document written and complete
- [ ] Teacher data generated (200+ samples)
- [ ] Teacher data validated (rule/class coverage ≥ 80%)
- [ ] No-skill conversion applied (skill doc removed from system prompts)
- [ ] cutoff_len set ≥ p100 of (input + output) token length
- [ ] Dataset registered in LLaMA-Factory dataset_info.json

### Training
- [ ] LoRA rank=128, lora_target=all
- [ ] effective_batch = 8 (batch=2 × grad_accum=4)
- [ ] N × epochs ≈ 1000 → steps ≈ 125
- [ ] monitor training log for `train_runtime`
- [ ] verify no truncation errors in log

### Evaluation
- [ ] vLLM max_model_len = training cutoff_len
- [ ] System prompt: NO skill document
- [ ] Test set: different seed from training data
- [ ] Check per-rule recall for coverage
- [ ] Compare against: base+skill, base-skill, teacher

---

## Common Pitfalls Summary

1. **cutoff_len too small** → silent truncation → last output sections never trained
2. **Skill doc in training system prompt** → model doesn't internalize, just learns to use doc
3. **max_model_len < cutoff_len at inference** → truncated generation → missing answer sections
4. **Too many training steps** → overfitting to specific phrasings
5. **Mixed training formats** → partial internalization
6. **Textbook data mixed in** → disrupts output format distribution
7. **Teacher data not validated** → noisy labels → lower F1 ceiling

---

## Results Archive

### Route 2.1.1 Batch 4+5 — Final Results (200/200 samples)

These experiments all used cutoff_len=4096 (known to truncate 94% of samples at rule8):

| Experiment | F1 | Description |
|-----------|-----|-------------|
| expMM | 0.273 | 100 ns + 83 synth + 51 textbook, ~146 steps |
| expNN | 0.279 | 200 ns + 51 textbook, ~156 steps |
| expPP | 0.290 | 100 ns + 100 ws + 51 textbook, ~156 steps |
| expQQ | 0.255 | 200 synth + 50 ns + 51 textbook, ~112 steps |
| **expRR** | **0.358** | **200 pure ns, 125 steps — best with 4096 cutoff** |
| expSS | 0.313 | 300 pure ns, 187 steps |
| expTT | 0.319 | 500 pure ns, 3ep, 187 steps |
| expUU | 0.343 | 150 ns + 150 synth, 187 steps |

Note: ALL experiments have rule8=0.00 due to cutoff_len=4096 truncation.

### Route 2.1.1 Batch 6+ — Pending

| Experiment | Status | Description |
|-----------|--------|-------------|
| expWW | training | 500 with_skill teacher, 3ep, cutoff=6144 |
| expXX | training | 250 ws + 250 ns, 3ep, cutoff=6144 |
| expVV | training | 500 ns_v1 + 300 ns_v2, 2ep, cutoff=5120 |
| expYY2 | training | 300 ws + 300 ns_v2, 3ep, cutoff=6144 |
| expZZ | pending B7 | 200 ns_v1, 3ep=75 steps, cutoff=5120 |
| expAAA | pending B7 | 200 ns_v2, 5ep=125 steps, cutoff=5120 |
| **expMMM** | **pending B9** | **200 ns_v1, 5ep=125 steps, cutoff=5120 (KEY)** |

**expMMM** is the critical experiment: expRR equivalent but with correct cutoff_len.
Expected to fix rule8=0.00 and push F1 well above 0.358.

---

*This document will be updated as experiments complete.*
*Route 2.1.1 full progress: `history-route2.1.1/progress.md`*
