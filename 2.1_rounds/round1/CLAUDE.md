# industrial-llm-demo 执行手册（Claude Code 操作指南）

本项目实现 route2.1 Demo 阶段：SPC 异常判断 SFT 微调，验证训练链路可行性。

---

## 快速索引

| 阶段 | 关键脚本 | 检验命令 |
|------|---------|---------|
| Phase 0 | 手动下载模型 | `ls /data/models/Qwen3-32B/*.safetensors \| wc -l` |
| Phase 1 | `tools/spc/generator.py` | `python tools/spc/generator.py --validate data/demo/train_N4.jsonl` |
| Phase 2 | 内置验证逻辑 | 通过率 ≥ 80% 才继续 |
| Phase 3 | `tools/train/deploy_vllm.py` + `tools/eval/spc_eval.py` | 检查 results/demo/baseline_*.json |
| Phase 4 | LLaMA-Factory + 上述脚本 | 检查 results/demo/sft_*.json |
| Phase 5 | `tools/eval/scaling_curve.py` + `tools/eval/generate_summary.py` | 查看 reports/ |

---

## Phase 0：环境准备

```bash
pip install -r requirements.txt

# 后台下载模型（约 90 分钟）
modelscope download --model Qwen/Qwen3-32B --local_dir /data/models/Qwen3-32B &
modelscope download --model Qwen/Qwen3-14B --local_dir /data/models/Qwen3-14B &

# 验证
ls /data/models/Qwen3-32B/*.safetensors | wc -l   # 预期约 16 个分片
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('/data/models/Qwen3-32B')"
```

---

## Phase 1：生成数据

```bash
python tools/spc/generator.py --n 100  --output data/demo/train_N1.jsonl  --seed 1
python tools/spc/generator.py --n 200  --output data/demo/train_N2.jsonl  --seed 2
python tools/spc/generator.py --n 300  --output data/demo/train_N3.jsonl  --seed 3
python tools/spc/generator.py --n 500  --output data/demo/train_N4.jsonl  --seed 4
python tools/spc/generator.py --n 200  --output data/demo/test.jsonl      --seed 99

# 验证格式和覆盖率
python -c "import json; [json.loads(l) for l in open('data/demo/train_N4.jsonl')]"
python tools/spc/generator.py --validate data/demo/train_N4.jsonl
```

**检验标准**：rule1~rule8 均有覆盖，正常样本约 20%。

---

## Phase 2：质量关卡

```bash
# 从 N4 随机抽 50 条，自动验证 ground_truth 与规则引擎输出一致性 + 处置建议质量
python tools/eval/quality_gate.py --file data/demo/train_N4.jsonl --n 50
```

> ⚠ 若通过率 < 80%，分析失败样本后修复 `tools/spc/generator.py` 并重新执行 Phase 1。

---

## Phase 3：基座基线

```bash
python tools/train/deploy_vllm.py --model /data/models/Qwen3-32B --port 8010
python tools/train/deploy_vllm.py --model /data/models/Qwen3-14B --port 8020

python tools/eval/spc_eval.py --model_url http://localhost:8010 \
  --model_name qwen3-32b-base --test data/demo/test.jsonl \
  --output results/demo/baseline_32b.json

python tools/eval/spc_eval.py --model_url http://localhost:8020 \
  --model_name qwen3-14b-base --test data/demo/test.jsonl \
  --output results/demo/baseline_14b.json
```

**检验标准**：rule_f1 ∈ [0.05, 0.30]，has_reasoning ≈ 0。

---

## Phase 4：N1~N4 训练+评测循环

以 N1 为例（N2~N4 替换 NODE/N/PORT）：

```bash
NODE=N1; N=100; PORT=8011

python tools/train/register_dataset.py --name spc_${NODE} --file data/demo/train_${NODE}.jsonl

llamafactory-cli train configs/lora_base.yaml \
  --dataset spc_${NODE} \
  --output_dir /data/checkpoints/demo-${NODE}

python tools/train/merge_adapter.py \
  --base /data/models/Qwen3-32B \
  --adapter /data/checkpoints/demo-${NODE} \
  --output /data/checkpoints/demo-${NODE}-merged

python tools/train/deploy_vllm.py \
  --model /data/checkpoints/demo-${NODE}-merged --port ${PORT}

python tools/eval/spc_eval.py \
  --model_url http://localhost:${PORT} \
  --model_name qwen3-32b-sft-${NODE} \
  --test data/demo/test.jsonl \
  --output results/demo/sft_${NODE}.json

python tools/eval/scaling_curve.py \
  --results_dir results/demo/ --output reports/scaling_curve.png
```

**N1 完成后**：读 `results/demo/sft_N1.json` 中 `per_rule_recall`，
找出 recall 最低的 3 条规则，构造 20 条针对性黄金样本追加到 `data/demo/train_N4.jsonl`。

**N4 还需**运行 Qwen3-14B 对比（端口 8024，model_name `qwen3-14b-sft-N4`）。

---

## Phase 5：最终报告

```bash
python tools/eval/scaling_curve.py \
  --results_dir results/demo/ --target_n 100000 \
  --output reports/scaling_curve.png

python tools/eval/generate_summary.py \
  --results_dir results/demo/ \
  --output reports/demo_summary.md
```

---

## 关键文件说明

| 文件 | 作用 |
|------|------|
| `tools/spc/rules.py` | Nelson 规则引擎 + CPK，generator 和 eval 共用 |
| `tools/spc/generator.py` | 生成 SPC 训练样本（含 \<think\> 推理链） |
| `tools/spc/formatter.py` | 结构化答案 → 自然语言（模板 + 可选 LLM 润色） |
| `tools/eval/extractor.py` | 从模型输出提取 violations 和 CPK |
| `tools/eval/spc_eval.py` | 批量评测，输出 JSON 指标 |
| `tools/eval/scaling_curve.py` | 幂律拟合 + 6 子图 |
| `tools/eval/generate_summary.py` | 生成 demo_summary.md |
| `tools/train/register_dataset.py` | 向 LLaMA-Factory 注册数据集 |
| `tools/train/merge_adapter.py` | llamafactory-cli export 封装 |
| `tools/train/deploy_vllm.py` | 启动 vLLM，健康检查后返回 URL |
| `configs/lora_base.yaml` | LoRA 超参（4 节点共用） |
| `configs/models.yaml` | 模型路径和端口分配 |
| `configs/demo.yaml` | 节点规模、seed、质量关卡阈值 |

---

## 常见问题

**Q: rule_f1 基座 > 0.30？**
检查 `tools/eval/spc_eval.py` 的 `call_model()` 是否在 prompt 中泄露了答案格式。

**Q: N1 rule_f1 < 0.35？**
检查 `data/demo/train_N1.jsonl` 的 output 字段是否有 `<think>` 块；
确认 LLaMA-Factory 使用的是 `qwen` template。

**Q: 质量关卡通过率 < 80%？**
用 `python tools/spc/generator.py --validate data/demo/train_N4.jsonl` 查看
哪些规则注入失败，再调整 `tools/spc/generator.py` 的 `_inject_rule*` 函数。
