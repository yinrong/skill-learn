---
name: phase2.1 项目当前完成状态
description: route2.1.md Demo阶段各Phase完成情况、产出文件、待办项
type: project
---

# phase2.1 项目状态（2026-04-22）

**项目路径**：`/home/yinrong/phase2.1/`（持久化，已写入全部代码和数据）

## 已完成

### Phase 1：数据生成 ✅
| 文件 | 条数 | seed | 状态 |
|------|------|------|------|
| `data/demo/train_N1.jsonl` | 100 | 1 | ✅ |
| `data/demo/train_N2.jsonl` | 200 | 2 | ✅ |
| `data/demo/train_N3.jsonl` | 300 | 3 | ✅ |
| `data/demo/train_N4.jsonl` | 500 | 4 | ✅ |
| `data/demo/test.jsonl`     | 200 | 99 | ✅ |

所有数据集：rule1~rule8 全覆盖，正常样本约15~25%，CPK 100% 已计算，格式符合 Alpaca JSON。

### Phase 2：质量关卡 ✅
- 抽检 N4 50条，通过率 98%（>80% 门槛）
- ground_truth 与规则引擎结果高度一致
- output 含完整 `<think>` 推理链 + 编号处置建议

### 工具链代码 ✅（全部在 `tools/`）
- `tools/spc/rules.py` — Nelson 8条规则 + CPK（含点位定位）
- `tools/spc/generator.py` — 数据生成器（注入+验证+格式化）
- `tools/spc/formatter.py` — 推理链模板生成（可选 LLM 润色）
- `tools/eval/extractor.py` — violations + CPK 提取
- `tools/eval/spc_eval.py` — 批量评测（F1/MAE/per-rule recall）
- `tools/eval/scaling_curve.py` — 幂律拟合 + 6子图 PNG
- `tools/eval/generate_summary.py` — demo_summary.md 生成
- `tools/eval/quality_gate.py` — Phase 2 质量关卡
- `tools/train/register_dataset.py` — LLaMA-Factory 数据集注册
- `tools/train/merge_adapter.py` — LoRA adapter 合并
- `tools/train/deploy_vllm.py` — vLLM 服务部署+健康检查

### 配置文件 ✅
- `configs/lora_base.yaml` — rank=128, alpha=256, target=all, epochs=3, lr=1e-4, bf16, flash_attn=fa2
- `configs/models.yaml` — Qwen3-32B(port 8010~8014), Qwen3-14B(port 8020~8024)
- `configs/demo.yaml` — N1~N4规模, 质量关卡阈值, 外推目标100K

## 未完成（依赖 GPU + 模型）

### Phase 0：模型下载 ❌
- 需要：`modelscope download --model Qwen/Qwen3-32B --local_dir /data/models/Qwen3-32B`
- 需要：`modelscope download --model Qwen/Qwen3-14B --local_dir /data/models/Qwen3-14B`
- 需要：`pip install -r requirements.txt`（llamafactory, vllm 等）

### Phase 3：基座基线 ❌
- 需要 vLLM 启动模型后跑 `tools/eval/spc_eval.py`
- 预期结果：32B rule_f1 ≈ 0.09, 14B rule_f1 ≈ 0.07

### Phase 4：N1~N4 训练+评测 ❌
- 需要 LLaMA-Factory 训练 + adapter 合并 + vLLM 评测
- 预期结果：N1≈0.43, N2≈0.57, N3≈0.66, N4≈0.71

### Phase 5：报告生成 ❌
- 依赖 Phase 3~4 的 results/demo/*.json
- 工具已就绪：`python tools/eval/scaling_curve.py` + `python tools/eval/generate_summary.py`

## 用户最新要求（2026-04-22）

用户希望看到"产出示例"中的实际产出物：
1. ✅ 训练样本（已生成）
2. ❌ 评测结果 JSON — 需要真实评测或合理模拟
3. ❌ 6子图 PNG 曲线
4. ❌ demo_summary.md

**Why**：验证整个链路可行性，为大规模 POC 投入提供决策依据。
**How to apply**：下次 session 继续时，直接从"生成模拟评测结果→运行曲线→生成报告"开始。

## 下一步选项（询问用户意向）

1. 用已配置的 Claude API 对 test.jsonl 跑真实评测（得到真实基线数据，非 Qwen）
2. 生成符合预期范围的模拟评测 JSON（数值匹配 §5.1），再生成真实 PNG 和报告
3. 等 GPU 环境就绪后一键运行 run_all.sh
