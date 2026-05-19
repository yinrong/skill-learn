"""
训练数据增强：model_id 多样化

目的：防止模型记住"UPH→model_id=12198"这样的固定映射，强迫它每次都读
get_object_type_metrics 的返回值来确定 model_id。

策略：
  对每条包含 get_idi_model_data 的训练样本，生成 N 个变体：
  - 把 get_object_type_metrics 响应里的 modelId 替换成随机假 ID
  - 把 get_idi_model_data 调用里的 model_id 参数替换成同一假 ID
  - 其他内容（数据值、最终答案）保持不变
  这样模型无法记住"问这类问题→用 12198"，只能学会"用 metrics 返回的那个 ID"

用法：
  python round4/scripts/augment_trajectories.py \
    --input round4/data/train_ns_all.jsonl \
    --output round4/data/train_ns_augmented.jsonl \
    --variants 5 \
    --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
import copy
from pathlib import Path
from typing import Any


def extract_model_ids_from_metrics_response(content: str) -> list[int]:
    """从 get_object_type_metrics 的响应中提取所有 modelId 值。"""
    try:
        resp = json.loads(content)
        data = resp.get("data", [])
        ids = []
        if isinstance(data, list):
            for item in data:
                for binding in item.get("modelBindings", []):
                    mid = binding.get("modelId")
                    if mid is not None:
                        ids.append(int(mid))
        return ids
    except Exception:
        # 对截断或格式不正确的响应，用正则兜底
        matches = re.findall(r'"modelId"\s*:\s*(\d+)', content)
        return [int(m) for m in matches]


def replace_model_ids_in_metrics_response(content: str, mapping: dict[int, int]) -> str:
    """把 metrics 响应里的 modelId 值按 mapping 替换。"""
    result = content
    for old_id, new_id in mapping.items():
        # 替换 "modelId": 12198 → "modelId": 99999
        result = re.sub(
            rf'"modelId"\s*:\s*{old_id}\b',
            f'"modelId": {new_id}',
            result,
        )
    return result


def replace_model_id_in_idi_args(arguments_str: str, mapping: dict[int, int]) -> str:
    """把 get_idi_model_data 参数里的 model_id 按 mapping 替换。"""
    try:
        args = json.loads(arguments_str)
        old_id = args.get("model_id")
        if old_id is not None and int(old_id) in mapping:
            args["model_id"] = mapping[int(old_id)]
            return json.dumps(args, ensure_ascii=False)
    except Exception:
        pass
    # 兜底：正则替换
    for old_id, new_id in mapping.items():
        arguments_str = re.sub(
            rf'"model_id"\s*:\s*{old_id}\b',
            f'"model_id": {new_id}',
            arguments_str,
        )
    return arguments_str


def generate_fake_model_id(existing_ids: set[int], rng: random.Random) -> int:
    """生成一个不与现有 ID 冲突的假 model_id（10000-99999 范围）。"""
    while True:
        fake = rng.randint(10000, 99999)
        if fake not in existing_ids:
            existing_ids.add(fake)
            return fake


def build_id_mapping(
    sample: dict,
    rng: random.Random,
    used_ids: set[int],
) -> dict[int, int]:
    """
    扫描样本，找出所有 get_idi_model_data 调用里的 model_id，
    为每个 ID 生成一个假 ID，返回 {原始ID: 假ID} 映射。
    """
    real_ids: set[int] = set()
    msgs = sample.get("messages", [])
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc["function"]["name"] == "get_idi_model_data":
                    try:
                        args = json.loads(tc["function"].get("arguments", "{}"))
                        mid = args.get("model_id")
                        if mid is not None:
                            real_ids.add(int(mid))
                    except Exception:
                        pass
    if not real_ids:
        return {}
    return {rid: generate_fake_model_id(used_ids, rng) for rid in real_ids}


def inject_model_id_into_metrics_response(content: str, old_id: int, new_id: int) -> str:
    """
    在 get_object_type_metrics 的响应里注入新的 model_id 绑定。

    策略：
    1. 如果响应里已有 modelId=old_id，直接替换为 new_id
    2. 如果没有（典型情况：metrics 里的 ID 和 idi 里的 ID 本来就不同），
       则在第一个 modelBindings 数组里追加一个新绑定
    """
    # 先尝试直接替换已有的 old_id
    if f'"modelId": {old_id}' in content or f'"modelId":{old_id}' in content:
        return replace_model_ids_in_metrics_response(content, {old_id: new_id})

    # old_id 不在响应里：注入一个新绑定
    try:
        resp = json.loads(content)
        data = resp.get("data", [])
        if isinstance(data, list) and data:
            # 在第一个对象的 modelBindings 里追加
            first_item = data[0]
            bindings = first_item.get("modelBindings", [])
            bindings.append({
                "modelId": new_id,
                "modelName": f"核心指标模型_{new_id}",
                "joinKey": "line_code",
                "metrics": [
                    {"fieldName": "value", "desc": "指标值", "fieldIsValidated": True}
                ],
                "dimension": [
                    {"fieldName": "line_code", "desc": "线体编码", "type": "string"},
                    {"fieldName": "shift_date", "desc": "日期", "type": "date"},
                ],
                "params": [
                    {"name": "start_shift_date", "type": "date", "optional": False},
                    {"name": "end_shift_date", "type": "date", "optional": False},
                    {"name": "line", "type": "string", "optional": True},
                ],
            })
            first_item["modelBindings"] = bindings
            resp["data"] = data
            return json.dumps(resp, ensure_ascii=False)
    except Exception:
        pass
    # JSON 截断/无法解析：用文本追加（兜底，通常不走到这里）
    return content


def augment_sample(sample: dict, mapping: dict[int, int]) -> dict:
    """
    用给定的 ID 映射创建样本变体：
    1. 修改紧接 get_idi_model_data 之前的最近一个 get_object_type_metrics 响应，
       注入新的 model_id 绑定，建立因果链
    2. 修改 get_idi_model_data 的调用参数，使用新 model_id
    """
    if not mapping:
        return copy.deepcopy(sample)

    new_sample = copy.deepcopy(sample)
    msgs = new_sample["messages"]

    # 找出每个 get_idi_model_data 调用的位置
    idi_positions: list[int] = []
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc["function"]["name"] == "get_idi_model_data":
                    idi_positions.append(i)

    # 对每个 get_idi_model_data，找到它之前最近的 get_object_type_metrics 响应
    # 并注入新 model_id
    for idi_pos in idi_positions:
        old_id = None
        new_id = None
        try:
            args = json.loads(msgs[idi_pos]["tool_calls"][0]["function"].get("arguments", "{}"))
            old_id = int(args.get("model_id", 0))
            new_id = mapping.get(old_id)
        except Exception:
            pass
        if old_id is None or new_id is None:
            continue

        # 找 idi_pos 之前最近的 get_object_type_metrics tool 响应
        metrics_resp_pos = None
        for j in range(idi_pos - 1, -1, -1):
            m = msgs[j]
            if m.get("role") == "tool" and j > 0:
                prev = msgs[j - 1]
                if prev.get("role") == "assistant" and prev.get("tool_calls"):
                    if any(tc["function"]["name"] == "get_object_type_metrics"
                           for tc in prev["tool_calls"]):
                        metrics_resp_pos = j
                        break

        if metrics_resp_pos is not None:
            msgs[metrics_resp_pos]["content"] = inject_model_id_into_metrics_response(
                msgs[metrics_resp_pos].get("content", ""), old_id, new_id
            )

    # 修改所有 get_idi_model_data 的 model_id 参数
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc["function"]["name"] == "get_idi_model_data":
                    tc["function"]["arguments"] = replace_model_id_in_idi_args(
                        tc["function"].get("arguments", "{}"), mapping
                    )

    return new_sample


def process_file(
    input_path: Path,
    output_path: Path,
    variants: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    used_ids: set[int] = set()

    with open(input_path) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(samples)} samples from {input_path}")

    augmented: list[dict] = []
    n_augmented = 0
    n_skipped = 0

    for sample in samples:
        # 原始样本始终保留
        augmented.append(sample)

        # 只对含 get_idi_model_data 的样本做增强
        has_idi = any(
            tc["function"]["name"] == "get_idi_model_data"
            for m in sample.get("messages", [])
            if m.get("role") == "assistant" and m.get("tool_calls")
            for tc in m["tool_calls"]
        )
        if not has_idi:
            n_skipped += 1
            continue

        base_mapping = build_id_mapping(sample, rng, used_ids)
        if not base_mapping:
            n_skipped += 1
            continue

        for _ in range(variants):
            # 每个变体用不同的假 ID
            variant_mapping = {
                real: generate_fake_model_id(used_ids, rng)
                for real in base_mapping
            }
            augmented.append(augment_sample(sample, variant_mapping))
            n_augmented += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in augmented:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Original: {len(samples)} samples")
    print(f"Augmented variants created: {n_augmented} (from {len(samples)-n_skipped} eligible samples)")
    print(f"Skipped (no get_idi_model_data): {n_skipped}")
    print(f"Total output: {len(augmented)} samples → {output_path}")

    # 验证：检查几个变体的 model_id 一致性
    print("\nValidation: checking model_id consistency in variants...")
    errors = 0
    for s in augmented:
        msgs = s["messages"]
        # 收集 metrics 响应里的 modelIds
        metrics_ids: set[int] = set()
        idi_ids: set[int] = set()
        for i, m in enumerate(msgs):
            if m.get("role") == "tool" and i > 0:
                prev = msgs[i-1]
                if prev.get("role") == "assistant" and prev.get("tool_calls"):
                    if any(tc["function"]["name"] == "get_object_type_metrics"
                           for tc in prev["tool_calls"]):
                        metrics_ids.update(extract_model_ids_from_metrics_response(
                            m.get("content", "")
                        ))
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc["function"]["name"] == "get_idi_model_data":
                        try:
                            args = json.loads(tc["function"].get("arguments", "{}"))
                            mid = args.get("model_id")
                            if mid is not None:
                                idi_ids.add(int(mid))
                        except Exception:
                            pass
        # idi 调用的 model_id 应该都出现在 metrics 响应里
        if idi_ids and metrics_ids:
            diff = idi_ids - metrics_ids
            if diff:
                errors += 1

    if errors == 0:
        print(f"✓ All {len(augmented)} samples pass model_id consistency check")
    else:
        print(f"✗ {errors} samples have inconsistent model_ids (idi_id not in metrics response)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="round4/data/train_ns_all.jsonl")
    parser.add_argument("--output", default="round4/data/train_ns_augmented.jsonl")
    parser.add_argument("--variants", type=int, default=5,
                        help="每条 idi 样本生成的变体数量")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    process_file(
        Path(args.input),
        Path(args.output),
        args.variants,
        args.seed,
    )


if __name__ == "__main__":
    main()
