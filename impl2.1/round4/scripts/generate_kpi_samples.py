"""
生成 KPI 类合成训练样本

每条样本都有完整因果链：
  list_object_types → get_object_type_metrics（含随机 model_id）
                    → get_idi_model_data（使用 metrics 返回的那个 model_id）
                    → 最终答案

这样模型无法记住"UPH→12202"，只能学会读 metrics 返回值。

生成的 model_id 范围：10000-99999（避开真实 ID）
"""
from __future__ import annotations

import argparse
import json
import random
import math
from pathlib import Path


# ── 工厂数据模板 ──────────────────────────────────────────────────────────────

LINES = ["S03", "S04", "S05", "S06", "S07", "S08"]
PROJECTS = ["PRAGUE", "P17", "P18", "N16", "DUBAI"]
SHIFTS = ["D", "N", "ALL"]

# 指标类型配置：字段名、合理值域、答案模板
METRIC_TYPES = [
    {
        "name": "UPH",
        "keyword": "线体",
        "api_name": "line_operation",
        "model_name": "线体小时产出UPH",
        "fields": ["line_code", "shift_date", "shift", "uph", "target_uph", "achieve_flag"],
        "question_templates": [
            "查询{line}线{date}的UPH数据",
            "查看{line}线{date} {shift}班的UPH",
            "{line}线{date}的小时产出是多少",
            "查询{line}线{date}{hour}:00-{hour_end}:00的UPH数据，包含UPH实际值、标准值{target}、达标情况",
        ],
        "answer_templates": [
            "{line}线{date}的UPH为 **{uph}**，目标值{target}，{achieve_flag}。",
            "查询结果：{line}线{date} {shift}班 UPH = {uph}（目标：{target}）。",
        ],
        "value_gen": lambda rng: {
            "uph": round(rng.uniform(150, 600), 1),
            "target_uph": rng.choice([200, 300, 400, 500, 570]),
        },
    },
    {
        "name": "良率",
        "keyword": "线体",
        "api_name": "line_operation",
        "model_name": "线体良率统计",
        "fields": ["line_code", "shift_date", "process_section_code", "first_pass_rate", "target_first_pass_rate", "final_pass_rate"],
        "question_templates": [
            "查询{line}线{date}的良率",
            "{line}线{date}一次良率是多少",
            "查看{line}线{date}的直通率数据",
        ],
        "answer_templates": [
            "{line}线{date}一次良率为 **{rate:.1f}%**（目标：{rate_target:.1f}%），{rate_flag}。",
            "查询结果：{line}线{date} 直通率 = {rate:.1f}%。",
        ],
        "value_gen": lambda rng: {
            "first_pass_rate": round(rng.uniform(0.80, 0.99), 4),
            "target_first_pass_rate": 0.80,
            "final_pass_rate": round(rng.uniform(0.92, 0.99), 4),
        },
    },
    {
        "name": "产出达成",
        "keyword": "线体",
        "api_name": "line_operation",
        "model_name": "线体产出达成率",
        "fields": ["line_code", "shift_date", "project", "plan_cnt", "output_cnt", "achieve_rate"],
        "question_templates": [
            "查询{line}线{date}的产出达成情况",
            "{line}线{date}计划完成了多少",
            "查看{line}线{date}的排产达成率",
        ],
        "answer_templates": [
            "{line}线{date}计划{plan}件，实际产出{output}件，达成率{achieve:.1f}%。",
            "查询结果：{line}线{date} 产出达成 = {achieve:.1f}%（{output}/{plan}）。",
        ],
        "value_gen": lambda rng: {
            "plan_cnt": rng.randint(200, 1000),
            "output_cnt": rng.randint(150, 900),
        },
    },
    {
        "name": "抛料",
        "keyword": "物料",
        "api_name": "mat_fail_detail",
        "model_name": "线体抛料统计",
        "fields": ["line_code", "shift_date", "mat_type", "throw_mat_cnt", "throw_mat_rate", "warning_std"],
        "question_templates": [
            "查询{line}线{date}的抛料情况",
            "{line}线{date}抛料率是多少",
            "查看{line}线{date}的物料损耗",
        ],
        "answer_templates": [
            "{line}线{date}抛料{cnt}个，抛料率{rate:.4f}（预警阈值{std:.4f}）。",
            "查询结果：{line}线{date} 抛料率 = {rate:.4f}。",
        ],
        "value_gen": lambda rng: {
            "throw_mat_cnt": rng.randint(50, 500),
            "throw_mat_rate": round(rng.uniform(0.0001, 0.001), 6),
            "warning_std": 0.0003,
        },
    },
]

# 工具定义（简化版，保持与训练数据一致）
TOOLS = [
    {"type": "function", "function": {
        "name": "list_object_types",
        "description": "搜索并列出可用的对象类型（apiName、displayName、desc）",
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string", "description": "搜索关键词"}
        }, "required": ["keyword"]},
    }},
    {"type": "function", "function": {
        "name": "get_object_type_metrics",
        "description": "获取对象类型绑定的指标模型（modelId、params、metrics）",
        "parameters": {"type": "object", "properties": {
            "api_names": {"type": "string", "description": "apiName，多个用逗号分隔"}
        }, "required": ["api_names"]},
    }},
    {"type": "function", "function": {
        "name": "get_idi_model_data",
        "description": "查询指标数据（通过 model_id 指定模型）",
        "parameters": {"type": "object", "properties": {
            "model_id": {"type": "integer"},
            "params_json": {"type": "string", "description": "查询参数 JSON 数组"},
            "limit": {"type": "integer", "default": 1000},
        }, "required": ["model_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_object_type_data",
        "description": "查询对象实体数据",
        "parameters": {"type": "object", "properties": {
            "api_name": {"type": "string"},
            "filter_json": {"type": "string"},
            "limit": {"type": "integer", "default": 100},
        }, "required": ["api_name"]},
    }},
    {"type": "function", "function": {
        "name": "get_object_type_detail",
        "description": "获取对象类型的字段定义和参数",
        "parameters": {"type": "object", "properties": {
            "api_names": {"type": "string"}
        }, "required": ["api_names"]},
    }},
]

SYSTEM_MSG = "你是制造业数据分析助手，通过调用工具查询工厂数据来回答用户问题。"


def ok(data):
    return json.dumps({"code": 0, "message": "Success", "data": data}, ensure_ascii=False)


def make_list_object_types_response(keyword: str, api_name: str) -> str:
    return ok([
        {"apiName": api_name, "displayName": api_name, "desc": ""},
        {"apiName": "User", "displayName": "用户", "desc": ""},
    ])


def make_get_object_type_metrics_response(api_name: str, model_id: int, model_name: str, params: list) -> str:
    return ok([{
        "apiName": api_name,
        "modelBindings": [{
            "modelId": model_id,
            "modelName": model_name,
            "joinKey": "line_code",
            "metrics": [{"fieldName": f, "desc": f, "fieldIsValidated": True} for f in ["value", "value2"]],
            "dimension": [
                {"fieldName": "line_code", "desc": "线体编码", "type": "string"},
                {"fieldName": "shift_date", "desc": "日期", "type": "date"},
            ],
            "params": params,
        }],
    }])


def make_idi_response(model_id: int, row: dict) -> str:
    return ok({"data": [row], "count": 1})


def gen_date(rng: random.Random) -> str:
    y, m, d = 2026, rng.randint(4, 5), rng.randint(1, 28)
    return f"{y}-{m:02d}-{d:02d}"


def gen_hour(rng: random.Random) -> tuple[int, int]:
    h = rng.randint(0, 22)
    return h, h + 1


def generate_sample(metric: dict, rng: random.Random, used_ids: set) -> dict:
    """生成一条完整的 KPI 查询训练样本。"""
    # 随机上下文
    line = rng.choice(LINES)
    date = gen_date(rng)
    shift = rng.choice(SHIFTS)
    project = rng.choice(PROJECTS)
    hour, hour_end = gen_hour(rng)
    vals = metric["value_gen"](rng)

    # 随机 model_id（10000-99999，避开真实 ID）
    while True:
        mid = rng.randint(10000, 99999)
        if mid not in used_ids:
            used_ids.add(mid)
            break

    # 构造问题
    qtpl = rng.choice(metric["question_templates"])
    target = vals.get("target_uph", vals.get("target_first_pass_rate", 0))
    question = qtpl.format(
        line=line, date=date, shift=shift, hour=hour, hour_end=hour_end,
        target=target if isinstance(target, int) else round(target * 100, 1),
    )

    # params for metrics binding
    params = [
        {"name": "start_shift_date", "type": "date", "optional": False},
        {"name": "end_shift_date", "type": "date", "optional": False},
        {"name": "line", "type": "string", "optional": True},
        {"name": "shift", "type": "string", "optional": True},
    ]

    # params_json for get_idi_model_data call
    params_json = json.dumps([
        {"name": "start_shift_date", "value": date},
        {"name": "end_shift_date", "value": date},
        {"name": "line", "value": line},
        {"name": "shift", "value": shift},
    ], ensure_ascii=False)

    # 数据行
    row = {"line_code": line, "shift_date": date, "project": project, "shift": shift}
    row.update(vals)

    # 构造答案
    atpl = rng.choice(metric["answer_templates"])
    uph = vals.get("uph", 0)
    rate = round(vals.get("first_pass_rate", 0) * 100, 1)
    plan = vals.get("plan_cnt", 0)
    output = vals.get("output_cnt", 0)
    achieve = round(output / plan * 100, 1) if plan > 0 else 0
    target_disp = target if isinstance(target, int) else round(target * 100, 1)
    achieve_flag = "✓ 达标" if uph >= (target if isinstance(target, (int, float)) else 0) else "✗ 未达标"
    rate_target = round(vals.get("target_first_pass_rate", 0.8) * 100, 1)
    rate_flag = "达标" if rate >= rate_target else "未达标"
    answer = atpl.format(
        line=line, date=date, shift=shift, uph=uph,
        target=target_disp, achieve_flag=achieve_flag,
        rate=rate, rate_target=rate_target, rate_flag=rate_flag,
        plan=plan, output=output, achieve=achieve,
        cnt=vals.get("throw_mat_cnt", 0),
        rate2=vals.get("throw_mat_rate", 0),
        std=vals.get("warning_std", 0),
    )

    # 构造消息序列
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "list_object_types",
                    "arguments": json.dumps({"keyword": metric["keyword"]}, ensure_ascii=False),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": make_list_object_types_response(metric["keyword"], metric["api_name"]),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "get_object_type_metrics",
                    "arguments": json.dumps({"api_names": metric["api_name"]}, ensure_ascii=False),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": make_get_object_type_metrics_response(
                metric["api_name"], mid, metric["model_name"], params
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_3",
                "type": "function",
                "function": {
                    "name": "get_idi_model_data",
                    "arguments": json.dumps({
                        "model_id": mid,
                        "params_json": params_json,
                        "limit": 1000,
                    }, ensure_ascii=False),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_3",
            "content": make_idi_response(mid, row),
        },
        {"role": "assistant", "content": answer},
    ]

    return {"messages": messages, "tools": TOOLS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="round4/data/train_kpi_synthetic.jsonl")
    parser.add_argument("--count", type=int, default=600,
                        help="生成样本总数（均匀分布到各指标类型）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    used_ids: set[int] = set()
    # 避开真实训练数据里的 model_ids
    for real_id in [45, 732, 6668, 8065, 11288, 12089, 12192, 12194,
                    12198, 12202, 12203, 12204, 12207, 12214, 12217,
                    12223, 12365, 12385, 12389]:
        used_ids.add(real_id)

    per_type = args.count // len(METRIC_TYPES)
    samples = []

    for metric in METRIC_TYPES:
        for _ in range(per_type):
            s = generate_sample(metric, rng, used_ids)
            samples.append(s)

    # 打乱顺序
    rng.shuffle(samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 验证：每条样本的 model_id 必须来自 metrics 响应
    import re
    errors = 0
    for s in samples:
        msgs = s["messages"]
        metrics_ids = set()
        idi_ids = set()
        for i, m in enumerate(msgs):
            if m.get("role") == "tool" and i > 0 and msgs[i-1].get("role") == "assistant":
                if any(tc["function"]["name"] == "get_object_type_metrics"
                       for tc in msgs[i-1].get("tool_calls", [])):
                    matches = re.findall(r'"modelId"\s*:\s*(\d+)', m.get("content", ""))
                    metrics_ids.update(int(x) for x in matches)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc["function"]["name"] == "get_idi_model_data":
                        try:
                            a = json.loads(tc["function"].get("arguments", "{}"))
                            if a.get("model_id"):
                                idi_ids.add(int(a["model_id"]))
                        except:
                            pass
        if idi_ids and not idi_ids.issubset(metrics_ids):
            errors += 1

    print(f"Generated: {len(samples)} synthetic KPI samples → {args.output}")
    print(f"Metric types: {[m['name'] for m in METRIC_TYPES]}")
    print(f"Unique model_ids used: {len(used_ids) - 19} (all random, 10000-99999)")
    if errors == 0:
        print("✓ All samples pass model_id consistency check (idi_id ∈ metrics_response)")
    else:
        print(f"✗ {errors} samples have inconsistent model_ids")


if __name__ == "__main__":
    main()
