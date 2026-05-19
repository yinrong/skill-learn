"""
generate_data.py — SFT training data generator for skill internalization.

Generates multi-turn agentic traces by calling Claude API with a skill doc present,
then saves raw traces (ns conversion strips the skill doc later).

Usage:
    python generate_data.py --skill <SKILL_NAME> --count <N> --output <PATH>

Arguments:
    --skill   : one of the supported skill names
    --count   : number of samples to generate (default: 20)
    --output  : output jsonl file path
    --seed    : random seed (default: 42)
"""

import argparse
import json
import os
import random
import re
import time
import uuid
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKILL_DOCS_DIR = Path("/home/yinrong/post-train/impl2.1/round5/data/skill_docs")

VALID_SKILLS = [
    "general-kpi-query",
    "line-operation-skill",
    "equipment-cpk-query",
    "object-data-query",
    "lineside-material-query",
    "line-attendance-query",
    "line-exemption-query",
    "分板机过站明细查询",
    "workstion-kpi-query",
]

# Type A: skills that have a fixed model_id table (no dynamic discovery needed)
TYPE_A_SKILLS = {
    "line-operation-skill",
    "line-attendance-query",
    "line-exemption-query",
    "lineside-material-query",
    "workstion-kpi-query",
    "分板机过站明细查询",
}

# Type B: skills that discover model_id dynamically via get_object_type_metrics
TYPE_B_SKILLS = {"general-kpi-query", "equipment-cpk-query"}

# No-idi: skills that don't call get_idi_model_data
NO_IDI_SKILLS = {"object-data-query"}

# Real model IDs to avoid in Type B random assignment
REAL_MODEL_IDS = {
    12202, 12198, 12204, 12203, 12385, 12389, 12194, 12223,
    12336, 12214, 12192, 12328,
}

# ---------------------------------------------------------------------------
# Type A model definitions
# ---------------------------------------------------------------------------

TYPE_A_MODELS = {
    "line-operation-skill": [
        {
            "modelId": 12202,
            "modelName": "线体小时别UPH",
            "params": [
                {"name": "start_shift_date", "type": "date", "optional": False},
                {"name": "end_shift_date", "type": "date", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
        {
            "modelId": 12198,
            "modelName": "线体一次良率",
            "params": [
                {"name": "start_time", "type": "datetime", "optional": False},
                {"name": "end_time", "type": "datetime", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
        {
            "modelId": 12204,
            "modelName": "线体产出达成",
            "params": [
                {"name": "start_shift_date", "type": "date", "optional": False},
                {"name": "end_shift_date", "type": "date", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
        {
            "modelId": 12203,
            "modelName": "线体OEE",
            "params": [
                {"name": "start_shift_date", "type": "date", "optional": False},
                {"name": "end_shift_date", "type": "date", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
        {
            "modelId": 12385,
            "modelName": "线体出勤率",
            "params": [
                {"name": "shift_date", "type": "date", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
        {
            "modelId": 12389,
            "modelName": "员工出勤打卡记录",
            "params": [
                {"name": "shift_date", "type": "date", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
        {
            "modelId": 12194,
            "modelName": "线体抛料率",
            "params": [
                {"name": "start_time", "type": "datetime", "optional": False},
                {"name": "end_time", "type": "datetime", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
        {
            "modelId": 12223,
            "modelName": "线体TOP异常任务",
            "params": [
                {"name": "start_time", "type": "datetime", "optional": False},
                {"name": "end_time", "type": "datetime", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
    ],
    "line-attendance-query": [
        {
            "modelId": 12385,
            "modelName": "线体出勤率",
            "params": [
                {"name": "shift_date", "type": "date", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
        {
            "modelId": 12389,
            "modelName": "员工出勤打卡记录",
            "params": [
                {"name": "shift_date", "type": "date", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
    ],
    "lineside-material-query": [
        {
            "modelId": 12336,
            "modelName": "线边仓物料剩余",
            "params": [
                {"name": "shift_date", "type": "date", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        }
    ],
    "workstion-kpi-query": [
        {
            "modelId": 12214,
            "modelName": "工站OEE",
            "params": [
                {"name": "start_shift_date", "type": "date", "optional": False},
                {"name": "end_shift_date", "type": "date", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        }
    ],
    "line-exemption-query": [
        {
            "modelId": 12192,
            "modelName": "设备异常工单",
            "params": [
                {"name": "start_time", "type": "datetime", "optional": False},
                {"name": "end_time", "type": "datetime", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
        {
            "modelId": 12223,
            "modelName": "装备故障TOP",
            "params": [
                {"name": "start_time", "type": "datetime", "optional": False},
                {"name": "end_time", "type": "datetime", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        },
    ],
    "分板机过站明细查询": [
        {
            "modelId": 12328,
            "modelName": "分板机铣刀寿命",
            "params": [
                {"name": "shift_date", "type": "date", "optional": False},
                {"name": "line", "type": "string", "optional": True},
            ],
        }
    ],
}

# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

LINES = ["S03", "S04", "S05"]
DAYS = list(range(1, 20))
DATES = ["今天", "昨天", "今天白班", "昨晚夜班"] + [f"2026-05-{d:02d}" for d in DAYS]

QUESTIONS_BY_SKILL = {
    "general-kpi-query": [
        "{line}线今天的UPH",
        "{line}线{date}良率数据",
        "查一下{date}{line}线的OEE",
        "{line}线{date}的产出达成情况",
        "{line}线今天白班的抛料率",
    ],
    "line-operation-skill": [
        "查询{line}线今天的综合运营情况",
        "出一份{line}线{date}的运营报告",
        "{line}线{date}各项指标汇总",
        "{line}线昨晚夜班运营情况怎么样",
    ],
    "equipment-cpk-query": [
        "查一下印刷机的CPK",
        "看下{date}回焊炉的CPK",
        "昨天AOI的SPC数据",
        "{line}线印刷机{date}的CP和CPK",
        "S04线回焊炉今天的炉温CPK",
    ],
    "object-data-query": [
        "{line}线今天的排产",
        "查一下{line}线绑定了哪些指标模型",
        "{line}线当前在产项目",
        "查询{line}线的线体实体信息",
    ],
    "lineside-material-query": [
        "{line}线今天线边仓物料剩余情况",
        "查一下{line}线的热熔胶用量",
        "{line}线线边仓今天有没有物料告警",
    ],
    "line-attendance-query": [
        "{line}线今天出勤情况",
        "查一下{line}线今天白班的出勤人数和出勤率",
        "{line}线和{line2}线今日出勤对比",
    ],
    "line-exemption-query": [
        "{line}线本周设备异常情况",
        "查询{line}线上板机的异常记录",
        "{line}线本周故障改善趋势",
    ],
    "分板机过站明细查询": [
        "{line}线分板机铣刀剩余使用寿命",
        "看下{line}线今天分板机左右台面铣刀寿命",
        "{line}线分板机铣刀最新数据",
    ],
    "workstion-kpi-query": [
        "{line}线各工站今天的OEE",
        "查一下{line}线今天各工站产出数据",
        "{line}线工站级别的OEE数据",
    ],
}


def generate_question(skill: str, rng: random.Random) -> str:
    """Generate a random question for the given skill."""
    templates = QUESTIONS_BY_SKILL[skill]
    tmpl = rng.choice(templates)
    line = rng.choice(LINES)
    line2 = rng.choice([l for l in LINES if l != line])
    date = rng.choice(DATES)
    return tmpl.format(line=line, date=date, line2=line2)


def new_type_b_model_id(rng: random.Random, used: set) -> int:
    """Pick a random model_id in 10000-99999 that avoids real IDs and already-used IDs."""
    while True:
        mid = rng.randint(10000, 99999)
        if mid not in REAL_MODEL_IDS and mid not in used:
            return mid


# ---------------------------------------------------------------------------
# Tool simulator
# ---------------------------------------------------------------------------

def _extract_line_from_params(params_json: str) -> str:
    """Extract line_code from params_json if present, else return S04."""
    try:
        params = json.loads(params_json)
        for p in params:
            if isinstance(p, dict) and p.get("name") == "line":
                return p.get("value", "S04")
    except Exception:
        pass
    return "S04"


def simulate_tool_response(
    tool_name: str,
    args: dict,
    skill: str,
    pre_assigned_model_id: int | None,
    rng: random.Random,
) -> str:
    """Simulate a realistic tool response as a JSON string."""

    # -----------------------------------------------------------------------
    # list_object_types
    # -----------------------------------------------------------------------
    if tool_name == "list_object_types":
        keyword = args.get("keyword", "")
        if any(kw in keyword for kw in ["线体", "线"]):
            api_name, display_name = "line", "线体"
        elif any(kw in keyword for kw in ["装备", "device", "设备"]):
            api_name, display_name = "device", "装备"
        elif any(kw in keyword for kw in ["物料", "material"]):
            api_name, display_name = "mat_fail_detail", "物料失效明细"
        elif any(kw in keyword for kw in ["分板机", "铣刀"]):
            api_name, display_name = "pcb_router", "分板机"
        else:
            api_name, display_name = "line", "线体"
        return json.dumps(
            {
                "code": 0,
                "message": "success",
                "data": [{"apiName": api_name, "displayName": display_name, "desc": ""}],
            },
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------------
    # get_object_type_metrics
    # -----------------------------------------------------------------------
    if tool_name == "get_object_type_metrics":
        api_names = args.get("api_names", "")
        api_name = api_names.split(",")[0].strip()

        if skill in TYPE_A_SKILLS:
            models = TYPE_A_MODELS.get(skill, [])
            bindings = [
                {
                    "modelId": m["modelId"],
                    "modelName": m["modelName"],
                    "params": m["params"],
                }
                for m in models
            ]
        else:
            # Type B: return the pre-assigned model_id with sensible params
            mid = pre_assigned_model_id or 50000
            if skill == "equipment-cpk-query":
                params = [
                    {"name": "start_time", "type": "datetime", "optional": False},
                    {"name": "end_time", "type": "datetime", "optional": False},
                    {"name": "device_code", "type": "string", "optional": True},
                ]
                model_name = "设备CPK指标"
            else:
                params = [
                    {"name": "start_time", "type": "datetime", "optional": False},
                    {"name": "end_time", "type": "datetime", "optional": False},
                    {"name": "line", "type": "string", "optional": True},
                ]
                model_name = "线体KPI指标"
            bindings = [{"modelId": mid, "modelName": model_name, "params": params}]

        return json.dumps(
            {
                "code": 0,
                "message": "success",
                "data": [{"apiName": api_name, "modelBindings": bindings}],
            },
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------------
    # get_idi_model_data
    # -----------------------------------------------------------------------
    if tool_name == "get_idi_model_data":
        model_id = args.get("model_id", 0)
        params_json = args.get("params_json", "[]")
        line_code = _extract_line_from_params(params_json)

        if model_id == 12202:  # UPH
            uph = rng.randint(300, 600)
            row = {
                "line_code": line_code,
                "shift_date": "2026-05-14",
                "shift": rng.choice(["D", "N"]),
                "uph": uph,
                "target_uph": 500,
            }
            data = {"data": [row], "count": 1}

        elif model_id == 12198:  # first-pass yield
            rate = round(rng.uniform(0.85, 0.99), 4)
            row = {"line_code": line_code, "first_pass_rate": rate}
            data = {"data": [row], "count": 1}

        elif model_id == 12204:  # output achievement
            plan = rng.randint(200, 1000)
            output = rng.randint(int(plan * 0.7), plan)
            rate = round(output / plan, 4)
            row = {
                "line_code": line_code,
                "plan_cnt": plan,
                "output_cnt": output,
                "achieve_rate": rate,
            }
            data = {"data": [row], "count": 1}

        elif model_id == 12385:  # attendance rate
            roll = rng.randint(10, 20)
            row = {
                "line_code": line_code,
                "roll_call_con": roll,
                "roll_call_rate": 1.0,
            }
            data = {"data": [row], "count": 1}

        elif model_id == 12389:  # employee clock-in records
            n = rng.randint(2, 3)
            rows = [
                {
                    "employee_id": f"EMP{1000 + i}",
                    "employee_name": f"员工{1000 + i}",
                    "line_code": line_code,
                    "clock_in": "2026-05-14 07:58:00",
                    "clock_out": "2026-05-14 20:02:00",
                }
                for i in range(n)
            ]
            data = {"data": rows, "count": n}

        elif model_id == 12194:  # scrap rate
            cnt = rng.randint(5, 50)
            rate = round(cnt / 1000, 4)
            row = {
                "line_code": line_code,
                "throw_mat_cnt": cnt,
                "throw_mat_rate": rate,
            }
            data = {"data": [row], "count": 1}

        elif model_id == 12203:  # OEE
            oee = round(rng.uniform(0.65, 0.95), 4)
            row = {"line_code": line_code, "oee": oee}
            data = {"data": [row], "count": 1}

        elif model_id == 12223:  # anomaly tasks
            n = rng.randint(1, 2)
            rows = [
                {
                    "line_code": line_code,
                    "anomaly_type": rng.choice(["设备故障", "质量异常", "物料短缺"]),
                    "anomaly_desc": f"异常任务{i+1}",
                    "create_time": "2026-05-14 10:00:00",
                    "status": rng.choice(["处理中", "已关闭"]),
                }
                for i in range(n)
            ]
            data = {"data": rows, "count": n}

        elif model_id == 12328:  # drill bit life
            row = {
                "line_code": line_code,
                "device_code": f"PCB-ROUTER-{line_code}",
                "remaining_life": rng.randint(20, 100),
                "total_life": 200,
            }
            data = {"data": [row], "count": 1}

        elif model_id == 12336:  # lineside material
            row = {
                "line_code": line_code,
                "material_code": f"MAT-{rng.randint(1000, 9999)}",
                "remaining_qty": rng.randint(10, 500),
                "unit": "pcs",
            }
            data = {"data": [row], "count": 1}

        elif model_id == 12214:  # workstation OEE
            n = rng.randint(2, 4)
            rows = [
                {
                    "line_code": line_code,
                    "station_code": f"WS-{i+1:02d}",
                    "oee": round(rng.uniform(0.60, 0.95), 4),
                }
                for i in range(n)
            ]
            data = {"data": rows, "count": n}

        elif model_id == 12192:  # equipment anomaly work order
            n = rng.randint(1, 3)
            rows = [
                {
                    "line_code": line_code,
                    "work_order_id": f"WO-{rng.randint(10000, 99999)}",
                    "device_code": f"DEV-{rng.randint(100, 999)}",
                    "anomaly_desc": rng.choice(["机械故障", "电气故障", "程序异常"]),
                    "create_time": "2026-05-14 09:00:00",
                }
                for i in range(n)
            ]
            data = {"data": rows, "count": n}

        else:
            # Generic data for Type B random model IDs
            if skill == "equipment-cpk-query":
                row = {
                    "device_code": f"DEV-{rng.randint(100, 999)}",
                    "cp": round(rng.uniform(1.0, 2.0), 3),
                    "cpk": round(rng.uniform(0.8, 1.8), 3),
                    "measure_time": "2026-05-14 10:00:00",
                }
            else:
                row = {
                    "line_code": line_code,
                    "value": round(rng.uniform(0.5, 1.0), 4),
                    "measure_time": "2026-05-14 10:00:00",
                }
            data = {"data": [row], "count": 1}

        return json.dumps({"code": 0, "message": "success", "data": data}, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # get_object_type_data
    # -----------------------------------------------------------------------
    if tool_name == "get_object_type_data":
        api_name = args.get("api_name", "line")
        if api_name == "line":
            rows = [
                {
                    "line_code": rng.choice(LINES),
                    "line_name": f"SMT{rng.choice(['A', 'B', 'C'])}线体",
                    "project": f"项目{rng.choice(['P17', 'O1', 'O3'])}",
                }
                for _ in range(rng.randint(1, 2))
            ]
        elif api_name == "device":
            rows = [
                {
                    "device_code": f"DEV-{rng.randint(100, 999)}",
                    "device_name": rng.choice(["印刷机", "贴片机", "回焊炉", "AOI"]),
                    "line_code": rng.choice(LINES),
                }
                for _ in range(rng.randint(2, 3))
            ]
        else:
            rows = [
                {
                    "id": rng.randint(1000, 9999),
                    "name": f"记录{i+1}",
                    "value": round(rng.uniform(0, 100), 2),
                }
                for i in range(rng.randint(1, 2))
            ]
        return json.dumps(
            {"code": 0, "message": "success", "data": rows}, ensure_ascii=False
        )

    # -----------------------------------------------------------------------
    # get_object_type_detail
    # -----------------------------------------------------------------------
    if tool_name == "get_object_type_detail":
        api_names = args.get("api_names", "line")
        api = api_names.split(",")[0].strip()
        schema = {
            "line": {
                "primaryKey": "line_code",
                "columns": [
                    {"name": "line_code", "type": "string"},
                    {"name": "line_name", "type": "string"},
                    {"name": "project", "type": "string"},
                ],
                "params": [],
                "count": 20,
            },
            "device": {
                "primaryKey": "device_code",
                "columns": [
                    {"name": "device_code", "type": "string"},
                    {"name": "device_name", "type": "string"},
                    {"name": "line_code", "type": "string"},
                ],
                "params": [],
                "count": 50,
            },
        }.get(
            api,
            {
                "primaryKey": "id",
                "columns": [{"name": "id", "type": "integer"}, {"name": "value", "type": "float"}],
                "params": [],
                "count": 10,
            },
        )
        return json.dumps(
            {"code": 0, "message": "success", "data": [{"apiName": api, **schema}]},
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------------
    # format_feishu_card — acknowledge without error
    # -----------------------------------------------------------------------
    if tool_name == "format_feishu_card":
        return json.dumps({"code": 0, "message": "卡片已渲染", "data": {}}, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # Fallback for any other tools
    # -----------------------------------------------------------------------
    return json.dumps({"code": 0, "message": "success", "data": {}}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Claude API client
# ---------------------------------------------------------------------------

def build_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY environment variable is required"
        )
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    return anthropic.Anthropic(api_key=api_key, base_url=base_url)


SYSTEM_PROMPT = (
    "你是一个熟练的工厂数据分析助手。"
    "你的任务是根据 skill 文档严格执行数据查询。"
    "请在 thinking 中展示你的推理过程，特别是如何从工具响应中确定参数（如 model_id）。"
)


def call_claude_with_retry(
    client: anthropic.Anthropic,
    messages: list,
    tools: list,
    max_retries: int = 3,
) -> anthropic.types.Message:
    """Call Claude API with exponential backoff on failures."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="ppio/pa/claude-sonnet-4-6",
                max_tokens=4000,
                thinking={"type": "enabled", "budget_tokens": 1024},
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
            )
            return response
        except anthropic.APIStatusError as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error ({e.status_code}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Connection error, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def parse_response(response: anthropic.types.Message) -> tuple[str, list[dict], bool]:
    """
    Parse response content blocks.

    Returns:
        assistant_content: str with <think>...</think> wrapped thinking + text
        tool_calls: list of OpenAI-format tool call dicts
        has_tool_calls: bool
    """
    assistant_content = ""
    tool_calls = []

    for block in response.content:
        if block.type == "thinking":
            assistant_content += f"<think>\n{block.thinking}\n</think>\n"
        elif block.type == "text":
            assistant_content += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )

    return assistant_content, tool_calls, len(tool_calls) > 0


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def get_skill_type(skill: str) -> str:
    """Return 'type_a', 'type_b', or 'no_idi'."""
    if skill in TYPE_A_SKILLS:
        return "type_a"
    if skill in TYPE_B_SKILLS:
        return "type_b"
    return "no_idi"


def generate_sample(
    skill: str,
    skill_doc: str,
    tools: list,
    client: anthropic.Anthropic,
    rng: random.Random,
    used_model_ids: set,
) -> dict | None:
    """
    Run one multi-turn agentic loop and return a sample dict or None if invalid.
    """
    skill_type = get_skill_type(skill)
    question = generate_question(skill, rng)

    # Pre-assign model_id for Type B
    pre_assigned_model_id = None
    if skill_type == "type_b":
        pre_assigned_model_id = new_type_b_model_id(rng, used_model_ids)
        used_model_ids.add(pre_assigned_model_id)



    # Build initial messages
    # api_messages: Claude-native format for API calls
    # out_messages: OpenAI format for training data output
    user_content = f"{skill_doc}\n\n## 用户问题\n{question}"
    api_messages = [{"role": "user", "content": user_content}]
    out_messages = [
        {"role": "system", "content": "你是制造业数据分析助手，通过调用工具查询工厂数据来回答用户问题。"},
        {"role": "user", "content": user_content},
    ]

    # Track all model_ids returned by get_object_type_metrics (for Type B validation)
    metrics_returned_model_ids: set[int] = set()

    max_iters = 20
    final_text = ""
    has_any_thinking = False

    for _iter in range(max_iters):
        response = call_claude_with_retry(client, api_messages, tools)
        assistant_content, tool_calls, has_tool_calls = parse_response(response)

        # Check for thinking content
        if "<think>" in assistant_content and "</think>" in assistant_content:
            think_matches = re.findall(r"<think>\n?(.*?)\n?</think>", assistant_content, re.DOTALL)
            if any(m.strip() for m in think_matches):
                has_any_thinking = True

        if has_tool_calls:
            # === API messages (Claude format) ===
            # Claude wants tool_use as content blocks in assistant message
            # IMPORTANT: thinking blocks must include signature field as returned
            api_assistant_content = []
            for block in response.content:
                if block.type == "thinking":
                    thinking_block = {"type": "thinking", "thinking": block.thinking}
                    if hasattr(block, "signature") and block.signature:
                        thinking_block["signature"] = block.signature
                    api_assistant_content.append(thinking_block)
                elif block.type == "text" and block.text.strip():
                    api_assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    api_assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            api_messages.append({"role": "assistant", "content": api_assistant_content})

            # === Output messages (OpenAI format) ===
            out_assistant_msg: dict = {"role": "assistant"}
            if assistant_content.strip():
                out_assistant_msg["content"] = assistant_content
            out_assistant_msg["tool_calls"] = tool_calls
            out_messages.append(out_assistant_msg)

            # Process each tool call
            tool_results_for_api = []
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    tool_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}

                tool_result = simulate_tool_response(
                    tool_name, tool_args, skill, pre_assigned_model_id, rng
                )

                # Track model_ids from get_object_type_metrics
                if tool_name == "get_object_type_metrics":
                    sim_data = json.loads(tool_result)
                    for entry in sim_data.get("data", []):
                        for binding in entry.get("modelBindings", []):
                            mid = binding.get("modelId")
                            if mid is not None:
                                metrics_returned_model_ids.add(mid)

                tool_results_for_api.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": tool_result,
                })

                # OpenAI format: role="tool"
                out_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

            # Claude format: all tool results as a single user message
            api_messages.append({"role": "user", "content": tool_results_for_api})

        else:
            # Final text response — end the loop
            final_text = assistant_content.strip()
            out_messages.append({"role": "assistant", "content": assistant_content})
            break

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    # Final answer must be substantial and contain at least one digit

    if len(final_text) < 50:
        return None
    if not re.search(r"\d", final_text):
        return None

    # At least one assistant message must have non-empty thinking
    if not has_any_thinking:
        return None

    # Type B: all get_idi_model_data model_ids must appear in a preceding metrics response
    if skill_type == "type_b" and metrics_returned_model_ids:
        for msg in out_messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc["function"]["name"] == "get_idi_model_data":
                        try:
                            args = json.loads(tc["function"]["arguments"])
                            mid = args.get("model_id")
                            if mid is not None and mid not in metrics_returned_model_ids:
                                return None
                        except json.JSONDecodeError:
                            pass

    return {
        "messages": out_messages,
        "tools": tools,
        "metadata": {
            "skill": skill,
            "question": question,
            "skill_type": skill_type,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data for skill internalization")
    parser.add_argument(
        "--skill",
        required=True,
        choices=VALID_SKILLS,
        help="Skill name to generate data for",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of samples to generate (default: 20)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load skill doc
    skill_doc_path = SKILL_DOCS_DIR / f"{args.skill}.txt"
    if not skill_doc_path.exists():
        raise FileNotFoundError(f"Skill doc not found: {skill_doc_path}")
    skill_doc = skill_doc_path.read_text(encoding="utf-8")

    # Load tools and convert OpenAI format → Claude native format
    tools_path = SKILL_DOCS_DIR / f"{args.skill}_tools.json"
    if not tools_path.exists():
        raise FileNotFoundError(f"Tools file not found: {tools_path}")
    raw_tools = json.loads(tools_path.read_text(encoding="utf-8"))
    tools = []
    for t in raw_tools:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        elif "name" in t and "input_schema" in t:
            tools.append(t)  # already Claude format
    # Also store original OpenAI tools for saving in output samples
    openai_tools = raw_tools

    # Build client
    client = build_client()

    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    used_model_ids: set[int] = set()

    with output_path.open("w", encoding="utf-8") as out_f:
        for i in range(args.count):
            try:
                sample = generate_sample(
                    skill=args.skill,
                    skill_doc=skill_doc,
                    tools=tools,           # Claude format for API calls
                    client=client,
                    rng=rng,
                    used_model_ids=used_model_ids,
                )
                # Replace with OpenAI format tools for training data output
                if sample:
                    sample["tools"] = openai_tools
            except Exception as e:
                print(f"[{i+1}/{args.count}] skill={args.skill} ERROR: {e}")
                sample = None

            # Determine question for display (re-generate doesn't advance rng state the
            # same way, but we capture from metadata if available)
            q_display = sample["metadata"]["question"][:40] if sample else "?"
            status = "OK" if sample else "SKIP"
            print(f"[{i+1}/{args.count}] skill={args.skill} q={q_display}... {status}")

            if sample:
                # Write without metadata key to keep output clean (just messages + tools)
                out_record = {"messages": sample["messages"], "tools": sample["tools"]}
                out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                valid_count += 1

    print(f"\nGenerated {valid_count} valid samples → {output_path}")


if __name__ == "__main__":
    main()
