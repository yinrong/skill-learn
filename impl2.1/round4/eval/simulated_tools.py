"""
模拟工具引擎：GT 轨迹重放 + 合成兜底

策略：
  1. 优先重放：从对应测试样本的 ws 轨迹中按顺序取真实工具响应
  2. 合成兜底：GT 响应耗尽后，根据工具名 + 参数生成上下文相关的合成数据

用法：
  engine = ReplayToolEngine(ws_sample)  # ws_sample 来自 test_ns_all.jsonl
  response_str = engine.respond("list_object_types", {"keyword": "线体"})
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import deque
from typing import Any


# ── Synthetic data templates ──────────────────────────────────────────────────

# list_object_types: keyword → 候选对象类型列表
_KEYWORD_TO_OBJECTS: dict[str, list[dict]] = {
    "线体": [
        {"apiName": "line", "displayName": "线体", "desc": "生产线体信息"},
        {"apiName": "ShiftLineProductionPlan", "displayName": "线体排产计划", "desc": "班次线体排产计划"},
    ],
    "装备": [
        {"apiName": "device_fail_item", "displayName": "装备故障明细", "desc": "装备故障记录"},
        {"apiName": "device", "displayName": "装备", "desc": "装备基本信息"},
        {"apiName": "mat_fail_detail", "displayName": "上料接料报错明细", "desc": "上料接料报错明细"},
    ],
    "故障": [
        {"apiName": "device_fail_item", "displayName": "装备故障明细", "desc": "装备故障记录"},
        {"apiName": "mat_fail_detail", "displayName": "上料接料报错明细", "desc": "上料接料报错明细"},
        {"apiName": "exemption_task", "displayName": "异常工单", "desc": "设备异常工单"},
    ],
    "物料": [
        {"apiName": "material", "displayName": "物料信息", "desc": "物料基本信息"},
        {"apiName": "expired_material", "displayName": "物料过期明细", "desc": "过期物料记录"},
        {"apiName": "mat_fail_detail", "displayName": "上料接料报错明细", "desc": "上料接料报错明细"},
        {"apiName": "material_bind_detail", "displayName": "物料绑定明细", "desc": "物料绑定记录"},
    ],
    "工站": [
        {"apiName": "work_station", "displayName": "工站", "desc": "工站信息"},
        {"apiName": "station_code", "displayName": "站位信息", "desc": "站位与装备绑定"},
    ],
    "工段": [
        {"apiName": "process_section_code", "displayName": "工段", "desc": "工段信息"},
    ],
    "项目": [
        {"apiName": "project", "displayName": "项目", "desc": "项目基本信息"},
        {"apiName": "idi_project", "displayName": "IDI项目", "desc": "IDI项目信息"},
    ],
    "异常": [
        {"apiName": "exemption_task", "displayName": "异常工单", "desc": "设备异常工单"},
        {"apiName": "idi_alarm", "displayName": "告警记录", "desc": "系统告警记录"},
    ],
    "产品": [
        {"apiName": "SN", "displayName": "SN信息", "desc": "产品序列号"},
        {"apiName": "sn_process_history", "displayName": "SN过站历史", "desc": "产品过站记录"},
    ],
    "排产": [
        {"apiName": "ShiftLineProductionPlan", "displayName": "线体排产计划", "desc": "班次线体排产计划"},
        {"apiName": "mes_order_no", "displayName": "工单", "desc": "MES工单"},
    ],
    "仓库": [
        {"apiName": "wh_name", "displayName": "仓库", "desc": "仓库基本信息"},
        {"apiName": "wh_area", "displayName": "库区", "desc": "库区信息"},
        {"apiName": "in_bound_r", "displayName": "入库记录", "desc": "物料入库记录"},
        {"apiName": "out_bound_r", "displayName": "出库记录", "desc": "物料出库记录"},
    ],
}

_DEFAULT_OBJECTS = [
    {"apiName": "line", "displayName": "线体", "desc": "生产线体信息"},
    {"apiName": "device_fail_item", "displayName": "装备故障明细", "desc": "装备故障记录"},
    {"apiName": "exemption_task", "displayName": "异常工单", "desc": "设备异常工单"},
]

# get_object_type_detail: apiName → schema
_SCHEMAS: dict[str, dict] = {
    "line": {
        "apiName": "line", "displayName": "线体", "desc": "生产线体信息",
        "primaryKey": "line_code",
        "columns": [
            {"name": "line_code", "displayName": "线体编码", "type": "string"},
            {"name": "line_name", "displayName": "线体名称", "type": "string"},
            {"name": "process_section_code", "displayName": "工段编码", "type": "string"},
            {"name": "process_section_name", "displayName": "工段名称", "type": "string"},
            {"name": "project", "displayName": "项目", "type": "string"},
            {"name": "project_code", "displayName": "项目编码", "type": "string"},
        ],
        "params": [], "exampleData": [],
    },
    "device_fail_item": {
        "apiName": "device_fail_item", "displayName": "装备故障明细", "desc": "装备故障记录",
        "primaryKey": "id",
        "columns": [
            {"name": "line_code", "displayName": "线体编码", "type": "string"},
            {"name": "device_code", "displayName": "装备编码", "type": "string"},
            {"name": "start_tm", "displayName": "故障开始时间", "type": "datetime"},
            {"name": "end_tm", "displayName": "故障结束时间", "type": "datetime"},
            {"name": "fault_code", "displayName": "故障代码", "type": "string"},
            {"name": "fault_desc", "displayName": "故障描述", "type": "string"},
            {"name": "duration_min", "displayName": "持续时长(分钟)", "type": "float"},
        ],
        "params": [], "exampleData": [],
    },
    "exemption_task": {
        "apiName": "exemption_task", "displayName": "异常工单", "desc": "设备异常工单",
        "primaryKey": "id",
        "columns": [
            {"name": "line", "displayName": "线体", "type": "string"},
            {"name": "machine_code", "displayName": "装备编码", "type": "string"},
            {"name": "workstation_code", "displayName": "工站编码", "type": "string"},
            {"name": "exception_desc", "displayName": "异常描述", "type": "string"},
            {"name": "exception_level", "displayName": "异常级别", "type": "string"},
            {"name": "create_time", "displayName": "创建时间", "type": "datetime"},
        ],
        "params": [], "exampleData": [],
    },
    "ShiftLineProductionPlan": {
        "apiName": "ShiftLineProductionPlan", "displayName": "线体排产计划", "desc": "班次线体排产计划",
        "primaryKey": "id",
        "columns": [
            {"name": "line_code", "displayName": "线体编码", "type": "string"},
            {"name": "shift", "displayName": "班次", "type": "string"},
            {"name": "shift_date", "displayName": "日期", "type": "date"},
            {"name": "project", "displayName": "项目", "type": "string"},
            {"name": "plan_cnt", "displayName": "计划数量", "type": "int"},
            {"name": "plan_start_tm", "displayName": "计划开始时间", "type": "datetime"},
            {"name": "plan_end_tm", "displayName": "计划结束时间", "type": "datetime"},
        ],
        "params": [
            {"name": "shift_date", "type": "date", "optional": False},
            {"name": "shift", "type": "string", "optional": True},
        ],
        "exampleData": [],
    },
    "mat_fail_detail": {
        "apiName": "mat_fail_detail", "displayName": "上料接料报错明细", "desc": "上料接料报错明细",
        "primaryKey": "id",
        "columns": [
            {"name": "line_code", "displayName": "线体编码", "type": "string"},
            {"name": "station_code", "displayName": "站位编码", "type": "string"},
            {"name": "shift_date", "displayName": "日期", "type": "date"},
            {"name": "shift", "displayName": "班次", "type": "string"},
            {"name": "mat_type", "displayName": "物料类型", "type": "string"},
            {"name": "fail_cnt", "displayName": "报错次数", "type": "int"},
        ],
        "params": [], "exampleData": [],
    },
}

_DEFAULT_SCHEMA = {
    "columns": [
        {"name": "id", "displayName": "主键", "type": "int"},
        {"name": "line_code", "displayName": "线体编码", "type": "string"},
        {"name": "create_time", "displayName": "创建时间", "type": "datetime"},
    ],
    "params": [], "exampleData": [],
}

# get_object_type_metrics: apiName → modelBindings
_METRICS: dict[str, list[dict]] = {
    # "line" is the api_name returned by list_object_types(keyword="线体")
    # Must mirror line_operation so models calling either api_name succeed
    "line": [
        {"modelId": 12202, "modelName": "线体小时别UPH",
         "params": [{"name": "start_shift_date", "type": "date", "optional": False},
                    {"name": "end_shift_date", "type": "date", "optional": False},
                    {"name": "shift", "type": "string", "optional": True},
                    {"name": "line", "type": "string", "optional": True}]},
        {"modelId": 12198, "modelName": "线体一次良率",
         "params": [{"name": "start_time", "type": "datetime", "optional": False},
                    {"name": "end_time", "type": "datetime", "optional": False},
                    {"name": "line", "type": "string", "optional": True}]},
        {"modelId": 12204, "modelName": "线体产出达成",
         "params": [{"name": "start_shift_date", "type": "date", "optional": False},
                    {"name": "end_shift_date", "type": "date", "optional": False},
                    {"name": "line", "type": "string", "optional": True}]},
    ],
    # "device" is the api_name returned by list_object_types(keyword="装备")
    "device": [
        {"modelId": 12365, "modelName": "装备CPK统计",
         "params": [{"name": "startTm", "type": "datetime", "optional": False},
                    {"name": "endTm", "type": "datetime", "optional": False},
                    {"name": "device_code", "type": "string", "optional": True}]},
    ],
    "line_operation": [
        {"modelId": 12202, "modelName": "线体运营核心指标", "joinKey": "line_code",
         "metrics": [{"fieldName": "uph", "desc": "UPH", "type": "float"},
                     {"fieldName": "oee", "desc": "OEE", "type": "float"},
                     {"fieldName": "yield_rate", "desc": "良率", "type": "float"}],
         "dimension": [{"fieldName": "line_code", "desc": "线体编码", "type": "string"},
                       {"fieldName": "shift_date", "desc": "日期", "type": "date"}],
         "params": [{"name": "start_shift_date", "type": "date", "optional": False},
                    {"name": "end_shift_date", "type": "date", "optional": False},
                    {"name": "shift", "type": "string", "optional": True},
                    {"name": "line", "type": "string", "optional": True}]},
    ],
    "device_fail_item": [
        {"modelId": 9923, "modelName": "装备故障统计", "joinKey": "device_code",
         "metrics": [{"fieldName": "fault_duration_min", "desc": "故障时长(分钟)", "type": "float"},
                     {"fieldName": "fault_cnt", "desc": "故障次数", "type": "int"}],
         "dimension": [{"fieldName": "line_code", "desc": "线体编码", "type": "string"},
                       {"fieldName": "device_code", "desc": "装备编码", "type": "string"}],
         "params": [{"name": "date_7", "type": "date", "optional": True}]},
    ],
    "exemption_task": [
        {"modelId": 9924, "modelName": "异常工单统计", "joinKey": "line",
         "metrics": [{"fieldName": "exception_cnt", "desc": "异常数量", "type": "int"}],
         "dimension": [{"fieldName": "line", "desc": "线体", "type": "string"}],
         "params": []},
    ],
}

# get_idi_model_data: model_id → sample row template + field list
_IDI_MODEL_TEMPLATES: dict[int, dict] = {
    12204: {  # 产出达成
        "fields": ["process", "line_code", "project", "plan_cnt", "output_cnt", "day_achieve_rate", "input_cnt"],
        "sample": {"process": "主板SMT_B", "line_code": "S04", "project": "PRAGUE",
                   "plan_cnt": 4200, "output_cnt": 3780, "day_achieve_rate": 0.9, "input_cnt": 3800},
    },
    12385: {  # 出勤
        "fields": ["line_code", "shift_date", "roll_call_con", "roll_call_rate", "actl_att_con"],
        "sample": {"line_code": "S04", "shift_date": "2026-05-13", "roll_call_con": 12, "roll_call_rate": 1.0, "actl_att_con": 12},
    },
    12389: {  # 人员技能
        "fields": ["line_code", "emp_code", "shift_date", "shift", "skill_name", "emp_name"],
        "sample": {"line_code": "S04", "emp_code": "12345", "shift_date": "2026-05-13", "shift": "D", "skill_name": "上料技能", "emp_name": "张三"},
    },
    12198: {  # 良率
        "fields": ["line_code", "process_section_code", "project", "first_pass_rate", "target_first_pass_rate", "final_pass_rate"],
        "sample": {"line_code": "S04", "process_section_code": "SMT", "project": "PRAGUE",
                   "first_pass_rate": 0.932, "target_first_pass_rate": 0.80, "final_pass_rate": 0.975},
    },
    12194: {  # 抛料
        "fields": ["line_code", "mat_type", "project_code", "throw_mat_cnt", "throw_mat_rate", "warning_std"],
        "sample": {"line_code": "S04", "mat_type": "A", "project_code": "PRAGUE",
                   "throw_mat_cnt": 131, "throw_mat_rate": 0.000196, "warning_std": 0.0003},
    },
    12223: {  # 装备 TOP 异常
        "fields": ["top_machine_code", "line", "exception_desc", "exception_count", "avg_processing_time_minutes"],
        "sample": {"top_machine_code": "M-SMT-PCBC-A-ZMO-0401", "line": "S04",
                   "exception_desc": "Mark NG 取料吸附失败", "exception_count": 10, "avg_processing_time_minutes": 2.5},
    },
    12202: {  # OEE/UPH
        "fields": ["line_code", "shift_date", "shift", "uph", "target_uph", "oee", "yield_rate"],
        "sample": {"line_code": "S04", "shift_date": "2026-05-13", "shift": "D",
                   "uph": 185, "target_uph": 200, "oee": 0.72, "yield_rate": 0.975},
    },
    12203: {  # 线体综合指标
        "fields": ["line_code", "shift_date", "project", "plan_cnt", "output_cnt", "achieve_rate"],
        "sample": {"line_code": "S04", "shift_date": "2026-05-13", "project": "PRAGUE",
                   "plan_cnt": 500, "output_cnt": 450, "achieve_rate": 0.90},
    },
}

_DEFAULT_IDI_SAMPLE = {"line_code": "S04", "shift_date": "2026-05-13", "value": 0, "count": 0}


# ── Synthetic response builders ───────────────────────────────────────────────

def _ok(data: Any) -> str:
    return json.dumps({"code": 0, "message": "Success", "data": data}, ensure_ascii=False)


def _synth_list_object_types(args: dict) -> str:
    keyword = args.get("keyword", "")
    # Try to match keyword to known types
    for key, objs in _KEYWORD_TO_OBJECTS.items():
        if key in keyword or keyword in key:
            return _ok(objs)
    return _ok(_DEFAULT_OBJECTS)


def _synth_get_object_type_detail(args: dict) -> str:
    api_names_str = args.get("api_names", "")
    api_names = [a.strip() for a in api_names_str.split(",") if a.strip()]
    result = []
    for an in api_names:
        if an in _SCHEMAS:
            result.append({**_SCHEMAS[an], "count": 0, "exampleData": []})
        else:
            result.append({
                "apiName": an, "displayName": an, "desc": "",
                "primaryKey": "id",
                **_DEFAULT_SCHEMA,
                "count": 0,
            })
    return _ok(result if result else [{"apiName": api_names_str, **_DEFAULT_SCHEMA, "count": 0}])


def _synth_get_object_type_metrics(args: dict) -> str:
    api_names_str = args.get("api_names", "")
    api_names = [a.strip() for a in api_names_str.split(",") if a.strip()]
    result = []
    for an in api_names:
        bindings = _METRICS.get(an, [])
        result.append({"apiName": an, "modelBindings": bindings})
    return _ok(result)


def _extract_filter_context(filter_json_str: str) -> dict:
    """
    Extract line_code and time window from filter_json.
    Handles patterns like: start_tm < T2 AND end_tm > T1 (overlap window [T1, T2])
    """
    ctx: dict = {"line_code": "S04", "start_tm": None, "end_tm": None, "shift_date": None}
    try:
        filters = json.loads(filter_json_str) if filter_json_str else []
        if isinstance(filters, list):
            for f in filters:
                conds = f.get("conditions", [])
                for c in conds:
                    col = c.get("column", "")
                    op = c.get("operator", "")
                    val = c.get("value", "")
                    if col in ("line_code", "line") and val:
                        ctx["line_code"] = str(val)
                    elif col == "shift_date" and val:
                        ctx["shift_date"] = str(val)
                    # start_tm < T2 → window starts at T2 - 1h (find the window lower bound)
                    # end_tm > T1 → T1 is the window lower bound
                    elif col == "end_tm" and op in ("gt", "gte") and val:
                        ctx["start_tm"] = str(val)  # window lower bound
                    elif col == "start_tm" and op in ("lt", "lte") and val:
                        ctx["end_tm"] = str(val)    # window upper bound
                    elif col == "start_tm" and op in ("eq", "gte", "gt") and val:
                        ctx["start_tm"] = str(val)
    except Exception:
        pass
    return ctx


def _synth_get_object_type_data(args: dict) -> str:
    api_name = args.get("api_name", "")
    filter_json = args.get("filter_json", "")
    params_json = args.get("params_json", "")

    ctx = _extract_filter_context(filter_json)
    line_code = ctx["line_code"]
    # Derive a date from start_tm if available
    start_tm = ctx.get("start_tm") or "2026-05-13 08:00:00"
    shift_date = ctx.get("shift_date") or start_tm[:10]

    if api_name == "line":
        suffix = line_code[-2:] if len(line_code) >= 2 else "04"
        rows = [{"line_code": line_code, "line_name": f"1F-SMT{suffix}",
                 "process_section_code": "SMT", "process_section_name": "贴片段",
                 "project": "PRAGUE", "project_code": ""}]

    elif api_name == "device_fail_item":
        # Generate 2-3 fault records in the queried time range
        base_date = start_tm[:10]
        base_hour = int(start_tm[11:13]) if len(start_tm) > 12 else 8
        rows = [
            {"line_code": line_code, "device_code": "M-SMT-PCBC-A-ZMO-0401",
             "start_tm": f"{base_date} {base_hour:02d}:15:00",
             "end_tm": f"{base_date} {base_hour:02d}:42:00",
             "fault_code": "E001", "fault_desc": "Mark NG 取料吸附失败",
             "duration_min": 27.0, "work_station_code": "SPI", "station_code": "ALC"},
            {"line_code": line_code, "device_code": "M-SMT-PCBC-B-ZMO-0402",
             "start_tm": f"{base_date} {(base_hour+0):02d}:30:00",
             "end_tm": f"{base_date} {(base_hour+0):02d}:55:00",
             "fault_code": "E002", "fault_desc": "轨道堵塞",
             "duration_min": 25.0, "work_station_code": "AOI", "station_code": "ALC"},
        ]

    elif api_name == "exemption_task":
        rows = [{"line": line_code, "machine_code": "M-SMT-PCBC-A-ZMO-0401",
                 "workstation_code": "SPI", "exception_desc": "贴片异常",
                 "exception_level": "P2", "create_time": f"{shift_date} 09:00:00"}]

    elif api_name == "ShiftLineProductionPlan":
        rows = [
            {"line_code": line_code, "shift": "D", "shift_date": shift_date,
             "project": "PRAGUE", "plan_cnt": 500,
             "plan_start_tm": f"{shift_date} 08:00:00", "plan_end_tm": f"{shift_date} 20:00:00"},
            {"line_code": line_code, "shift": "N", "shift_date": shift_date,
             "project": "PRAGUE", "plan_cnt": 480,
             "plan_start_tm": f"{shift_date} 20:00:00",
             "plan_end_tm": f"{shift_date[:8]}{int(shift_date[8:10])+1:02d} 08:00:00"},
        ]

    elif api_name == "mat_fail_detail":
        rows = [{"line_code": line_code, "station_code": "ALC", "shift_date": shift_date,
                 "shift": "D", "mat_type": "A", "fail_cnt": 3}]

    elif api_name == "mat_occupy_pos":
        rows = [{"line_code": line_code, "mat_code": "COMP-0001",
                 "mat_name": "贴片电容100nF", "remain_qty": 5000,
                 "warn_qty": 2000, "shift_date": shift_date}]

    elif api_name in ("wh_name", "wh_area", "out_bound_r", "in_bound_r"):
        rows = [{"id": 1, "line_code": line_code, "shift_date": shift_date,
                 "create_time": f"{shift_date} 10:00:00", "qty": 100}]

    else:
        rows = [{"id": 1, "line_code": line_code, "shift_date": shift_date,
                 "create_time": f"{shift_date} 10:00:00"}]

    return _ok({"data": rows, "count": len(rows)})


def _synth_get_idi_model_data(args: dict) -> str:
    model_id = int(args.get("model_id", 0))
    params_json_str = args.get("params_json", "[]")

    # Extract line, dates from params
    line_code = "S04"
    start_date = "2026-05-13"
    end_date = "2026-05-13"
    try:
        params = json.loads(params_json_str) if params_json_str else []
        for p in params:
            name = p.get("name", "")
            value = p.get("value", "")
            if name == "line" and value:
                line_code = str(value)
            elif name in ("start_shift_date", "start_tm") and value:
                start_date = str(value)[:10]
            elif name in ("end_shift_date", "end_tm") and value:
                end_date = str(value)[:10]
    except Exception:
        pass

    tmpl = _IDI_MODEL_TEMPLATES.get(model_id)
    if tmpl:
        row = {**tmpl["sample"]}
        # Update contextual fields
        row["line_code"] = line_code
        if "line" in tmpl["sample"]:
            row["line"] = line_code
        if "shift_date" in tmpl["sample"]:
            row["shift_date"] = start_date
        if "start_shift_date" in tmpl["sample"]:
            row["start_shift_date"] = start_date
        rows = [row]
    else:
        rows = [{**_DEFAULT_IDI_SAMPLE, "line_code": line_code,
                 "model_id": model_id, "shift_date": start_date}]

    return _ok({"data": rows, "count": len(rows)})


def synthetic_response(tool_name: str, args: dict) -> str:
    """Generate a realistic synthetic response when GT replay is exhausted."""
    dispatch = {
        "list_object_types": _synth_list_object_types,
        "get_object_type_detail": _synth_get_object_type_detail,
        "get_object_type_metrics": _synth_get_object_type_metrics,
        "get_object_type_data": _synth_get_object_type_data,
        "get_idi_model_data": _synth_get_idi_model_data,
    }
    fn = dispatch.get(tool_name)
    if fn:
        return fn(args)
    # Unknown tool: return empty success
    return _ok([])


# ── Replay Engine ─────────────────────────────────────────────────────────────

class ReplayToolEngine:
    """
    GT 轨迹重放引擎。

    对每个工具名维护一个 FIFO 队列（来自 ws_sample 的真实响应）。
    模型调用工具时：
      1. 从对应工具名的队列中 popleft()（真实响应）
      2. 队列为空时，调用 synthetic_response() 兜底
    """

    def __init__(self, ws_sample: dict):
        self._queues: dict[str, deque] = {}
        self._build(ws_sample)

    def _build(self, ws_sample: dict) -> None:
        msgs = ws_sample.get("messages", [])
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                if i + 1 < len(msgs) and msgs[i + 1].get("role") == "tool":
                    tool_resp = msgs[i + 1].get("content", "{}")
                    for tc in m["tool_calls"]:
                        name = tc["function"]["name"]
                        if name not in self._queues:
                            self._queues[name] = deque()
                        self._queues[name].append(tool_resp)

    def respond(self, tool_name: str, args: dict) -> str:
        q = self._queues.get(tool_name)
        while q:
            candidate = q.popleft()
            try:
                json.loads(candidate)  # validate — truncated responses will fail
                return candidate
            except (json.JSONDecodeError, ValueError):
                continue  # skip truncated, try next GT or fall to synthetic
        return synthetic_response(tool_name, args)

    def stats(self) -> dict:
        return {name: len(q) for name, q in self._queues.items()}
