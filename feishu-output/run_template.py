#!/usr/bin/env python3
"""上传"训练数据生成 SKILL 需求表"到飞书。"""
import sys, os, json
SKILL_DIR = os.path.expanduser("~/.claude/skills/feishu-doc")
sys.path.insert(0, os.path.join(SKILL_DIR, "scripts"))
from feishu_api import FeishuAuth, process_blocks

with open(os.path.join(SKILL_DIR, "config.json")) as f:
    config = json.load(f)

TEMPLATE = """\
# 技能名称
填写

# 关联技能文档
填写（文档名称或链接，规则/知识已在此文档中）

# 智能体工作流
按顺序列出步骤，例如：输入 → 任务规划 → 工具调用×N → 最终输出
填写

# 终止条件
完成标志（什么情况视为任务成功完成）
  填写
失败标志（什么情况视为任务失败，需提前终止）
  填写
最大步骤数（允许的最多轮次 / 工具调用次数）
  填写


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一、用户输入模拟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心变量（什么参数在不同输入间变化，以及各自的取值范围）
  -
  -

必须覆盖的情形（边界、特殊子类型、正负样本要求等）
  -
  -

输入格式示例（贴 1–2 个完整的用户输入文本）

  [示例 1]


  [示例 2]



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二、输出步骤与评分
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─ 最终结果评分（必填）─────────────

输出格式
  填写（字段和结构，可贴示例）

标准答案来源（正确答案怎么算出来）
  填写（例如：跑规则引擎算法 / 查现有业务系统 / 人工标注 / 确定性公式）

评分方式（如何对比模型输出与标准答案）
  填写（例如：集合完全匹配得 1 分；字段缺失得 0 分；数值误差<0.1 得满分）

满分条件
  填写

零分 / 失败条件
  填写


─ 任务规划 DAG 评分（可选，不涉及则整块删除）─

计划输出格式
  填写

必须包含的步骤/节点
  填写

不允许出现的错误
  填写

标准答案来源
  填写

评分方式
  填写


─ 工具调用评分（可选，不涉及则整块删除）──────

工具签名（每个工具的函数名、参数名/类型/含义、返回值格式）

  工具 1：
    函数名：填写
    参数：填写（例如：sensor_id: str — 传感器编号；start_time: datetime — 查询起始时间）
    返回值：填写（例如：{"values": [float], "timestamps": [str]}）

  工具 2：
    函数名：填写
    参数：填写
    返回值：填写

调用顺序要求（有序 / 无序，有顺序要求则列出）
  填写

参数正确性评判（哪些参数必须精确，哪些允许模糊）
  填写

标准答案来源（正确的工具调用序列从哪里获取）
  填写

评分方式
  填写


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三、完整案例（至少 1 个）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[案例 1]

用户输入
  填写

步骤期望输出与得分（按工作流顺序逐步填写）
  步骤 1：填写输出 → 期望得分 填写
  步骤 2：填写输出 → 期望得分 填写
  最终结果：填写输出 → 期望得分 填写
"""

blocks = [
    {"type": "document_title", "text": "训练数据生成 SKILL 需求表"},
    {"type": "text", "elements": [
        {"text": "使用方式：", "bold": True},
        {"text": "复制下方模板，填写「填写」标记处，不涉及的可选块整块删除，完成后发给 AI 训练团队。"},
    ]},
    {"type": "code", "language": "plain", "content": TEMPLATE},
]

auth = FeishuAuth()
results = process_blocks(
    auth,
    blocks,
    default_owner=config.get("default_owner") or config.get("default_owner_email") or None,
)
print(json.dumps(results, ensure_ascii=False, indent=2))
