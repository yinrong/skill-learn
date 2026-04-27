"""结构化答案 → 含 <think> 推理链的自然语言输出。

多样性设计：每种表达均有多个语言变体，训练时随机选取，避免样本文字雷同。
"""
from __future__ import annotations
import os
import random
from typing import Optional


# ── 规则简称（用于输出中的标识）─────────────────────────────────────────────────
RULE_DESC = {
    "rule1": "任意1点超出UCL或LCL（突发性异常）",
    "rule2": "连续9点在均值同侧（均值漂移）",
    "rule3": "连续6点单调递增或递减（趋势性漂移）",
    "rule4": "连续14点交替升降（周期性干扰）",
    "rule5": "连续3点中2点在同侧2σ外（过程能力早期恶化）",
    "rule6": "连续5点中4点在同侧1σ外（均值偏移预警）",
    "rule7": "连续15点在1σ以内（测量系统分辨率不足或数据造假）",
    "rule8": "连续8点在1σ外且两侧均有（双峰分布/混线混料）",
}

# ── 规则完整定义（用于 no_skill 风格嵌入 think 块）───────────────────────────────
RULE_FULL_DEF = {
    "rule1": "任意1点超出 UCL（均值+3σ）或 LCL（均值-3σ）→ 突发异常",
    "rule2": "连续9点全部在均值同一侧（全>均值 或 全<均值）→ 均值系统性漂移",
    "rule3": "连续6点严格单调递增 或 严格单调递减 → 趋势性漂移",
    "rule4": "连续14点严格交替升降（每相邻两点一高一低）→ 周期性干扰",
    "rule5": "连续3点中有2点在同侧2σ～3σ区间（超过2σ但未超3σ）→ 过程能力早期恶化",
    "rule6": "连续5点中有4点在同侧1σ～3σ区间（超过1σ但未超3σ）→ 均值偏移预警",
    "rule7": "连续15点全部在均值±1σ以内（|偏差|<σ）→ 测量分辨率不足或数据被人为压缩",
    "rule8": "连续8点全在均值±1σ以外，且上下两侧都有分布 → 双峰分布或混线混料",
}

# ── 规则 think 行——未触发，多种表达 ──────────────────────────────────────────────
_NOT_TRIGGERED: dict[str, list[str]] = {
    "rule1": [
        "  rule1：{desc} → 未触发。",
        "  rule1（突发异常检测）：所有点均在控制限 [{lcl}, {ucl}] 内，未触发。",
        "  rule1：检查UCL={ucl}、LCL={lcl}，25点无超限，✓ 未触发。",
        "  rule1 ✗ 未触发：数据范围内无点超出3σ控制限。",
        "  rule1：最大值={max_val:.3f}，最小值={min_val:.3f}，UCL={ucl}，LCL={lcl} → 未触发。",
    ],
    "rule2": [
        "  rule2：{desc} → 未触发。",
        "  rule2（均值漂移）：连续9点同侧条件未满足，过程围绕均值波动，未触发。",
        "  rule2：未发现连续9点全在均值 {mu} 同侧，✓ 未触发。",
        "  rule2 ✗ 未触发：数据在均值两侧正常交替分布。",
        "  rule2：逐段检查9点窗口，未发现持续性单侧集中，未触发。",
    ],
    "rule3": [
        "  rule3：{desc} → 未触发。",
        "  rule3（趋势检测）：未发现连续6点单调序列，趋势检测通过，未触发。",
        "  rule3：检查所有6点滑动窗口，无严格单调递增/递减序列，✓ 未触发。",
        "  rule3 ✗ 未触发：过程无明显漂移趋势。",
        "  rule3：数据波动随机，无连续单向趋势，未触发。",
    ],
    "rule4": [
        "  rule4：{desc} → 未触发。",
        "  rule4（周期干扰）：未发现连续14点严格交替升降模式，未触发。",
        "  rule4：检查14点zigzag窗口，无周期性振荡，✓ 未触发。",
        "  rule4 ✗ 未触发：相邻点涨跌方向不构成持续交替。",
        "  rule4：过程波动不呈现规律性交替，未触发。",
    ],
    "rule5": [
        "  rule5：{desc} → 未触发。",
        "  rule5（能力恶化）：3点窗口内未出现2点同侧超2σ，未触发。",
        "  rule5：2σ区间=[{two_sigma_lo:.3f},{two_sigma_hi:.3f}]，3点中无2点集中于同侧外圈，✓ 未触发。",
        "  rule5 ✗ 未触发：2σ外侧点分布稀少或不集中。",
        "  rule5：逐3点窗口检查，无同侧2σ外2点集中，未触发。",
    ],
    "rule6": [
        "  rule6：{desc} → 未触发。",
        "  rule6（偏移预警）：5点窗口内未出现4点同侧超1σ，未触发。",
        "  rule6：1σ区间=[{one_sigma_lo:.3f},{one_sigma_hi:.3f}]，5点中无4点偏向同侧1σ外，✓ 未触发。",
        "  rule6 ✗ 未触发：1σ外侧点无明显单侧集中趋势。",
        "  rule6：5点滑动检查，无连续4点同侧偏移信号，未触发。",
    ],
    "rule7": [
        "  rule7：{desc} → 未触发。",
        "  rule7（分辨率检测）：未出现连续15点全在±1σ以内，数据离散度正常，未触发。",
        "  rule7：1σ带=[{one_sigma_lo:.3f},{one_sigma_hi:.3f}]，15点内1σ集中条件不满足，✓ 未触发。",
        "  rule7 ✗ 未触发：数据标准差在正常范围内，测量系统无分辨率问题。",
        "  rule7：数据超过±1σ的点足够多，无异常紧缩现象，未触发。",
    ],
    "rule8": [
        "  rule8：{desc} → 未触发。",
        "  rule8（双峰检测）：未发现连续8点均在1σ外且两侧兼有，未触发。",
        "  rule8：检查8点窗口，无全部在±1σ外且双侧分布的模式，✓ 未触发。",
        "  rule8 ✗ 未触发：数据不呈双峰分布特征。",
        "  rule8：8点持续外侧双峰条件不满足，未触发。",
    ],
}

# ── 规则 think 行——已触发，多种表达 ─────────────────────────────────────────────
_TRIGGERED: dict[str, list[str]] = {
    "rule1": [
        "  rule1：{desc} → **触发**（{loc}）。",
        "  rule1（突发异常）→ 触发！{loc}，超出控制限。",
        "  rule1：检测到点位超出UCL/LCL控制限 → 触发。位置：{loc}。",
        "  rule1 ⚠ 触发：{loc}，发生突发性异常，须立即响应。",
        "  rule1：{loc} → 超限，判定为突发异常，触发。",
    ],
    "rule2": [
        "  rule2：{desc} → **触发**（{loc}）。",
        "  rule2（均值漂移）→ 触发！{loc}，连续点单侧集中。",
        "  rule2：发现连续9点在均值同侧 → 触发。区段：{loc}。",
        "  rule2 ⚠ 触发：{loc}，过程均值发生系统性偏移。",
        "  rule2：{loc} → 9点持续同侧，均值漂移信号，触发。",
    ],
    "rule3": [
        "  rule3：{desc} → **触发**（{loc}）。",
        "  rule3（趋势漂移）→ 触发！{loc}，单调趋势已成形。",
        "  rule3：检测到连续6点单调序列 → 触发。位置：{loc}。",
        "  rule3 ⚠ 触发：{loc}，发现持续性趋势，需排查过程变化。",
        "  rule3：{loc} → 6点单调变化，趋势漂移确认，触发。",
    ],
    "rule4": [
        "  rule4：{desc} → **触发**（{loc}）。",
        "  rule4（周期干扰）→ 触发！{loc}，规律性交替振荡。",
        "  rule4：检测到14点严格zigzag模式 → 触发。区段：{loc}。",
        "  rule4 ⚠ 触发：{loc}，周期性干扰信号，需排查振荡源。",
        "  rule4：{loc} → 14点交替升降，周期干扰确认，触发。",
    ],
    "rule5": [
        "  rule5：{desc} → **触发**（{loc}）。",
        "  rule5（能力早期恶化）→ 触发！{loc}，3点中2点超2σ。",
        "  rule5：发现3点窗口内2点在同侧2σ外 → 触发。位置：{loc}。",
        "  rule5 ⚠ 触发：{loc}，过程能力早期恶化信号。",
        "  rule5：{loc} → 2点同侧超2σ，能力恶化预警，触发。",
    ],
    "rule6": [
        "  rule6：{desc} → **触发**（{loc}）。",
        "  rule6（偏移预警）→ 触发！{loc}，5点中4点偏向同侧1σ外。",
        "  rule6：发现5点窗口内4点在同侧1σ外 → 触发。位置：{loc}。",
        "  rule6 ⚠ 触发：{loc}，均值偏移预警信号。",
        "  rule6：{loc} → 4点持续同侧超1σ，均值偏移确认，触发。",
    ],
    "rule7": [
        "  rule7：{desc} → **触发**（{loc}）。",
        "  rule7（分辨率/造假）→ 触发！{loc}，数据异常紧缩在1σ内。",
        "  rule7：检测到连续15点全在±1σ以内 → 触发。区段：{loc}。",
        "  rule7 ⚠ 触发：{loc}，测量系统或数据可信度存疑。",
        "  rule7：{loc} → 15点连续紧缩，测量分辨率不足信号，触发。",
    ],
    "rule8": [
        "  rule8：{desc} → **触发**（{loc}）。",
        "  rule8（双峰分布）→ 触发！{loc}，8点在1σ外且两侧均有。",
        "  rule8：检测到8点连续在±1σ外且双侧分布 → 触发。位置：{loc}。",
        "  rule8 ⚠ 触发：{loc}，双峰或混线混料信号，须立即排查。",
        "  rule8：{loc} → 8点持续外侧且两侧兼有，双峰确认，触发。",
    ],
}

# ── think 块开头语，多种 ───────────────────────────────────────────────────────
_THINK_OPENINGS = [
    "逐条检查 Nelson 规则：",
    "依次对 Nelson 8条规则进行判断：",
    "Nelson 规则检查（rule1～rule8）：",
    "按顺序逐条核查8条控制图规则：",
    "SPC 异常检测——逐一验证 Nelson 规则：",
]

_THINK_OPENINGS_NOSKILL = [
    "回忆 Nelson 8条规则定义并逐一检查：",
    "根据记忆中的 Nelson 规则定义，逐条检验：",
    "内化 SPC 知识，依次检查 Nelson 8条规则：",
    "从记忆提取 Nelson 规则，逐项核对数据：",
    "调用 Nelson 规则知识，逐一判断当前数据：",
]

_CPK_OPENINGS = [
    "计算 CPK：",
    "过程能力计算（CPK）：",
    "CPK 指数计算：",
    "计算过程能力指数 CPK：",
]

# ── 正文：异常判断句，多种 ───────────────────────────────────────────────────────
_ABNORMAL_TMPL = [
    "**异常判断**：检出{n}条违规——{descs}。",
    "**异常检测结果**：共发现{n}条规则触发——{descs}。",
    "**SPC 判定**：触发{n}条 Nelson 规则：{descs}。",
    "**控制图异常**：{n}条规则报警——{descs}。",
]

_NORMAL_TMPL = [
    "**异常判断**：全部8条 Nelson 规则均未触发，当前过程受控。",
    "**SPC 判定**：8条 Nelson 规则全部通过，过程处于受控状态。",
    "**控制图检查**：无异常规则触发，过程受控✓。",
    "**异常检测结果**：未检出任何规则违规，当前过程正常。",
]

_MULTI_RULE_SUFFIX = [
    "多条规则同时触发表明过程存在系统性异常，需立即处置。",
    "多规则并发提示过程已出现系统性问题，须紧急响应。",
    "同时触发多条规则，说明过程异常原因可能较为复杂，需综合排查。",
]

# ── 正文：CPK 描述，多种 ─────────────────────────────────────────────────────────
_CPK_SECTION_TMPL = [
    "**过程能力**：CPK={cpk}，等级{grade}（{action}）。",
    "**CPK 指数**：{cpk}，过程能力等级：{grade}，建议：{action}。",
    "**过程能力评估**：CPK = {cpk}（{grade}级，{action}）。",
    "**CPK**：{cpk} → 等级 {grade}，{action}。",
]

# ── 各规则处置建议（每条规则多个池，随机选其中一个池）─────────────────────────────
DISPOSAL_POOLS: dict[str, list[list[str]]] = {
    "rule1": [
        [
            "通知当班工艺工程师立即到现场确认该批次是否继续流转",
            "封存异常点前后5个采样点对应的实物，等待品质工程师判定",
            "排查该时段的原料批次号、设备参数记录和操作员操作日志",
            "根因确认后填写《异常处理报告》，完成闭环方可恢复正常生产",
        ],
        [
            "立即暂停该批次生产，通知质量工程师进行紧急评审",
            "隔离异常点对应时间段内的所有产品，待复测确认合格后方可放行",
            "调取异常点前1小时内的设备运行参数曲线，查找突变来源",
            "完成根因分析并经工艺主管审批后方可恢复正常检验节拍",
        ],
        [
            "第一时间通知现场质量人员，禁止该批次在制品继续流转",
            "复测当前点及相邻3点，确认是否为偶发还是系统性问题",
            "检查测量设备是否正常、操作方法是否一致",
            "记录异常发生时间、工位、操作员及设备编号，纳入异常台账",
        ],
    ],
    "rule2": [
        [
            "通知当班工艺工程师检查过程均值是否发生系统性偏移",
            "对比异常窗口前后的原料批次记录和设备保养记录",
            "将检验频次加倍，直到连续5点均回落到均值附近",
            "若偏移来自设备磨损，安排预防性维修并重新进行过程能力确认",
        ],
        [
            "确认过去9个采样点的实物是否已发生质量偏移，必要时全检该批次",
            "核查是否在漂移起点前后发生了换料、换班或工艺参数调整",
            "加密采样频率，连续观察5点是否恢复到均值两侧正常波动",
            "若均值持续偏移超过1σ，申请工艺参数变更评审",
        ],
    ],
    "rule3": [
        [
            "通知工艺工程师确认是否存在刀具磨损、夹具松动或材料渐变等因素",
            "立即核查趋势起点前后的设备参数调整记录",
            "在趋势未超出规格限前提前补偿工艺参数，防止产生不合格品",
            "记录趋势持续点数和斜率，更新预警阈值以便早期干预",
        ],
        [
            "立即检查趋势方向对应的设备磨损件（如刀片、导轨、夹具）",
            "评估若趋势延续，距离规格限还有多少裕量，计算剩余可用时间",
            "在趋势未失控前提前调整工艺补偿量，并记录调整时间和幅度",
            "复核前6点数据的测量环境（温湿度、量具温漂），排除测量误差",
        ],
    ],
    "rule4": [
        [
            "排查是否存在周期性干扰源（换班、循环补料、环境温湿度波动）",
            "与生产排班记录对比，确认振荡周期是否与换班/换料周期吻合",
            "对干扰源实施物理隔离或工艺参数补偿",
            "处置后连续采集20点，验证周期性振荡消失后方可降低检验频次",
        ],
        [
            "分析交替振荡的周期，与换班时间、补料节拍、冷却循环等对比",
            "检查设备是否有周期性震动、热膨胀或液压/气压脉冲问题",
            "隔离可疑干扰源，观察振荡是否消失",
            "建立专项监控，连续10次检验结果稳定后方可关闭该项监控",
        ],
    ],
    "rule5": [
        [
            "立即复测同批次相邻3个点，确认是否为测量误差",
            "若复测确认异常，通知工艺工程师检查设备精度和原料质量",
            "将检验频次加倍，连续监控后续5个采样点",
            "启动过程能力专项分析，评估是否需要调整控制限或工艺参数",
        ],
        [
            "对触发的2个外侧点进行复测，排除单次测量误差",
            "检查同批次原料的来料检验报告，是否存在批间差异",
            "加密后续监控频次，若连续3组同样触发则上报质量主管",
            "分析2σ外侧点的具体数值与均值的偏差，评估过程能力恶化速度",
        ],
    ],
    "rule6": [
        [
            "通知工艺工程师检查均值偏移是否由原料批次或设备参数变化引起",
            "核查触发窗口对应时段的换料记录、设备温度和压力参数",
            "将检验频次加倍，观察后续5点是否持续偏移",
            "若确认均值偏移，调整中心值并重新计算控制限，填写变更记录",
        ],
        [
            "确认4个偏移点对应的生产时间段，查找同期工艺参数变化",
            "检查偏移方向（上偏/下偏）对应可能的物理原因（如温度、压力、材料）",
            "在恢复正常前保持加密检验，连续5点回归正常后恢复标准频次",
            "若偏移持续，启动过程重新定中（re-centering）程序",
        ],
    ],
    "rule7": [
        [
            "立即对测量设备进行量具重复性与再现性（GR&R）分析",
            "核查近期是否有操作员更换或测量方法变更",
            "抽取同批次产品送实验室复测，与在线测量结果对比",
            "若确认测量系统失效，停止使用当前量具并安排计量校准",
        ],
        [
            "排查数据采集是否存在人为填写、自动复制或系统故障问题",
            "更换测量员重新抽测5个随机点，与原始数据对比",
            "检查量具分辨率是否满足被测量变差的要求（一般要求分辨率<10%）",
            "若确认数据造假，按质量体系不符合项处理，启动内部调查程序",
        ],
    ],
    "rule8": [
        [
            "立即核查当前生产批次是否存在混料情况，隔离可疑批次",
            "检查是否有两条以上生产线的产品混入同一控制图",
            "对当前批次全检，按生产来源分类统计各自的均值和标准差",
            "确认混料根因后，建立物料防混管控流程，必要时分批建立独立控制图",
        ],
        [
            "分析8个外侧点的上下分布规律，判断是否呈现两个不同的分布中心",
            "追溯这8个点对应的原料批号、设备参数和操作人员，查找分层因素",
            "将数据按可能的分层因素拆分后分别绘制控制图，验证双峰假设",
            "若确认混线，立即物理分隔两类产品来源，独立建立各自的控制图",
        ],
    ],
    "normal": [
        [
            "当前过程受控，维持现有检验频次和工艺参数",
            "继续按计划进行定期预防性维护和量具校准",
            "持续累积数据，在下一计划点重新评估过程能力指数",
            "若CPK等级为一般（1.00~1.33），可安排工艺改善立项以提升稳定性",
        ],
        [
            "无需立即干预，按既定生产计划继续执行",
            "维持当前检验频次，下一轮计划保养后复核过程能力",
            "建立数据历史档案，持续跟踪CPK趋势，提前发现潜在恶化",
            "当CPK下降至1.33以下时，触发过程能力改善立项",
        ],
        [
            "过程处于受控状态，保持现有操作规范不变",
            "定期按保养计划维护设备，避免引入新的变差源",
            "可考虑适当降低检验频次以节约资源，但须保证统计可信度",
            "持续监控趋势，CPK > 1.67 时可进一步优化检验策略",
        ],
    ],
}

# ── CPK 等级 ──────────────────────────────────────────────────────────────────
def cpk_grade(cpk: float) -> tuple[str, str]:
    if cpk >= 1.67:
        return "优秀", "可降低检验频次"
    elif cpk >= 1.33:
        return "良好", "维持现有策略"
    elif cpk >= 1.00:
        return "一般", "加强监控"
    else:
        return "不合格", "立即排查，暂停出货"


# ── 核心格式化函数 ─────────────────────────────────────────────────────────────
def format_output(
    data: list[float],
    mu: float,
    sigma: float,
    usl: float,
    lsl: float,
    violations: list[str],
    cpk_detail: dict,
    locations: Optional[dict] = None,
    style: str = "detailed",
    rng: Optional[random.Random] = None,
) -> str:
    """生成含 <think> 块的多样化推理链输出。

    style:
        "detailed"   - 逐条列举所有规则，每行随机短语（默认）
        "brief"      - 只列触发规则，其余合并
        "calc_first" - 先算控制限再检规则
        "no_skill"   - 无Skill文档：think块内嵌完整规则定义
    rng: 随机数生成器；None 时使用全局 random（每次不同）
    """
    if locations is None:
        locations = {}
    if rng is None:
        rng = random.Random()

    ucl = round(mu + 3 * sigma, 3)
    lcl = round(mu - 3 * sigma, 3)

    xbar = cpk_detail.get("xbar")
    s    = cpk_detail.get("s")
    cpu  = cpk_detail.get("cpu")
    cpl  = cpk_detail.get("cpl")
    cpk  = cpk_detail.get("cpk")

    s_f = float(s) if s is not None else sigma
    xbar_f = float(xbar) if xbar is not None else mu
    one_sigma_lo = round(xbar_f - s_f, 3)
    one_sigma_hi = round(xbar_f + s_f, 3)
    two_sigma_lo = round(xbar_f - 2 * s_f, 3)
    two_sigma_hi = round(xbar_f + 2 * s_f, 3)
    max_val = max(data)
    min_val = min(data)

    def _fmt_rule_line(rule_id: str) -> str:
        """生成单条规则检查行，随机选短语变体。"""
        desc = RULE_DESC[rule_id]
        fmt_vars = dict(
            ucl=ucl, lcl=lcl, mu=mu,
            one_sigma_lo=one_sigma_lo, one_sigma_hi=one_sigma_hi,
            two_sigma_lo=two_sigma_lo, two_sigma_hi=two_sigma_hi,
            max_val=max_val, min_val=min_val,
        )
        if rule_id in violations:
            loc = locations.get(rule_id, "触发")
            tmpl = rng.choice(_TRIGGERED[rule_id])
            return tmpl.format(desc=desc, loc=loc, **fmt_vars)
        else:
            tmpl = rng.choice(_NOT_TRIGGERED[rule_id])
            return tmpl.format(desc=desc, **fmt_vars)

    # ── Style: brief ───────────────────────────────────────────────────────────
    if style == "brief":
        think_lines = [
            f"控制限：UCL={ucl}，LCL={lcl}，均值={xbar}，s={s}"
        ]
        if violations:
            think_lines.append("触发规则：")
            for rule_id in violations:
                loc = locations.get(rule_id, "触发")
                desc = RULE_DESC[rule_id]
                think_lines.append(f"  {rule_id}（{desc}）→ {loc}")
            untriggered = [r for r in [f"rule{i}" for i in range(1, 9)] if r not in violations]
            think_lines.append(f"其余规则（{', '.join(untriggered)}）：均未触发")
        else:
            think_lines.append("rule1～rule8 全部通过，无触发。")
        think_lines += ["", f"{rng.choice(_CPK_OPENINGS)}", f"  CPK=min({cpu},{cpl})={cpk}"]

    # ── Style: calc_first ──────────────────────────────────────────────────────
    elif style == "calc_first":
        think_lines = [
            "第一步：计算控制限和过程能力",
            f"  样本均值 x̄ = {xbar}，样本标准差 s = {s}",
            f"  UCL = x̄ + 3s = {xbar} + 3×{s} = {ucl}",
            f"  LCL = x̄ - 3s = {xbar} - 3×{s} = {lcl}",
            f"  CPU = (USL-x̄)/(3s) = ({usl}-{xbar})/(3×{s}) = {cpu}",
            f"  CPL = (x̄-LSL)/(3s) = ({xbar}-{lsl})/(3×{s}) = {cpl}",
            f"  CPK = min(CPU, CPL) = min({cpu},{cpl}) = {cpk}",
            "",
            "第二步：逐条检查 Nelson 规则（UCL={ucl}，LCL={lcl}）：".format(ucl=ucl, lcl=lcl),
        ]
        for rule_id in [f"rule{i}" for i in range(1, 9)]:
            think_lines.append(_fmt_rule_line(rule_id))

    # ── Style: no_skill ────────────────────────────────────────────────────────
    elif style == "no_skill":
        opening = rng.choice(_THINK_OPENINGS_NOSKILL)
        think_lines = [
            opening,
            f"  （计算基准：UCL={ucl}，LCL={lcl}，均值={xbar}，σ={s}，"
            f"1σ带=[{one_sigma_lo},{one_sigma_hi}]，2σ带=[{two_sigma_lo},{two_sigma_hi}]）",
        ]
        for rule_id in [f"rule{i}" for i in range(1, 9)]:
            defn = RULE_FULL_DEF[rule_id]
            loc = locations.get(rule_id, "触发")
            if rule_id in violations:
                tmpl = rng.choice([
                    f"  {rule_id}：{defn} → 触发（{loc}）。",
                    f"  {rule_id}：[定义：{defn}] 检查结果：触发，{loc}。",
                    f"  {rule_id} ⚠ 触发：{loc}。规则定义：{defn}。",
                    f"  {rule_id}：{loc} → 符合触发条件（{defn}）。",
                ])
            else:
                tmpl = rng.choice([
                    f"  {rule_id}：{defn} → 未触发。",
                    f"  {rule_id}：[定义：{defn}] 检查结果：正常，未触发。",
                    f"  {rule_id} ✓ 未触发：不满足 {defn.split('→')[0].strip()} 的条件。",
                    f"  {rule_id}：当前数据不满足触发条件（{defn}），通过。",
                ])
            think_lines.append(tmpl)
        think_lines += [
            "",
            rng.choice(_CPK_OPENINGS),
            f"  CPU = ({usl}-{xbar})/(3×{s}) = {cpu}",
            f"  CPL = ({xbar}-{lsl})/(3×{s}) = {cpl}",
            f"  CPK = min({cpu},{cpl}) = {cpk}",
        ]

    # ── Style: detailed（默认）────────────────────────────────────────────────
    else:
        opening = rng.choice(_THINK_OPENINGS)
        think_lines = [opening]
        for rule_id in [f"rule{i}" for i in range(1, 9)]:
            think_lines.append(_fmt_rule_line(rule_id))
        think_lines += ["", rng.choice(_CPK_OPENINGS)]
        if xbar is not None and s is not None and s_f > 0:
            # 随机选CPK计算写法
            cpk_fmt = rng.choice([
                [f"  x̄={xbar}，s={s}", f"  CPU=({usl}-{xbar})/(3×{s})={cpu}",
                 f"  CPL=({xbar}-{lsl})/(3×{s})={cpl}", f"  CPK=min({cpu},{cpl})={cpk}"],
                [f"  均值={xbar}，标准差={s}",
                 f"  CPU = (USL-x̄)/(3s) = ({usl}-{xbar})/(3×{s}) = {cpu}",
                 f"  CPL = (x̄-LSL)/(3s) = ({xbar}-{lsl})/(3×{s}) = {cpl}",
                 f"  CPK = min(CPU,CPL) = min({cpu},{cpl}) = {cpk}"],
                [f"  x̄={xbar}  s={s}  UCL={ucl}  LCL={lcl}",
                 f"  上侧：CPU=({usl}-{xbar})/(3×{s})={cpu}",
                 f"  下侧：CPL=({xbar}-{lsl})/(3×{s})={cpl}",
                 f"  CPK=min({cpu},{cpl})={cpk}"],
            ])
            think_lines.extend(cpk_fmt)
        else:
            think_lines.append("  标准差为0，CPK无法计算。")

    think_block = "\n".join(think_lines)

    # ── 正文：异常判断 ─────────────────────────────────────────────────────────
    if violations:
        n = len(violations)
        descs = "；".join(f"{v}（{RULE_DESC[v]}）" for v in violations)
        tmpl = rng.choice(_ABNORMAL_TMPL)
        abnormal_section = tmpl.format(n=n, descs=descs)
        if n >= 2:
            abnormal_section += rng.choice(_MULTI_RULE_SUFFIX)
    else:
        abnormal_section = rng.choice(_NORMAL_TMPL)

    # ── 正文：CPK ─────────────────────────────────────────────────────────────
    if cpk is not None:
        grade_name, grade_action = cpk_grade(cpk)
        tmpl = rng.choice(_CPK_SECTION_TMPL)
        cpk_section = tmpl.format(cpk=cpk, grade=grade_name, action=grade_action)
    else:
        cpk_section = "**过程能力**：CPK无法计算（数据标准差为0）。"

    # ── 正文：处置建议 ─────────────────────────────────────────────────────────
    if violations:
        primary = violations[0]
        pools = DISPOSAL_POOLS.get(primary, DISPOSAL_POOLS["normal"])
        items = list(rng.choice(pools))  # 随机选一个池
        for v in violations[1:]:
            extra_pools = DISPOSAL_POOLS.get(v, [])
            if extra_pools:
                extra_pool = rng.choice(extra_pools)
                items.append(extra_pool[0])
    else:
        items = list(rng.choice(DISPOSAL_POOLS["normal"]))

    # 随机顺序选3~4条（不完全固定顺序）
    n_items = rng.randint(3, min(4, len(items)))
    selected = items[:n_items]

    # 随机选列表格式（①②③ 或 1. 2. 3. 或 - ）
    fmt_style = rng.choice(["circle", "number", "dash"])
    if fmt_style == "circle":
        _prefix = ["①", "②", "③", "④", "⑤"]
        disposal_lines = ["**处置建议**："]
        for idx, item in enumerate(selected):
            disposal_lines.append(f"{_prefix[idx]} {item}")
    elif fmt_style == "number":
        disposal_lines = ["**处置建议**（按优先级）："]
        for idx, item in enumerate(selected):
            disposal_lines.append(f"{idx+1}. {item}")
    else:
        disposal_lines = ["**处置建议**："]
        for item in selected:
            disposal_lines.append(f"- {item}")

    disposal_section = "\n".join(disposal_lines)

    output = (
        f"<think>\n{think_block}\n</think>\n\n"
        f"{abnormal_section}\n\n"
        f"{cpk_section}\n\n"
        f"{disposal_section}"
    )
    return output


def format_output_with_llm(
    data: list[float],
    mu: float,
    sigma: float,
    usl: float,
    lsl: float,
    violations: list[str],
    cpk_detail: dict,
    locations: Optional[dict] = None,
    style: str = "detailed",
    rng: Optional[random.Random] = None,
) -> str:
    """先生成模板输出，若环境变量中有 API 配置则调用 LLM 润色。"""
    raw = format_output(data, mu, sigma, usl, lsl, violations, cpk_detail, locations, style=style, rng=rng)
    api_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    if api_url and api_key:
        try:
            return _llm_polish(raw, api_url, api_key)
        except Exception:
            pass
    return raw


def _llm_polish(raw: str, api_url: str, api_key: str) -> str:
    import json, urllib.request
    prompt = (
        "下面是一段SPC分析报告，请在保持所有数值、规则编号和格式标记（<think>、**加粗**、编号）"
        "完全不变的前提下，对文字表述做适当润色使其更自然流畅。直接输出润色后的内容，不加说明。\n\n"
        + raw
    )
    body = json.dumps({
        "model": "claude-sonnet-4-6", "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/v1/messages",
        data=body,
        headers={"Content-Type": "application/json", "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result["content"][0]["text"]
