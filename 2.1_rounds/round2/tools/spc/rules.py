"""Nelson规则引擎 + CPK计算，被 generator 和 eval 共用。"""
from __future__ import annotations
import statistics
from typing import Optional


def check_nelson_rules(data: list[float], mu: float, sigma: float) -> list[str]:
    """检查全部8条Nelson规则，返回违规规则名列表。

    Args:
        data:  测量值序列（通常25点）
        mu:    过程均值（中心线 CL）
        sigma: 过程标准差

    Returns:
        触发的规则名列表，如 ["rule1", "rule3"]，无违规则返回 []
    """
    if sigma <= 0:
        return []

    violations: list[str] = []
    n = len(data)

    ucl = mu + 3 * sigma
    lcl = mu - 3 * sigma
    z2 = mu + 2 * sigma
    z2l = mu - 2 * sigma
    z1 = mu + 1 * sigma
    z1l = mu - 1 * sigma

    # Rule 1：任意1点超出 UCL 或 LCL
    for x in data:
        if x > ucl or x < lcl:
            violations.append("rule1")
            break

    # Rule 2：连续9点在均值同侧
    for i in range(n - 8):
        w = data[i:i + 9]
        if all(x > mu for x in w) or all(x < mu for x in w):
            violations.append("rule2")
            break

    # Rule 3：连续6点单调递增或递减
    for i in range(n - 5):
        w = data[i:i + 6]
        if all(w[j] < w[j + 1] for j in range(5)):
            violations.append("rule3")
            break
        if all(w[j] > w[j + 1] for j in range(5)):
            violations.append("rule3")
            break

    # Rule 4：连续14点交替升降
    for i in range(n - 13):
        w = data[i:i + 14]
        # 模式1：先升后降交替
        p1 = all(
            (w[j] < w[j + 1]) if j % 2 == 0 else (w[j] > w[j + 1])
            for j in range(13)
        )
        # 模式2：先降后升交替
        p2 = all(
            (w[j] > w[j + 1]) if j % 2 == 0 else (w[j] < w[j + 1])
            for j in range(13)
        )
        if p1 or p2:
            violations.append("rule4")
            break

    # Rule 5：连续3点中2点在同侧2σ外
    for i in range(n - 2):
        w = data[i:i + 3]
        above = sum(1 for x in w if x > z2)
        below = sum(1 for x in w if x < z2l)
        if above >= 2 or below >= 2:
            violations.append("rule5")
            break

    # Rule 6：连续5点中4点在同侧1σ外
    for i in range(n - 4):
        w = data[i:i + 5]
        above = sum(1 for x in w if x > z1)
        below = sum(1 for x in w if x < z1l)
        if above >= 4 or below >= 4:
            violations.append("rule6")
            break

    # Rule 7：连续15点在1σ以内（μ-σ < x < μ+σ）
    for i in range(n - 14):
        w = data[i:i + 15]
        if all(z1l < x < z1 for x in w):
            violations.append("rule7")
            break

    # Rule 8：连续8点在1σ外且两侧均有
    for i in range(n - 7):
        w = data[i:i + 8]
        all_outside = all(x > z1 or x < z1l for x in w)
        if all_outside:
            has_above = any(x > z1 for x in w)
            has_below = any(x < z1l for x in w)
            if has_above and has_below:
                violations.append("rule8")
                break

    return violations


def compute_cpk(data: list[float], usl: float, lsl: float) -> Optional[float]:
    """计算CPK，返回 None 表示无法计算（标准差为0或数据太少）。"""
    result = compute_cpk_detailed(data, usl, lsl)
    return result.get("cpk")


def compute_cpk_detailed(data: list[float], usl: float, lsl: float) -> dict:
    """返回 xbar, s, cpu, cpl, cpk 五个字段。"""
    if len(data) < 2:
        return {"xbar": None, "s": None, "cpu": None, "cpl": None, "cpk": None}

    xbar = sum(data) / len(data)
    s = statistics.stdev(data)

    if s == 0:
        return {"xbar": round(xbar, 3), "s": 0.0,
                "cpu": None, "cpl": None, "cpk": None}

    cpu = (usl - xbar) / (3 * s)
    cpl = (xbar - lsl) / (3 * s)
    cpk = min(cpu, cpl)

    return {
        "xbar": round(xbar, 3),
        "s":    round(s, 3),
        "cpu":  round(cpu, 3),
        "cpl":  round(cpl, 3),
        "cpk":  round(cpk, 3),
    }


def find_rule_locations(data: list[float], mu: float, sigma: float) -> dict:
    """返回每条触发规则的具体点位信息（用于 think 块生成）。"""
    if sigma <= 0:
        return {}

    n = len(data)
    locations: dict[str, str] = {}

    ucl = mu + 3 * sigma
    lcl = mu - 3 * sigma
    z2 = mu + 2 * sigma
    z2l = mu - 2 * sigma
    z1 = mu + 1 * sigma
    z1l = mu - 1 * sigma

    # Rule 1
    for i, x in enumerate(data):
        if x > ucl:
            locations["rule1"] = f"第{i + 1}点{x:.3f}>{ucl:.3f}(UCL)"
            break
        if x < lcl:
            locations["rule1"] = f"第{i + 1}点{x:.3f}<{lcl:.3f}(LCL)"
            break

    # Rule 2
    for i in range(n - 8):
        w = data[i:i + 9]
        if all(x > mu for x in w):
            locations["rule2"] = f"第{i + 1}~{i + 9}点连续在均值上方"
            break
        if all(x < mu for x in w):
            locations["rule2"] = f"第{i + 1}~{i + 9}点连续在均值下方"
            break

    # Rule 3
    for i in range(n - 5):
        w = data[i:i + 6]
        if all(w[j] < w[j + 1] for j in range(5)):
            locations["rule3"] = f"第{i + 1}~{i + 6}点连续单调递增"
            break
        if all(w[j] > w[j + 1] for j in range(5)):
            locations["rule3"] = f"第{i + 1}~{i + 6}点连续单调递减"
            break

    # Rule 4
    for i in range(n - 13):
        w = data[i:i + 14]
        p1 = all(
            (w[j] < w[j + 1]) if j % 2 == 0 else (w[j] > w[j + 1])
            for j in range(13)
        )
        p2 = all(
            (w[j] > w[j + 1]) if j % 2 == 0 else (w[j] < w[j + 1])
            for j in range(13)
        )
        if p1 or p2:
            locations["rule4"] = f"第{i + 1}~{i + 14}点交替升降"
            break

    # Rule 5
    for i in range(n - 2):
        w = data[i:i + 3]
        above_idx = [i + j + 1 for j, x in enumerate(w) if x > z2]
        below_idx = [i + j + 1 for j, x in enumerate(w) if x < z2l]
        if len(above_idx) >= 2:
            pts = "、".join(f"第{p}点" for p in above_idx)
            locations["rule5"] = f"第{i + 1}~{i + 3}点中{pts}超2σ上侧"
            break
        if len(below_idx) >= 2:
            pts = "、".join(f"第{p}点" for p in below_idx)
            locations["rule5"] = f"第{i + 1}~{i + 3}点中{pts}超2σ下侧"
            break

    # Rule 6
    for i in range(n - 4):
        w = data[i:i + 5]
        above_idx = [i + j + 1 for j, x in enumerate(w) if x > z1]
        below_idx = [i + j + 1 for j, x in enumerate(w) if x < z1l]
        if len(above_idx) >= 4:
            pts = "、".join(f"第{p}点" for p in above_idx)
            locations["rule6"] = f"第{i + 1}~{i + 5}点中{pts}超1σ上侧"
            break
        if len(below_idx) >= 4:
            pts = "、".join(f"第{p}点" for p in below_idx)
            locations["rule6"] = f"第{i + 1}~{i + 5}点中{pts}超1σ下侧"
            break

    # Rule 7
    for i in range(n - 14):
        w = data[i:i + 15]
        if all(z1l < x < z1 for x in w):
            locations["rule7"] = f"第{i + 1}~{i + 15}点全部在1σ以内"
            break

    # Rule 8
    for i in range(n - 7):
        w = data[i:i + 8]
        all_outside = all(x > z1 or x < z1l for x in w)
        if all_outside:
            has_above = any(x > z1 for x in w)
            has_below = any(x < z1l for x in w)
            if has_above and has_below:
                locations["rule8"] = f"第{i + 1}~{i + 8}点全在1σ外且两侧均有"
                break

    return locations
