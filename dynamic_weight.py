"""V7.0.0 决策引擎 —— 动态权重模块（dynamic weight）。

需求第七、八、十二节：

* 历史证据质量高（HIGH）时，历史相似情景必须是权重最大且 ≥50%；
* 历史证据不足（LOW）时，禁止强行把历史模块提高到 50%，
  自动降级为“资讯 + 市场反应 + 当前行情 + 原有因子”为主；
* 所有权重总和恒等于 100%，并且输出实际使用的权重与理由。

三档设计（A/B/C）：

====  =====================  =====================================
档位  历史权重区间            触发条件
====  =====================  =====================================
A     50% — 65%              证据 HIGH；证据分越高权重越高
B     20% — 50%              证据 MEDIUM
C     0% — 20%               证据 LOW；证据分越低权重越低
====  =====================  =====================================

剩余权重在其余三个模块（最新资讯与事件反应、当前行情与技术形态、
原有因子交叉验证）之间按基础占比分配；某模块数据不可用时，其份额
按比例转给其余可用模块，保证总和始终为 100%。
"""

from __future__ import annotations

from typing import Any

import numpy as np

MODULE_KEYS = ("historical", "news_event", "market_technical", "legacy_factor")

MODULE_NAMES = {
    "historical": "历史相似情景",
    "news_event": "最新资讯与事件—市场反应",
    "market_technical": "当前行情与技术形态",
    "legacy_factor": "原有因子交叉验证",
}

# 非历史模块之间的基础占比（历史权重确定后按此瓜分剩余权重）。
RESIDUAL_BASE_SHARES: dict[str, float] = {
    "news_event": 0.38,
    "market_technical": 0.34,
    "legacy_factor": 0.28,
}

HIGH_HISTORICAL_MIN = 0.50
HIGH_HISTORICAL_MAX = 0.65
MEDIUM_HISTORICAL_MIN = 0.20
MEDIUM_HISTORICAL_MAX = 0.50
LOW_HISTORICAL_MAX = 0.20


def _historical_weight(level: str, evidence_score: float) -> tuple[float, str]:
    if level == "HIGH":
        span = HIGH_HISTORICAL_MAX - HIGH_HISTORICAL_MIN
        extra = float(np.clip((evidence_score - 45.0) / 55.0, 0.0, 1.0)) * span
        weight = HIGH_HISTORICAL_MIN + extra
        reason = f"证据HIGH：历史权重提升到{weight:.1%}（50%—65%档），为最大权重模块。"
        return weight, reason
    if level == "MEDIUM":
        span = MEDIUM_HISTORICAL_MAX - MEDIUM_HISTORICAL_MIN
        extra = float(np.clip((evidence_score - 25.0) / 25.0, 0.0, 1.0)) * span
        weight = MEDIUM_HISTORICAL_MIN + extra
        reason = f"证据MEDIUM：历史权重{weight:.1%}（20%—50%档），不强制成为最大模块。"
        return weight, reason
    weight = float(np.clip(evidence_score / 25.0, 0.0, 1.0)) * LOW_HISTORICAL_MAX
    reason = f"证据LOW：历史权重限制在{weight:.1%}（0%—20%档），禁止强行提高到50%。"
    return weight, reason


def determine_weights(
    evidence_level: str,
    evidence_score: float,
    modules_available: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """根据证据等级与模块可用性计算四模块权重。

    Args:
        evidence_level: HIGH / MEDIUM / LOW。
        evidence_score: 0—100 的历史证据分。
        modules_available: 各模块是否可用；historical 的可用性由证据等级
            决定，其余模块不可用时权重被重新分配。
    """
    availability = {key: True for key in MODULE_KEYS}
    if modules_available:
        for key in MODULE_KEYS:
            availability[key] = bool(modules_available.get(key, True))

    historical_weight, historical_reason = _historical_weight(evidence_level, float(evidence_score))
    if not availability["historical"]:
        historical_weight = 0.0
        historical_reason = "历史模块不可用：权重为0%，其余模块按基础占比分配。"
    residual = 1.0 - historical_weight
    residual_modules = [key for key in MODULE_KEYS[1:] if availability.get(key, True)]
    reasons = [historical_reason]

    weights: dict[str, float] = {key: 0.0 for key in MODULE_KEYS}
    weights["historical"] = historical_weight if availability["historical"] else 0.0
    if not availability["historical"]:
        residual = 1.0
        reasons.append("历史模块不可用，其权重已转给其余模块。")

    if not residual_modules:
        # 极端情况：其余模块全部不可用——把全部权重交回历史（若可用），否则均分。
        if availability["historical"]:
            weights["historical"] = 1.0
            reasons.append("其余模块均不可用，决策完全依赖历史证据。")
        else:
            for key in MODULE_KEYS:
                weights[key] = 0.25
            reasons.append("所有模块均不可用，使用等权占位并应显著降低置信度。")
    else:
        base_total = sum(RESIDUAL_BASE_SHARES[key] for key in residual_modules)
        unavailable_names = [
            MODULE_NAMES[key] for key in MODULE_KEYS[1:] if not availability.get(key, True)
        ]
        if unavailable_names:
            reasons.append("、".join(unavailable_names) + "数据不可用，其权重已按比例转给其余模块。")
        for key in residual_modules:
            weights[key] = residual * RESIDUAL_BASE_SHARES[key] / base_total

    # 数值规范化：四舍五入到 0.1% 后再把残差加回最大权重，保证严格求和为1。
    rounded = {key: round(value, 3) for key, value in weights.items()}
    remainder = round(1.0 - sum(rounded.values()), 3)
    if abs(remainder) >= 0.0005:
        largest = max(rounded, key=rounded.get)
        rounded[largest] = round(rounded[largest] + remainder, 3)

    total = float(sum(rounded.values()))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"动态权重总和为{total}，偏离100%，拒绝输出。")
    if evidence_level == "HIGH" and availability["historical"]:
        if rounded["historical"] < HIGH_HISTORICAL_MIN - 1e-9:
            raise ValueError("证据HIGH但历史权重低于50%，违反决策规则。")
        other_maximum = max(rounded[key] for key in MODULE_KEYS[1:])
        if rounded["historical"] < other_maximum - 1e-9:
            raise ValueError("证据HIGH但历史模块不是最大权重模块，违反决策规则。")

    return {
        "weights": rounded,
        "weights_pct": {key: round(value * 100, 1) for key, value in rounded.items()},
        "weight_names": {MODULE_NAMES[key]: round(value * 100, 1) for key, value in rounded.items()},
        "tier": {"HIGH": "A", "MEDIUM": "B", "LOW": "C"}.get(evidence_level, "C"),
        "evidence_level": evidence_level,
        "evidence_score": float(evidence_score),
        "modules_available": availability,
        "reasons": reasons,
        "total": total,
    }
