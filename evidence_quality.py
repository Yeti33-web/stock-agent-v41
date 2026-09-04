"""V7.0.0 决策引擎 —— 历史证据质量评估（evidence quality）。

需求第四节：历史相似度 ≠ 历史案例可信度。本模块把证据拆成三层：

    Historical Evidence Score = Similarity × Reliability × DataQuality

* **Similarity（相似度）**：被选中案例的平均相似度（0—100）。
* **Reliability（可信度）**：样本数量、结果一致性、方向跨期限一致性、
  相似度分布集中程度。
* **DataQuality（数据质量）**：可计算特征覆盖率、情景维度组覆盖率、
  历史长度、行情新鲜度。历史不可得的维度（宏观/行业/政策/事件/预期）
  在这里体现为数据质量下降，而不是被悄悄忽略。

分级阈值（可解释、可回测调整）：

* HIGH：Evidence ≥ 45 且 样本 ≥ 8 且 平均相似度 ≥ 62；
* MEDIUM：Evidence ≥ 25 且 样本 ≥ 6；
* LOW：其余（包括样本不足、相似度过低、维度缺失严重）。

只有 HIGH 才允许历史模块成为最大权重模块（≥50%），
LOW 时历史权重上限 20%，由 ``dynamic_weight`` 执行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from historical_context import CONTEXT_GROUPS, HISTORICAL_GROUP_KEYS

REGISTERED_FEATURE_COUNT = sum(
    len(group.features)
    for group in CONTEXT_GROUPS
    if group.key in HISTORICAL_GROUP_KEYS
)

HIGH_EVIDENCE_THRESHOLD = 45.0
MEDIUM_EVIDENCE_THRESHOLD = 25.0
HIGH_MIN_SAMPLES = 8
HIGH_MIN_SIMILARITY = 62.0
MEDIUM_MIN_SAMPLES = 6


@dataclass
class EvidenceAssessment:
    level: str
    evidence_score: float
    similarity_score: float
    reliability_score: float
    data_quality_score: float
    sample_count: int
    selection_mode: str
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    unavailable_dimensions: list[dict[str, str]] = field(default_factory=list)


def _reliability(
    sample_count: int,
    similarities: list[float],
    horizon_results: list[dict[str, Any]],
    direction_strength: float,
) -> tuple[float, dict[str, float]]:
    n_factor = float(np.clip(sample_count / 15.0, 0.0, 1.0))

    anchor = next(
        (item for item in horizon_results if item.get("available") and int(item.get("days")) == 20),
        next((item for item in horizon_results if item.get("available")), None),
    )
    consistency = 0.5
    if anchor:
        dispersion = float(anchor.get("q75_return", 0.0) - anchor.get("q25_return", 0.0))
        expected_scale = 0.04 * np.sqrt(max(int(anchor["days"]), 1))
        consistency = float(np.clip(1.0 - dispersion / (2.0 * expected_scale), 0.0, 1.0))

    available_horizons = [item for item in horizon_results if item.get("available")]
    if available_horizons and abs(direction_strength) > 1e-9:
        overall_sign = np.sign(direction_strength)
        agreeing = sum(
            1
            for item in available_horizons
            if np.sign(float(item.get("median_return", 0.0))) == overall_sign
        )
        agreement = agreeing / len(available_horizons)
    else:
        agreement = 0.5

    similarity_array = np.asarray(similarities, dtype="float64")
    if len(similarity_array) >= 2:
        concentration = float(np.clip(1.0 - similarity_array.std(ddof=1) / 15.0, 0.0, 1.0))
    else:
        concentration = 0.5

    reliability = (
        0.35 * n_factor
        + 0.25 * consistency
        + 0.20 * agreement
        + 0.20 * concentration
    )
    components = {
        "样本数量因子": n_factor,
        "结果一致性": consistency,
        "跨期限方向一致性": agreement,
        "相似度集中度": concentration,
    }
    return float(np.clip(reliability, 0.0, 1.0)), components


def _data_quality(
    features_used: dict[str, int],
    available_groups: list[str],
    rows: int,
    latest_lag_days: int,
) -> tuple[float, dict[str, float]]:
    feature_coverage = float(
        np.clip(sum(features_used.values()) / max(REGISTERED_FEATURE_COUNT, 1), 0.0, 1.0)
    )
    group_coverage = float(
        np.clip(len(available_groups) / max(len(HISTORICAL_GROUP_KEYS), 1), 0.0, 1.0)
    )
    history_coverage = float(np.clip(rows / 750.0, 0.0, 1.0))
    if latest_lag_days <= 5:
        freshness = 1.0
    else:
        freshness = float(max(0.4, 1.0 - (latest_lag_days - 5) * 0.05))
    quality = (
        0.35 * feature_coverage
        + 0.25 * group_coverage
        + 0.25 * history_coverage
        + 0.15 * freshness
    )
    components = {
        "特征覆盖率": feature_coverage,
        "维度组覆盖率": group_coverage,
        "历史长度覆盖": history_coverage,
        "行情新鲜度": freshness,
    }
    return float(np.clip(quality, 0.0, 1.0)), components


def assess_evidence(
    similarity_result: Any,
    outcome_result: dict[str, Any],
    rows: int,
    latest_lag_days: int,
) -> EvidenceAssessment:
    """汇总相似度、可信度与数据质量，输出证据等级。"""
    reasons: list[str] = []
    unavailable_dimensions = list(
        (similarity_result.context.unavailable_groups if similarity_result.context else [])
    )

    if not getattr(similarity_result, "available", False) or not similarity_result.selected:
        reasons.append(similarity_result.reason or "没有可用的历史相似案例。")
        if unavailable_dimensions:
            reasons.append(
                "宏观、行业、政策、事件与预期维度缺少历史时点数据，"
                "已被明确标记为不可用，不参与相似度，也不伪造。"
            )
        return EvidenceAssessment(
            level="LOW",
            evidence_score=0.0,
            similarity_score=0.0,
            reliability_score=0.0,
            data_quality_score=0.0,
            sample_count=0,
            selection_mode=getattr(similarity_result, "selection_mode", "未形成样本"),
            reasons=reasons,
            unavailable_dimensions=unavailable_dimensions,
        )

    selected_similarities = [float(item["similarity"]) for item in similarity_result.selected]
    similarity_score = float(np.mean(selected_similarities))
    sample_count = len(similarity_result.selected)
    horizon_results = list(outcome_result.get("horizons", []))
    direction_strength = float(outcome_result.get("direction_strength", 0.0))

    reliability, reliability_components = _reliability(
        sample_count, selected_similarities, horizon_results, direction_strength
    )
    available_groups = list(similarity_result.context.available_groups)
    quality, quality_components = _data_quality(
        dict(similarity_result.features_used),
        available_groups,
        rows,
        latest_lag_days,
    )
    evidence_score = float(similarity_score / 100.0 * reliability * quality * 100.0)

    level = "LOW"
    if (
        evidence_score >= HIGH_EVIDENCE_THRESHOLD
        and sample_count >= HIGH_MIN_SAMPLES
        and similarity_score >= HIGH_MIN_SIMILARITY
    ):
        level = "HIGH"
    elif evidence_score >= MEDIUM_EVIDENCE_THRESHOLD and sample_count >= MEDIUM_MIN_SAMPLES:
        level = "MEDIUM"

    if level == "LOW":
        if sample_count < MEDIUM_MIN_SAMPLES:
            reasons.append(f"有效案例{sample_count}个，低于{MEDIUM_MIN_SAMPLES}个下限。")
        if similarity_score < HIGH_MIN_SIMILARITY:
            reasons.append(f"平均相似度{similarity_score:.3f}分偏低。")
        if "market_env" not in available_groups:
            reasons.append("缺少市场基准维度，情景证据不完整。")
        reasons.append("历史证据=LOW：禁止把历史模块权重强行提高到50%。")
    elif level == "MEDIUM":
        reasons.append("历史证据=MEDIUM：历史模块参与决策，但权重不超过50%。")
    else:
        reasons.append("历史证据=HIGH：历史相似情景成为权重最大的决策模块（≥50%）。")
    if unavailable_dimensions:
        reasons.append(
            f"{len(unavailable_dimensions)}个情景维度（"
            + "、".join(item["name"] for item in unavailable_dimensions)
            + "）历史数据暂不可用，已计入数据质量扣分。"
        )

    return EvidenceAssessment(
        level=level,
        evidence_score=evidence_score,
        similarity_score=similarity_score,
        reliability_score=reliability,
        data_quality_score=quality,
        sample_count=sample_count,
        selection_mode=similarity_result.selection_mode,
        components={
            "可信度构成": reliability_components,
            "数据质量构成": quality_components,
        },
        reasons=reasons,
        unavailable_dimensions=unavailable_dimensions,
    )
