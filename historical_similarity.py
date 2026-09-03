"""V7.0.0 决策引擎 —— 历史相似情景数学模型（historical similarity）。

需求第三节要求相似度“可计算、权重可解释、缺失不崩溃、维度标准化、
不允许因缺失而人为制造高相似度”。本模块的实现：

* **标准化**：每个特征先按候选集合的 IQR（四分位距）做稳健标准化，
  IQR 退化时退回标准差，再退化为常数 1（该特征差异为 0，不产生距离）；
  标准化差值截断在 ±8，防止极端异常日主导结果。
* **距离**：组内加权欧氏距离 ``d_g = sqrt(Σ w_i · z_i²)``，
  组内权重在“当前与候选都可计算”的特征上重新归一。
* **相似度**：``Sim_g = 100 · exp(-λ · d_g)``，λ=0.40；
  总相似度 = 各可得组相似度的加权平均（股票自身状态 0.65、
  市场环境 0.35，缺失组不参与且不计入分母——缺失不会抬高相似度）。
* **防前视**：计算前先把行情截断到分析日 T；候选案例必须早于
  ``T - max(outcome_horizons)``，保证选中后才观察结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from historical_context import ContextFrame, build_context_frame
from lookahead_guard import (
    TimeBoundary,
    build_boundary,
    ensure_selection_dates_within_boundary,
    truncate_to_boundary,
)

GROUP_WEIGHTS: dict[str, float] = {
    "stock_state": 0.65,
    "market_env": 0.35,
}

FEATURE_WEIGHTS: dict[str, float] = {
    # 股票自身状态
    "return_1": 0.02,
    "return_5": 0.05,
    "return_20": 0.09,
    "return_60": 0.08,
    "return_120": 0.04,
    "volatility_20": 0.10,
    "volatility_60": 0.06,
    "downside_volatility_60": 0.05,
    "drawdown_60": 0.07,
    "drawdown_120": 0.06,
    "drawdown_250": 0.05,
    "price_ma20": 0.06,
    "price_ma60": 0.06,
    "price_ma120": 0.05,
    "volume_ratio": 0.04,
    "volume_z": 0.03,
    "rsi_14": 0.05,
    "macd_hist": 0.04,
    # 市场整体环境
    "benchmark_return_20": 0.14,
    "benchmark_return_60": 0.13,
    "benchmark_return_120": 0.08,
    "benchmark_volatility_20": 0.12,
    "benchmark_volatility_60": 0.08,
    "benchmark_drawdown_120": 0.12,
    "benchmark_ma60": 0.11,
    "benchmark_ma250": 0.10,
    "relative_return_20": 0.07,
    "relative_return_60": 0.05,
}

DEFAULT_OUTCOME_HORIZONS: tuple[int, ...] = (1, 5, 20, 60, 120)


@dataclass(frozen=True)
class SimilarityConfig:
    warmup_rows: int = 130
    min_gap_from_today: int = 20
    min_gap_between_matches: int = 20
    max_matches: int = 12
    min_matches: int = 6
    primary_threshold: float = 62.0
    fallback_threshold: float = 52.0
    decay: float = 0.40
    clip_z: float = 8.0
    min_features_total: int = 8
    min_features_per_group: int = 4
    min_candidates: int = 30


@dataclass
class SimilarityResult:
    available: bool
    boundary: TimeBoundary | None
    context: ContextFrame | None
    similarities: pd.Series
    selected: list[dict[str, Any]] = field(default_factory=list)
    selection_mode: str = "未形成样本"
    selection_threshold: float = 0.0
    group_similarity_by_match: dict[str, dict[str, float]] = field(default_factory=dict)
    current_group_similarity: dict[str, float] = field(default_factory=dict)
    features_used: dict[str, int] = field(default_factory=dict)
    candidate_count: int = 0
    reason: str = ""
    notes: list[str] = field(default_factory=list)


def _unavailable(reason: str, notes: list[str] | None = None) -> SimilarityResult:
    return SimilarityResult(
        available=False,
        boundary=None,
        context=None,
        similarities=pd.Series(dtype="float64"),
        reason=reason,
        notes=list(notes or []),
    )


def _group_distance_and_similarity(
    candidate_matrix: pd.DataFrame,
    current_vector: pd.Series,
    columns: list[str],
    config: SimilarityConfig,
) -> tuple[pd.Series, int]:
    usable = [
        column
        for column in columns
        if column in candidate_matrix.columns
        and column in current_vector.index
        and np.isfinite(current_vector[column])
    ]
    if len(usable) < config.min_features_per_group:
        return pd.Series(dtype="float64"), len(usable)
    frame = candidate_matrix[usable].dropna()
    if len(frame) < max(10, config.min_candidates // 3):
        return pd.Series(dtype="float64"), len(usable)
    first_quartile = frame.quantile(0.25)
    third_quartile = frame.quantile(0.75)
    scale = third_quartile - first_quartile
    fallback = frame.std().replace(0, np.nan)
    scale = scale.where(scale.abs() > 1e-9, fallback).replace(0, np.nan).fillna(1.0)
    differences = ((frame - current_vector[usable]) / scale).clip(-config.clip_z, config.clip_z)
    weights = pd.Series(
        {column: FEATURE_WEIGHTS.get(column, 1.0 / len(usable)) for column in usable},
        dtype="float64",
    )
    weights = weights / weights.sum()
    distance = np.sqrt(differences.pow(2).mul(weights, axis=1).sum(axis=1))
    similarity = 100.0 * np.exp(-config.decay * distance)
    return similarity, len(usable)


def _select_spaced(
    similarities: pd.Series,
    positions: dict[pd.Timestamp, int],
    minimum_gap: int,
    limit: int,
    threshold: float,
) -> list[tuple[pd.Timestamp, int, float]]:
    selected: list[tuple[pd.Timestamp, int, float]] = []
    for timestamp, similarity in similarities.sort_values(ascending=False).items():
        if float(similarity) < threshold:
            continue
        position = positions.get(pd.Timestamp(timestamp))
        if position is None:
            continue
        if any(abs(position - existing) < minimum_gap for _, existing, _ in selected):
            continue
        selected.append((pd.Timestamp(timestamp), position, float(similarity)))
        if len(selected) >= limit:
            break
    return selected


def search_historical_cases(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    config: SimilarityConfig | None = None,
    analysis_position: int | None = None,
    outcome_horizons: tuple[int, ...] = DEFAULT_OUTCOME_HORIZONS,
) -> SimilarityResult:
    """在 T 时刻可见数据内检索历史相似案例。

    Args:
        stock / benchmark: 原始行情（日期、开盘、收盘、成交量等）。
        analysis_position: 分析日 T 的位置；None 表示最后一个可得交易日。
        outcome_horizons: 之后要观察的结果窗口；候选必须能完整观察。
    """
    settings = config or SimilarityConfig()
    notes: list[str] = []

    prepared = stock.copy()
    prepared["日期"] = pd.to_datetime(prepared["日期"], errors="coerce")
    prepared = prepared.dropna(subset=["日期", "收盘"]).drop_duplicates("日期").sort_values("日期")
    if len(prepared) < settings.warmup_rows + max(outcome_horizons) + settings.min_gap_from_today:
        return _unavailable(
            f"可得行情{len(prepared)}个交易日，不足以同时建立特征窗口"
            f"（约{settings.warmup_rows}日）和观察最长{max(outcome_horizons)}日结果。",
            notes,
        )

    boundary = build_boundary(pd.DatetimeIndex(prepared["日期"]), analysis_position)
    truncated_stock = truncate_to_boundary(prepared, boundary)
    truncated_benchmark = truncate_to_boundary(benchmark, boundary) if benchmark is not None else None

    context = build_context_frame(truncated_stock, truncated_benchmark)
    notes.extend(context.warnings)
    features = context.features
    close = context.close
    rows = len(close)

    maximum_forward = max(outcome_horizons)
    latest_candidate_position = rows - 1 - maximum_forward
    latest_candidate_position = min(latest_candidate_position, rows - 1 - settings.min_gap_from_today)
    if latest_candidate_position < settings.warmup_rows:
        return _unavailable(
            "截断到分析日后，剩余历史不足以形成候选案例区间。",
            notes,
        )

    candidate_positions = range(settings.warmup_rows, latest_candidate_position + 1)
    candidate_dates = close.index[list(candidate_positions)]
    if len(candidate_dates) < settings.min_candidates:
        return _unavailable(
            f"候选历史时点只有{len(candidate_dates)}个，少于最低要求{settings.min_candidates}个。",
            notes,
        )
    candidate_frame = features.loc[candidate_dates]
    current_vector = features.iloc[-1]

    group_similarities: dict[str, pd.Series] = {}
    features_used: dict[str, int] = {}
    for group_key, columns in context.group_columns.items():
        similarity, used = _group_distance_and_similarity(
            candidate_frame, current_vector, columns, settings
        )
        features_used[group_key] = used
        if not similarity.empty:
            group_similarities[group_key] = similarity
        else:
            notes.append(
                f"维度组[{group_key}]可比较特征不足{settings.min_features_per_group}项，"
                "该组不参与相似度（不会人为补足）。"
            )

    if not group_similarities or sum(features_used.values()) < settings.min_features_total:
        return _unavailable(
            "可计算的历史情景特征不足，无法形成可靠的相似度；已拒绝输出，避免虚假高相似度。",
            notes,
        )

    total_weight = sum(GROUP_WEIGHTS[key] for key in group_similarities)
    similarities = pd.Series(0.0, index=candidate_dates)
    for group_key, series in group_similarities.items():
        similarities = similarities.add(series * (GROUP_WEIGHTS[group_key] / total_weight), fill_value=0.0)
    similarities = similarities.sort_index()

    positions = {pd.Timestamp(timestamp): index for index, timestamp in enumerate(close.index)}
    strict = _select_spaced(
        similarities,
        positions,
        settings.min_gap_between_matches,
        settings.max_matches,
        settings.primary_threshold,
    )
    selected = strict
    selection_mode = "严格样本"
    selection_threshold = settings.primary_threshold
    if len(strict) < settings.min_matches:
        fallback = _select_spaced(
            similarities,
            positions,
            settings.min_gap_between_matches,
            settings.max_matches,
            settings.fallback_threshold,
        )
        if len(fallback) > len(strict):
            selected = fallback
            selection_mode = "放宽样本"
            selection_threshold = settings.fallback_threshold
            notes.append(
                f"严格阈值{settings.primary_threshold:.0f}分只有{len(strict)}个样本，"
                f"已放宽到{settings.fallback_threshold:.0f}分，证据质量将相应下调。"
            )
    if not selected:
        result = _unavailable(
            f"没有相似度达到{settings.fallback_threshold:.0f}分的历史案例；"
            f"最高相似度{float(similarities.max()):.3f}分。",
            notes,
        )
        result.boundary = boundary
        result.context = context
        result.similarities = similarities
        result.candidate_count = int(len(similarities))
        result.features_used = features_used
        return result

    ensure_selection_dates_within_boundary([timestamp for timestamp, _, _ in selected], boundary)

    selected_rows: list[dict[str, Any]] = []
    for timestamp, position, similarity in selected:
        selected_rows.append(
            {
                "anchor_date": timestamp,
                "position": position,
                "similarity": similarity,
                "context_window_start": close.index[max(0, position - 59)],
            }
        )

    return SimilarityResult(
        available=True,
        boundary=boundary,
        context=context,
        similarities=similarities,
        selected=selected_rows,
        selection_mode=selection_mode,
        selection_threshold=selection_threshold,
        features_used=features_used,
        candidate_count=int(len(similarities)),
        notes=notes,
    )
