"""V7.0.0 决策引擎 —— 历史案例后续表现分析（historical outcome）。

需求第五节：找到历史案例后，必须统计事件后 1/5/20/60/120 日表现，
且不能只统计最终涨跌。本模块输出：

* 各期限：平均/中位收益、上涨概率（历史频率）、10/25/75 分位、
  相似度加权中位收益；
* 路径统计：最大上涨幅度、最大回撤、最大亏损、峰值出现时间、
  先跌后涨/先涨后跌等路径形态、成交量变化、是否突破前高；
* 汇总的历史路径分析文本素材。

注意：这里统计的“上涨概率”是历史样本频率，不是校准后的真实概率，
报告会明确这一点（沿用 V6.5.2 的口径）。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20, 60, 120)
HORIZON_WEIGHTS: dict[int, float] = {1: 0.10, 5: 0.15, 20: 0.30, 60: 0.30, 120: 0.15}


def _path_pattern(values: np.ndarray) -> str:
    """把事件后路径归为五类形态。"""
    if len(values) < 2:
        return "路径过短"
    midpoint = len(values) // 2
    first_half = float(values[:midpoint].min()) if midpoint > 0 else 0.0
    second_half_change = float(values[-1] - values[midpoint - 1]) if midpoint > 0 else float(values[-1])
    final = float(values[-1])
    if first_half < -0.02 and final > 0.01:
        return "先跌后涨"
    if first_half > 0.02 and final < -0.01:
        return "先涨后跌"
    if final > 0.02 and float(values.min()) > -0.02:
        return "单边上涨"
    if final < -0.02 and float(values.max()) < 0.02:
        return "单边下跌"
    return "震荡反复"


def _evaluate_one_horizon(
    close: pd.Series,
    volume: pd.Series,
    matches: Sequence[dict[str, Any]],
    days: int,
) -> dict[str, Any]:
    outcomes: list[float] = []
    similarities: list[float] = []
    max_gains: list[float] = []
    worst_losses: list[float] = []
    drawdowns: list[float] = []
    peak_days: list[int] = []
    volume_changes: list[float] = []
    breakouts = 0
    patterns: dict[str, int] = {}

    for match in matches:
        position = int(match["position"])
        entry = float(close.iloc[position])
        if entry <= 0 or position + days >= len(close):
            continue
        future = close.iloc[position + 1 : position + days + 1]
        if len(future) < days:
            continue
        normalized = future.to_numpy(dtype="float64") / entry - 1
        outcomes.append(float(normalized[-1]))
        similarities.append(float(match["similarity"]))
        max_gains.append(float(normalized.max()))
        worst_losses.append(float(normalized.min()))
        running_max = np.maximum.accumulate(np.concatenate(([1.0], 1.0 + normalized)))
        drawdowns.append(float(((1.0 + normalized) / running_max[1:] - 1.0).min()))
        peak_days.append(int(normalized.argmax()) + 1)
        pattern = _path_pattern(normalized)
        patterns[pattern] = patterns.get(pattern, 0) + 1

        baseline_volume = float(volume.iloc[max(0, position - 19) : position + 1].mean())
        reaction_volume = float(volume.iloc[position + 1 : position + 1 + min(5, days)].mean())
        if baseline_volume > 0 and np.isfinite(reaction_volume):
            volume_changes.append(reaction_volume / baseline_volume - 1.0)

        prior_high = float(close.iloc[max(0, position - 60) : position + 1].max())
        if float(future.max()) > prior_high * 1.001:
            breakouts += 1

    sample_count = len(outcomes)
    if sample_count == 0:
        return {
            "days": days,
            "available": False,
            "sample_count": 0,
            "reason": "该期限没有可完整观察的历史案例。",
        }
    outcomes_array = np.asarray(outcomes)
    similarities_array = np.asarray(similarities)
    weighted_median = float(
        outcomes_array[np.argsort(outcomes_array)][
            min(
                sample_count - 1,
                int(np.searchsorted(np.cumsum(similarities_array[np.argsort(outcomes_array)]) / similarities_array.sum(), 0.5)),
            )
        ]
    )
    return {
        "days": days,
        "available": True,
        "sample_count": sample_count,
        "mean_return": float(outcomes_array.mean()),
        "median_return": float(np.median(outcomes_array)),
        "weighted_median_return": weighted_median,
        "win_rate": float((outcomes_array > 0).mean()),
        "q10_return": float(np.quantile(outcomes_array, 0.10)),
        "q25_return": float(np.quantile(outcomes_array, 0.25)),
        "q75_return": float(np.quantile(outcomes_array, 0.75)),
        "median_max_gain": float(np.median(max_gains)),
        "median_max_drawdown": float(np.median(drawdowns)),
        "median_worst_loss": float(np.median(worst_losses)),
        "median_peak_day": float(np.median(peak_days)),
        "median_volume_change": float(np.median(volume_changes)) if volume_changes else None,
        "breakout_ratio": float(breakouts / sample_count),
        "patterns": patterns,
        "dominant_pattern": max(patterns, key=patterns.get),
        "reason": "",
    }


def evaluate_outcomes(
    close: pd.Series,
    volume: pd.Series,
    matches: Sequence[dict[str, Any]],
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    """对已选中的历史案例统计后续真实走势。

    ``matches`` 必须来自 ``historical_similarity.search_historical_cases``，
    即先选择、后观察——顺序由调用链结构保证，本函数不会回头修改选择。
    """
    horizon_results = [
        _evaluate_one_horizon(close, volume, matches, int(days)) for days in horizons
    ]
    available = [item for item in horizon_results if item["available"]]
    if not available:
        return {
            "available": False,
            "horizons": horizon_results,
            "direction_summary": "样本不足，不形成历史路径结论。",
            "direction_strength": 0.0,
            "notes": ["所有结果窗口都没有完整可观察的历史案例。"],
        }

    strengths: list[float] = []
    for item in available:
        days = int(item["days"])
        expected_scale = 0.02 * np.sqrt(max(days, 1))
        win_component = float(np.clip((item["win_rate"] - 0.5) * 2.0, -1.0, 1.0))
        return_component = float(np.clip(item["median_return"] / expected_scale, -1.0, 1.0))
        strengths.append(0.5 * win_component + 0.5 * return_component)
    weights = np.asarray(
        [HORIZON_WEIGHTS.get(int(item["days"]), 0.1) for item in available],
        dtype="float64",
    )
    weights = weights / weights.sum()
    direction_strength = float(np.clip(float(np.dot(weights, strengths)), -1.0, 1.0))

    if direction_strength >= 0.25:
        summary = "历史相似案例之后整体偏正面"
    elif direction_strength <= -0.25:
        summary = "历史相似案例之后整体偏负面"
    else:
        summary = "历史相似案例之后方向分歧明显"

    lines = [f"历史相似案例 N={len(matches)}"]
    for item in available:
        lines.append(
            f"{item['days']}日：平均收益{item['mean_return']:+.3%}，"
            f"中位收益{item['median_return']:+.3%}，上涨频率{item['win_rate']:.1%}"
        )
    return {
        "available": True,
        "horizons": horizon_results,
        "direction_summary": summary,
        "direction_strength": direction_strength,
        "path_report": lines,
        "notes": [
            "上涨频率是历史样本频率，不是经过校准的上涨概率。",
            "结果窗口在选择案例之后读取，选择过程没有使用这些结果。",
        ],
    }
