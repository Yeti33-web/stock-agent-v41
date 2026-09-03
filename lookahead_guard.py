"""V7.0.0 决策引擎 —— 数据时间边界守卫（lookahead guard）。

这是最高优先级模块：保证“历史相似情景”检索严格无前视。

核心原则（与需求文档第六节一致）：

    分析日 T 只能使用 T 及 T 以前已经公开的信息。
    禁止使用 T+1、T+5、T+20、T+60、T+120 的信息
    来决定某个历史案例是否与当前相似。

正确流程：

    T 时刻信息 → 计算当前状态 → 寻找历史相似案例 →
    确定历史案例 → 再观察这些案例之后的真实走势

本模块提供两类保护：

1. ``TimeBoundary``：显式记录分析日 T、数据起点与数据末端，
   所有候选案例必须位于 [数据起点, T] 之内；
2. 结构性守卫函数：在选择完成后校验“选择所用日期均不晚于 T”、
   “结果观察窗口只出现在选择之后”，任何违反都会抛出
   ``LookaheadViolation``，让前视错误在测试与生产中都可见。

为了让守卫真正生效，``historical_similarity`` 在计算特征前会先把
行情截断到 ``analysis_position``（即 T），因此 T 之后的任何数据
（哪怕被人为篡改）都不可能进入相似度计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

DECISION_ENGINE_VERSION = "V7.0.0"


class LookaheadViolation(RuntimeError):
    """检测到未来数据可能进入模型时抛出。"""


@dataclass(frozen=True)
class TimeBoundary:
    """一次分析允许使用的时间边界。

    Attributes:
        analysis_position: 分析日 T 在行情序列中的整数位置（含）。
        analysis_date: 分析日 T 的时间戳。
        first_date / last_date: 实际可得行情的首末日期。
        rows: 截断到 T 之后的行情行数。
    """

    analysis_position: int
    analysis_date: pd.Timestamp
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    rows: int


def build_boundary(dates: pd.DatetimeIndex, analysis_position: int | None = None) -> TimeBoundary:
    """根据行情日期序列构造时间边界。

    ``analysis_position=None`` 表示使用当前最后一个可得交易日作为 T。
    任何大于最后位置的取值都会被拒绝，避免凭空引用未来。
    """
    if len(dates) == 0:
        raise ValueError("行情为空，无法建立时间边界。")
    last_position = len(dates) - 1
    position = last_position if analysis_position is None else int(analysis_position)
    if position < 0 or position > last_position:
        raise LookaheadViolation(
            f"分析位置{position}超出可得行情范围[0, {last_position}]，拒绝引用未来数据。"
        )
    return TimeBoundary(
        analysis_position=position,
        analysis_date=pd.Timestamp(dates[position]),
        first_date=pd.Timestamp(dates[0]),
        last_date=pd.Timestamp(dates[position]),
        rows=position + 1,
    )


def truncate_to_boundary(frame: pd.DataFrame, boundary: TimeBoundary) -> pd.DataFrame:
    """把行情/特征表截断到 T（含）。这是防前视的第一道结构保护。"""
    if frame is None or frame.empty:
        return frame
    if len(frame) <= boundary.rows:
        return frame
    return frame.iloc[: boundary.rows].copy()


def ensure_selection_dates_within_boundary(
    selected_dates: Iterable[pd.Timestamp],
    boundary: TimeBoundary,
) -> None:
    """校验所有被选中的历史案例日期都不晚于分析日 T。"""
    for timestamp in selected_dates:
        if pd.Timestamp(timestamp) > boundary.analysis_date:
            raise LookaheadViolation(
                f"历史案例日期{timestamp}晚于分析日{boundary.analysis_date}，"
                "说明选择过程使用了未来信息。"
            )


def ensure_outcome_observed_after_selection(
    candidate_position: int,
    horizon_days: int,
    rows_at_analysis: int,
) -> bool:
    """候选案例必须能在不越过 T 的前提下观察到完整结果窗口。

    返回 False 时该候选不应进入结果统计——不是程序错误，而是
    它的未来还没有走完，强行读取会引入截断偏差。
    """
    return candidate_position + int(horizon_days) < int(rows_at_analysis)


def ensure_feature_window_ends_at_or_before(
    feature_dates: Sequence[pd.Timestamp],
    candidate_date: pd.Timestamp,
) -> None:
    """校验某案例的特征序列全部由不晚于该案例日期的数据构成。

    所有情景特征都由滚动窗口（rolling）在候选日收尾计算得到，
    该函数负责在测试中对这一结构假设做显式验证。
    """
    candidate_ts = pd.Timestamp(candidate_date)
    for timestamp in feature_dates:
        if pd.Timestamp(timestamp) > candidate_ts:
            raise LookaheadViolation(
                f"特征序列中出现晚于候选案例日期{candidate_ts}的数据{timestamp}。"
            )


def guard_summary(boundary: TimeBoundary) -> str:
    """生成面向用户的时间边界说明。"""
    return (
        f"本次分析日 T={boundary.analysis_date.date()}；"
        f"相似案例只能来自{boundary.first_date.date()}至{boundary.analysis_date.date()}，"
        "且案例选择只使用案例当日及以前的公开数据；"
        "案例被选中之后才读取其后续真实走势，禁止反向挑选。"
    )
