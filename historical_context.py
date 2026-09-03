"""V7.0.0 决策引擎 —— 历史情景维度登记与特征构造（historical context）。

需求文档第二节要求从 7 个维度刻画“投资环境”。本模块把 7 个维度
登记成显式的组（group），并对每个组标注 **历史可得性**：

============  =====================================================  ==========
组            内容                                                    历史可得
============  =====================================================  ==========
stock_state   股票自身状态：多期限收益、波动、回撤、均线、量能、       可得
              RSI、MACD、趋势位置
market_env    市场整体环境：基准收益/波动/回撤/均线结构、相对强弱      可得
industry_env  行业环境：行业指数、行业估值、行业资金流                 暂不可得
macro_env     宏观环境：增长、利率、流动性、政策周期                   历史快照不可得
policy_env    政策环境：货币/财政/产业/监管方向与强度                  暂不可得
event         事件性质：公司/行业重大事件类别                          历史新闻不可得
expectation   市场预期：一致预期与预期差                               暂不可得
============  =====================================================  ==========

纪律（需求第十七节）：

* 历史侧不可得的维度 **绝不回填、绝不伪造**，只在当前侧描述；
* 这些维度不参与历史相似度计算，只通过 ``evidence_quality``
  的数据质量分体现“证据不完整”，从而自动压低历史证据等级。

所有历史可得特征都只用“候选日 t 及以前”的滚动窗口计算，
由 ``lookahead_guard`` 负责结构性保证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContextGroup:
    key: str
    name: str
    history_available: bool
    features: tuple[str, ...] = ()
    unavailable_reason: str = ""


# 7 个情景维度。历史不可得的组必须写明原因，禁止参与历史相似度。
CONTEXT_GROUPS: tuple[ContextGroup, ...] = (
    ContextGroup(
        key="stock_state",
        name="股票自身状态",
        history_available=True,
        features=(
            "return_1",
            "return_5",
            "return_20",
            "return_60",
            "return_120",
            "volatility_20",
            "volatility_60",
            "downside_volatility_60",
            "drawdown_60",
            "drawdown_120",
            "drawdown_250",
            "price_ma20",
            "price_ma60",
            "price_ma120",
            "volume_ratio",
            "volume_z",
            "rsi_14",
            "macd_hist",
        ),
    ),
    ContextGroup(
        key="market_env",
        name="市场整体环境",
        history_available=True,
        features=(
            "benchmark_return_20",
            "benchmark_return_60",
            "benchmark_return_120",
            "benchmark_volatility_20",
            "benchmark_volatility_60",
            "benchmark_drawdown_120",
            "benchmark_ma60",
            "benchmark_ma250",
            "relative_return_20",
            "relative_return_60",
        ),
    ),
    ContextGroup(
        key="industry_env",
        name="行业环境",
        history_available=False,
        unavailable_reason=(
            "当前免费数据通道没有稳定的行业指数与行业估值历史，"
            "不使用无关股票冒充行业样本；该维度仅在报告中说明，不参与历史相似度。"
        ),
    ),
    ContextGroup(
        key="macro_env",
        name="宏观环境",
        history_available=False,
        unavailable_reason=(
            "宏观序列（LPR、美债收益率等）只能取到最新发布值，"
            "没有可靠的历史时点快照；为避免用今天的数据回填过去，"
            "宏观维度只在当前侧描述，不参与历史相似度。"
        ),
    ),
    ContextGroup(
        key="policy_env",
        name="政策环境",
        history_available=False,
        unavailable_reason="缺少结构化的历史政策数据库，无法还原历史政策方向与强度。",
    ),
    ContextGroup(
        key="event",
        name="事件性质",
        history_available=False,
        unavailable_reason="历史新闻事件库不在当前数据通道内，不做主观回填。",
    ),
    ContextGroup(
        key="expectation",
        name="市场预期",
        history_available=False,
        unavailable_reason=(
            "分析师一致预期与预期差缺少历史时点数据；"
            "当前侧用事件前价格漂移近似提示“可能已被提前交易”，不伪造具体预期值。"
        ),
    ),
)


GROUP_KEYS = tuple(group.key for group in CONTEXT_GROUPS)
HISTORICAL_GROUP_KEYS = tuple(group.key for group in CONTEXT_GROUPS if group.history_available)


@dataclass
class ContextFrame:
    """一次情景构造的完整输出。"""

    features: pd.DataFrame
    close: pd.Series
    volume: pd.Series
    dates: pd.DatetimeIndex
    group_columns: dict[str, list[str]] = field(default_factory=dict)
    available_groups: list[str] = field(default_factory=list)
    unavailable_groups: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _rolling_downside_volatility(returns: pd.Series, window: int) -> pd.Series:
    def calculate(values: np.ndarray) -> float:
        downside = values[np.isfinite(values) & (values < 0)]
        if len(downside) < 5:
            return np.nan
        return float(np.std(downside, ddof=1) * np.sqrt(252))

    return returns.rolling(window).apply(calculate, raw=True)


def _rsi_14(close: pd.Series) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0.0)
    loss = (-change).clip(lower=0.0)
    average_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    # 窗口内没有下跌：持续上涨记100，完全走平记50，避免除零伪影。
    boundary_values = pd.Series(
        np.where(average_gain > 0, 100.0, 50.0),
        index=close.index,
    )
    return rsi.where(average_loss != 0, boundary_values)


def _macd_histogram(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    histogram = dif - dea
    # 用价格百分比化，跨股票可比。
    return histogram / close.replace(0, np.nan)


def build_context_frame(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame | None,
) -> ContextFrame:
    """从行情构造 7 维度情景特征表（只含历史可得组的数值列）。

    输入必须已经截断到分析日 T（由调用方通过 ``lookahead_guard``
    完成）；本函数内所有滚动窗口都以“当前行”收尾，结构上不含未来。
    """
    prices = stock.copy()
    prices["日期"] = pd.to_datetime(prices["日期"], errors="coerce")
    prices = prices.dropna(subset=["日期", "收盘"]).drop_duplicates("日期").sort_values("日期")
    if prices.empty:
        raise ValueError("行情为空，无法构造历史情景特征。")
    close = pd.to_numeric(prices.set_index("日期")["收盘"], errors="coerce").dropna()
    close = close[close > 0]
    volume = pd.to_numeric(prices.set_index("日期")["成交量"], errors="coerce").reindex(close.index).fillna(0.0)
    returns = close.pct_change(fill_method=None)

    features = pd.DataFrame(index=close.index)
    for days in (1, 5, 20, 60, 120):
        features[f"return_{days}"] = close.pct_change(days, fill_method=None)
    features["volatility_20"] = returns.rolling(20).std() * np.sqrt(252)
    features["volatility_60"] = returns.rolling(60).std() * np.sqrt(252)
    features["downside_volatility_60"] = _rolling_downside_volatility(returns, 60)
    features["drawdown_60"] = close / close.rolling(60).max() - 1
    features["drawdown_120"] = close / close.rolling(120).max() - 1
    features["drawdown_250"] = close / close.rolling(250).max() - 1
    features["price_ma20"] = close / close.rolling(20).mean() - 1
    features["price_ma60"] = close / close.rolling(60).mean() - 1
    features["price_ma120"] = close / close.rolling(120).mean() - 1
    average_volume_20 = volume.rolling(20).mean()
    average_volume_60 = volume.rolling(60).mean()
    features["volume_ratio"] = average_volume_20 / average_volume_60.replace(0, np.nan)
    volume_std_60 = volume.rolling(60).std()
    features["volume_z"] = (volume - average_volume_60) / volume_std_60.replace(0, np.nan)
    features["rsi_14"] = _rsi_14(close)
    features["macd_hist"] = _macd_histogram(close)

    warnings: list[str] = []
    group_columns: dict[str, list[str]] = {}
    available_groups: list[str] = []
    unavailable_groups: list[dict[str, str]] = []

    stock_state_columns = [
        column for column in CONTEXT_GROUPS[0].features if column in features.columns
    ]
    group_columns["stock_state"] = stock_state_columns
    available_groups.append("stock_state")

    benchmark_close = pd.Series(dtype="float64")
    if benchmark is not None and not benchmark.empty:
        bench = benchmark.copy()
        bench["日期"] = pd.to_datetime(bench["日期"], errors="coerce")
        bench = bench.dropna(subset=["日期", "收盘"]).drop_duplicates("日期").sort_values("日期")
        bench_close = pd.to_numeric(bench.set_index("日期")["收盘"], errors="coerce").dropna()
        bench_close = bench_close[bench_close > 0].reindex(close.index).ffill()
        if bench_close.notna().sum() >= 60:
            benchmark_close = bench_close
            bench_returns = bench_close.pct_change(fill_method=None)
            features["benchmark_return_20"] = bench_close.pct_change(20, fill_method=None)
            features["benchmark_return_60"] = bench_close.pct_change(60, fill_method=None)
            features["benchmark_return_120"] = bench_close.pct_change(120, fill_method=None)
            features["benchmark_volatility_20"] = bench_returns.rolling(20).std() * np.sqrt(252)
            features["benchmark_volatility_60"] = bench_returns.rolling(60).std() * np.sqrt(252)
            features["benchmark_drawdown_120"] = bench_close / bench_close.rolling(120).max() - 1
            features["benchmark_ma60"] = bench_close / bench_close.rolling(60).mean() - 1
            features["benchmark_ma250"] = bench_close / bench_close.rolling(250).mean() - 1
            features["relative_return_20"] = features["return_20"] - features["benchmark_return_20"]
            features["relative_return_60"] = features["return_60"] - features["benchmark_return_60"]
            market_columns = [
                column for column in CONTEXT_GROUPS[1].features if column in features.columns
            ]
            group_columns["market_env"] = market_columns
            available_groups.append("market_env")
        else:
            warnings.append("市场基准数据过短，市场环境维度不参与历史相似度。")
    else:
        warnings.append("市场基准缺失，市场环境维度不参与历史相似度。")

    for group in CONTEXT_GROUPS:
        if not group.history_available:
            unavailable_groups.append(
                {"key": group.key, "name": group.name, "reason": group.unavailable_reason}
            )

    features = features.replace([np.inf, -np.inf], np.nan)
    return ContextFrame(
        features=features,
        close=close,
        volume=volume,
        dates=pd.DatetimeIndex(close.index),
        group_columns=group_columns,
        available_groups=available_groups,
        unavailable_groups=unavailable_groups,
        warnings=warnings,
    )


def current_state_display(context: ContextFrame) -> dict[str, Any]:
    """输出当前（T 时刻）状态的可读快照，供报告引用。"""
    if context.features.empty:
        return {}
    current = context.features.iloc[-1]

    def value(name: str) -> float | None:
        item = current.get(name, np.nan)
        return float(item) if item is not None and np.isfinite(item) else None

    ma60 = value("price_ma60")
    ma120 = value("price_ma120")
    return_60 = value("return_60")
    if ma60 is None or return_60 is None:
        trend = "数据不足"
    elif return_60 > 0 and ma60 > 0:
        trend = "偏强上行"
    elif return_60 < 0 and ma60 < 0:
        trend = "偏弱下行"
    else:
        trend = "震荡整理"
    drawdown_250 = value("drawdown_250")
    if drawdown_250 is None:
        position_text = "回撤数据不足"
    elif drawdown_250 <= -0.30:
        position_text = "深度回撤区（接近底部区域，但需量能确认）"
    elif drawdown_250 <= -0.12:
        position_text = "中等回撤区"
    elif drawdown_250 >= -0.03:
        position_text = "接近一年高点（顶部区域风险需警惕）"
    else:
        position_text = "高位震荡区"
    return {
        "trend": trend,
        "position": position_text,
        "近20日收益": value("return_20"),
        "近60日收益": return_60,
        "20日年化波动": value("volatility_20"),
        "距120日高点": value("drawdown_120"),
        "距250日高点": drawdown_250,
        "相对MA60": ma60,
        "相对MA120": ma120,
        "RSI14": value("rsi_14"),
        "MACD柱(价格百分比)": value("macd_hist"),
        "20日/60日量比": value("volume_ratio"),
        "近5日量能z值": value("volume_z"),
    }
