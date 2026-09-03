"""V7.0.0 决策引擎 —— 事件—市场反应模块（event reaction）。

需求第九节：新闻分析不能只判断利好/利空/中性，必须继续判断：

    事件 → 市场预期 → 实际价格反应 → 成交量 → 资金行为

本模块把每条重要资讯映射到其发布后的第一个交易日，读取当日的
开盘跳空、收盘反应、冲高回落幅度、量能变化，以及事件前的价格
漂移（作为“是否已被提前交易”的可计算近似），然后归类为：

* 利好 + 放量上涨          → 正向确认
* 重大利好 + 高开低走      → 可能利好兑现
* 利好 + 前期已大涨且当日不涨 → 利好可能已被提前计价
* 利好 + 股价不涨          → 利好钝化
* 利空 + 放量下跌          → 风险确认
* 利空 + 股价不跌          → 可能已被提前计价

纪律：不混同“新闻情绪”与“股票未来收益”；情绪只描述事件方向，
价格反应才描述市场是否认同。所有规则为显式阈值，可测试。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


RELEVANCE_FLOOR = 0.65
SENTIMENT_FLOOR = 0.25
MAX_EVENT_AGE_DAYS = 60
POSITIVE_CONFIRM_RETURN = 0.010
POSITIVE_CONFIRM_VOLUME = 1.20
FADE_THRESHOLD = -0.015
BLUNT_THRESHOLD = 0.005
PRICED_IN_DRIFT = 0.06
NEGATIVE_CONFIRM_RETURN = -0.010


def _prepare_prices(stock: pd.DataFrame) -> pd.DataFrame:
    prices = stock.copy()
    prices["日期"] = pd.to_datetime(prices["日期"], errors="coerce")
    prices = prices.dropna(subset=["日期", "收盘"]).drop_duplicates("日期").sort_values("日期")
    for column in ("开盘", "最高", "最低", "收盘", "成交量"):
        if column in prices.columns:
            prices[column] = pd.to_numeric(prices[column], errors="coerce")
    return prices.reset_index(drop=True)


def _map_event_day(prices: pd.DataFrame, published_at: pd.Timestamp) -> int | None:
    """返回资讯发布后的第一个交易日位置；发布晚于数据末端时返回 None。"""
    publish_day = pd.Timestamp(published_at).normalize()
    dates = pd.DatetimeIndex(prices["日期"]).normalize()
    matches = np.where(dates >= publish_day)[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def _classify(
    sentiment_sign: int,
    day0_return: float,
    gap_open: float | None,
    intraday: float | None,
    volume_ratio: float | None,
    pre_drift: float | None,
) -> tuple[str, float]:
    """返回（判定, 对模块分的调整）。"""
    if sentiment_sign > 0:
        if (
            pre_drift is not None
            and pre_drift >= PRICED_IN_DRIFT
            and day0_return < BLUNT_THRESHOLD
        ):
            return "利好可能已被提前计价", -6.0
        if day0_return >= POSITIVE_CONFIRM_RETURN * 2 and intraday is not None and intraday <= FADE_THRESHOLD:
            return "可能利好兑现（高开低走）", -5.0
        if (
            day0_return >= POSITIVE_CONFIRM_RETURN
            and volume_ratio is not None
            and volume_ratio >= POSITIVE_CONFIRM_VOLUME
            and (intraday is None or intraday > FADE_THRESHOLD)
        ):
            return "正向确认（放量上涨）", 8.0
        if abs(day0_return) < BLUNT_THRESHOLD:
            return "利好钝化（股价不涨）", -4.0
        if day0_return > 0:
            return "温和正面反应", 3.0
        return "利好但股价下跌（分歧）", -3.0

    if (
        pre_drift is not None
        and pre_drift <= -PRICED_IN_DRIFT
        and day0_return > -BLUNT_THRESHOLD
    ):
        return "利空可能已被提前计价", 2.0
    if (
        day0_return <= NEGATIVE_CONFIRM_RETURN
        and volume_ratio is not None
        and volume_ratio >= POSITIVE_CONFIRM_VOLUME
    ):
        return "风险确认（放量下跌）", -10.0
    if day0_return > -BLUNT_THRESHOLD:
        return "利空但股价不跌（可能已计价）", 2.0
    if day0_return < 0:
        return "温和负面反应", -3.0
    return "利空但股价上涨（分歧）", 3.0


def assess_event_reaction(
    news_result: Mapping[str, Any] | None,
    stock: pd.DataFrame,
) -> dict[str, Any]:
    """把资讯情绪与实际价格反应联合判断，输出 0—100 的资讯模块方向分。"""
    if not news_result:
        return {
            "available": False,
            "score": None,
            "reason": "没有资讯输入。",
            "events": [],
            "expectation_flags": [],
        }
    items = [dict(item) for item in news_result.get("items") or [] if isinstance(item, Mapping)]
    prices = _prepare_prices(stock)
    if len(prices) < 30:
        return {
            "available": False,
            "score": None,
            "reason": "行情不足30个交易日，无法观察事件反应。",
            "events": [],
            "expectation_flags": [],
        }

    closes = pd.to_numeric(prices["收盘"], errors="coerce").to_numpy(dtype="float64")
    opens = prices["开盘"].to_numpy(dtype="float64") if "开盘" in prices.columns else None
    volumes = prices["成交量"].to_numpy(dtype="float64") if "成交量" in prices.columns else None

    events: list[dict[str, Any]] = []
    expectation_flags: list[str] = []
    adjustment_total = 0.0
    weight_total = 0.0
    skipped_not_observable = 0

    for item in items:
        sentiment_score = float(item.get("sentiment_score") or 0.0)
        relevance = float(item.get("relevance_score") or 0.0)
        if abs(sentiment_score) < SENTIMENT_FLOOR or relevance < RELEVANCE_FLOOR:
            continue
        published_raw = item.get("published_at")
        if not published_raw:
            continue
        try:
            published = pd.Timestamp(published_raw)
        except (TypeError, ValueError):
            continue
        event_position = _map_event_day(prices, published)
        title = str(item.get("title") or "（无标题）")
        if event_position is None:
            skipped_not_observable += 1
            events.append(
                {
                    "title": title,
                    "sentiment": item.get("sentiment"),
                    "reaction": "尚未观察到价格反应窗口",
                    "adjustment": 0.0,
                }
            )
            continue
        if event_position < 21 or len(prices) - 1 - event_position > MAX_EVENT_AGE_DAYS:
            continue
        if event_position >= len(closes) or event_position < 1:
            continue

        previous_close = closes[event_position - 1]
        if previous_close <= 0 or not np.isfinite(previous_close):
            continue
        day0_return = float(closes[event_position] / previous_close - 1)
        gap_open = None
        intraday = None
        if opens is not None and np.isfinite(opens[event_position]) and opens[event_position] > 0:
            gap_open = float(opens[event_position] / previous_close - 1)
            intraday = float(closes[event_position] / opens[event_position] - 1)
        volume_ratio = None
        if volumes is not None:
            baseline = float(np.nanmean(volumes[max(0, event_position - 20) : event_position]))
            if baseline > 0 and np.isfinite(volumes[event_position]):
                volume_ratio = float(volumes[event_position] / baseline)
        pre_drift = None
        if event_position >= 6 and closes[event_position - 6] > 0:
            pre_drift = float(closes[event_position - 1] / closes[event_position - 6] - 1)

        sentiment_sign = 1 if sentiment_score > 0 else -1
        reaction, adjustment = _classify(
            sentiment_sign, day0_return, gap_open, intraday, volume_ratio, pre_drift
        )

        if sentiment_sign > 0 and pre_drift is not None and pre_drift >= PRICED_IN_DRIFT:
            expectation_flags.append(f"《{title[:30]}》发布前5日已上涨{pre_drift:.1%}，利好可能已被提前交易。")
        if sentiment_sign < 0 and pre_drift is not None and pre_drift <= -PRICED_IN_DRIFT:
            expectation_flags.append(f"《{title[:30]}》发布前5日已下跌{abs(pre_drift):.1%}，利空可能已被提前交易。")

        recency = float(item.get("recency_weight") or 0.5)
        weight = recency * relevance
        adjustment_total += adjustment * weight
        weight_total += weight
        events.append(
            {
                "title": title,
                "published_at": str(published_raw),
                "event_date": str(pd.Timestamp(prices['日期'].iloc[event_position]).date()),
                "sentiment": item.get("sentiment"),
                "day0_return": day0_return,
                "gap_open": gap_open,
                "intraday_fade": intraday,
                "volume_ratio": volume_ratio,
                "pre_drift_5": pre_drift,
                "reaction": reaction,
                "adjustment": adjustment,
            }
        )

    sentiment_component = 0.0
    try:
        net_sentiment = float(news_result.get("net_sentiment_score") or 0.0)
        if news_result.get("usable_for_score"):
            sentiment_component = float(np.clip(net_sentiment, -1.0, 1.0)) * 18.0
    except (TypeError, ValueError):
        sentiment_component = 0.0

    raw_reaction = adjustment_total / weight_total if weight_total > 0 else 0.0
    if weight_total > 0:
        # 已有实际价格反应时，市场反应主导：情绪减半、反应放大。
        sentiment_component *= 0.5
        reaction_component = float(np.clip(raw_reaction * 1.5, -15.0, 15.0))
    else:
        reaction_component = 0.0
    if not events or weight_total <= 0:
        return {
            "available": False,
            "score": None,
            "reason": (
                "没有可观察价格反应的重要资讯（可能全部发布于数据末端之后，"
                f"或相关度/情绪强度不足；未观察条数{skipped_not_observable}）。"
            ),
            "events": events,
            "expectation_flags": expectation_flags,
            "sentiment_component": sentiment_component,
        }

    score = float(np.clip(50.0 + sentiment_component + reaction_component, 0.0, 100.0))
    return {
        "available": True,
        "score": score,
        "reason": "",
        "events": events,
        "expectation_flags": expectation_flags,
        "sentiment_component": sentiment_component,
        "reaction_component": reaction_component,
        "not_observable_count": skipped_not_observable,
        "method_note": (
            "情绪分只描述事件方向；事件反应分描述市场是否认同。"
            "两者分开计算后相加，避免把新闻情绪直接当成未来收益。"
        ),
    }
