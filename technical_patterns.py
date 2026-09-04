"""V7.0.0 决策引擎 —— 技术形态分析（technical patterns）。

需求第十节：保留并完善技术分析，重点增加红三兵／三连阳识别、
趋势位置、MA20/60/120、突破、RSI、MACD、动量、波动率、回撤。

核心纪律：**红三兵不能简单判断为买入**。本模块按十项背景条件给
形态打“有效性分”，再归类为：趋势反转、超跌反弹、消息刺激、
主力试盘、诱多、上涨末端或普通延续。所有规则透明、可测试。

模块输出 0—100 的方向分（50 为中性），供动态权重层使用；
该模块只读行情，不修改 V6.5.2 的任何原有评分。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def _prepare(stock: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    prices = stock.copy()
    prices["日期"] = pd.to_datetime(prices["日期"], errors="coerce")
    prices = prices.dropna(subset=["日期", "收盘"]).drop_duplicates("日期").sort_values("日期")
    frame = prices.set_index("日期")
    close = pd.to_numeric(frame["收盘"], errors="coerce")
    open_price = pd.to_numeric(frame.get("开盘"), errors="coerce").reindex(close.index)
    high = pd.to_numeric(frame.get("最高"), errors="coerce").reindex(close.index)
    low = pd.to_numeric(frame.get("最低"), errors="coerce").reindex(close.index)
    volume = pd.to_numeric(frame.get("成交量"), errors="coerce").reindex(close.index).fillna(0.0)
    return close, open_price, high, low, volume


def _rsi_14(close: pd.Series) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0.0)
    loss = (-change).clip(lower=0.0)
    average_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    boundary_values = pd.Series(np.where(average_gain > 0, 100.0, 50.0), index=close.index)
    return rsi.where(average_loss != 0, boundary_values)


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    return dif, dea, dif - dea


def classify_trend_position(close: pd.Series) -> dict[str, Any]:
    """把当前走势归类为：底部区域／上涨早段／上涨中段／顶部区域／下跌阶段／震荡。"""
    if len(close) < 130:
        return {"label": "数据不足", "score": 0.0, "detail": "行情不足130个交易日。"}
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    ma120 = float(close.rolling(120).mean().iloc[-1])
    price = float(close.iloc[-1])
    range_250 = close.tail(250)
    percentile = float((price - range_250.min()) / (range_250.max() - range_250.min())) if range_250.max() > range_250.min() else 0.5
    drawdown_120 = float(price / close.tail(120).max() - 1)
    return_60 = float(close.iloc[-1] / close.iloc[-61] - 1)
    return_120 = float(close.iloc[-1] / close.iloc[-121] - 1)

    if price >= ma20 >= ma60 >= ma120:
        if percentile >= 0.85:
            label, score = "顶部区域（多头排列但位置偏高）", 6.0
        elif return_120 > 0.35:
            label, score = "上涨中段", 12.0
        else:
            label, score = "上涨早段", 10.0
    elif price <= ma20 <= ma60 <= ma120:
        if percentile <= 0.20 and drawdown_120 <= -0.25:
            label, score = "底部区域（空头排列但深度回撤）", -2.0
        else:
            label, score = "下跌阶段", -12.0
    elif price >= ma60 and return_60 > 0:
        label, score = "强势震荡", 5.0
    elif price <= ma60 and return_60 < 0:
        label, score = "弱势震荡", -5.0
    else:
        label, score = "震荡整理", 0.0
    return {
        "label": label,
        "score": score,
        "percentile_in_250_range": percentile,
        "drawdown_120": drawdown_120,
        "return_60": return_60,
        "return_120": return_120,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
        "price": price,
    }


def detect_three_white_soldiers(
    stock: pd.DataFrame,
    news_dates: Sequence[pd.Timestamp] | None = None,
) -> dict[str, Any]:
    """识别最近三根K线是否为红三兵，并按背景条件评估有效性。

    有效性检查清单（需求第十节）：
    前期趋势、三根K线位置、实体大小、上下影线、成交量、
    是否突破压力位、是否处于长期下跌后的底部、是否与重大新闻时间吻合。
    """
    close, open_price, high, low, volume = _prepare(stock)
    if len(close) < 70:
        return {"detected": False, "reason": "行情不足70个交易日，无法可靠识别形态。"}

    last3_close = close.iloc[-3:].to_numpy(dtype="float64")
    last3_open = open_price.iloc[-3:].to_numpy(dtype="float64")
    last3_high = high.iloc[-3:].to_numpy(dtype="float64")
    last3_low = low.iloc[-3:].to_numpy(dtype="float64")
    if np.isnan(last3_open).any() or np.isnan(last3_close).any():
        return {"detected": False, "reason": "最近三根K线开盘或收盘价缺失。"}

    bullish = last3_close > last3_open
    rising_close = np.all(np.diff(last3_close) > 0)
    detected = bool(bullish.all() and rising_close)
    if not detected:
        return {"detected": False, "reason": "最近三根K线不满足红三兵定义（连续收阳且收盘价逐步抬高）。"}

    bodies = last3_close - last3_open
    ranges = np.maximum(last3_high - last3_low, 1e-9)
    atr14 = float((high - low).tail(14).mean())
    upper_shadow_ratio = float(np.mean((last3_high - last3_close) / ranges))
    body_atr_ratio = float(bodies.mean() / atr14) if atr14 > 0 else 0.0
    body_shrinking = bool(bodies[-1] < bodies[0] * 0.7)

    volume_3 = float(volume.iloc[-3:].mean())
    volume_20 = float(volume.iloc[-23:-3].mean())
    volume_expansion = volume_3 / volume_20 if volume_20 > 0 else None

    before_close = close.iloc[:-3]
    prior_return_20 = float(before_close.iloc[-1] / before_close.iloc[-21] - 1)
    prior_return_60 = float(before_close.iloc[-1] / before_close.iloc[-61] - 1)
    range_250 = close.tail(250)
    position_in_range = float(
        (before_close.iloc[-1] - range_250.min()) / (range_250.max() - range_250.min())
    ) if range_250.max() > range_250.min() else 0.5

    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    resistance_60 = float(before_close.tail(60).max())
    latest_price = float(close.iloc[-1])
    broke_resistance = latest_price > resistance_60
    deep_prior_decline = prior_return_60 <= -0.15 and position_in_range <= 0.30

    news_aligned = False
    if news_dates:
        recent_dates = close.index[-6:]
        news_aligned = any(
            pd.Timestamp(item).normalize() in recent_dates
            or any(abs((pd.Timestamp(item).normalize() - day).days) <= 2 for day in recent_dates)
            for item in news_dates
        )

    validity = 0.0
    checks: list[str] = []
    if deep_prior_decline:
        validity += 22
        checks.append("处于长期下跌后的低位区域，反转意义更强")
    elif prior_return_60 > 0.15 and position_in_range >= 0.80:
        validity -= 18
        checks.append("出现在明显上涨后的高位，诱多／上涨末端风险上升")
    else:
        validity += 8
        checks.append("前期趋势中性，形态需要量能进一步确认")

    if body_atr_ratio >= 0.9:
        validity += 15
        checks.append("三根K线实体相对正常波幅足够大")
    elif body_atr_ratio >= 0.5:
        validity += 8
        checks.append("实体中等")
    else:
        validity -= 8
        checks.append("实体偏小，推动力不足")

    if upper_shadow_ratio <= 0.15:
        validity += 10
        checks.append("上影线较短，抛压不明显")
    elif upper_shadow_ratio >= 0.35:
        validity -= 12
        checks.append("上影线偏长，上方抛压明显")

    if volume_expansion is None:
        checks.append("成交量数据不足，未计分量")
    elif volume_expansion >= 1.3:
        validity += 18
        checks.append("成交量较前期明显放大")
    elif volume_expansion >= 1.0:
        validity += 8
        checks.append("成交量温和")
    else:
        validity -= 10
        checks.append("成交量萎缩，资金参与度不足")

    if broke_resistance:
        validity += 15
        checks.append("突破近60日压力位")
    else:
        checks.append("尚未突破近60日压力位")

    if body_shrinking:
        validity -= 8
        checks.append("第三根K线实体明显缩小，上攻动能减弱")

    if news_aligned:
        validity += 10
        checks.append("与近期重大新闻时间吻合")

    validity = float(np.clip(validity, 0.0, 100.0))

    if deep_prior_decline and validity >= 55:
        interpretation = "趋势反转（底部区域放量收复）"
        bias = +1
    elif prior_return_20 <= -0.10:
        interpretation = "超跌反弹"
        bias = +1 if validity >= 45 else 0
    elif news_aligned:
        interpretation = "消息刺激"
        bias = +1 if validity >= 45 else 0
    elif position_in_range >= 0.85 and (body_shrinking or (volume_expansion is not None and volume_expansion < 1.0)):
        interpretation = "上涨末端／诱多风险"
        bias = -1
    elif volume_expansion is not None and volume_expansion < 0.9:
        interpretation = "主力试盘（量能不足，持续性待验证）"
        bias = 0
    else:
        interpretation = "上涨趋势延续"
        bias = +1 if validity >= 55 else 0

    return {
        "detected": True,
        "validity_score": validity,
        "interpretation": interpretation,
        "bias": bias,
        "checks": checks,
        "body_atr_ratio": body_atr_ratio,
        "upper_shadow_ratio": upper_shadow_ratio,
        "volume_expansion": volume_expansion,
        "prior_return_20": prior_return_20,
        "prior_return_60": prior_return_60,
        "position_in_250_range": position_in_range,
        "broke_60d_resistance": broke_resistance,
        "news_aligned": news_aligned,
    }


def assess_technical_patterns(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    news_dates: Sequence[pd.Timestamp] | None = None,
) -> dict[str, Any]:
    """综合趋势位置、红三兵、RSI、MACD、动量与波动率，输出 0—100 方向分。"""
    close, _, _, _, volume = _prepare(stock)
    if len(close) < 130:
        return {
            "available": False,
            "score": None,
            "reason": f"行情仅{len(close)}个交易日，至少需要130个才能完成技术形态分析。",
        }

    trend = classify_trend_position(close)
    soldiers = detect_three_white_soldiers(stock, news_dates)
    rsi = float(_rsi_14(close).iloc[-1])
    _, _, histogram = _macd(close)
    macd_hist = float(histogram.iloc[-1])
    macd_hist_prev = float(histogram.iloc[-2])
    returns = close.pct_change(fill_method=None)
    volatility_20 = float(returns.tail(20).std() * np.sqrt(252))
    return_20 = float(close.iloc[-1] / close.iloc[-21] - 1)

    score = 50.0
    reasons: list[str] = []

    score += float(trend["score"])
    reasons.append(f"趋势位置：{trend['label']}（{trend['score']:+.1f}分）")

    if soldiers["detected"]:
        bias = int(soldiers["bias"])
        contribution = bias * (float(soldiers["validity_score"]) - 40.0) * 0.30
        contribution = float(np.clip(contribution, -12.0, 12.0))
        score += contribution
        reasons.append(
            f"红三兵：{soldiers['interpretation']}，有效性{soldiers['validity_score']:.1f}/100"
            f"（{contribution:+.1f}分）"
        )
    else:
        reasons.append(f"红三兵：未出现（{soldiers['reason']}）")

    if rsi >= 80:
        score -= 6
        reasons.append(f"RSI14={rsi:.1f}，严重超买，短期追高风险（-6分）")
    elif rsi >= 70:
        score -= 3
        reasons.append(f"RSI14={rsi:.1f}，超买区（-3分）")
    elif rsi <= 20:
        score += 4
        reasons.append(f"RSI14={rsi:.1f}，严重超卖（+4分）")
    elif rsi <= 30:
        score += 2
        reasons.append(f"RSI14={rsi:.1f}，超卖区（+2分）")
    else:
        reasons.append(f"RSI14={rsi:.1f}，中性区")

    if macd_hist > 0 and macd_hist > macd_hist_prev:
        score += 5
        reasons.append("MACD红柱放大（+5分）")
    elif macd_hist > 0:
        score += 2
        reasons.append("MACD红柱但动能减弱（+2分）")
    elif macd_hist < 0 and macd_hist < macd_hist_prev:
        score -= 5
        reasons.append("MACD绿柱放大（-5分）")
    else:
        score -= 2
        reasons.append("MACD绿柱但跌势收敛（-2分）")

    expected_move = volatility_20 * np.sqrt(20 / 252)
    if expected_move > 0:
        momentum_signal = float(np.clip(return_20 / (2 * expected_move), -1, 1))
        momentum_points = momentum_signal * 6.0
        score += momentum_points
        reasons.append(f"20日波动调整动量{momentum_signal:+.2f}（{momentum_points:+.1f}分）")

    score = float(np.clip(score, 0.0, 100.0))
    return {
        "available": True,
        "score": score,
        "trend": trend,
        "three_white_soldiers": soldiers,
        "rsi_14": rsi,
        "macd_hist": macd_hist,
        "volatility_20": volatility_20,
        "return_20": return_20,
        "reasons": reasons,
    }
