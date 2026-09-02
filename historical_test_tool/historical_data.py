from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

import agent_core as default_core


PRICE_COLUMNS = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]


@dataclass
class HistoricalBundleResult:
    bundle: Any
    requested_date: date
    actual_trading_date: date
    requested_start: date


def fetch_historical_fx(core_module: Any, market: str, cutoff: date) -> dict[str, Any] | None:
    """Return the latest public FX close not later than T for holding conversion."""

    if market == "A股":
        return None
    if market == "美股":
        symbol, pair = "CNY=X", "美元兑人民币"
    elif market == "港股":
        symbol, pair = "HKDCNY=X", "港元兑人民币"
    else:
        raise ValueError("市场仅支持A股、美股或港股。")

    start = (pd.Timestamp(cutoff) - pd.Timedelta(days=20)).date().isoformat()
    data, _ = core_module.fetch_yahoo_chart_history(symbol, start, cutoff.isoformat())
    data = _cut_at(data, cutoff, pair)
    latest = data.iloc[-1]
    rate = float(latest["收盘"])
    if not np.isfinite(rate) or rate <= 0:
        raise RuntimeError(f"{pair}在历史分析日前没有有效数据。")
    return {
        "rate": rate,
        "pair": pair,
        "date": pd.Timestamp(latest["日期"]).date(),
        "provider": f"Yahoo Finance历史汇率（截至{cutoff.isoformat()}）",
    }


def _empty_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


def _cut_at(frame: pd.DataFrame, cutoff: date, label: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise RuntimeError(f"{label}没有返回可用行情。")
    if "日期" not in frame.columns or "收盘" not in frame.columns:
        raise RuntimeError(f"{label}缺少日期或收盘价字段。")

    result = frame.copy()
    result["日期"] = pd.to_datetime(result["日期"], errors="coerce").dt.tz_localize(None)
    result = result.dropna(subset=["日期", "收盘"])
    result = result[result["日期"] <= pd.Timestamp(cutoff)]
    result = result.drop_duplicates("日期", keep="last").sort_values("日期").reset_index(drop=True)
    for column in ("开盘", "最高", "最低", "收盘", "成交量"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["收盘"])
    if result.empty:
        raise RuntimeError(f"{label}在 {cutoff.isoformat()} 当天及以前没有可用行情。")
    if pd.Timestamp(result["日期"].max()).date() > cutoff:
        raise RuntimeError(f"{label}截断失败：仍存在历史分析日之后的数据。")
    return result


def _cut_benchmark(frame: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_prices()
    try:
        return _cut_at(frame, cutoff, "基准指数")
    except RuntimeError:
        return _empty_prices()


def assert_bundle_cutoff(bundle: Any, cutoff: date) -> None:
    stock_max = pd.to_datetime(bundle.stock["日期"], errors="coerce").max()
    if pd.isna(stock_max) or stock_max.date() > cutoff:
        raise RuntimeError("防泄漏检查失败：个股行情包含分析日之后的数据。")
    if bundle.benchmark is not None and not bundle.benchmark.empty:
        benchmark_max = pd.to_datetime(bundle.benchmark["日期"], errors="coerce").max()
        if pd.isna(benchmark_max) or benchmark_max.date() > cutoff:
            raise RuntimeError("防泄漏检查失败：基准行情包含分析日之后的数据。")


def fetch_historical_bundle(
    market: str,
    raw_code: str,
    requested_date: date,
    core_module: Any = default_core,
) -> HistoricalBundleResult:
    """Fetch only data requested through T, then enforce a second row-level cutoff."""

    requested_ts = pd.Timestamp(requested_date).normalize()
    start_ts = requested_ts - pd.DateOffset(years=5)
    start_text = start_ts.date().isoformat()
    end_text = requested_ts.date().isoformat()
    warnings: list[str] = []

    if market == "A股":
        code = core_module.normalize_a_code(raw_code)
        stock, name, provider = core_module.fetch_a_security(code, start_text, end_text)
        benchmark_name = "沪深300"
        try:
            benchmark = core_module.fetch_a_benchmark(start_text, end_text)
        except Exception as exc:
            benchmark = _empty_prices()
            warnings.append(f"沪深300基准暂不可用：{exc}")
        is_fund = bool(core_module.is_exchange_traded_fund_code(code))
        asset_type = "场内基金" if is_fund else "A股个股"
        price_unit = "人民币元"
    elif market == "美股":
        code = core_module.normalize_us_code(raw_code)
        stock, name, provider = core_module.fetch_us_security(code, start_text, end_text)
        benchmark_name = "标普500代理ETF（SPY）"
        try:
            benchmark = core_module.fetch_us_benchmark(start_text, end_text)
        except Exception as exc:
            benchmark = _empty_prices()
            warnings.append(f"标普500代理基准暂不可用：{exc}")
        asset_type = "美股个股"
        price_unit = "美元"
    elif market == "港股":
        code = core_module.normalize_hk_code(raw_code)
        stock, name, provider = core_module.fetch_hk_security(code, start_text, end_text)
        benchmark_name = "恒生指数（HSI）"
        try:
            benchmark = core_module.fetch_hk_benchmark(start_text, end_text)
        except Exception as exc:
            benchmark = _empty_prices()
            warnings.append(f"恒生指数基准暂不可用：{exc}")
        asset_type = "港股个股"
        price_unit = "港元"
    else:
        raise ValueError("市场仅支持A股、美股或港股。")

    stock = _cut_at(stock, requested_ts.date(), "个股")
    actual_date = pd.Timestamp(stock["日期"].max()).date()
    benchmark = _cut_benchmark(benchmark, actual_date)

    first_date = pd.Timestamp(stock["日期"].min())
    expected_span = max(1, (pd.Timestamp(actual_date) - start_ts).days)
    actual_span = max(0, (pd.Timestamp(actual_date) - first_date).days)
    coverage_ratio = float(np.clip(actual_span / expected_span, 0, 1))
    history_complete = bool(first_date <= start_ts + pd.Timedelta(days=45))
    if not history_complete:
        warnings.append("该股票在分析日之前的可得历史不足五年，数据完整度会按原规则降低。")
    if (requested_ts - pd.Timestamp(actual_date)).days > 7:
        warnings.append("请求日期与实际采用交易日相隔超过7天，请核对停牌、退市或数据源状态。")

    bundle = core_module.PriceBundle(
        stock=stock,
        benchmark=benchmark,
        code=code,
        name=str(name or code),
        provider=str(provider),
        benchmark_name=benchmark_name,
        asset_type=asset_type,
        price_unit=price_unit,
        warnings=warnings,
        requested_start=start_text,
        requested_end=actual_date.isoformat(),
        history_complete=history_complete,
        coverage_ratio=coverage_ratio,
    )
    assert_bundle_cutoff(bundle, actual_date)
    return HistoricalBundleResult(
        bundle=bundle,
        requested_date=requested_ts.date(),
        actual_trading_date=actual_date,
        requested_start=start_ts.date(),
    )
