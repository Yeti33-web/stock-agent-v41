from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from io import StringIO
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import baostock as bs
except ImportError:
    bs = None

try:
    import yfinance as yf
except ImportError:
    yf = None


HISTORY_YEARS = 5
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 StockResearchAgent/4.0"}
SEC_HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "IndividualInvestorResearchAgent/4.0 educational-use")}


HORIZONS: list[dict[str, Any]] = [
    {
        "name": "1个交易日",
        "days": 1,
        "fast": 5,
        "slow": 20,
        "minimum_rows": 60,
        "intraday_required": True,
        "review": "下一个交易日前复核",
    },
    {
        "name": "2—5个交易日",
        "days": 5,
        "fast": 5,
        "slow": 20,
        "minimum_rows": 120,
        "intraday_required": False,
        "review": "每个交易日复核",
    },
    {
        "name": "2—4周",
        "days": 20,
        "fast": 20,
        "slow": 60,
        "minimum_rows": 120,
        "intraday_required": False,
        "review": "每周复核",
    },
    {
        "name": "1—3个月",
        "days": 60,
        "fast": 20,
        "slow": 120,
        "minimum_rows": 250,
        "intraday_required": False,
        "review": "每月或重大公告后复核",
    },
    {
        "name": "3—12个月",
        "days": 120,
        "fast": 60,
        "slow": 250,
        "minimum_rows": 500,
        "intraday_required": False,
        "review": "每季度及财报后复核",
    },
    {
        "name": "1—3年",
        "days": 250,
        "fast": 120,
        "slow": 250,
        "minimum_rows": 750,
        "intraday_required": False,
        "review": "每季度及投资逻辑变化时复核",
    },
]


GOAL_PRIORS = {
    "保值为主": {"2—5个交易日": -8, "2—4周": -3, "1—3个月": 2, "3—12个月": 9, "1—3年": 8},
    "股息／稳健收益": {"2—5个交易日": -8, "2—4周": -4, "1—3个月": 2, "3—12个月": 9, "1—3年": 13},
    "长期增值": {"2—5个交易日": -9, "2—4周": -4, "1—3个月": 2, "3—12个月": 10, "1—3年": 14},
    "波段操作": {"2—5个交易日": 3, "2—4周": 12, "1—3个月": 13, "3—12个月": 2, "1—3年": -6},
    "短线交易": {"2—5个交易日": 14, "2—4周": 8, "1—3个月": -1, "3—12个月": -8, "1—3年": -12},
}


@dataclass
class PriceBundle:
    stock: pd.DataFrame
    benchmark: pd.DataFrame
    code: str
    name: str
    provider: str
    benchmark_name: str
    asset_type: str
    price_unit: str
    warnings: list[str] = field(default_factory=list)
    requested_start: str = ""
    requested_end: str = ""
    history_complete: bool = True
    coverage_ratio: float = 1.0


@dataclass
class EvidenceSnapshot:
    available: bool
    provider: str
    fields: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    positives: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "" or pd.isna(value):
            return None
        result = float(str(value).replace(",", "").replace("%", ""))
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def ratio_fraction(value: Any) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    if abs(number) > 5:
        return number / 100
    return number


def calculate_holding_values(
    market: str,
    shares: float,
    latest_price: float,
    cost_price: float | None = None,
    usd_cny_rate: float | None = None,
) -> dict[str, Any]:
    share_count = float(shares)
    price = float(latest_price)
    if share_count <= 0:
        raise ValueError("持股数量必须大于0。")
    if price <= 0:
        raise ValueError("最新公开价格无效，暂时无法计算持仓市值。")
    if market not in {"A股", "美股"}:
        raise ValueError("暂不支持该市场的持仓换算。")

    current_native = share_count * price
    rate = 1.0
    if market == "美股":
        rate = float(usd_cny_rate or 0.0)
        if rate <= 0:
            raise ValueError("美元兑人民币汇率暂不可用，请改用“按持仓金额填写”。")
    current_rmb = current_native * rate

    parsed_cost = float(cost_price or 0.0)
    cost_total_native = share_count * parsed_cost if parsed_cost > 0 else None
    profit_native = current_native - cost_total_native if cost_total_native is not None else None
    return_rate = price / parsed_cost - 1 if parsed_cost > 0 else None
    return {
        "method": "按持股数量填写",
        "shares": share_count,
        "latest_price": price,
        "native_currency": "人民币元" if market == "A股" else "美元",
        "current_native": current_native,
        "current_rmb": current_rmb,
        "cost_price": parsed_cost if parsed_cost > 0 else None,
        "cost_total_native": cost_total_native,
        "cost_total_rmb": cost_total_native * rate if cost_total_native is not None else None,
        "profit_native": profit_native,
        "profit_rmb": profit_native * rate if profit_native is not None else None,
        "return_rate": return_rate,
        "usd_cny_rate": rate if market == "美股" else None,
    }


def calculate_amount_holding_values(
    current_rmb: float,
    total_cost_rmb: float | None = None,
) -> dict[str, Any]:
    current_value = float(current_rmb)
    if current_value <= 0:
        raise ValueError("当前持仓市值必须大于0。")

    parsed_cost = float(total_cost_rmb or 0.0)
    profit_rmb = current_value - parsed_cost if parsed_cost > 0 else None
    return_rate = current_value / parsed_cost - 1 if parsed_cost > 0 else None
    return {
        "method": "按持仓金额填写",
        "shares": None,
        "latest_price": None,
        "native_currency": "人民币元",
        "current_native": current_value,
        "current_rmb": current_value,
        "cost_price": None,
        "cost_total_native": parsed_cost if parsed_cost > 0 else None,
        "cost_total_rmb": parsed_cost if parsed_cost > 0 else None,
        "profit_native": profit_rmb,
        "profit_rmb": profit_rmb,
        "return_rate": return_rate,
        "usd_cny_rate": None,
    }


def normalize_a_code(raw_code: str) -> str:
    code = raw_code.strip().upper()
    for suffix in (".SZ", ".SH", ".SS", ".BJ"):
        code = code.replace(suffix, "")
    if not (len(code) == 6 and code.isdigit()):
        raise ValueError("A股代码应为6位数字，例如000001、600519、300750或北交所代码。")
    return code


def normalize_us_code(raw_code: str) -> str:
    code = raw_code.strip().upper()
    if not code or len(code) > 15 or not all(char.isalnum() or char in ".-" for char in code):
        raise ValueError("请输入有效美股代码，例如AAPL、MSFT、NVDA或BRK-B。")
    return code


def is_exchange_traded_fund_code(code: str) -> bool:
    return code.startswith(("1", "5"))


def a_share_exchange(code: str) -> str:
    if code.startswith(("4", "8", "92")):
        return "bj"
    if code[0] in {"5", "6", "9"}:
        return "sh"
    return "sz"


def a_share_baostock_code(code: str) -> str:
    return f"{a_share_exchange(code)}.{code}"


def a_share_yahoo_ticker(code: str) -> str:
    suffix = ".BJ" if a_share_exchange(code) == "bj" else ".SS" if a_share_exchange(code) == "sh" else ".SZ"
    return f"{code}{suffix}"


def clean_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘", "成交量"])
    result = data.copy()
    result["日期"] = pd.to_datetime(result["日期"], errors="coerce").dt.tz_localize(None)
    for column in ["开盘", "最高", "最低", "收盘", "成交量"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["日期", "开盘", "最高", "最低", "收盘"])
    result = result[(result["收盘"] > 0) & (result["最高"] >= result["最低"])]
    result = result[result["日期"] <= pd.Timestamp(date.today()) + pd.Timedelta(days=1)]
    result["成交量"] = result["成交量"].fillna(0).clip(lower=0)
    return result.sort_values("日期").drop_duplicates("日期", keep="last").reset_index(drop=True)


def standardize_chinese_ohlcv(raw: pd.DataFrame, volume_in_lots: bool = False) -> pd.DataFrame:
    required = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"行情接口返回数据缺少字段：{', '.join(missing)}")
    data = raw[required].copy()
    if volume_in_lots:
        data["成交量"] = pd.to_numeric(data["成交量"], errors="coerce") * 100
    return clean_ohlcv(data)


def standardize_english_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return clean_ohlcv(pd.DataFrame())
    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index() if not isinstance(data.index, pd.RangeIndex) else data
    lower_map = {str(column).lower(): column for column in data.columns}
    date_column = lower_map.get("date") or lower_map.get("datetime")
    mapping = {
        date_column: "日期",
        lower_map.get("open"): "开盘",
        lower_map.get("high"): "最高",
        lower_map.get("low"): "最低",
        lower_map.get("close"): "收盘",
        lower_map.get("volume"): "成交量",
    }
    if any(key is None for key in mapping):
        raise ValueError("英文行情接口返回字段不完整。")
    return clean_ohlcv(data[list(mapping)].rename(columns=mapping))


def filter_dates(data: pd.DataFrame, start_text: str, end_text: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_text)
    end_ts = pd.Timestamp(end_text)
    return data[(data["日期"] >= start_ts) & (data["日期"] <= end_ts)].copy().reset_index(drop=True)


def _baostock_result_to_frame(result: Any) -> pd.DataFrame:
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields) if rows else pd.DataFrame(columns=result.fields)


def fetch_baostock_history(symbol: str, start_text: str, end_text: str, adjustflag: str = "2") -> pd.DataFrame:
    if bs is None:
        return pd.DataFrame()
    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"BaoStock登录失败：{login_result.error_msg}")
    try:
        result = bs.query_history_k_data_plus(
            symbol,
            "date,open,high,low,close,volume",
            start_date=start_text,
            end_date=end_text,
            frequency="d",
            adjustflag=adjustflag,
        )
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock查询失败：{result.error_msg}")
        raw = _baostock_result_to_frame(result).rename(
            columns={"date": "日期", "open": "开盘", "high": "最高", "low": "最低", "close": "收盘", "volume": "成交量"}
        )
        return clean_ohlcv(raw)
    finally:
        bs.logout()


def fetch_yfinance_history(symbol: str, start_text: str, end_text: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    end_exclusive = (pd.Timestamp(end_text) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        symbol,
        start=start_text,
        end=end_exclusive,
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=15,
    )
    return standardize_english_ohlcv(raw)


def _get_with_retry(url: str, *, params: dict[str, Any] | None = None, timeout: int = 20) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=timeout)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_error = RuntimeError(f"HTTP {response.status_code}")
                if attempt < 2:
                    time.sleep(0.6 * (2**attempt))
                    continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.6 * (2**attempt))
    raise RuntimeError(f"公开行情接口连续重试后仍失败：{last_error}")


def fetch_yahoo_chart_history(symbol: str, start_text: str, end_text: str) -> tuple[pd.DataFrame, str]:
    period1 = int(pd.Timestamp(start_text, tz="UTC").timestamp())
    period2 = int((pd.Timestamp(end_text, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    response = _get_with_retry(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        },
        timeout=20,
    )
    payload = response.json().get("chart", {})
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    results = payload.get("result") or []
    if not results:
        return pd.DataFrame(), symbol
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjusted_list = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or quotes.get("close") or []
    if not timestamps or not quotes.get("close"):
        return pd.DataFrame(), symbol
    raw_close = pd.Series(quotes.get("close"), dtype="float64")
    adjusted_close = pd.Series(adjusted_list, dtype="float64")
    factor = adjusted_close / raw_close.replace(0, np.nan)
    frame = pd.DataFrame(
        {
            "日期": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
            "开盘": pd.Series(quotes.get("open"), dtype="float64") * factor,
            "最高": pd.Series(quotes.get("high"), dtype="float64") * factor,
            "最低": pd.Series(quotes.get("low"), dtype="float64") * factor,
            "收盘": adjusted_close,
            "成交量": pd.Series(quotes.get("volume"), dtype="float64"),
        }
    )
    meta = result.get("meta", {})
    name = str(meta.get("longName") or meta.get("shortName") or symbol)
    return clean_ohlcv(frame), name


def fetch_stooq_history(symbol: str, start_text: str, end_text: str) -> pd.DataFrame:
    normalized = normalize_us_code(symbol).lower().replace("-", ".")
    response = _get_with_retry(
        "https://stooq.com/q/d/l/",
        params={
            "s": f"{normalized}.us",
            "d1": start_text.replace("-", ""),
            "d2": end_text.replace("-", ""),
            "i": "d",
        },
        timeout=20,
    )
    if not response.text.strip() or response.text.lstrip().lower().startswith("no data"):
        return pd.DataFrame()
    raw = pd.read_csv(StringIO(response.text))
    return standardize_english_ohlcv(raw)


def fetch_usd_cny_rate() -> dict[str, Any]:
    end_text = date.today().strftime("%Y-%m-%d")
    start_text = (pd.Timestamp(end_text) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    errors: list[str] = []
    candidates: list[dict[str, Any]] = []

    try:
        data, _ = fetch_yahoo_chart_history("CNY=X", start_text, end_text)
        if not data.empty:
            latest = data.iloc[-1]
            candidates.append(
                {
                    "rate": float(latest["收盘"]),
                    "date": pd.Timestamp(latest["日期"]),
                    "provider": "Yahoo Finance美元兑人民币公开日线（CNY=X）",
                }
            )
    except Exception as exc:
        errors.append(f"Yahoo图表接口：{exc}")

    try:
        data = fetch_yfinance_history("CNY=X", start_text, end_text)
        if not data.empty:
            latest = data.iloc[-1]
            candidates.append(
                {
                    "rate": float(latest["收盘"]),
                    "date": pd.Timestamp(latest["日期"]),
                    "provider": "yfinance美元兑人民币备用日线（CNY=X）",
                }
            )
    except Exception as exc:
        errors.append(f"yfinance：{exc}")

    try:
        response = _get_with_retry(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": "DEXCHUS"},
            timeout=20,
        )
        raw = pd.read_csv(StringIO(response.text))
        date_column = next((column for column in raw.columns if "date" in str(column).lower()), None)
        value_column = next((column for column in raw.columns if str(column).upper() == "DEXCHUS"), None)
        if date_column and value_column:
            values = raw[[date_column, value_column]].copy()
            values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
            values[date_column] = pd.to_datetime(values[date_column], errors="coerce")
            values = values.dropna()
            if not values.empty:
                latest = values.iloc[-1]
                candidates.append(
                    {
                        "rate": float(latest[value_column]),
                        "date": pd.Timestamp(latest[date_column]),
                        "provider": "美国FRED美元兑人民币参考汇率（DEXCHUS）",
                    }
                )
    except Exception as exc:
        errors.append(f"FRED：{exc}")

    valid = [item for item in candidates if 4.0 <= float(item["rate"]) <= 12.0]
    if not valid:
        detail = "；".join(errors) if errors else "公开接口没有返回有效汇率"
        raise RuntimeError(f"美元兑人民币汇率获取失败。{detail}")
    return max(valid, key=lambda item: pd.Timestamp(item["date"]))


def fetch_akshare_us_history(symbol: str, start_text: str, end_text: str) -> tuple[pd.DataFrame, str]:
    if ak is None:
        return pd.DataFrame(), ""
    last_error: Exception | None = None
    candidates = [symbol] if symbol[:3].isdigit() and "." in symbol else [f"105.{symbol}", f"106.{symbol}", f"107.{symbol}"]
    for internal_code in candidates:
        try:
            raw = ak.stock_us_hist(
                symbol=internal_code,
                period="daily",
                start_date=start_text.replace("-", ""),
                end_date=end_text.replace("-", ""),
                adjust="qfq",
            )
            data = standardize_chinese_ohlcv(raw)
            if not data.empty:
                return data, internal_code
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.DataFrame(), ""


def _candidate_quality(item: tuple[pd.DataFrame, str, str]) -> tuple[int, int]:
    data = item[0]
    if data.empty:
        return (-10_000, 0)
    lag = max(0, (pd.Timestamp(date.today()) - data["日期"].max()).days)
    return (len(data) - lag * 5, len(data))


def fetch_us_security(symbol: str, start_text: str, end_text: str) -> tuple[pd.DataFrame, str, str]:
    code = normalize_us_code(symbol)
    candidates: list[tuple[pd.DataFrame, str, str]] = []
    errors: list[str] = []
    try:
        data, name = fetch_yahoo_chart_history(code, start_text, end_text)
        if not data.empty:
            candidates.append((data, name, "Yahoo Finance图表公开接口（复权日线）"))
            if len(data) >= 500 and (pd.Timestamp(date.today()) - data["日期"].max()).days <= 10:
                return candidates[-1]
    except Exception as exc:
        errors.append(f"Yahoo图表接口：{exc}")
    for source_name, loader in [
        ("yfinance备用日线（自动复权）", lambda: fetch_yfinance_history(code, start_text, end_text)),
        ("Stooq独立备用日线", lambda: fetch_stooq_history(code, start_text, end_text)),
    ]:
        try:
            data = loader()
            if not data.empty:
                candidates.append((data, code, source_name))
        except Exception as exc:
            errors.append(f"{source_name.split('公开')[0]}：{exc}")
    try:
        data, internal_code = fetch_akshare_us_history(code, start_text, end_text)
        if not data.empty:
            candidates.append((data, code, f"AKShare／东方财富美股日线（{internal_code}）"))
    except Exception as exc:
        errors.append(f"AKShare备用：{exc}")
    if candidates:
        return max(candidates, key=_candidate_quality)
    detail = "；".join(errors) if errors else "相关组件未安装或接口无数据"
    raise RuntimeError(f"美股数据通道均未返回有效数据。{detail}")


def _fetch_sina_fund(code: str, start_text: str, end_text: str) -> pd.DataFrame:
    if ak is None:
        return pd.DataFrame()
    exchange = "sh" if code.startswith("5") else "sz"
    raw = ak.fund_etf_hist_sina(symbol=f"{exchange}{code}")
    return filter_dates(standardize_english_ohlcv(raw), start_text, end_text)


def fetch_a_security(code: str, start_text: str, end_text: str) -> tuple[pd.DataFrame, str, str]:
    normalized = normalize_a_code(code)
    candidates: list[tuple[pd.DataFrame, str, str]] = []
    errors: list[str] = []
    if is_exchange_traded_fund_code(normalized):
        try:
            data = _fetch_sina_fund(normalized, start_text, end_text)
            if not data.empty:
                candidates.append((data, normalized, "AKShare／新浪场内基金日线（未复权）"))
        except Exception as exc:
            errors.append(f"新浪基金：{exc}")
    try:
        bao_code = a_share_baostock_code(normalized)
        data = fetch_baostock_history(bao_code, start_text, end_text, adjustflag="2")
        if not data.empty:
            candidates.append((data, normalized, f"BaoStock公开行情（{bao_code}，前复权日线）"))
    except Exception as exc:
        errors.append(f"BaoStock：{exc}")
    if ak is not None:
        try:
            if is_exchange_traded_fund_code(normalized):
                raw = ak.fund_etf_hist_em(
                    symbol=normalized,
                    period="daily",
                    start_date=start_text.replace("-", ""),
                    end_date=end_text.replace("-", ""),
                    adjust="qfq",
                )
            else:
                raw = ak.stock_zh_a_hist(
                    symbol=normalized,
                    period="daily",
                    start_date=start_text.replace("-", ""),
                    end_date=end_text.replace("-", ""),
                    adjust="qfq",
                    timeout=15,
                )
            data = standardize_chinese_ohlcv(raw, volume_in_lots=True)
            if not data.empty:
                candidates.append((data, normalized, "AKShare／东方财富公开行情（前复权日线）"))
        except Exception as exc:
            errors.append(f"AKShare／东方财富：{exc}")
    try:
        yahoo_symbol = a_share_yahoo_ticker(normalized)
        data, yahoo_name = fetch_yahoo_chart_history(yahoo_symbol, start_text, end_text)
        if not data.empty:
            candidates.append((data, yahoo_name or normalized, f"Yahoo Finance图表备用行情（{yahoo_symbol}）"))
            if len(data) >= 500 and (pd.Timestamp(date.today()) - data["日期"].max()).days <= 10:
                return candidates[-1]
    except Exception as exc:
        errors.append(f"Yahoo图表备用：{exc}")
    try:
        yahoo_symbol = a_share_yahoo_ticker(normalized)
        data = fetch_yfinance_history(yahoo_symbol, start_text, end_text)
        if not data.empty:
            candidates.append((data, normalized, f"yfinance备用行情（{yahoo_symbol}）"))
    except Exception as exc:
        errors.append(f"yfinance备用：{exc}")
    if candidates:
        return max(candidates, key=_candidate_quality)
    detail = "；".join(errors) if errors else "相关组件未安装或接口无数据"
    raise RuntimeError(f"A股数据通道均未返回有效数据。{detail}")


def fetch_a_benchmark(start_text: str, end_text: str) -> pd.DataFrame:
    errors: list[str] = []
    try:
        data = fetch_baostock_history("sh.000300", start_text, end_text, adjustflag="3")
        if not data.empty:
            return data
    except Exception as exc:
        errors.append(str(exc))
    if ak is not None:
        try:
            raw = ak.index_zh_a_hist(
                symbol="000300",
                period="daily",
                start_date=start_text.replace("-", ""),
                end_date=end_text.replace("-", ""),
            )
            data = standardize_chinese_ohlcv(raw)
            if not data.empty:
                return data
        except Exception as exc:
            errors.append(str(exc))
    try:
        data, _ = fetch_yahoo_chart_history("000300.SS", start_text, end_text)
        if not data.empty:
            return data
    except Exception as exc:
        errors.append(str(exc))
    try:
        data = fetch_yfinance_history("000300.SS", start_text, end_text)
        if not data.empty:
            return data
    except Exception as exc:
        errors.append(str(exc))
    raise RuntimeError("沪深300基准获取失败。" + "；".join(errors))


def fetch_us_benchmark(start_text: str, end_text: str) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    try:
        data, _ = fetch_yahoo_chart_history("SPY", start_text, end_text)
        if not data.empty:
            return data
    except Exception:
        pass
    for loader in (
        lambda: fetch_yfinance_history("SPY", start_text, end_text),
        lambda: fetch_stooq_history("SPY", start_text, end_text),
    ):
        try:
            data = loader()
            if not data.empty:
                candidates.append(data)
        except Exception:
            pass
    if not candidates:
        try:
            data, _ = fetch_akshare_us_history("SPY", start_text, end_text)
            if not data.empty:
                candidates.append(data)
        except Exception:
            pass
    if not candidates:
        raise RuntimeError("标普500代理基准SPY获取失败。")
    return max(candidates, key=len)


def _bao_first_row(result: Any) -> dict[str, Any]:
    if result.error_code != "0":
        return {}
    frame = _baostock_result_to_frame(result)
    if frame.empty:
        return {}
    return frame.iloc[-1].to_dict()


def _recent_report_quarters(count: int = 10) -> list[tuple[int, int]]:
    today = date.today()
    year = today.year
    quarter = max(1, (today.month - 1) // 3)
    result: list[tuple[int, int]] = []
    for _ in range(count):
        result.append((year, quarter))
        quarter -= 1
        if quarter == 0:
            year -= 1
            quarter = 4
    return result


def _score_fundamentals(fields: dict[str, Any]) -> tuple[float | None, list[str], list[str]]:
    measurable = 0
    score = 50.0
    positives: list[str] = []
    risks: list[str] = []

    roe = ratio_fraction(fields.get("净资产收益率"))
    if roe is not None:
        measurable += 1
        if roe >= 0.15:
            score += 10
            positives.append("净资产收益率相对较高")
        elif roe >= 0.08:
            score += 5
        elif roe < 0:
            score -= 12
            risks.append("净资产收益率为负")

    margin = ratio_fraction(fields.get("净利率"))
    if margin is not None:
        measurable += 1
        if margin >= 0.15:
            score += 8
            positives.append("净利率相对较高")
        elif margin >= 0.05:
            score += 3
        elif margin < 0:
            score -= 10
            risks.append("净利率为负")

    profit_growth = ratio_fraction(fields.get("净利润同比"))
    if profit_growth is not None:
        measurable += 1
        if profit_growth >= 0.15:
            score += 8
            positives.append("净利润同比增长")
        elif profit_growth < 0:
            score -= 9
            risks.append("净利润同比下降")

    revenue_growth = ratio_fraction(fields.get("营收同比"))
    if revenue_growth is not None:
        measurable += 1
        if revenue_growth >= 0.10:
            score += 6
            positives.append("营业收入保持增长")
        elif revenue_growth < 0:
            score -= 6
            risks.append("营业收入同比下降")

    debt_ratio = ratio_fraction(fields.get("资产负债率"))
    if debt_ratio is not None:
        measurable += 1
        if debt_ratio >= 0.75:
            score -= 10
            risks.append("资产负债率较高，需结合行业判断")
        elif debt_ratio >= 0.55:
            score -= 4
        elif debt_ratio <= 0.35:
            score += 4

    cash_quality = ratio_fraction(fields.get("经营现金流／净利润"))
    if cash_quality is not None:
        measurable += 1
        if cash_quality >= 1:
            score += 7
            positives.append("经营现金流对利润覆盖较好")
        elif cash_quality < 0:
            score -= 9
            risks.append("经营现金流与净利润方向不一致")

    pe = safe_float(fields.get("市盈率TTM"))
    if pe is not None:
        measurable += 1
        if pe <= 0:
            score -= 7
            risks.append("市盈率不可用或公司处于亏损状态")
        elif pe > 80:
            score -= 6
            risks.append("市盈率较高，估值容错空间可能较低")
        elif pe < 20:
            score += 3

    if measurable < 2:
        return None, positives, risks
    return float(np.clip(score, 0, 100)), positives, risks


def fetch_a_fundamentals(code: str, last_price: float, asset_type: str) -> EvidenceSnapshot:
    if asset_type != "A股个股":
        return EvidenceSnapshot(False, "不适用", notes=["场内基金不使用单一上市公司的财务报表评分。"])
    if bs is None:
        return EvidenceSnapshot(False, "BaoStock未安装", notes=["未取得财务数据，不参与评分。"])
    bao_code = a_share_baostock_code(code)
    login_result = bs.login()
    if login_result.error_code != "0":
        return EvidenceSnapshot(False, "BaoStock", notes=[f"财务接口登录失败：{login_result.error_msg}"])
    fields: dict[str, Any] = {}
    notes: list[str] = []
    name = code
    industry = "未取得"
    try:
        basic = _bao_first_row(bs.query_stock_basic(code=bao_code))
        name = str(basic.get("code_name") or code)
        industry_row = _bao_first_row(bs.query_stock_industry(code=bao_code))
        industry = str(industry_row.get("industry") or "未取得")
        latest_period = ""
        for year, quarter in _recent_report_quarters():
            profit = _bao_first_row(bs.query_profit_data(code=bao_code, year=year, quarter=quarter))
            if not profit:
                continue
            growth = _bao_first_row(bs.query_growth_data(code=bao_code, year=year, quarter=quarter))
            balance = _bao_first_row(bs.query_balance_data(code=bao_code, year=year, quarter=quarter))
            cashflow = _bao_first_row(bs.query_cash_flow_data(code=bao_code, year=year, quarter=quarter))
            latest_period = str(profit.get("statDate") or f"{year}Q{quarter}")
            previous_profit = _bao_first_row(bs.query_profit_data(code=bao_code, year=year - 1, quarter=quarter))
            current_revenue = safe_float(profit.get("MBRevenue"))
            previous_revenue = safe_float(previous_profit.get("MBRevenue"))
            revenue_growth = (
                current_revenue / previous_revenue - 1
                if current_revenue is not None and previous_revenue not in {None, 0}
                else None
            )
            fields.update(
                {
                    "公司名称": name,
                    "行业": industry,
                    "报告期": latest_period,
                    "净资产收益率": safe_float(profit.get("roeAvg")),
                    "净利率": safe_float(profit.get("npMargin")),
                    "净利润同比": safe_float(growth.get("YOYNI")),
                    "营收同比": revenue_growth,
                    "资产负债率": safe_float(balance.get("liabilityToAsset")),
                    "经营现金流／净利润": safe_float(cashflow.get("CFOToNP")),
                    "每股收益TTM": safe_float(profit.get("epsTTM")),
                }
            )
            break
        eps = safe_float(fields.get("每股收益TTM"))
        fields["市盈率TTM"] = last_price / eps if eps is not None and eps != 0 else None
        if not latest_period:
            notes.append("公开接口没有返回近期财务报告。")
    except Exception as exc:
        notes.append(f"部分财务数据获取失败：{exc}")
    finally:
        bs.logout()
    score, positives, risks = _score_fundamentals(fields)
    return EvidenceSnapshot(score is not None, "BaoStock公开财务数据", fields, score, positives, risks, notes)


def _sec_fact_series(facts: dict[str, Any], tags: tuple[str, ...], unit_names: tuple[str, ...]) -> list[dict[str, Any]]:
    for tag in tags:
        fact = facts.get(tag)
        if not fact:
            continue
        units = fact.get("units", {})
        for unit_name in unit_names:
            observations = units.get(unit_name)
            if observations:
                annual = [item for item in observations if item.get("form") in {"10-K", "20-F", "40-F"} and item.get("fp") == "FY"]
                annual.sort(key=lambda item: (str(item.get("end", "")), str(item.get("filed", ""))))
                unique_by_end: dict[str, dict[str, Any]] = {}
                for item in annual:
                    unique_by_end[str(item.get("end", ""))] = item
                return list(unique_by_end.values())
    return []


def _last_value(series: list[dict[str, Any]]) -> float | None:
    return safe_float(series[-1].get("val")) if series else None


def fetch_sec_fundamentals(symbol: str, last_price: float | None) -> EvidenceSnapshot:
    ticker_response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=20)
    ticker_response.raise_for_status()
    ticker_items = ticker_response.json().values()
    matched = next((item for item in ticker_items if str(item.get("ticker", "")).upper() == symbol.upper()), None)
    if not matched:
        return EvidenceSnapshot(False, "美国SEC", notes=["SEC代码表未找到该证券，可能是ETF或非美国申报主体。"])
    cik = f"{int(matched['cik_str']):010d}"
    facts_response = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=SEC_HEADERS, timeout=30)
    facts_response.raise_for_status()
    payload = facts_response.json()
    facts = payload.get("facts", {}).get("us-gaap", {})
    revenue_series = _sec_fact_series(facts, ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"), ("USD",))
    income_series = _sec_fact_series(facts, ("NetIncomeLoss", "ProfitLoss"), ("USD",))
    equity_series = _sec_fact_series(facts, ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), ("USD",))
    assets_series = _sec_fact_series(facts, ("Assets",), ("USD",))
    liabilities_series = _sec_fact_series(facts, ("Liabilities",), ("USD",))
    cashflow_series = _sec_fact_series(facts, ("NetCashProvidedByUsedInOperatingActivities",), ("USD",))
    eps_series = _sec_fact_series(facts, ("EarningsPerShareDiluted", "EarningsPerShareBasic"), ("USD/shares", "USD / shares"))
    revenue = _last_value(revenue_series)
    prior_revenue = safe_float(revenue_series[-2].get("val")) if len(revenue_series) >= 2 else None
    income = _last_value(income_series)
    prior_income = safe_float(income_series[-2].get("val")) if len(income_series) >= 2 else None
    equity = _last_value(equity_series)
    assets = _last_value(assets_series)
    liabilities = _last_value(liabilities_series)
    cashflow = _last_value(cashflow_series)
    eps = _last_value(eps_series)
    report_end = revenue_series[-1].get("end") if revenue_series else income_series[-1].get("end") if income_series else "最近年度"
    fields = {
        "公司名称": payload.get("entityName") or matched.get("title") or symbol,
        "行业": "请结合SEC申报行业另行核对",
        "报告期": report_end,
        "净资产收益率": income / equity if income is not None and equity not in {None, 0} else None,
        "净利率": income / revenue if income is not None and revenue not in {None, 0} else None,
        "净利润同比": income / prior_income - 1 if income is not None and prior_income not in {None, 0} else None,
        "营收同比": revenue / prior_revenue - 1 if revenue is not None and prior_revenue not in {None, 0} else None,
        "资产负债率": liabilities / assets if liabilities is not None and assets not in {None, 0} else None,
        "经营现金流／净利润": cashflow / income if cashflow is not None and income not in {None, 0} else None,
        "市盈率TTM": last_price / eps if last_price is not None and eps not in {None, 0} else None,
    }
    score, positives, risks = _score_fundamentals(fields)
    notes = ["SEC公司事实数据采用最近可得年度申报口径，市盈率为最新价格除以年度每股收益的简化值。"]
    return EvidenceSnapshot(score is not None, "美国SEC Companyfacts公开申报数据", fields, score, positives, risks, notes)


def fetch_us_fundamentals(symbol: str, last_price: float | None = None) -> EvidenceSnapshot:
    sec_notes: list[str] = []
    try:
        sec_result = fetch_sec_fundamentals(symbol, last_price)
        if sec_result.available:
            return sec_result
        sec_notes.extend(sec_result.notes)
    except Exception as exc:
        sec_notes.append(f"SEC财务接口暂不可用：{exc}")
    if yf is None:
        return EvidenceSnapshot(False, "美国SEC／yfinance", notes=sec_notes + ["未取得财务数据，不参与评分。"])
    fields: dict[str, Any] = {}
    notes: list[str] = []
    try:
        info = yf.Ticker(symbol).get_info()
        fields = {
            "公司名称": info.get("longName") or info.get("shortName") or symbol,
            "行业": info.get("industry") or info.get("sector") or "未取得",
            "报告期": "Yahoo Finance最近可得口径",
            "净资产收益率": safe_float(info.get("returnOnEquity")),
            "净利率": safe_float(info.get("profitMargins")),
            "净利润同比": safe_float(info.get("earningsGrowth")),
            "营收同比": safe_float(info.get("revenueGrowth")),
            "资产负债率": None,
            "经营现金流／净利润": None,
            "市盈率TTM": safe_float(info.get("trailingPE")),
            "市净率": safe_float(info.get("priceToBook")),
            "总市值": safe_float(info.get("marketCap")),
        }
        debt_to_equity = safe_float(info.get("debtToEquity"))
        if debt_to_equity is not None:
            fields["债务／权益"] = debt_to_equity / 100 if abs(debt_to_equity) > 5 else debt_to_equity
    except Exception as exc:
        notes.append(f"Yahoo财务接口暂不可用：{exc}")
    score, positives, risks = _score_fundamentals(fields)
    return EvidenceSnapshot(score is not None, "Yahoo Finance公开公司资料", fields, score, positives, risks, sec_notes + notes)


def derive_market_regime(benchmark: pd.DataFrame) -> dict[str, Any]:
    if benchmark is None or benchmark.empty:
        return {
            "市场状态": "基准数据暂不可用",
            "市场分": 50,
            "近60日收益": np.nan,
            "近250日收益": np.nan,
            "近60日年化波动": np.nan,
        }
    close = benchmark.set_index("日期")["收盘"].sort_index()
    returns = close.pct_change().dropna()
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else close.mean()
    ma250 = close.rolling(250).mean().iloc[-1] if len(close) >= 250 else close.mean()
    r60 = close.iloc[-1] / close.iloc[-min(61, len(close))] - 1 if len(close) > 1 else 0.0
    r250 = close.iloc[-1] / close.iloc[-min(251, len(close))] - 1 if len(close) > 1 else 0.0
    volatility = returns.tail(min(60, len(returns))).std() * np.sqrt(252) if not returns.empty else np.nan
    if close.iloc[-1] >= ma60 and close.iloc[-1] >= ma250 and r60 > 0:
        regime = "偏强"
        score = 60
    elif close.iloc[-1] < ma60 and close.iloc[-1] < ma250 and r60 < 0:
        regime = "偏弱"
        score = 38
    else:
        regime = "震荡／方向不明"
        score = 50
    if np.isfinite(volatility) and volatility > 0.30:
        score -= 5
    return {
        "市场状态": regime,
        "市场分": int(np.clip(score, 0, 100)),
        "近60日收益": r60,
        "近250日收益": r250,
        "近60日年化波动": volatility,
    }


def _find_column(frame: pd.DataFrame, patterns: tuple[str, ...]) -> str | None:
    for column in frame.columns:
        text = str(column).lower()
        if any(pattern.lower() in text for pattern in patterns):
            return str(column)
    return None


def fetch_macro_snapshot(market: str, benchmark: pd.DataFrame) -> EvidenceSnapshot:
    regime = derive_market_regime(benchmark)
    fields: dict[str, Any] = dict(regime)
    notes: list[str] = []
    score = float(regime["市场分"])
    if benchmark is None or benchmark.empty:
        notes.append("市场基准暂未取得，宏观环境仅按中性处理，不影响个股行情继续分析。")
    if market == "A股" and ak is not None:
        try:
            raw = ak.macro_china_lpr()
            date_col = _find_column(raw, ("trade_date", "date", "日期"))
            rate_col = _find_column(raw, ("lpr1y", "1年", "1y"))
            if date_col and rate_col:
                temp = raw[[date_col, rate_col]].copy()
                temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
                temp[rate_col] = pd.to_numeric(temp[rate_col], errors="coerce")
                temp = temp.dropna().sort_values(date_col)
                if not temp.empty:
                    latest = float(temp[rate_col].iloc[-1])
                    previous = float(temp[rate_col].iloc[max(0, len(temp) - 7)])
                    direction = "下降" if latest < previous else "上升" if latest > previous else "持平"
                    fields.update({"利率指标": "1年期LPR", "最新利率": latest, "近半年方向": direction})
                    score += 3 if direction == "下降" else -3 if direction == "上升" else 0
        except Exception as exc:
            notes.append(f"LPR数据暂不可用：{exc}")
    elif market == "美股":
        try:
            response = requests.get(
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
                headers=HTTP_HEADERS,
                timeout=15,
            )
            response.raise_for_status()
            temp = pd.read_csv(StringIO(response.text))
            value_col = _find_column(temp, ("dgs10",))
            if value_col:
                values = pd.to_numeric(temp[value_col], errors="coerce").dropna()
                if not values.empty:
                    latest = float(values.iloc[-1])
                    previous = float(values.iloc[max(0, len(values) - 60)])
                    direction = "下降" if latest < previous - 0.10 else "上升" if latest > previous + 0.10 else "持平"
                    fields.update({"利率指标": "美国10年期国债收益率", "最新利率": latest, "近阶段方向": direction})
                    score += 3 if direction == "下降" else -3 if direction == "上升" else 0
        except Exception as exc:
            notes.append(f"美国利率数据暂不可用：{exc}")
    score = float(np.clip(score, 0, 100))
    return EvidenceSnapshot(True, "市场基准与公开宏观数据", fields, score, notes=notes)


def fetch_price_bundle(market: str, raw_code: str) -> PriceBundle:
    end_text = date.today().strftime("%Y-%m-%d")
    start_text = (pd.Timestamp(end_text) - pd.DateOffset(years=HISTORY_YEARS)).strftime("%Y-%m-%d")
    warnings: list[str] = []
    if market == "A股":
        code = normalize_a_code(raw_code)
        stock, name, provider = fetch_a_security(code, start_text, end_text)
        try:
            benchmark = fetch_a_benchmark(start_text, end_text)
        except Exception as exc:
            benchmark = pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘", "成交量"])
            warnings.append(f"沪深300基准暂不可用：{exc}")
        asset_type = "场内基金" if is_exchange_traded_fund_code(code) else "A股个股"
        bundle = PriceBundle(stock, benchmark, code, name, provider, "沪深300", asset_type, "人民币元", warnings)
    else:
        code = normalize_us_code(raw_code)
        stock, name, provider = fetch_us_security(code, start_text, end_text)
        try:
            benchmark = fetch_us_benchmark(start_text, end_text)
        except Exception as exc:
            benchmark = pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘", "成交量"])
            warnings.append(f"标普500代理基准SPY暂不可用：{exc}")
        bundle = PriceBundle(stock, benchmark, code, name, provider, "标普500代理ETF（SPY）", "美股个股", "美元", warnings)

    first_date = pd.Timestamp(bundle.stock["日期"].min())
    requested_start = pd.Timestamp(start_text)
    expected_span = max(1, (pd.Timestamp(end_text) - requested_start).days)
    actual_span = max(0, (pd.Timestamp(end_text) - first_date).days)
    bundle.requested_start = start_text
    bundle.requested_end = end_text
    bundle.coverage_ratio = float(np.clip(actual_span / expected_span, 0, 1))
    bundle.history_complete = bool(first_date <= requested_start + pd.Timedelta(days=45))
    if not bundle.history_complete:
        bundle.warnings.append("该股票可得历史不足五年。数据不足，无法准确判断，结果仅作低置信度参考。")
    return bundle


def calculate_quant_metrics(stock: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, Any]:
    close = stock.set_index("日期")["收盘"].sort_index()
    volume = stock.set_index("日期")["成交量"].sort_index()
    returns = close.pct_change().dropna()
    drawdown = close / close.cummax() - 1
    recent_returns = returns.tail(min(252, len(returns)))
    if benchmark is None or benchmark.empty:
        benchmark_close = pd.Series(dtype="float64", name="收盘")
        benchmark_returns = pd.Series(dtype="float64")
    else:
        benchmark_close = benchmark.set_index("日期")["收盘"].sort_index()
        benchmark_returns = benchmark_close.pct_change().dropna()
    aligned = pd.concat([returns.rename("stock"), benchmark_returns.rename("benchmark")], axis=1).dropna().tail(756)
    beta = np.nan
    correlation = np.nan
    if len(aligned) >= 60 and aligned["benchmark"].var() > 0:
        beta = aligned["stock"].cov(aligned["benchmark"]) / aligned["benchmark"].var()
        correlation = aligned["stock"].corr(aligned["benchmark"])
    volatility = recent_returns.std() * np.sqrt(252) if not recent_returns.empty else np.nan
    downside = recent_returns[recent_returns < 0].std() * np.sqrt(252) if (recent_returns < 0).any() else 0.0
    value_at_risk = recent_returns.quantile(0.05) if not recent_returns.empty else np.nan
    abnormal_days = int((returns.abs() > 0.70).sum())
    latest_lag = max(0, (pd.Timestamp(date.today()) - close.index.max()).days)
    volume20 = volume.tail(20).mean()
    volume60 = volume.tail(60).mean()
    volume_ratio = volume20 / volume60 if volume60 > 0 else np.nan
    return {
        "close": close,
        "volume": volume,
        "returns": returns,
        "drawdown": drawdown,
        "latest_price": float(close.iloc[-1]),
        "first_date": close.index.min(),
        "last_date": close.index.max(),
        "rows": len(close),
        "annual_volatility": float(volatility) if np.isfinite(volatility) else np.nan,
        "downside_volatility": float(downside) if np.isfinite(downside) else np.nan,
        "max_drawdown": float(drawdown.min()),
        "var95_daily": float(value_at_risk) if np.isfinite(value_at_risk) else np.nan,
        "beta": float(beta) if np.isfinite(beta) else np.nan,
        "correlation": float(correlation) if np.isfinite(correlation) else np.nan,
        "volume_ratio": float(volume_ratio) if np.isfinite(volume_ratio) else np.nan,
        "abnormal_days": abnormal_days,
        "latest_lag": latest_lag,
        "benchmark_close": benchmark_close,
    }


def score_stock_risk(metrics: dict[str, Any]) -> tuple[int, int, str, list[str]]:
    volatility = metrics["annual_volatility"]
    max_drawdown = abs(metrics["max_drawdown"])
    downside = metrics["downside_volatility"]
    beta = 1.0 if pd.isna(metrics["beta"]) else max(0.0, metrics["beta"])
    vol_part = np.clip((volatility - 0.12) / 0.58 * 35, 0, 35) if np.isfinite(volatility) else 20
    drawdown_part = np.clip((max_drawdown - 0.12) / 0.68 * 35, 0, 35)
    downside_part = np.clip((downside - 0.08) / 0.52 * 20, 0, 20) if np.isfinite(downside) else 10
    beta_part = np.clip((beta - 0.6) / 1.8 * 10, 0, 10)
    score = int(round(np.clip(vol_part + drawdown_part + downside_part + beta_part, 0, 100)))
    if score <= 18:
        level, level_number = "R2（中低风险）", 2
    elif score <= 38:
        level, level_number = "R3（中等风险）", 3
    elif score <= 63:
        level, level_number = "R4（中高风险）", 4
    else:
        level, level_number = "R5（高风险）", 5
    reasons = [f"近一年年化波动率约{volatility:.1%}" if np.isfinite(volatility) else "波动率数据不足", f"近五年或上市以来最大回撤约{metrics['max_drawdown']:.1%}"]
    if beta > 1.2:
        reasons.append("相对市场的Beta较高")
    if metrics["abnormal_days"]:
        reasons.append("历史数据中存在极端单日变动，需核对复权或公司事件")
    return score, level_number, level, reasons


def score_investor(profile: dict[str, Any]) -> tuple[int, int, str, str, list[str]]:
    source_points = {"闲置自有资金": 15, "未来有明确用途的资金": 8, "应急资金": 0, "借款／融资资金": 0}[profile["fund_source"]]
    reserve_points = {"不足3个月": 0, "3—6个月": 8, "6个月以上": 15}[profile["emergency_reserve"]]
    need_points = {"1周内": 0, "1个月内": 3, "3个月内": 7, "1年内": 12, "3年内": 15, "没有明确时间": 15}[profile["earliest_need"]]
    loss_points = {"立即全部卖出": 1, "大部分减仓": 4, "先复核原因再决定": 8, "继续按原计划持有": 11, "在条件允许时分批增加": 13}[profile["loss_response"]]
    max_loss_points = {"不超过5%": 0, "5%—10%": 4, "10%—20%": 8, "20%—30%": 12, "超过30%": 15}[profile.get("max_loss", "10%—20%")]
    goal_points = {"保值为主": 0, "股息／稳健收益": 3, "长期增值": 6, "波段操作": 8, "短线交易": 10}[profile["goal"]]
    stability_points = {"不稳定": 2, "较稳定": 7, "稳定": 10}[profile["income_stability"]]
    knowledge_points = {"没有经验": 0, "不足1年": 2, "1—3年": 4, "3年以上": 5}[profile["experience"]]
    score = int(np.clip(source_points + reserve_points + need_points + loss_points + max_loss_points + goal_points + stability_points + knowledge_points, 0, 100))
    if score <= 25:
        level_number, level = 1, "C1（保守型）"
    elif score <= 45:
        level_number, level = 2, "C2（谨慎型）"
    elif score <= 65:
        level_number, level = 3, "C3（平衡型）"
    elif score <= 89:
        level_number, level = 4, "C4（积极型）"
    else:
        level_number, level = 5, "C5（激进型）"

    if profile["experience"] in {"没有经验", "不足1年"}:
        style = "基础型"
    elif profile["goal"] == "短线交易" or profile["trade_frequency"] == "几乎每天":
        style = "活跃交易型"
    elif profile["goal"] in {"股息／稳健收益", "长期增值"}:
        style = "长期配置型"
    else:
        style = "进阶研究型"

    flags: list[str] = []
    if profile["fund_source"] in {"应急资金", "借款／融资资金"}:
        flags.append("资金来源不适合承担单只股票波动")
    if profile["emergency_reserve"] == "不足3个月":
        flags.append("应急储备不足3个月")
    if profile.get("existing_concentration") in {"30%—50%", "超过50%"}:
        flags.append("现有投资的单只股票集中度较高")
    if profile.get("leverage") == "是":
        flags.append("计划使用杠杆会放大损失")
    return score, level_number, level, style, flags


def _return_over(close: pd.Series, days: int) -> float:
    if len(close) <= days:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


def score_horizons(
    metrics: dict[str, Any],
    fundamental: EvidenceSnapshot,
    macro: EvidenceSnapshot,
) -> list[dict[str, Any]]:
    close: pd.Series = metrics["close"]
    benchmark_close: pd.Series = metrics["benchmark_close"]
    results: list[dict[str, Any]] = []
    for config in HORIZONS:
        if config["intraday_required"]:
            results.append({**config, "available": False, "score": None, "label": "需要分钟级／实时数据", "reasons": ["当前免费接口只提供日线，不能可靠判断下一交易日涨跌。"]})
            continue
        if len(close) < config["minimum_rows"]:
            results.append({**config, "available": False, "score": None, "label": "历史数据不足", "reasons": [f"需要至少{config['minimum_rows']}个交易日，当前只有{len(close)}个。"]})
            continue
        fast_ma = close.rolling(config["fast"]).mean().iloc[-1]
        slow_ma = close.rolling(config["slow"]).mean().iloc[-1]
        stock_return = _return_over(close, config["days"])
        benchmark_return = _return_over(benchmark_close, config["days"])
        score = 50.0
        reasons: list[str] = []
        if close.iloc[-1] >= fast_ma:
            score += 8
            reasons.append(f"现价位于{config['fast']}日均线之上")
        else:
            score -= 8
            reasons.append(f"现价位于{config['fast']}日均线之下")
        if fast_ma >= slow_ma:
            score += 8
            reasons.append("快慢均线结构偏强")
        else:
            score -= 8
            reasons.append("快慢均线结构偏弱")
        scale = 220 if config["days"] <= 5 else 100 if config["days"] <= 20 else 50 if config["days"] <= 60 else 25 if config["days"] <= 120 else 15
        if np.isfinite(stock_return):
            score += float(np.clip(stock_return * scale, -12, 12))
        if np.isfinite(stock_return) and np.isfinite(benchmark_return):
            excess = stock_return - benchmark_return
            score += float(np.clip(excess * 50, -10, 10))
            reasons.append("同期跑赢市场基准" if excess > 0 else "同期弱于市场基准")
        if np.isfinite(metrics["volume_ratio"]):
            if metrics["volume_ratio"] >= 1.2:
                score += 4
                reasons.append("近期成交量有所放大")
            elif metrics["volume_ratio"] <= 0.75:
                score -= 3
        macro_weight = 0.10 if config["days"] <= 20 else 0.16
        if macro.score is not None:
            score += (macro.score - 50) * macro_weight
        fundamental_weight = 0.04 if config["days"] <= 20 else 0.10 if config["days"] <= 60 else 0.20
        if fundamental.score is not None:
            score += (fundamental.score - 50) * fundamental_weight
            reasons.append("基本面评分提供支持" if fundamental.score >= 55 else "基本面评分未形成明显支持")
        score_int = int(round(np.clip(score, 0, 100)))
        label = "条件较积极" if score_int >= 68 else "中性偏积极" if score_int >= 56 else "中性观察" if score_int >= 42 else "偏弱／暂缓"
        stress_returns = close.pct_change(config["days"]).dropna()
        stress_loss = abs(float(stress_returns.quantile(0.05))) if not stress_returns.empty else np.nan
        historical_worst = float(stress_returns.min()) if not stress_returns.empty else np.nan
        results.append(
            {
                **config,
                "available": True,
                "score": score_int,
                "label": label,
                "stock_return": stock_return,
                "benchmark_return": benchmark_return,
                "stress_loss": stress_loss,
                "historical_worst": historical_worst,
                "reasons": reasons,
            }
        )
    return results


def _maximum_horizon_rank(earliest_need: str) -> int:
    return {"1周内": 0, "1个月内": 1, "3个月内": 2, "1年内": 3, "3年内": 4, "没有明确时间": 4}[earliest_need]


def choose_horizon(horizon_scores: list[dict[str, Any]], profile: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    available = [item for item in horizon_scores if item["available"]]
    if not available:
        return None, ["所有可用持有周期的数据都不足。"]
    max_rank = _maximum_horizon_rank(profile["earliest_need"])
    ranked_names = ["2—5个交易日", "2—4周", "1—3个月", "3—12个月", "1—3年"]
    allowed_names = set(ranked_names[: max_rank + 1])
    candidates = [item for item in available if item["name"] in allowed_names]
    notes: list[str] = []
    if not candidates:
        candidates = available[:1]
        notes.append("资金使用时间与可用行情周期存在冲突，只能提供最短可分析周期。")
    priors = GOAL_PRIORS[profile["goal"]]
    for item in candidates:
        operational = 0
        if item["name"] in {"2—5个交易日", "2—4周"}:
            if profile["monitor_time"] == "不足15分钟":
                operational -= 9
            if profile.get("stop_loss") in {"没有明确规则", "有规则但经常改变"}:
                operational -= 7
            if profile["experience"] in {"没有经验", "不足1年"}:
                operational -= 6
        item["selection_score"] = item["score"] + priors.get(item["name"], 0) + operational
    selected = max(candidates, key=lambda item: item["selection_score"])
    if profile["goal"] == "短线交易" and selected["name"] != "2—5个交易日":
        notes.append("短线意向与看盘条件、纪律或市场信号不匹配，因此Agent没有选择最短周期。")
    if profile["earliest_need"] in {"1周内", "1个月内"}:
        notes.append("资金近期可能使用，任何股票持有周期都存在被迫在不利价格退出的风险。")
    return selected, notes


def calculate_data_confidence(
    metrics: dict[str, Any],
    fundamental: EvidenceSnapshot,
    macro: EvidenceSnapshot,
    asset_type: str,
) -> tuple[int, list[str]]:
    score = 0
    notes: list[str] = []
    rows = metrics["rows"]
    score += 40 if rows >= 750 else 34 if rows >= 500 else 26 if rows >= 250 else 16 if rows >= 120 else 8
    if metrics["latest_lag"] <= 5:
        score += 20
    elif metrics["latest_lag"] <= 12:
        score += 12
        notes.append("最新行情可能存在延迟。")
    else:
        score += 3
        notes.append("最新行情日期较旧。")
    score += 15 if np.isfinite(metrics["beta"]) else 6
    if asset_type == "场内基金":
        score += 12
        notes.append("基金未使用单一公司财务评分，需另行核对跟踪标的和费率。")
    elif fundamental.available:
        score += 18
    else:
        score += 5
        notes.append("财务或估值数据未取得，结论主要基于行情。")
    score += 7 if macro.available else 2
    if metrics["abnormal_days"]:
        score -= min(10, metrics["abnormal_days"] * 2)
        notes.append("检测到极端日收益，可能包含公司事件或复权差异。")
    if not metrics.get("history_complete", True):
        rows = int(metrics.get("rows", 0))
        cap = 30 if rows < 60 else 45 if rows < 250 else 65 if rows < 750 else 78
        score = min(score, cap)
        notes.append("该股票可得历史不足五年。数据不足，无法准确判断。")
    if not metrics.get("benchmark_available", True):
        score = min(score, 72)
        notes.append("市场基准暂不可用，Beta和相对强弱未参与判断。")
    return int(np.clip(score, 0, 100)), notes


def assess_suitability(
    profile: dict[str, Any],
    investor_level_number: int,
    stock_risk_number: int,
    selected_horizon: dict[str, Any] | None,
    data_confidence: int,
) -> dict[str, Any]:
    hard_reasons: list[str] = []
    if profile["fund_source"] == "借款／融资资金":
        hard_reasons.append("不使用借款或融资资金承担单只股票风险")
    if profile["fund_source"] == "应急资金":
        hard_reasons.append("应急资金应优先保持安全性和流动性")
    if profile["emergency_reserve"] == "不足3个月" and profile["planned_amount"] / profile["investable_assets"] > 0.10:
        hard_reasons.append("应急储备不足且计划投入比例较高")
    if data_confidence < 35 or selected_horizon is None:
        return {"fit": "证据不足", "fit_reason": "公开数据不足以形成可靠判断", "hard_reasons": hard_reasons}
    if hard_reasons:
        return {"fit": "不适配", "fit_reason": hard_reasons[0], "hard_reasons": hard_reasons}
    gap = investor_level_number - stock_risk_number
    if gap <= -2:
        fit, reason = "不适配", "用户风险承受能力与该股票风险等级差距较大"
    elif gap == -1:
        fit, reason = "有限适配", "股票风险比用户风险等级高一级，只适合严格限制风险预算"
    else:
        fit, reason = "适配", "用户风险承受能力覆盖该股票的模型风险等级"
    return {"fit": fit, "fit_reason": reason, "hard_reasons": hard_reasons}


def position_budget(
    profile: dict[str, Any],
    investor_level_number: int,
    stock_risk_number: int,
    suitability_result: dict[str, Any],
    selected_horizon: dict[str, Any] | None,
) -> dict[str, Any]:
    if selected_horizon is None or suitability_result["fit"] in {"不适配", "证据不足"}:
        return {"lower_pct": 0.0, "upper_pct": 0.0, "lower_amount": 0.0, "upper_amount": 0.0, "stress_loss": None}
    risk_budget = {1: 0.0025, 2: 0.005, 3: 0.010, 4: 0.020, 5: 0.030}[investor_level_number]
    cap = {1: 0.03, 2: 0.05, 3: 0.10, 4: 0.15, 5: 0.20}[investor_level_number]
    stress_loss = max(float(selected_horizon.get("stress_loss") or 0.0), 0.08)
    upper = min(risk_budget / stress_loss, cap)
    if stock_risk_number == 5:
        upper *= 0.75
    if suitability_result["fit"] == "有限适配":
        upper *= 0.50
    concentration = profile["existing_concentration"]
    if concentration in {"30%—50%", "超过50%"}:
        upper *= 0.60
    timing_score = selected_horizon["score"]
    if timing_score < 42:
        upper = 0.0
    elif timing_score < 56:
        upper *= 0.50
    lower = upper * 0.40 if upper > 0 else 0.0
    assets = float(profile["investable_assets"])
    return {
        "lower_pct": lower,
        "upper_pct": upper,
        "lower_amount": lower * assets,
        "upper_amount": upper * assets,
        "stress_loss": stress_loss,
    }


def build_final_conclusion(
    suitability_result: dict[str, Any],
    selected_horizon: dict[str, Any] | None,
    position: dict[str, Any],
) -> tuple[str, str]:
    fit = suitability_result["fit"]
    if fit == "证据不足" or selected_horizon is None:
        return "证据不足，暂不形成判断", "公开数据完整度不足，需要更可靠的数据源。"
    if fit == "不适配":
        return "不适合该用户", suitability_result["fit_reason"]
    if selected_horizon["score"] < 42:
        return "个人条件可讨论，但当前时点暂缓", "股票与用户并非绝对不匹配，但当前多周期信号偏弱。"
    if selected_horizon["score"] < 56 or position["upper_pct"] <= 0.02:
        return "可以小仓观察", "当前证据尚不支持较高风险预算，重点观察后续信号。"
    if fit == "有限适配":
        return "条件适配，可进一步研究", "风险等级略高于用户等级，需要严格限制仓位并按期复核。"
    return "在风险预算内相对适配", "个人条件与股票风险基本匹配，但仍需满足仓位和复核条件。"


def analyze_all(
    bundle: PriceBundle,
    profile: dict[str, Any],
    fundamental: EvidenceSnapshot,
    macro: EvidenceSnapshot,
) -> dict[str, Any]:
    metrics = calculate_quant_metrics(bundle.stock, bundle.benchmark)
    metrics["history_complete"] = bundle.history_complete
    metrics["coverage_ratio"] = bundle.coverage_ratio
    metrics["benchmark_available"] = bundle.benchmark is not None and not bundle.benchmark.empty
    investor_score, investor_number, investor_level, style, profile_flags = score_investor(profile)
    stock_risk_score, stock_risk_number, stock_risk_level, risk_reasons = score_stock_risk(metrics)
    horizon_scores = score_horizons(metrics, fundamental, macro)
    selected_horizon, horizon_notes = choose_horizon(horizon_scores, profile)
    confidence, confidence_notes = calculate_data_confidence(metrics, fundamental, macro, bundle.asset_type)
    suitability_result = assess_suitability(profile, investor_number, stock_risk_number, selected_horizon, confidence)
    position = position_budget(profile, investor_number, stock_risk_number, suitability_result, selected_horizon)
    conclusion, conclusion_reason = build_final_conclusion(suitability_result, selected_horizon, position)
    return {
        "metrics": metrics,
        "investor_score": investor_score,
        "investor_number": investor_number,
        "investor_level": investor_level,
        "style": style,
        "profile_flags": profile_flags,
        "stock_risk_score": stock_risk_score,
        "stock_risk_number": stock_risk_number,
        "stock_risk_level": stock_risk_level,
        "risk_reasons": risk_reasons,
        "horizon_scores": horizon_scores,
        "selected_horizon": selected_horizon,
        "horizon_notes": horizon_notes,
        "data_confidence": confidence,
        "confidence_notes": confidence_notes,
        "suitability": suitability_result,
        "position": position,
        "conclusion": conclusion,
        "conclusion_reason": conclusion_reason,
        "fundamental": fundamental,
        "macro": macro,
    }
