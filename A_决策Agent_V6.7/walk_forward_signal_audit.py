"""Sealed, read-only walk-forward audit for Agent A V6.6.

This file is never imported by either Streamlit app.  At every test date T it
rebuilds the score from the preceding five years only.  The later outcome is
read only after the T-date score has been stored in the in-memory result row.
It compares the former signal-separation gate with the fused V6.6 gate; it
does not optimise or write production weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import agent_core


SEALED_SYMBOLS = ("AAPL", "MSFT", "JPM", "XOM")
BENCHMARK = "SPY"
HORIZONS = (5, 20, 60, 120)
EVALUATION_START = pd.Timestamp("2023-01-01")


@dataclass(frozen=True)
class AuditRow:
    symbol: str
    date_t: pd.Timestamp
    horizon: int
    variant: str
    score: float
    validation_status: str
    direction: int
    stock_return: float
    benchmark_return: float
    excess_return: float
    forward_max_drawdown: float


def _rank_ic(signal: pd.Series, outcome: pd.Series) -> float | None:
    pair = pd.concat([signal.rename("signal"), outcome.rename("outcome")], axis=1).dropna()
    if len(pair) < 8 or pair["signal"].nunique() < 2 or pair["outcome"].nunique() < 2:
        return None
    value = pair["signal"].rank(method="average").corr(pair["outcome"].rank(method="average"))
    return float(value) if value is not None and np.isfinite(value) else None


def _legacy_global_gate(frame: pd.DataFrame, days: int) -> dict[str, Any]:
    """The fixed V6.5 signal-separation gate, without V6.6 local calibration."""

    usable = frame[["raw_score", "future_target"]].replace([np.inf, -np.inf], np.nan).dropna()
    usable = usable.iloc[:: max(5, int(round(days / 4)))].copy()
    samples = int(len(usable))
    minimum = 40 if days <= 20 else 30 if days <= 60 else 24 if days <= 120 else 16
    signal = usable["raw_score"] - 50.0
    outcome = usable["future_target"]
    ic = _rank_ic(signal, outcome)
    midpoint = samples // 2
    first_ic = _rank_ic(signal.iloc[:midpoint], outcome.iloc[:midpoint]) if midpoint >= 8 else None
    second_ic = _rank_ic(signal.iloc[midpoint:], outcome.iloc[midpoint:]) if samples - midpoint >= 8 else None
    active = usable.loc[signal.abs() >= 4.0]
    hit_rate = None
    if len(active) >= 12:
        hit_rate = float(
            (((active["raw_score"] - 50.0) > 0) == (active["future_target"] > 0)).mean()
        )
    spread = None
    if samples >= 16 and signal.nunique() >= 5:
        lower, upper = signal.quantile(0.25), signal.quantile(0.75)
        low = outcome[signal <= lower]
        high = outcome[signal >= upper]
        if not low.empty and not high.empty:
            spread = float(high.median() - low.median())
    passed = bool(
        samples >= minimum
        and ic is not None
        and ic >= 0.03
        and first_ic is not None
        and first_ic > 0
        and second_ic is not None
        and second_ic > 0
        and spread is not None
        and spread > 0
        and hit_rate is not None
        and hit_rate >= 0.52
    )
    limited = bool(
        not passed
        and samples >= minimum
        and ic is not None
        and ic > 0
        and second_ic is not None
        and second_ic >= 0
        and spread is not None
        and spread > 0
        and hit_rate is not None
        and hit_rate >= 0.50
    )
    return {
        "status": "通过" if passed else "有限通过" if limited else "未通过",
        "reliability_multiplier": 1.0 if passed else 0.5 if limited else 0.0,
    }


def _candidate_recent_gate(frame: pd.DataFrame, days: int) -> dict[str, Any]:
    """Research-only stricter gate that demands recent and sign-specific support."""

    base = agent_core._validate_timing_signal(frame, days)
    if base["status"] == "未通过":
        return {"status": "未通过", "reliability_multiplier": 0.0}
    usable = frame[["raw_score", "future_target"]].replace([np.inf, -np.inf], np.nan).dropna()
    usable = usable.iloc[:: max(5, int(round(days / 4)))].copy()
    minimum = 40 if days <= 20 else 30 if days <= 60 else 24 if days <= 120 else 16
    recent_count = min(len(usable), max(minimum, int(np.ceil(len(usable) * 0.35))))
    recent = usable.tail(recent_count)
    signal = recent["raw_score"] - 50.0
    outcome = recent["future_target"]
    recent_ic = _rank_ic(signal, outcome)
    lower, upper = signal.quantile(0.25), signal.quantile(0.75)
    low, high = outcome[signal <= lower], outcome[signal >= upper]
    recent_spread = float(high.median() - low.median()) if not low.empty and not high.empty else None
    current_score = float(frame["raw_score"].dropna().iloc[-1])
    current_direction = int(np.sign(current_score - 50.0))
    same_sign = recent[(signal * current_direction >= 4.0)] if current_direction else recent.iloc[0:0]
    signed_outcomes = same_sign["future_target"] * current_direction
    sign_hit = float((signed_outcomes > 0).mean()) if len(same_sign) >= 12 else None
    sign_median = float(signed_outcomes.median()) if len(same_sign) >= 12 else None
    supported = bool(
        recent_ic is not None
        and recent_ic > 0
        and recent_spread is not None
        and recent_spread > 0
        and sign_hit is not None
        and sign_hit >= 0.52
        and sign_median is not None
        and sign_median > 0
    )
    if not supported:
        return {"status": "未通过", "reliability_multiplier": 0.0}
    return {
        "status": str(base["status"]),
        "reliability_multiplier": float(base["reliability_multiplier"]),
    }


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    data = frame.copy()
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce").dt.normalize()
    data[column] = pd.to_numeric(data[column], errors="coerce")
    return (
        data.dropna(subset=["日期", column])
        .drop_duplicates("日期", keep="last")
        .set_index("日期")[column]
        .sort_index()
    )


def _macro_points(benchmark_close: pd.Series, days: int) -> float:
    benchmark = pd.DataFrame({"日期": benchmark_close.index, "收盘": benchmark_close.values})
    score = float(agent_core.derive_market_regime(benchmark)["市场分"])
    weight = 0.08 if days <= 20 else 0.10
    return float(np.clip((score - 50.0) * weight, -2.0, 2.0))


def _direction(status: str, score: float, variant: str) -> int:
    allowed = {"通过", "有限通过"} if variant == "V6.5信号分离" else {"通过"}
    if status not in allowed:
        return 0
    if score >= 60.0:
        return 1
    if score < 45.0:
        return -1
    return 0


def _evaluate_symbol(
    symbol: str,
    stock_frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
) -> list[AuditRow]:
    close = _series(stock_frame, "收盘")
    volume = _series(stock_frame, "成交量").reindex(close.index)
    benchmark_raw = _series(benchmark_frame, "收盘")
    benchmark = benchmark_raw.reindex(close.index).ffill()
    rows: list[AuditRow] = []
    for days in HORIZONS:
        config = next(item for item in agent_core.HORIZONS if int(item["days"]) == days)
        stride = max(10, int(round(days / 2)))
        candidates = close.index[(close.index >= EVALUATION_START)][::stride]
        for date_t in candidates:
            location = int(close.index.get_loc(date_t))
            if location + days >= len(close):
                continue
            start_t = date_t - pd.DateOffset(years=5)
            close_t = close.loc[(close.index >= start_t) & (close.index <= date_t)]
            if len(close_t) < int(config["minimum_rows"]):
                continue
            volume_t = volume.reindex(close_t.index)
            benchmark_t = benchmark.reindex(close_t.index).ffill()
            if close_t.index.max() > date_t or benchmark_t.index.max() > date_t:
                raise RuntimeError("未来数据防护失败：T日评分阶段出现T日之后的行。")

            current_frame = agent_core._technical_timing_frame(close_t, volume_t, benchmark_t, config)
            current_validation = agent_core._validate_timing_signal(current_frame, days)
            current_raw = float(current_frame["raw_score"].dropna().iloc[-1])

            old_frame = current_frame.copy()
            old_frame["raw_score"] = (
                old_frame["raw_score"] - old_frame["short_reversal_adjustment"].fillna(0.0)
            )
            old_validation = _legacy_global_gate(old_frame, days)
            old_raw = float(old_frame["raw_score"].dropna().iloc[-1])
            macro = _macro_points(benchmark_t, days)

            # Store the T-date scores before the later path is accessed.
            stored_scores = (
                (
                    "V6.5信号分离",
                    old_validation,
                    50.0 + (old_raw - 50.0) * float(old_validation["reliability_multiplier"]) + macro,
                ),
                (
                    "V6.6融合校准",
                    current_validation,
                    50.0
                    + (current_raw - 50.0) * float(current_validation["reliability_multiplier"])
                    + macro,
                ),
                (
                    "候选_近期与同向闸门",
                    _candidate_recent_gate(current_frame, days),
                    0.0,
                ),
            )

            candidate_validation = stored_scores[-1][1]
            stored_scores = stored_scores[:-1] + (
                (
                    "候选_近期与同向闸门",
                    candidate_validation,
                    50.0
                    + (current_raw - 50.0)
                    * float(candidate_validation["reliability_multiplier"])
                    + macro,
                ),
            )

            end_t = close.index[location + days]
            stock_return = float(close.iloc[location + days] / close.iloc[location] - 1.0)
            benchmark_start = float(benchmark.iloc[location])
            benchmark_end = float(benchmark.iloc[location + days])
            benchmark_return = benchmark_end / benchmark_start - 1.0
            excess = stock_return - benchmark_return
            path = close.iloc[location : location + days + 1]
            forward_drawdown = float((path / path.cummax() - 1.0).min())
            for variant, validation, score in stored_scores:
                status = str(validation["status"])
                rows.append(
                    AuditRow(
                        symbol=symbol,
                        date_t=date_t,
                        horizon=days,
                        variant=variant,
                        score=float(np.clip(score, 0.0, 100.0)),
                        validation_status=status,
                        direction=_direction(status, score, variant),
                        stock_return=stock_return,
                        benchmark_return=benchmark_return,
                        excess_return=excess,
                        forward_max_drawdown=forward_drawdown,
                    )
                )
    return rows


def run_sealed_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    end = pd.Timestamp.today().date().isoformat()
    benchmark, _ = agent_core.fetch_yahoo_chart_history(BENCHMARK, "2017-01-01", end)
    rows: list[AuditRow] = []
    for symbol in SEALED_SYMBOLS:
        stock, _ = agent_core.fetch_yahoo_chart_history(symbol, "2017-01-01", end)
        rows.extend(_evaluate_symbol(symbol, stock, benchmark))
    detail = pd.DataFrame([item.__dict__ for item in rows])
    summaries: list[dict[str, Any]] = []
    for (variant, horizon), group in detail.groupby(["variant", "horizon"], sort=True):
        active = group[group["direction"] != 0]
        buys = active[active["direction"] > 0]
        signed = active["direction"] * active["excess_return"]
        summaries.append(
            {
                "variant": variant,
                "horizon": int(horizon),
                "test_dates": int(len(group)),
                "validated_dates": int((group["validation_status"] != "未通过").sum()),
                "active_signals": int(len(active)),
                "active_coverage": float(len(active) / len(group)) if len(group) else None,
                "directional_hit_rate": float((signed > 0).mean()) if len(active) else None,
                "mean_signed_excess": float(signed.mean()) if len(active) else None,
                "median_signed_excess": float(signed.median()) if len(active) else None,
                "buy_signals": int(len(buys)),
                "buy_mean_return": float(buys["stock_return"].mean()) if len(buys) else None,
                "buy_worst_forward_drawdown": float(buys["forward_max_drawdown"].min()) if len(buys) else None,
            }
        )
    return detail, pd.DataFrame(summaries)


def _display(summary: pd.DataFrame) -> None:
    printable = summary.copy()
    for column in (
        "active_coverage",
        "directional_hit_rate",
        "mean_signed_excess",
        "median_signed_excess",
        "buy_mean_return",
        "buy_worst_forward_drawdown",
    ):
        printable[column] = printable[column].map(
            lambda value: "—" if pd.isna(value) else f"{float(value):+.3%}"
        )
    print(printable.to_string(index=False))


if __name__ == "__main__":
    _, summary_frame = run_sealed_audit()
    _display(summary_frame)
