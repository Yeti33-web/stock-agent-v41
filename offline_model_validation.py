"""Offline, read-only factor research for Agent A.

This module never changes production weights and is not imported by the
Streamlit app.  Candidate rules are selected on a development universe ending
in 2022, then reported once on different stocks from 2023 onward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import agent_core


DEVELOPMENT_SYMBOLS = (
    "600519",
    "600036",
    "000858",
    "601318",
    "600276",
    "002594",
    "601899",
    "000333",
)
HOLDOUT_SYMBOLS = (
    "600900",
    "601088",
    "600309",
    "000651",
)
AUDIT_HORIZONS = (5, 20, 60, 120)
VARIANTS = (
    "signal_separation_base",
    "fused_20d_reversal",
    "plus_short_reversal",
    "plus_short_reversal_penalty",
    "plus_52_week_position",
    "plus_price_volume_confirmation",
    "plus_volatility_regime",
    "self_calibration_all_four",
)


@dataclass(frozen=True)
class AuditMetric:
    split: str
    variant: str
    horizon: int
    pairs: int
    observations: int
    median_rank_ic: float | None
    positive_ic_share: float | None
    median_high_low_spread: float | None
    directional_hit_rate: float | None


def _rank_ic(signal: pd.Series, outcome: pd.Series) -> float | None:
    pair = pd.concat([signal.rename("signal"), outcome.rename("outcome")], axis=1).dropna()
    if len(pair) < 12 or pair["signal"].nunique() < 3 or pair["outcome"].nunique() < 3:
        return None
    value = pair["signal"].rank().corr(pair["outcome"].rank())
    return float(value) if value is not None and np.isfinite(value) else None


def _candidate_factors(close: pd.Series, volume: pd.Series, days: int) -> pd.DataFrame:
    factors = pd.DataFrame(index=close.index)

    if days <= 60:
        ret_short = close.pct_change(10, fill_method=None)
        mean_short = ret_short.rolling(250, min_periods=60).mean()
        std_short = ret_short.rolling(250, min_periods=60).std(ddof=1)
        z_short = (ret_short - mean_short) / std_short.replace(0, np.nan)
        factors["short_reversal"] = np.where(
            z_short >= 1.0,
            -np.clip((z_short - 0.8) * 3.0, 0, 5),
            np.where(z_short <= -1.0, np.clip((0.8 - z_short) * 1.8, 0, 3), 0.0),
        )
    else:
        factors["short_reversal"] = 0.0
    factors["production_short_reversal"] = (
        factors["short_reversal"] if days == 20 else 0.0
    )
    if days <= 20:
        factors["short_reversal_penalty"] = -np.clip(
            (z_short - 0.8) * 1.5,
            0.0,
            3.0,
        )
    else:
        factors["short_reversal_penalty"] = 0.0

    high_252 = close.rolling(252, min_periods=252).max()
    distance = close / high_252.replace(0, np.nan) - 1
    factors["position_52_week"] = np.clip((distance + 0.30) / 0.30 * 6, -6, 6)

    ret10 = close.pct_change(10, fill_method=None)
    avg10 = volume.rolling(10, min_periods=10).mean()
    avg30 = volume.rolling(30, min_periods=30).mean()
    vol_short = avg10 / avg30.replace(0, np.nan)
    factors["price_volume_confirmation"] = np.where(
        (ret10 > 0.02) & (vol_short > 1.15),
        2.0,
        np.where(
            (ret10 > 0.02) & (vol_short < 0.95),
            -2.0,
            np.where((ret10 < -0.02) & (vol_short > 1.15), -2.0, 0.0),
        ),
    )

    returns = close.pct_change(fill_method=None)
    vol20 = returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    vol_rank = vol20.rolling(250, min_periods=60).rank(pct=True)
    factors["volatility_regime"] = np.where(
        vol_rank >= 0.85,
        -3.0,
        np.where(vol_rank <= 0.15, 1.0, 0.0),
    )
    return factors.replace([np.inf, -np.inf], np.nan)


def _variant_scores(base_score: pd.Series, candidate: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "signal_separation_base": base_score,
        "fused_20d_reversal": base_score + candidate["production_short_reversal"].fillna(0.0),
        "plus_short_reversal": base_score + candidate["short_reversal"].fillna(0.0),
        "plus_short_reversal_penalty": base_score + candidate["short_reversal_penalty"].fillna(0.0),
        "plus_52_week_position": base_score + candidate["position_52_week"].fillna(0.0),
        "plus_price_volume_confirmation": base_score + candidate["price_volume_confirmation"].fillna(0.0),
        "plus_volatility_regime": base_score + candidate["volatility_regime"].fillna(0.0),
        "self_calibration_all_four": base_score
        + candidate[
            [
                "short_reversal",
                "position_52_week",
                "price_volume_confirmation",
                "volatility_regime",
            ]
        ].fillna(0.0).sum(axis=1),
    }


def _pair_metrics(frame: pd.DataFrame, score_name: str) -> dict[str, Any] | None:
    clean = frame[[score_name, "future_excess"]].dropna()
    if len(clean) < 12:
        return None
    signal = clean[score_name] - 50.0
    outcome = clean["future_excess"]
    ic = _rank_ic(signal, outcome)
    if ic is None:
        return None
    low = signal.quantile(0.25)
    high = signal.quantile(0.75)
    low_outcome = outcome[signal <= low]
    high_outcome = outcome[signal >= high]
    spread = (
        float(high_outcome.median() - low_outcome.median())
        if not low_outcome.empty and not high_outcome.empty
        else None
    )
    active = clean.loc[signal.abs() >= 4.0]
    hit_rate = (
        float(((active[score_name] > 50.0) == (active["future_excess"] > 0.0)).mean())
        if len(active) >= 8
        else None
    )
    return {
        "ic": ic,
        "spread": spread,
        "hit_rate": hit_rate,
        "observations": int(len(clean)),
    }


def _build_symbol_frames(
    code: str,
    stock: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    stock_frame = stock.copy()
    benchmark_frame = benchmark.copy()
    stock_frame["日期"] = pd.to_datetime(stock_frame["日期"]).dt.normalize()
    benchmark_frame["日期"] = pd.to_datetime(benchmark_frame["日期"]).dt.normalize()
    stock_frame = stock_frame.drop_duplicates("日期", keep="last").set_index("日期").sort_index()
    benchmark_close = (
        benchmark_frame.drop_duplicates("日期", keep="last")
        .set_index("日期")["收盘"]
        .sort_index()
        .reindex(stock_frame.index)
        .ffill()
    )
    close = pd.to_numeric(stock_frame["收盘"], errors="coerce")
    volume = pd.to_numeric(stock_frame["成交量"], errors="coerce")
    output: dict[int, pd.DataFrame] = {}
    for days in AUDIT_HORIZONS:
        config = next(item for item in agent_core.HORIZONS if int(item["days"]) == days)
        base = agent_core._technical_timing_frame(close, volume, benchmark_close, config)
        candidate = _candidate_factors(close, volume, days)
        base_without_fusion = base["raw_score"] - base["short_reversal_adjustment"].fillna(0.0)
        variants = _variant_scores(base_without_fusion, candidate)
        future_stock = close.shift(-days) / close - 1
        future_benchmark = benchmark_close.shift(-days) / benchmark_close - 1
        frame = pd.DataFrame({**variants, "future_excess": future_stock - future_benchmark})
        frame["code"] = code
        stride = max(5, int(round(days / 4)))
        output[days] = frame.iloc[::stride].copy()
    return output


def _aggregate(
    frames: dict[str, dict[int, pd.DataFrame]],
    symbols: tuple[str, ...],
    split: str,
) -> list[AuditMetric]:
    rows: list[AuditMetric] = []
    for horizon in AUDIT_HORIZONS:
        for variant in VARIANTS:
            pair_rows: list[dict[str, Any]] = []
            for code in symbols:
                frame = frames[code][horizon]
                if split == "development":
                    frame = frame.loc[frame.index <= pd.Timestamp("2022-12-31")]
                else:
                    frame = frame.loc[frame.index >= pd.Timestamp("2023-01-01")]
                metrics = _pair_metrics(frame, variant)
                if metrics:
                    pair_rows.append(metrics)
            ics = [item["ic"] for item in pair_rows]
            spreads = [item["spread"] for item in pair_rows if item["spread"] is not None]
            hits = [item["hit_rate"] for item in pair_rows if item["hit_rate"] is not None]
            rows.append(
                AuditMetric(
                    split=split,
                    variant=variant,
                    horizon=horizon,
                    pairs=len(pair_rows),
                    observations=sum(int(item["observations"]) for item in pair_rows),
                    median_rank_ic=float(np.median(ics)) if ics else None,
                    positive_ic_share=float(np.mean(np.asarray(ics) > 0)) if ics else None,
                    median_high_low_spread=float(np.median(spreads)) if spreads else None,
                    directional_hit_rate=float(np.mean(hits)) if hits else None,
                )
            )
    return rows


def run_candidate_audit() -> pd.DataFrame:
    benchmark, _ = agent_core.fetch_yahoo_chart_history(
        "510300.SS",
        "2015-01-01",
        pd.Timestamp.today().date().isoformat(),
    )
    frames: dict[str, dict[int, pd.DataFrame]] = {}
    for code in DEVELOPMENT_SYMBOLS + HOLDOUT_SYMBOLS:
        symbol = agent_core.a_share_yahoo_ticker(code)
        stock, _ = agent_core.fetch_yahoo_chart_history(
            symbol,
            "2015-01-01",
            pd.Timestamp.today().date().isoformat(),
        )
        frames[code] = _build_symbol_frames(code, stock, benchmark)
    metrics = _aggregate(frames, DEVELOPMENT_SYMBOLS, "development")
    metrics.extend(_aggregate(frames, HOLDOUT_SYMBOLS, "holdout"))
    return pd.DataFrame([item.__dict__ for item in metrics])


def _display(frame: pd.DataFrame) -> None:
    columns = [
        "split",
        "horizon",
        "variant",
        "pairs",
        "observations",
        "median_rank_ic",
        "positive_ic_share",
        "median_high_low_spread",
        "directional_hit_rate",
    ]
    printable = frame[columns].copy()
    for column in (
        "median_rank_ic",
        "positive_ic_share",
        "median_high_low_spread",
        "directional_hit_rate",
    ):
        printable[column] = printable[column].map(
            lambda value: "—" if pd.isna(value) else f"{float(value):+.4f}"
        )
    print(printable.to_string(index=False))


if __name__ == "__main__":
    _display(run_candidate_audit())
