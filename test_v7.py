"""V7.0.0 决策引擎测试（需求第十六节全部 12 类）。

全部离线、确定性（固定随机种子），不访问网络：

1.  历史相似度计算测试
2.  历史案例筛选测试
3.  历史案例可信度测试
4.  高质量案例权重 ≥50% 测试
5.  低质量案例不得强行 ≥50% 测试
6.  动态权重总和 = 100% 测试
7.  新闻市场反应测试
8.  红三兵识别测试
9.  缺失数据测试
10. 异常数据测试
11. 未来数据泄露测试
12. 原有 V6.5.2/V7.0 功能回归测试
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import agent_core
from dynamic_weight import determine_weights
from evidence_quality import assess_evidence
from event_reaction import assess_event_reaction
from historical_decision import build_v7_decision
from historical_outcome import evaluate_outcomes
from historical_similarity import SimilarityConfig, search_historical_cases
from lookahead_guard import (
    LookaheadViolation,
    build_boundary,
    ensure_selection_dates_within_boundary,
)
from technical_patterns import assess_technical_patterns, detect_three_white_soldiers

ROOT = Path(__file__).resolve().parent


def synth_stock(rows: int = 900, seed: int = 7, start: str = "2021-06-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=rows)
    market = rng.normal(0.0003, 0.009, rows)
    stock_returns = 0.0004 + 0.8 * market + rng.normal(0, 0.011, rows)
    stock_close = 30 * np.exp(np.cumsum(stock_returns))
    bench_close = 100 * np.exp(np.cumsum(market))
    volume = rng.lognormal(14, 0.3, rows)
    stock = pd.DataFrame(
        {
            "日期": dates,
            "开盘": stock_close * 0.998,
            "最高": stock_close * 1.012,
            "最低": stock_close * 0.990,
            "收盘": stock_close,
            "成交量": volume,
        }
    )
    benchmark = pd.DataFrame(
        {
            "日期": dates,
            "开盘": bench_close * 0.999,
            "最高": bench_close * 1.006,
            "最低": bench_close * 0.995,
            "收盘": bench_close,
            "成交量": volume * 2,
        }
    )
    return stock, benchmark


def default_profile() -> dict:
    return {
        "asset_band": "20万—50万元",
        "investable_assets": 350_000.0,
        "fund_source": "闲置自有资金",
        "emergency_reserve": "6个月以上",
        "earliest_need": "3年内",
        "income_stability": "稳定",
        "max_loss": "10%—20%",
        "loss_response": "先复核原因再决定",
        "goal": "长期增值",
        "experience": "1—3年",
        "trade_frequency": "每月1—3次",
        "monitor_time": "15—30分钟",
        "existing_concentration": "10%—30%",
        "stop_loss": "有明确规则并能执行",
        "fx_acceptance": "只能接受较小波动",
        "fundamental_action": "会重新评估",
        "planned_amount": 50_000.0,
        "leverage": "否",
    }


# ---------------------------------------------------------------- 1. 相似度计算


def test_v7_01_similarity_repeats_identical_regime_high_score() -> None:
    """把同一段行情复制两次：当前状态应能找到高相似度的历史案例。"""
    rng = np.random.default_rng(42)
    base_returns = rng.normal(0.001, 0.012, 160)
    warm = rng.normal(0.0005, 0.010, 300)
    filler = rng.normal(0.0005, 0.010, 120)
    returns = np.concatenate([warm, base_returns, filler, base_returns])
    close = 40 * np.exp(np.cumsum(returns))
    dates = pd.bdate_range("2021-01-01", periods=len(close))
    volume = rng.lognormal(14, 0.25, len(close))
    stock = pd.DataFrame(
        {
            "日期": dates,
            "开盘": close * 0.998,
            "最高": close * 1.012,
            "最低": close * 0.990,
            "收盘": close,
            "成交量": volume,
        }
    )
    result = search_historical_cases(stock, None)
    assert result.available, result.reason
    assert result.selected, "应当找到至少一个相似案例"
    best = max(item["similarity"] for item in result.selected)
    assert best >= 75.0, f"完全复制的历史片段相似度应足够高，实际{best:.1f}"
    # 最相似案例应落在第一段复制区（位置300—460）附近。
    best_position = max(result.selected, key=lambda item: item["similarity"])["position"]
    assert 280 <= best_position <= 480
    assert result.features_used.get("stock_state", 0) >= 12


def test_v7_02_similarity_distant_state_scores_low() -> None:
    """构造一个与当前状态差异巨大的历史区间：其相似度应显著更低。"""
    rng = np.random.default_rng(5)
    calm = rng.normal(0.0002, 0.004, 400)  # 历史低波动横盘
    crash = np.full(60, -0.025)  # 当前处于暴跌
    rebound = np.full(20, 0.02)
    returns = np.concatenate([calm, crash, rebound])
    close = 100 * np.exp(np.cumsum(returns))
    dates = pd.bdate_range("2022-01-03", periods=len(close))
    stock = pd.DataFrame(
        {
            "日期": dates,
            "开盘": close * 0.995,
            "最高": close * 1.02,
            "最低": close * 0.98,
            "收盘": close,
            "成交量": np.full(len(close), 1_000_000.0),
        }
    )
    result = search_historical_cases(stock, None)
    assert result.similarities is not None and len(result.similarities) > 0
    calm_region = result.similarities.iloc[:300]
    current_best = float(result.similarities.max())
    assert float(calm_region.max()) <= current_best + 1e-9
    assert float(calm_region.mean()) < 70.0


# ---------------------------------------------------------------- 2. 案例筛选


def test_v7_03_case_selection_spacing_threshold_and_order() -> None:
    stock, benchmark = synth_stock(900, seed=7)
    result = search_historical_cases(stock, benchmark)
    assert result.available
    positions = sorted(item["position"] for item in result.selected)
    for left, right in zip(positions, positions[1:]):
        assert right - left >= 20, "被选案例之间必须保持至少20个交易日的间隔"
    similarities = [item["similarity"] for item in result.selected]
    assert similarities == sorted(similarities, reverse=True)
    assert all(item["similarity"] >= result.selection_threshold for item in result.selected)
    max_forward = 120
    assert all(item["position"] + max_forward < result.boundary.rows for item in result.selected)
    assert all(item["position"] <= result.boundary.rows - 1 - 20 for item in result.selected)
    ensure_selection_dates_within_boundary([item["anchor_date"] for item in result.selected], result.boundary)


# ---------------------------------------------------------------- 3. 可信度


def _fake_similarity(sample_count: int, similarity_mean: float, similarity_std: float) -> SimpleNamespace:
    rng = np.random.default_rng(3)
    selected = [
        {"anchor_date": pd.Timestamp("2022-01-01"), "position": i * 25, "similarity": float(rng.normal(similarity_mean, similarity_std))}
        for i in range(sample_count)
    ]
    context = SimpleNamespace(
        available_groups=["stock_state", "market_env"],
        unavailable_groups=[{"key": "macro_env", "name": "宏观环境", "reason": "历史快照不可得"}],
    )
    return SimpleNamespace(
        available=True,
        selected=selected,
        selection_mode="严格样本",
        features_used={"stock_state": 18, "market_env": 10},
        context=context,
        reason="",
    )


def _fake_outcomes(median_return: float, iqr_spread: float) -> dict:
    base = {
        "available": True,
        "sample_count": 10,
        "mean_return": median_return,
        "median_return": median_return,
        "win_rate": 0.5 + np.clip(median_return * 10, -0.45, 0.45),
        "q10_return": median_return - iqr_spread,
        "q25_return": median_return - iqr_spread / 2,
        "q75_return": median_return + iqr_spread / 2,
        "median_max_gain": median_return + 0.05,
        "median_max_drawdown": -0.05,
        "median_worst_loss": -iqr_spread,
        "median_peak_day": 8.0,
        "median_volume_change": 0.1,
        "breakout_ratio": 0.5,
        "patterns": {"单边上涨": 10},
        "dominant_pattern": "单边上涨",
        "reason": "",
    }
    return {
        "available": True,
        "horizons": [{**base, "days": 20}],
        "direction_summary": "偏正面" if median_return > 0 else "偏负面",
        "direction_strength": float(np.clip(median_return * 8, -1, 1)),
        "notes": [],
    }


def test_v7_04_reliability_penalizes_small_dispersed_samples() -> None:
    good = assess_evidence(_fake_similarity(12, 75.0, 3.0), _fake_outcomes(0.04, 0.03), rows=1100, latest_lag_days=1)
    bad = assess_evidence(_fake_similarity(3, 75.0, 12.0), _fake_outcomes(0.04, 0.16), rows=1100, latest_lag_days=1)
    assert good.reliability_score > bad.reliability_score
    assert good.evidence_score > bad.evidence_score
    assert good.level in {"HIGH", "MEDIUM"}
    assert bad.level == "LOW"


def test_v7_05_missing_dimension_lowers_data_quality_not_similarity() -> None:
    full = _fake_similarity(10, 72.0, 4.0)
    partial = _fake_similarity(10, 72.0, 4.0)
    partial.context = SimpleNamespace(
        available_groups=["stock_state"],
        unavailable_groups=[
            {"key": "market_env", "name": "市场整体环境", "reason": "基准缺失"},
            {"key": "macro_env", "name": "宏观环境", "reason": "历史快照不可得"},
        ],
    )
    partial.features_used = {"stock_state": 18}
    evidence_full = assess_evidence(full, _fake_outcomes(0.03, 0.05), rows=1100, latest_lag_days=1)
    evidence_partial = assess_evidence(partial, _fake_outcomes(0.03, 0.05), rows=1100, latest_lag_days=1)
    assert evidence_partial.data_quality_score < evidence_full.data_quality_score
    assert evidence_partial.similarity_score == evidence_full.similarity_score


# ---------------------------------------------------------------- 4/5/6. 动态权重


def test_v7_06_high_quality_history_weight_at_least_50_and_largest() -> None:
    result = determine_weights("HIGH", 60.0)
    weights = result["weights"]
    assert weights["historical"] >= 0.50
    assert all(weights["historical"] >= weights[key] for key in ("news_event", "market_technical", "legacy_factor"))
    assert result["tier"] == "A"


def test_v7_07_low_quality_history_weight_never_reaches_50() -> None:
    for evidence_score in (0.0, 5.0, 10.0, 20.0, 24.9, 25.0, 100.0):
        result = determine_weights("LOW", evidence_score)
        assert result["weights"]["historical"] <= 0.20 + 1e-9
    result_medium = determine_weights("MEDIUM", 30.0)
    assert 0.20 <= result_medium["weights"]["historical"] <= 0.50 + 1e-9


def test_v7_08_dynamic_weights_always_sum_to_100_percent() -> None:
    for level in ("HIGH", "MEDIUM", "LOW"):
        for evidence_score in range(0, 101, 5):
            for availability in (
                {"historical": True, "news_event": True, "market_technical": True, "legacy_factor": True},
                {"historical": True, "news_event": False, "market_technical": True, "legacy_factor": True},
                {"historical": True, "news_event": False, "market_technical": False, "legacy_factor": False},
                {"historical": False, "news_event": True, "market_technical": True, "legacy_factor": True},
            ):
                result = determine_weights(level, float(evidence_score), availability)
                assert abs(sum(result["weights"].values()) - 1.0) < 1e-6


# ---------------------------------------------------------------- 7. 事件—市场反应


def _news_item(title: str, sentiment_score: float, published: pd.Timestamp, recency: float = 1.0) -> dict:
    return {
        "title": title,
        "summary": title,
        "sentiment": "偏正面" if sentiment_score > 0 else "偏负面",
        "sentiment_score": sentiment_score,
        "relevance_score": 0.95,
        "recency_weight": recency,
        "published_at": published.isoformat(),
        "source": "公司公告",
    }


def _flat_stock(rows: int = 120, start: str = "2026-01-05", scale: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=rows)
    close = np.full(rows, 100.0)
    volume = np.full(rows, 1_000_000.0)
    return pd.DataFrame(
        {
            "日期": dates,
            "开盘": close,
            "最高": close * (1 + 0.001),
            "最低": close * (1 - 0.001),
            "收盘": close,
            "成交量": volume,
        }
    )


def test_v7_09_event_reaction_classifications() -> None:
    dates = pd.bdate_range("2026-01-05", periods=120)
    close = np.full(120, 100.0)
    opens = np.full(120, 100.0)
    volume = np.full(120, 1_000_000.0)

    # 事件A（位置60）：利好 + 放量上涨 → 正向确认
    close[60] = 103.5
    opens[60] = 101.0
    volume[60] = 2_500_000.0
    # 事件B（位置90）：利空 + 股价不跌 → 可能已被提前计价
    close[90] = 100.1

    stock = pd.DataFrame(
        {
            "日期": dates,
            "开盘": opens,
            "最高": np.maximum(close, opens) * 1.002,
            "最低": np.minimum(close, opens) * 0.998,
            "收盘": close,
            "成交量": volume,
        }
    )
    news = {
        "items": [
            _news_item("公司签订重大合同", 0.8, dates[60] + pd.Timedelta(hours=2)),
            _news_item("监管问询函", -0.7, dates[90] + pd.Timedelta(hours=2)),
        ],
        "net_sentiment_score": 0.05,
        "usable_for_score": True,
    }
    result = assess_event_reaction(news, stock)
    assert result["available"]
    reactions = {item["title"]: item["reaction"] for item in result["events"]}
    assert "正向确认" in reactions["公司签订重大合同"]
    assert "提前计价" in reactions["监管问询函"] or "不跌" in reactions["监管问询函"]
    assert 0 <= result["score"] <= 100


def test_v7_10_good_news_but_no_price_move_is_blunting() -> None:
    stock = _flat_stock()
    dates = pd.DatetimeIndex(stock["日期"])
    news = {
        "items": [_news_item("公司宣布回购", 0.9, dates[70] + pd.Timedelta(hours=1))],
        "net_sentiment_score": 0.5,
        "usable_for_score": True,
    }
    result = assess_event_reaction(news, stock)
    assert result["available"]
    reaction = result["events"][0]["reaction"]
    assert "钝化" in reaction or "不涨" in reaction
    assert result["score"] < 50, "利好钝化不应给出高于中性的资讯分"


# ---------------------------------------------------------------- 8. 红三兵


def test_v7_11_three_white_soldiers_detection_and_validity() -> None:
    rng = np.random.default_rng(21)
    rows = 200
    dates = pd.bdate_range("2025-01-01", periods=rows)
    decline = np.full(rows - 3, -0.004)
    body = np.array([0.030, 0.032, 0.034])
    returns = np.concatenate([decline, body])
    close = 80 * np.exp(np.cumsum(returns))
    opens = close[:-3] * (1 + rng.normal(0, 0.0005, rows - 3))
    # 显式构造最后三根阳线：大实体逐根放大，收盘逐步抬高。
    opens = np.append(opens, [close[-3] * 0.985, close[-2] * 0.984, close[-1] * 0.983])
    volume = np.full(rows, 1_000_000.0)
    volume[-3:] = 2_200_000.0
    stock = pd.DataFrame(
        {
            "日期": dates,
            "开盘": opens,
            "最高": np.maximum(close, opens) * 1.004,
            "最低": np.minimum(close, opens) * 0.997,
            "收盘": close,
            "成交量": volume,
        }
    )
    soldiers = detect_three_white_soldiers(stock)
    assert soldiers["detected"] is True
    assert soldiers["validity_score"] >= 55
    assert "反" in soldiers["interpretation"] or "反弹" in soldiers["interpretation"]

    flat = _flat_stock(rows)
    assert detect_three_white_soldiers(flat)["detected"] is False


def test_v7_12_technical_patterns_module_score_bounds() -> None:
    stock, benchmark = synth_stock(500, seed=13)
    result = assess_technical_patterns(stock, benchmark)
    assert result["available"]
    assert 0 <= result["score"] <= 100
    assert result["trend"]["label"]
    short = stock.iloc[:100].copy()
    short_result = assess_technical_patterns(short)
    assert short_result["available"] is False


# ---------------------------------------------------------------- 9/10. 缺失与异常数据


def test_v7_13_missing_optional_inputs_do_not_crash() -> None:
    stock, benchmark = synth_stock(900, seed=7)
    bundle = agent_core.PriceBundle(stock, benchmark, "600000", "测试", "synthetic", "基准", "A股个股", "人民币元")
    missing = agent_core.EvidenceSnapshot(False, "全部辅助通道关闭", score=None)
    analysis = agent_core.analyze_all(bundle, default_profile(), missing, missing)
    decision = build_v7_decision(bundle, analysis, news_result=None)
    assert decision["engine_version"] == "V7.0.0"
    assert abs(sum(decision["weights"]["weights"].values()) - 1.0) < 1e-6
    assert decision["recommendation"] in {"建议买入", "谨慎买入", "观望", "不建议买入"}

    short_stock = stock.iloc[:240].copy().reset_index(drop=True)
    short_bundle = agent_core.PriceBundle(short_stock, None, "600000", "测试", "synthetic", "基准", "A股个股", "人民币元")
    short_analysis = agent_core.analyze_all(short_bundle, default_profile(), missing, missing)
    short_decision = build_v7_decision(short_bundle, short_analysis, news_result={"items": []})
    assert short_decision["evidence"]["level"] == "LOW"
    assert short_decision["weights"]["weights"]["historical"] <= 0.20 + 1e-6
    assert abs(sum(short_decision["weights"]["weights"].values()) - 1.0) < 1e-6


def test_v7_14_abnormal_values_do_not_crash_or_fabricate() -> None:
    stock, benchmark = synth_stock(900, seed=17)
    stock = stock.copy()
    stock.loc[stock.index[::37], "成交量"] = 0.0
    stock.loc[stock.index[::53], "成交量"] = np.nan
    stock.loc[stock.index[400], "收盘"] = stock.loc[stock.index[400], "收盘"] * 3.0
    result = search_historical_cases(stock, benchmark)
    # 允许相似检索可用或明确不可用，但绝不能抛异常或产生 NaN 相似度。
    if result.available:
        for item in result.selected:
            assert np.isfinite(item["similarity"])

    tiny = stock.iloc[:40].copy()
    tiny_result = search_historical_cases(tiny, None)
    assert tiny_result.available is False
    assert tiny_result.reason


# ---------------------------------------------------------------- 11. 未来数据泄露


def test_v7_15_future_data_changes_do_not_affect_selection() -> None:
    stock, benchmark = synth_stock(900, seed=7)
    analysis_position = 700
    baseline = search_historical_cases(stock, benchmark, analysis_position=analysis_position)

    mutated = stock.copy()
    mutated.loc[mutated.index[analysis_position + 1 :], "收盘"] *= np.linspace(1.0, 4.0, len(mutated) - analysis_position - 1)
    mutated.loc[mutated.index[analysis_position + 1 :], "成交量"] *= 99.0
    mutated_benchmark = benchmark.copy()
    mutated_benchmark.loc[mutated_benchmark.index[analysis_position + 1 :], "收盘"] *= 0.2
    tampered = search_historical_cases(mutated, mutated_benchmark, analysis_position=analysis_position)

    key = lambda result: [(pd.Timestamp(item["anchor_date"]).isoformat(), round(item["similarity"], 9)) for item in result.selected]
    assert key(baseline) == key(tampered), "篡改分析日之后的数据不得影响案例选择"


def test_v7_16_boundary_rejects_future_positions() -> None:
    dates = pd.bdate_range("2026-01-01", periods=10)
    with pytest.raises(LookaheadViolation):
        build_boundary(pd.DatetimeIndex(dates), analysis_position=15)
    boundary = build_boundary(pd.DatetimeIndex(dates), analysis_position=5)
    with pytest.raises(LookaheadViolation):
        ensure_selection_dates_within_boundary([dates[7]], boundary)


def test_v7_17_outcome_window_must_be_fully_observable() -> None:
    stock, benchmark = synth_stock(900, seed=7)
    result = search_historical_cases(stock, benchmark)
    assert result.available
    for item in result.selected:
        assert item["position"] + 120 < result.boundary.rows
    outcomes = evaluate_outcomes(result.context.close, result.context.volume, result.selected)
    assert outcomes["available"]
    for horizon in outcomes["horizons"]:
        if horizon.get("available"):
            assert horizon["sample_count"] > 0


# ---------------------------------------------------------------- 12. 回归


def test_v7_18_original_pipeline_unchanged_by_decision_layer() -> None:
    """新增决策层不得改写 V6.5.2 原有评分管线的任何输出。"""
    stock, benchmark = synth_stock(900, seed=7)
    bundle = agent_core.PriceBundle(stock, benchmark, "600000", "测试", "synthetic", "基准", "A股个股", "人民币元")
    missing = agent_core.EvidenceSnapshot(False, "全部辅助通道关闭", score=None)
    analysis = agent_core.analyze_all(bundle, default_profile(), missing, missing)
    assert agent_core.MODEL_VERSION == "V6.5.2"
    assert "historical_decision" not in analysis
    selected = analysis["selected_horizon"]
    assert selected is not None
    assert selected["factor_contributions"]["volume"] == 0.0
    assert selected["factor_contributions"]["historical_analog"] == 0.0
    decision = build_v7_decision(bundle, analysis, news_result=None)
    rerun = agent_core.analyze_all(bundle, default_profile(), missing, missing)
    assert rerun["selected_horizon"]["score"] == analysis["selected_horizon"]["score"]
    assert rerun["conclusion"] == analysis["conclusion"]
    assert decision["legacy_factor"]["score"] == analysis["selected_horizon"]["score"]


def test_v7_19_snapshot_roundtrip_keeps_decision() -> None:
    from snapshot_codec import build_analysis_snapshot, restore_analysis_snapshot

    stock, benchmark = synth_stock(700, seed=23)
    bundle = agent_core.PriceBundle(stock, benchmark, "600000", "测试", "synthetic", "基准", "A股个股", "人民币元")
    missing = agent_core.EvidenceSnapshot(False, "全部辅助通道关闭", score=None)
    analysis = agent_core.analyze_all(bundle, default_profile(), missing, missing)
    analysis["news_analysis"] = {"items": [], "usable_for_score": False, "net_sentiment_score": 0.0}
    analysis["historical_decision"] = build_v7_decision(bundle, analysis, news_result=analysis["news_analysis"])
    payload = build_analysis_snapshot(
        bundle=bundle,
        analysis=analysis,
        profile=default_profile(),
        holding_state="尚未持有",
        holding_method="按持股数量填写",
        holding_snapshot=None,
    )
    restored = restore_analysis_snapshot(payload)
    restored_decision = restored["analysis"]["historical_decision"]
    assert restored_decision["engine_version"] == "V7.0.0"
    assert restored_decision["composite_score"] == analysis["historical_decision"]["composite_score"]
    assert restored_decision["weights"]["weights_pct"] == analysis["historical_decision"]["weights"]["weights_pct"]


def test_v7_20_app_wiring_contains_v7_decision_layer() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "from historical_decision import build_v7_decision" in source
    assert 'analysis["historical_decision"] = build_v7_decision(' in source
    assert '"历史情景决策"' in source
    assert "render_historical_decision" in source
    # 原有页面关键字符串必须保留（回归）
    assert "save_snapshot" in source and "restore_analysis_snapshot" in source
    assert 'modes = ["完整分析"]' in source


def test_v7_21_cross_validation_veto_caps_bullish_history() -> None:
    """历史情景看涨但基本面出现重大风险时，最终评分必须被压低。"""
    stock, benchmark = synth_stock(900, seed=7)
    bundle = agent_core.PriceBundle(stock, benchmark, "600000", "测试", "synthetic", "基准", "A股个股", "人民币元")
    missing = agent_core.EvidenceSnapshot(False, "全部辅助通道关闭", score=None)
    healthy = agent_core.EvidenceSnapshot(True, "synthetic", {}, 70.0, [], [], [])
    distressed = agent_core.EvidenceSnapshot(
        True,
        "synthetic",
        {},
        28.0,
        [],
        ["净资产收益率为负", "净利润同比下降", "经营现金流与净利润方向不一致"],
        [],
    )
    analysis_ok = agent_core.analyze_all(bundle, default_profile(), healthy, missing)
    analysis_bad = agent_core.analyze_all(bundle, default_profile(), distressed, missing)
    decision_ok = build_v7_decision(bundle, analysis_ok, news_result=None)
    decision_bad = build_v7_decision(bundle, analysis_bad, news_result=None)
    assert decision_bad["composite_score"] < decision_ok["composite_score"]
    assert decision_bad["veto"]["factor"] <= 0.75
    assert decision_bad["legacy_factor"]["cross_check"].startswith("存在冲突")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
