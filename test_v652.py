from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import agent_core


def price_frame(periods: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-09-02", periods=periods)
    close = 20 + np.linspace(0, 5, periods) + np.sin(np.arange(periods) / 8)
    return pd.DataFrame(
        {
            "日期": dates,
            "开盘": close * 0.995,
            "最高": close * 1.01,
            "最低": close * 0.99,
            "收盘": close,
            "成交量": np.full(periods, 1_000_000.0),
        }
    )


def test_a_price_prefers_fast_valid_chart_source(monkeypatch) -> None:
    frame = price_frame()
    calls = {"slow": 0}

    monkeypatch.setattr(agent_core, "fetch_yahoo_chart_history", lambda *args: (frame, "测试公司"))

    def slow_source(*args, **kwargs):
        calls["slow"] += 1
        raise AssertionError("有效快速行情存在时不应再调用慢速备用源")

    monkeypatch.setattr(agent_core, "fetch_baostock_history", slow_source)
    data, name, provider = agent_core.fetch_a_security("600519", "2025-01-01", "2026-09-02")
    assert len(data) == len(frame)
    assert name == "测试公司"
    assert "Yahoo Finance" in provider
    assert calls["slow"] == 0


def test_eastmoney_a_fundamental_fields_are_real_percentages(monkeypatch) -> None:
    raw = pd.DataFrame(
        [
            {
                "SECURITY_NAME_ABBR": "测试公司",
                "REPORT_DATE": "2026-06-30",
                "NOTICE_DATE": "2026-08-15",
                "REPORT_DATE_NAME": "2026中报",
                "ROEJQ": 12.0,
                "XSJLL": 4.0,
                "PARENTNETPROFITTZ": 3.0,
                "TOTALOPERATEREVETZ": 2.0,
                "ZCFZL": 45.0,
                "MGJYXJJE": 1.8,
                "EPSJB": 1.2,
            }
        ]
    )
    monkeypatch.setattr(
        agent_core,
        "ak",
        SimpleNamespace(stock_financial_analysis_indicator_em=lambda **kwargs: raw),
    )
    result = agent_core.fetch_a_fundamentals_eastmoney("600519", 100.0)
    assert result.available
    assert result.fields["净资产收益率"] == 0.12
    assert result.fields["净利率"] == 0.04
    assert result.fields["净利润同比"] == 0.03
    assert result.fields["营收同比"] == 0.02
    assert result.fields["资产负债率"] == 0.45
    assert result.fields["经营现金流／净利润"] == 1.5


def test_future_financial_row_is_not_selected() -> None:
    frame = pd.DataFrame(
        [
            {"NOTICE_DATE": "2026-08-15", "value": "published"},
            {"NOTICE_DATE": "2099-01-01", "value": "future"},
        ]
    )
    row = agent_core._latest_public_financial_row(frame, ("NOTICE_DATE",))
    assert row["value"] == "published"


def test_all_future_financial_rows_return_no_evidence() -> None:
    frame = pd.DataFrame([{"NOTICE_DATE": "2099-01-01", "value": "future"}])
    assert agent_core._latest_public_financial_row(frame, ("NOTICE_DATE",)) == {}


def test_missing_all_optional_evidence_still_produces_conclusion() -> None:
    stock = price_frame(900)
    benchmark = price_frame(900)
    bundle = agent_core.PriceBundle(
        stock,
        benchmark,
        "TEST",
        "测试股票",
        "测试行情",
        "测试基准",
        "美股个股",
        "美元",
    )
    profile = {
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
    missing = agent_core.EvidenceSnapshot(False, "全部辅助通道关闭", score=None)
    analysis = agent_core.analyze_all(bundle, profile, missing, missing)
    assert analysis["selected_horizon"] is not None
    assert not analysis["conclusion"].startswith("证据不足")
