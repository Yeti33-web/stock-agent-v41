from __future__ import annotations

from contextlib import contextmanager
from datetime import date as real_date
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

import agent_core
import factor_analysis
from news_analysis import assess_news

from historical_test_tool.historical_data import assert_bundle_cutoff, fetch_historical_bundle
from historical_test_tool.point_in_time import build_point_in_time_evidence, empty_historical_news_payload


PROFILE_PATH = Path(__file__).with_name("default_profile.json")


def load_test_profile() -> dict[str, Any]:
    with PROFILE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def frozen_agent_date(core_module: Any, frozen: real_date) -> Iterator[None]:
    """Temporarily replace only agent_core.date inside this standalone process."""

    original_date = core_module.date

    class FrozenDate(real_date):
        @classmethod
        def today(cls) -> real_date:
            return frozen

    core_module.date = FrozenDate
    try:
        yield
    finally:
        core_module.date = original_date


def _safe_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, real_date)):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return {str(key): _json_safe(item) for key, item in value.to_dict().items()}
    return _safe_number(value)


def _compact_horizons(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in analysis.get("horizon_scores") or []:
        rows.append(
            {
                "期限": item.get("name"),
                "交易日": item.get("days"),
                "是否可评分": bool(item.get("available")),
                "评分": _safe_number(item.get("score")),
                "标签": item.get("label"),
                "主要理由": list(item.get("reasons") or []),
            }
        )
    return rows


def _contribution_rows(analysis: dict[str, Any], profile: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "用户风险分": factor_analysis.investor_contribution_rows(profile),
        "股票风险分": factor_analysis.stock_risk_contribution_rows(analysis.get("metrics") or {}),
        "所选周期时点评分": factor_analysis.timing_contribution_rows(analysis),
    }


def run_historical_replay(
    market: str,
    raw_code: str,
    requested_date: real_date,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay V6.4 at T and return no post-T performance data."""

    active_profile = dict(profile or load_test_profile())

    # The fetchers themselves receive end=T. A second row-level cutoff is then
    # applied by historical_data.py, so future rows are never passed to Agent.
    with frozen_agent_date(agent_core, requested_date):
        historical = fetch_historical_bundle(market, raw_code, requested_date, agent_core)

    bundle = historical.bundle
    actual_date = historical.actual_trading_date
    assert_bundle_cutoff(bundle, actual_date)
    fundamental, macro, evidence_status = build_point_in_time_evidence(agent_core, bundle, actual_date)

    # This standalone Streamlit program is the independent process. The date
    # override exists only while V6.4 calculates the historical snapshot.
    with frozen_agent_date(agent_core, actual_date):
        analysis = agent_core.analyze_all(bundle, active_profile, fundamental, macro)

    selected_score = (analysis.get("selected_horizon") or {}).get("score")
    news_payload = empty_historical_news_payload(market, bundle.code, bundle.name, actual_date)
    analysis["news_analysis"] = assess_news(news_payload, selected_score)
    contributions = _contribution_rows(analysis, active_profile)
    selected = dict(analysis.get("selected_horizon") or {})
    stock_last = pd.Timestamp(bundle.stock["日期"].max()).date()
    benchmark_last = (
        pd.Timestamp(bundle.benchmark["日期"].max()).date()
        if bundle.benchmark is not None and not bundle.benchmark.empty
        else None
    )

    snapshot = {
        "tool_version": "独立历史时点复现工具V1.1",
        "回测范围": "只复现T日Agent判断；不读取、不计算、不评价T日之后走势",
        "测试条件": {
            "市场": market,
            "股票代码": bundle.code,
            "股票名称": bundle.name,
            "用户输入日期T": historical.requested_date,
            "实际采用交易日": actual_date,
            "基准指数": bundle.benchmark_name,
            "测试画像": active_profile.get("profile_name", "独立固定测试画像"),
        },
        "历史数据": {
            "请求起点": historical.requested_start,
            "个股首日": pd.Timestamp(bundle.stock["日期"].min()).date(),
            "个股末日": stock_last,
            "个股行数": int(len(bundle.stock)),
            "基准末日": benchmark_last,
            "数据源": bundle.provider,
            "五年覆盖率": float(bundle.coverage_ratio),
            "警告": list(bundle.warnings),
        },
        "当时Agent判断": {
            "结论": analysis.get("conclusion"),
            "结论理由": analysis.get("conclusion_reason"),
            "自动选择周期": selected.get("name"),
            "综合观察分_自动周期": _safe_number(selected.get("score")),
            "评分标签": selected.get("label"),
            "股票风险分": _safe_number(analysis.get("stock_risk_score")),
            "股票风险等级": analysis.get("stock_risk_level"),
            "用户风险分": _safe_number(analysis.get("investor_score")),
            "用户风险等级": analysis.get("investor_level"),
            "个人适配": (analysis.get("suitability") or {}).get("fit"),
            "数据完整度": _safe_number(analysis.get("data_confidence")),
            "主要时点理由": list(selected.get("reasons") or []),
            "主要风险理由": list(analysis.get("risk_reasons") or []),
            "数据限制": list(analysis.get("confidence_notes") or []),
        },
        "全部现有周期评分": _compact_horizons(analysis),
        "因子贡献": contributions,
        "历史证据状态": evidence_status,
        "防未来数据检查": {
            "冻结日期": actual_date,
            "个股最大日期": stock_last,
            "基准最大日期": benchmark_last,
            "T后行情传入Agent行数": 0,
            "T后财务_宏观利率_资讯传入": False,
            "H是否改变Agent结论": False,
        },
        "测试画像明细": active_profile,
    }
    return _json_safe(snapshot)


def snapshot_json(snapshot: dict[str, Any]) -> str:
    return json.dumps(_json_safe(snapshot), ensure_ascii=False, indent=2, sort_keys=False)
