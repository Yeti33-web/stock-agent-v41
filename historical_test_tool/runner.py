from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date as real_date
import json
from typing import Any, Iterator

import numpy as np
import pandas as pd

import agent_core
import factor_analysis
from news_analysis import assess_news

from historical_test_tool.historical_data import (
    assert_bundle_cutoff,
    fetch_historical_bundle,
    fetch_historical_fx,
)
from historical_test_tool.point_in_time import (
    build_point_in_time_evidence,
    build_point_in_time_news,
    empty_historical_news_payload,
)

# Agent A V7.0.0 新增的历史情景决策层。回测工具始终跟随当前A版本：
# 若分支中的A尚未包含该模块，则自动降级，不影响V6.5.2核心流程复现。
try:
    from historical_decision import build_v7_decision
except Exception:  # pragma: no cover - 仅在A版本较旧时触发
    build_v7_decision = None


@dataclass
class FullReplayResult:
    historical: Any
    bundle: Any
    analysis: dict[str, Any]
    profile: dict[str, Any]
    holding_snapshot: dict[str, Any] | None
    evidence_status: dict[str, Any]
    fx_snapshot: dict[str, Any] | None

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


def run_full_historical_agent(
    market: str,
    raw_code: str,
    requested_date: real_date,
    profile: dict[str, Any],
    holding_state: str = "尚未持有",
    holding_method: str = "按持股数量填写",
    share_count: float = 0.0,
    cost_price: float = 0.0,
    current_market_value: float = 0.0,
    total_cost: float = 0.0,
    additional_amount: float = 0.0,
) -> FullReplayResult:
    """Run the current Agent A analysis pipeline at T with the user's own profile."""

    active_profile = dict(profile)
    with frozen_agent_date(agent_core, requested_date):
        historical = fetch_historical_bundle(market, raw_code, requested_date, agent_core)

    bundle = historical.bundle
    actual_date = historical.actual_trading_date
    assert_bundle_cutoff(bundle, actual_date)
    last_price = float(bundle.stock["收盘"].iloc[-1])
    holding_snapshot: dict[str, Any] | None = None
    fx_snapshot: dict[str, Any] | None = None

    if holding_state == "已经持有":
        if holding_method == "按持股数量填写":
            if market in {"美股", "港股"}:
                with frozen_agent_date(agent_core, actual_date):
                    fx_snapshot = fetch_historical_fx(agent_core, market, actual_date)
            holding_snapshot = agent_core.calculate_holding_values(
                market,
                float(share_count),
                last_price,
                float(cost_price),
                usd_cny_rate=float(fx_snapshot["rate"]) if fx_snapshot and market == "美股" else None,
                hkd_cny_rate=float(fx_snapshot["rate"]) if fx_snapshot and market == "港股" else None,
            )
            if fx_snapshot:
                holding_snapshot["fx_provider"] = fx_snapshot["provider"]
                holding_snapshot["fx_date"] = fx_snapshot["date"].isoformat()
        else:
            holding_snapshot = agent_core.calculate_amount_holding_values(
                float(current_market_value),
                float(total_cost),
            )
        current_value = float(holding_snapshot["current_rmb"])
        active_profile["planned_amount"] = current_value + float(additional_amount)
        active_profile["current_holding_value"] = current_value
        active_profile["additional_amount"] = float(additional_amount)
        active_profile["holding_state"] = "已经持有"
    else:
        if float(active_profile.get("planned_amount") or 0.0) <= 0:
            raise ValueError("本次计划买入金额必须大于0。")
        active_profile["current_holding_value"] = 0.0
        active_profile["additional_amount"] = float(active_profile["planned_amount"])
        active_profile["holding_state"] = "尚未持有"

    fundamental, macro, evidence_status = build_point_in_time_evidence(agent_core, bundle, actual_date)
    news_payload, news_status = build_point_in_time_news(market, bundle.code, bundle.name, actual_date)
    evidence_status = dict(evidence_status)
    evidence_status["历史资讯"] = news_status

    with frozen_agent_date(agent_core, actual_date):
        analysis = agent_core.analyze_all(bundle, active_profile, fundamental, macro)
        selected_score = (analysis.get("selected_horizon") or {}).get("score")
        analysis["news_analysis"] = assess_news(news_payload, selected_score)
        if build_v7_decision is not None:
            try:
                analysis["historical_decision"] = build_v7_decision(
                    bundle, analysis, news_result=analysis["news_analysis"]
                )
            except Exception as decision_exc:
                analysis["historical_decision"] = {
                    "available": False,
                    "engine_version": "V7.0.0",
                    "reason": f"决策层属于新增展示模块，本次历史复现生成失败：{type(decision_exc).__name__}，不影响原有结论。",
                }
        analysis["factor_analysis"] = factor_analysis.build_factor_analysis(bundle, analysis, active_profile)
        current_holding = float(active_profile.get("current_holding_value") or 0.0)
        assets = float(active_profile.get("investable_assets") or 0.0)
        analysis["position"]["current_amount"] = current_holding
        analysis["position"]["current_pct"] = current_holding / assets if assets > 0 else 0.0
        analysis["position"]["remaining_upper_amount"] = max(
            float(analysis["position"]["upper_amount"]) - current_holding,
            0.0,
        )
        analysis["position"]["remaining_upper_pct"] = (
            analysis["position"]["remaining_upper_amount"] / assets if assets > 0 else 0.0
        )
        analysis["holding_snapshot"] = holding_snapshot
        analysis["sell_signals"] = (
            agent_core.analyze_sell_signals(bundle, analysis, active_profile, holding_snapshot)
            if holding_state == "已经持有"
            else None
        )

    return FullReplayResult(
        historical=historical,
        bundle=bundle,
        analysis=analysis,
        profile=active_profile,
        holding_snapshot=holding_snapshot,
        evidence_status=evidence_status,
        fx_snapshot=fx_snapshot,
    )


def _compact_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """Summarize the V7 decision layer for the replay snapshot without post-T data."""

    if not decision:
        return {"可用": False, "说明": "当前分支的Agent A未包含V7.0.0历史情景决策层，未参与本次复现。"}
    if not decision.get("available"):
        return {
            "可用": False,
            "引擎版本": decision.get("engine_version"),
            "说明": decision.get("reason") or "决策层各模块均不可用。",
        }
    weights = decision.get("weights") or {}
    evidence = decision.get("evidence") or {}
    historical = decision.get("historical") or {}
    return {
        "引擎版本": decision.get("engine_version"),
        "可用": True,
        "综合决策分": _safe_number(decision.get("composite_score")),
        "建议": decision.get("recommendation"),
        "建议理由": decision.get("recommendation_reason"),
        "置信度": _safe_number(decision.get("confidence")),
        "建议仓位上限比例": _safe_number(decision.get("suggested_position_pct")),
        "证据等级": evidence.get("level"),
        "证据分": _safe_number(evidence.get("evidence_score")),
        "动态权重_百分比": weights.get("weights_pct"),
        "模块得分": {key: _safe_number(value) for key, value in (decision.get("module_scores") or {}).items()},
        "历史相似样本数": historical.get("sample_count"),
        "防未来数据守卫": historical.get("guard"),
        "权重说明": list(weights.get("reasons") or []),
    }


def run_historical_replay(
    market: str,
    raw_code: str,
    requested_date: real_date,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Replay current Agent A at T and return no post-T performance data."""

    result = run_full_historical_agent(
        market=market,
        raw_code=raw_code,
        requested_date=requested_date,
        profile=dict(profile),
    )
    historical = result.historical
    bundle = result.bundle
    analysis = result.analysis
    active_profile = result.profile
    actual_date = historical.actual_trading_date
    evidence_status = result.evidence_status
    contributions = _contribution_rows(analysis, active_profile)
    selected = dict(analysis.get("selected_horizon") or {})
    stock_last = pd.Timestamp(bundle.stock["日期"].max()).date()
    benchmark_last = (
        pd.Timestamp(bundle.benchmark["日期"].max()).date()
        if bundle.benchmark is not None and not bundle.benchmark.empty
        else None
    )

    snapshot = {
        "tool_version": "完整界面独立历史时点测试工具V2.3（同步Agent A V7.0.0，含历史情景决策层）",
        "回测范围": "只复现T日Agent判断；不读取、不计算、不评价T日之后走势",
        "测试条件": {
            "市场": market,
            "股票代码": bundle.code,
            "股票名称": bundle.name,
            "用户输入日期T": historical.requested_date,
            "实际采用交易日": actual_date,
            "基准指数": bundle.benchmark_name,
            "测试画像": active_profile.get("profile_name", "用户本次填写的风险资料"),
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
            "行情方向": analysis.get("market_signal"),
            "方向可信度": analysis.get("signal_confidence"),
            "历史验证状态": (selected.get("signal_validation") or {}).get("status"),
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
        "历史情景决策层": _compact_decision(dict(analysis.get("historical_decision") or {})),
        "防未来数据检查": {
            "冻结日期": actual_date,
            "个股最大日期": stock_last,
            "基准最大日期": benchmark_last,
            "T后行情传入Agent行数": 0,
            "T后财务_宏观利率_资讯传入": False,
            "V7决策层输入": "仅使用T日及以前的行情、成交量与基准数据；历史相似检索在截断后序列内进行",
        },
        "测试画像明细": active_profile,
    }
    return _json_safe(snapshot)


def snapshot_json(snapshot: dict[str, Any]) -> str:
    return json.dumps(_json_safe(snapshot), ensure_ascii=False, indent=2, sort_keys=False)
