"""V7.0.0 决策引擎 —— 历史相似情景综合决策（historical decision）。

本模块把各计算模块按需求文档第一节的核心决策逻辑串起来：

    当前股票 → 识别当前投资环境 → 寻找历史最相似情景 →
    分析历史情景之后的表现 → 分析最新资讯与市场实际反应 →
    分析当前行情与技术形态 → 调用原有因子交叉验证 →
    动态确定权重 → 形成最终投资判断

纪律：

* 所有数学计算由程序完成；LLM 只在外层负责理解、解释与归纳。
* 原有 V6.5.2 评分管线（``analyze_all`` / ``score_horizons``）不被修改，
  本模块只读取其输出作为“原有因子交叉验证”模块的结果。
* 基本面出现重大风险时执行一票降权（需求第十一节）。
* 输出包含需求第十三节要求的全部字段与六段式“为什么”解释。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from dynamic_weight import MODULE_NAMES, determine_weights
from evidence_quality import assess_evidence
from event_reaction import assess_event_reaction
from historical_outcome import evaluate_outcomes
from historical_similarity import search_historical_cases
from lookahead_guard import DECISION_ENGINE_VERSION, guard_summary
from technical_patterns import assess_technical_patterns


def _py(value: Any) -> Any:
    """递归转换为 JSON 可序列化的原生类型（快照保存需要）。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _py(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_py(item) for item in value.tolist()]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _historical_module_score(outcome_result: dict[str, Any], reliability: float) -> float | None:
    if not outcome_result.get("available"):
        return None
    strength = float(outcome_result.get("direction_strength", 0.0))
    dampening = 0.5 + 0.5 * float(np.clip(reliability, 0.0, 1.0))
    return float(np.clip(50.0 + 35.0 * strength * dampening, 0.0, 100.0))


def _news_dates(news_result: Mapping[str, Any] | None) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    for item in (news_result or {}).get("items") or []:
        raw = item.get("published_at")
        if not raw:
            continue
        try:
            dates.append(pd.Timestamp(raw))
        except (TypeError, ValueError):
            continue
    return dates


def _recommendation(composite: float, suitability_fit: str | None) -> tuple[str, str]:
    if composite >= 65:
        base, reason = "建议买入", "综合评分进入积极区间。"
    elif composite >= 55:
        base, reason = "谨慎买入", "综合评分中性偏积极，但未达到积极区间。"
    elif composite >= 45:
        base, reason = "观望", "证据强度不足以支持新增风险敞口。"
    else:
        base, reason = "不建议买入", "综合评分偏弱，当前不支持买入。"
    if suitability_fit == "不适配":
        if composite >= 55:
            return "观望", "行情层面存在积极信号，但个人风险适配不足，仓位上限为0。"
        return "不建议买入", "综合评分不足，且个人风险适配不满足。"
    return base, reason


def _build_narrative(decision: dict[str, Any]) -> dict[str, str]:
    historical = decision["historical"]
    evidence = decision["evidence"]
    event = decision["event_reaction"]
    technical = decision["technical_patterns"]
    legacy = decision["legacy_factor"]
    weights = decision["weights"]

    if historical.get("available"):
        lines = [
            f"找到{historical['sample_count']}个历史相似案例"
            f"（{historical['selection_mode']}），平均相似度{evidence['similarity_score']:.3f}分，"
            f"历史证据等级={evidence['level']}，历史权重{weights['weights_pct']['historical']:.1f}%。"
        ]
        for horizon in historical.get("horizons", []):
            if horizon.get("available"):
                lines.append(
                    f"其后{horizon['days']}个交易日：平均收益{horizon['mean_return']:+.3%}，"
                    f"上涨频率{horizon['win_rate']:.1%}，最大涨幅中位{horizon['median_max_gain']:+.3%}，"
                    f"最大回撤中位{horizon['median_max_drawdown']:+.3%}。"
                )
        historical_text = "\n".join(lines)
    else:
        historical_text = f"没有可用历史相似案例：{historical.get('reason', '样本不足')}"

    if event.get("available"):
        event_lines = [
            f"{item['title'][:40]}｜{item.get('sentiment', '中性')}｜"
            f"当日{item.get('day0_return', 0.0):+.3%}｜判定：{item['reaction']}"
            for item in event["events"][:5]
            if "reaction" in item
        ]
        event_text = f"资讯模块分{event['score']:.1f}/100。\n" + "\n".join(event_lines)
        if event.get("expectation_flags"):
            event_text += "\n预期差提示：" + "；".join(event["expectation_flags"][:3])
    else:
        event_text = f"资讯模块不可用：{event.get('reason', '无有效资讯')}"

    if technical.get("available"):
        technical_text = (
            f"技术形态模块分{technical['score']:.1f}/100。"
            f"趋势位置：{technical['trend']['label']}。"
            + (
                f"出现红三兵：{technical['three_white_soldiers'].get('interpretation', '')}"
                if technical["three_white_soldiers"].get("detected")
                else "未出现红三兵。"
            )
        )
    else:
        technical_text = f"技术形态模块不可用：{technical.get('reason', '')}"

    legacy_text = (
        f"原V6.5.2时点评分{legacy['score']}/100（期限：{legacy['horizon_name']}），"
        f"基本面{legacy['fundamental_score'] if legacy['fundamental_score'] is not None else '数据暂不可用'}，"
        f"宏观市场{legacy['macro_score'] if legacy['macro_score'] is not None else '数据暂不可用'}。"
        f"交叉验证结论：{legacy['cross_check']}。"
    )

    future_lines = []
    for horizon in historical.get("horizons", []):
        if horizon.get("available"):
            future_lines.append(
                f"{horizon['days']}日：历史中位收益{horizon['median_return']:+.3%}"
                f"（上涨频率{horizon['win_rate']:.1%}）"
            )
    future_text = "；".join(future_lines) if future_lines else "缺少历史路径参考，只能按当前信号与风险预算判断"

    reasoning_chain = (
        f"① 当前发生了什么：{decision['current_state']}\n"
        f"② 历史上类似情况怎么样：{historical.get('direction_summary', '无历史参考')}\n"
        f"③ 当前市场是否正在重复类似路径：{'技术形态与量能' + ('确认' if decision['market_confirms'] else '尚未确认')}\n"
        f"④ 最新资讯是否支持：{event.get('reaction_summary', '资讯不可用')}\n"
        f"⑤ 当前行情是否确认：{technical.get('trend', {}).get('label', '数据不足')}\n"
        f"⑥ 原有因子是否存在冲突：{legacy['cross_check']}\n"
        f"⑦ 未来可能出现的情况（历史参考）：{future_text}\n"
        f"⑧ 最终是否值得现在买入：{decision['recommendation']}（{decision['recommendation_reason']}）"
    )

    return {
        "历史相似情景": historical_text,
        "当前资讯": event_text,
        "当前行情": technical_text,
        "原有因子": legacy_text,
        "最终推演": reasoning_chain,
    }


def build_v7_decision(
    bundle: Any,
    analysis: Mapping[str, Any],
    news_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """执行 V7.0.0 综合决策流程。只读输入，不修改任何传入对象。"""
    news = news_result if news_result is not None else dict(analysis.get("news_analysis") or {})
    metrics = dict(analysis.get("metrics") or {})
    latest_lag = int(metrics.get("latest_lag") or 0)

    similarity = search_historical_cases(bundle.stock, bundle.benchmark)
    boundary = similarity.boundary
    context = similarity.context

    if similarity.available:
        outcomes = evaluate_outcomes(
            context.close, context.volume, similarity.selected
        )
        rows = int(context.close.shape[0])
    else:
        outcomes = {
            "available": False,
            "horizons": [],
            "direction_summary": "无历史相似案例可参考。",
            "direction_strength": 0.0,
            "notes": [],
        }
        rows = int(len(bundle.stock))

    evidence = assess_evidence(similarity, outcomes, rows, latest_lag)

    news_dates = _news_dates(news)
    technical = assess_technical_patterns(bundle.stock, bundle.benchmark, news_dates)
    event = assess_event_reaction(news, bundle.stock)

    historical_score = _historical_module_score(outcomes, evidence.reliability_score)
    historical_available = bool(similarity.available and outcomes.get("available") and historical_score is not None)

    selected = analysis.get("selected_horizon") or {}
    legacy_score = selected.get("score")
    fundamental = analysis.get("fundamental")
    macro = analysis.get("macro")
    fundamental_score = getattr(fundamental, "score", None)
    macro_score = getattr(macro, "score", None)
    legacy_available = legacy_score is not None

    # 交叉验证（需求第十一节）：基本面重大风险一票降权。
    fundamental_risks = list(getattr(fundamental, "risks", []) or [])
    deterioration_terms = ("净资产收益率为负", "净利率为负", "净利润同比下降", "营业收入同比下降", "经营现金流与净利润方向不一致")
    deterioration_count = sum(
        1 for risk in fundamental_risks if any(term in str(risk) for term in deterioration_terms)
    )
    veto_factor = 1.0
    veto_reasons: list[str] = []
    if fundamental_score is not None and fundamental_score <= 35:
        veto_factor = 0.75
        veto_reasons.append(f"基本面评分{fundamental_score:.1f}≤35，存在重大财务风险，最终评分降权25%。")
    elif fundamental_score is not None and fundamental_score <= 45:
        veto_factor = 0.90
        veto_reasons.append(f"基本面评分{fundamental_score:.1f}≤45，最终评分降权10%。")
    elif deterioration_count >= 2:
        veto_factor = 0.85
        veto_reasons.append(f"基本面出现{deterioration_count}项经营恶化信号，最终评分降权15%。")
    if legacy_available and legacy_score < 42 and fundamental_score is not None and fundamental_score >= 55:
        veto_reasons.append("基本面未见恶化，但量价信号偏弱，保持谨慎。")

    if deterioration_count >= 2 or (fundamental_score is not None and fundamental_score <= 35):
        cross_check = "存在冲突——基本面出现重大风险，历史与技术面的乐观判断必须打折"
    elif fundamental_score is not None and fundamental_score >= 55 and legacy_available and legacy_score >= 55:
        cross_check = "相互支持——基本面与原时点评分均未反驳当前判断"
    elif fundamental_score is None:
        cross_check = "财务数据暂不可用——不构成反驳，但置信度相应下降"
    else:
        cross_check = "中性——基本面既未确认也未反驳"

    modules_available = {
        "historical": historical_available,
        "news_event": bool(event.get("available")),
        "market_technical": bool(technical.get("available")),
        "legacy_factor": legacy_available,
    }
    weights = determine_weights(evidence.level, evidence.evidence_score, modules_available)

    module_scores = {
        "historical": historical_score,
        "news_event": event.get("score"),
        "market_technical": technical.get("score"),
        "legacy_factor": float(legacy_score) if legacy_available else None,
    }
    composite = 0.0
    for key, weight in weights["weights"].items():
        if weight <= 0 or module_scores[key] is None:
            continue
        composite += weight * float(module_scores[key])
    composite = float(np.clip(composite * veto_factor, 0.0, 100.0))

    suitability = analysis.get("suitability") or {}
    suitability_fit = suitability.get("fit")
    recommendation, recommendation_reason = _recommendation(composite, suitability_fit)

    sell_signals = analysis.get("sell_signals") or {}
    if sell_signals and int(sell_signals.get("hard_count") or 0) >= 1 and composite > 45:
        composite = 45.0
        recommendation, recommendation_reason = "观望", "存在核心卖出信号，买入判断被压制到观望。"
        veto_reasons.append("已持有头寸触发核心卖出信号，新增买入不被支持。")

    data_confidence = int(analysis.get("data_confidence") or 0)
    confidence = float(np.clip(0.55 * data_confidence + 0.45 * evidence.evidence_score, 0, 100))
    if not historical_available:
        confidence = min(confidence, 65.0)

    position = dict(analysis.get("position") or {})
    upper_pct = float(position.get("upper_pct") or 0.0)
    if recommendation in {"观望", "不建议买入"} or suitability_fit == "不适配":
        suggested_position = 0.0
    else:
        suggested_position = upper_pct * float(np.clip(composite / 75.0, 0.0, 1.0))

    market_confirms = bool(
        technical.get("available")
        and float(technical.get("score") or 0) >= 55
    )
    event_summary = "资讯不可用"
    if event.get("available"):
        positive_confirm = sum(
            1 for item in event["events"] if "正向确认" in str(item.get("reaction", ""))
        )
        negative_confirm = sum(
            1 for item in event["events"] if "风险确认" in str(item.get("reaction", ""))
        )
        if positive_confirm > negative_confirm:
            event_summary = "市场实际反应整体确认正面事件"
        elif negative_confirm > positive_confirm:
            event_summary = "市场实际反应确认风险事件"
        else:
            event_summary = "市场反应与事件方向存在分歧或钝化"

    current_state_bits = []
    if context is not None:
        from historical_context import current_state_display

        state_display = current_state_display(context)
        current_state_bits.append(f"趋势：{state_display.get('trend', '数据不足')}")
        current_state_bits.append(f"位置：{state_display.get('position', '数据不足')}")
    if event.get("events"):
        latest_event = next((item for item in event["events"] if item.get("reaction")), None)
        if latest_event:
            current_state_bits.append(f"最近重要事件：{str(latest_event.get('title'))[:36]}（{latest_event['reaction']}）")

    historical_block = {
        "available": historical_available,
        "reason": similarity.reason,
        "sample_count": len(similarity.selected),
        "selection_mode": similarity.selection_mode,
        "selection_threshold": similarity.selection_threshold,
        "candidate_count": similarity.candidate_count,
        "horizons": outcomes.get("horizons", []),
        "direction_summary": outcomes.get("direction_summary"),
        "direction_strength": outcomes.get("direction_strength"),
        "path_report": outcomes.get("path_report", []),
        "matches": [
            {
                "anchor_date": pd.Timestamp(item["anchor_date"]).date().isoformat(),
                "similarity": float(item["similarity"]),
            }
            for item in similarity.selected
        ],
        "guard": guard_summary(boundary) if boundary else "",
        "notes": list(similarity.notes) + list(outcomes.get("notes", [])),
    }

    decision: dict[str, Any] = {
        "engine_version": DECISION_ENGINE_VERSION,
        "available": any(modules_available.values()),
        "composite_score": float(round(composite, 2)),
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "risk_level": analysis.get("stock_risk_level"),
        "confidence": float(round(confidence, 1)),
        "suggested_position_pct": float(round(suggested_position, 4)),
        "weights": weights,
        "module_scores": module_scores,
        "module_names": MODULE_NAMES,
        "modules_available": modules_available,
        "veto": {"factor": veto_factor, "reasons": veto_reasons},
        "evidence": {
            "level": evidence.level,
            "evidence_score": float(evidence.evidence_score),
            "similarity_score": float(evidence.similarity_score),
            "reliability_score": float(evidence.reliability_score),
            "data_quality_score": float(evidence.data_quality_score),
            "sample_count": evidence.sample_count,
            "reasons": evidence.reasons,
            "components": evidence.components,
            "unavailable_dimensions": evidence.unavailable_dimensions,
        },
        "historical": historical_block,
        "event_reaction": event,
        "technical_patterns": technical,
        "legacy_factor": {
            "score": legacy_score,
            "horizon_name": selected.get("name"),
            "fundamental_score": fundamental_score,
            "macro_score": macro_score,
            "deterioration_count": deterioration_count,
            "cross_check": cross_check,
        },
        "current_state": "；".join(current_state_bits) if current_state_bits else "数据不足",
        "market_confirms": market_confirms,
        "suitability_fit": suitability_fit,
        "data_confidence": data_confidence,
    }
    decision["narrative"] = _build_narrative(decision)
    return _py(decision)
