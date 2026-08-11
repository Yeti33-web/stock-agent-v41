from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent_core import (
    EvidenceSnapshot,
    analyze_all,
    calculate_amount_holding_values,
    calculate_holding_values,
    fetch_a_fundamentals,
    fetch_macro_snapshot,
    fetch_price_bundle,
    fetch_usd_cny_rate,
    fetch_us_fundamentals,
    normalize_a_code,
    normalize_us_code,
    score_investor,
)
from questionnaire import (
    QUESTIONS,
    answers_complete,
    answers_to_profile,
    compose_analysis_profile,
    first_unanswered_index,
    public_profile_rows,
)


st.set_page_config(
    page_title="个人投资者股票决策辅助 Agent V5.4",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {--brand:#1d4ed8; --ink:#172033; --muted:#667085; --line:#e4e7ec; --soft:#f6f8fb;}
    .stApp {background:linear-gradient(180deg,#f8fbff 0,#ffffff 360px); color:var(--ink);}
    .block-container {padding-top:1.35rem; padding-bottom:3rem; max-width:1220px;}
    .app-brand {font-size:.82rem; font-weight:700; color:#1d4ed8; letter-spacing:.08em; text-transform:uppercase;}
    .hero-card {padding:1.35rem 1.5rem; border:1px solid #dbe5f3; border-radius:20px; background:rgba(255,255,255,.92); margin:.7rem 0 1.2rem; box-shadow:0 12px 30px rgba(29,78,216,.06);}
    .question-card {padding:1.8rem; border:1px solid #dbe5f3; border-radius:22px; background:white; box-shadow:0 18px 50px rgba(16,24,40,.07); margin:1rem 0;}
    .result-card {padding:1.1rem 1.25rem; border-radius:16px; background:#f7f9fc; border-left:5px solid #2563eb;}
    .muted {color:var(--muted); font-size:.92rem;}
    div[data-testid="stMetric"] {border:1px solid #e4eaf2; padding:.9rem; border-radius:14px; background:white; box-shadow:0 5px 15px rgba(16,24,40,.035);}
    div.stButton > button {border-radius:12px; min-height:2.8rem;}
    div[data-testid="stForm"] {border:1px solid #e4eaf2; border-radius:18px; padding:1.25rem; background:white;}
    [data-testid="stSidebar"] {background:#f7f9fc;}
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_confirmed_stock() -> tuple[str, str]:
    market = st.session_state.confirmed_market
    code = st.session_state.confirmed_stock_code
    if not market or not code:
        st.error("尚未确认本次要分析的股票，请返回股票分析页重新选择。")
        if st.button("返回选择股票"):
            st.session_state.view = "analysis"
            st.rerun()
        st.stop()
    return str(market), str(code)


def validate_code(market: str, code: str) -> str:
    return normalize_a_code(code) if market == "A股" else normalize_us_code(code)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_price_bundle(market: str, code: str, request_token: int):
    del request_token
    return fetch_price_bundle(market, code)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamentals(market: str, code: str, last_price: float, asset_type: str) -> EvidenceSnapshot:
    if market == "A股":
        return fetch_a_fundamentals(code, last_price, asset_type)
    return fetch_us_fundamentals(code, last_price)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_macro(market: str, benchmark: pd.DataFrame) -> EvidenceSnapshot:
    return fetch_macro_snapshot(market, benchmark)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_usd_cny_rate() -> dict:
    return fetch_usd_cny_rate()


def initialize_v5_state() -> None:
    defaults = {
        "view": "analysis",
        "user_data_loaded": False,
        "profile_record": None,
        "saved_profile": None,
        "draft_record": None,
        "question_answers": {},
        "question_index": 0,
        "questionnaire_mode": "first",
        "confirm_profile_change": False,
        "confirmed_market": None,
        "confirmed_stock_code": None,
        "confirmed_holding_state": "尚未持有",
        "confirmed_holding_method": "按持股数量填写",
        "confirmed_share_count": 0.0,
        "confirmed_cost_price": 0.0,
        "confirmed_current_market_value": 0.0,
        "confirmed_total_cost": 0.0,
        "confirmed_additional_amount": 0.0,
        "holding_snapshot": None,
        "analysis_request_token": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.view == "analysis":
        widget_defaults = {
            "v5_market": "A股",
            "v5_stock_code": "",
            "v5_holding_state": "尚未持有",
            "v5_holding_method": "按持股数量填写",
            "v5_share_count": 0.0,
            "v5_cost_price": 0.0,
            "v5_current_value": 0.0,
            "v5_total_cost": 0.0,
            "v5_planned_action": "仅分析现有持仓",
            "v5_additional_amount": 0.0,
            "v5_planned_amount": 50_000.0,
            "v5_leverage": "否",
        }
        for key, value in widget_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value


def _session_load_profile() -> dict | None:
    return st.session_state.get("dev_profile_record")


def _session_load_draft() -> dict | None:
    return st.session_state.get("dev_draft_record")


def persist_draft(answers: dict, current_index: int) -> None:
    st.session_state.dev_draft_record = {
        "answers": dict(answers),
        "current_index": int(current_index),
        "updated_at": datetime.now().isoformat(),
    }


def remove_draft() -> None:
    st.session_state.pop("dev_draft_record", None)


def persist_profile(profile: dict, risk_score: int, risk_level: str, version: int) -> dict:
    record = {
        "profile_data": dict(profile),
        "risk_score": int(risk_score),
        "risk_level": risk_level,
        "version": int(version) + 1,
        "completed_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    st.session_state.dev_profile_record = record
    return record


def load_current_user_data() -> None:
    if st.session_state.user_data_loaded:
        return
    profile_record = _session_load_profile()
    draft_record = _session_load_draft()
    st.session_state.profile_record = profile_record
    st.session_state.saved_profile = profile_record.get("profile_data") if profile_record else None
    st.session_state.draft_record = draft_record
    st.session_state.question_answers = dict((draft_record or {}).get("answers") or {})
    st.session_state.question_index = int((draft_record or {}).get("current_index") or first_unanswered_index(st.session_state.question_answers))
    if not profile_record:
        st.session_state.questionnaire_mode = "first"
        st.session_state.view = "questionnaire"
    st.session_state.user_data_loaded = True


def render_brand(subtitle: str = "") -> None:
    st.markdown('<div class="app-brand">Five-year evidence · Personal suitability</div>', unsafe_allow_html=True)
    st.title("个人投资者股票决策辅助 Agent｜三位精度版 V5.4")
    st.caption(subtitle or "近五年真实公开行情 · 历史相似状态检索 · 个人风险适配 · 教学研究原型")


def question_option_label(option: str, current: str | None) -> str:
    return f"✓ 当前答案　{option}" if option == current else option


def questionnaire_page() -> None:
    answers = dict(st.session_state.question_answers)
    index = int(st.session_state.question_index)
    total = len(QUESTIONS)
    render_brand("本次网页会话首次测评提交后，日常换股不会再次要求填写")
    if index < total:
        question = QUESTIONS[index]
        st.progress((index + 1) / total, text=f"第 {index + 1} / {total} 题")
        st.markdown(
            f'<div class="question-card"><div class="muted">个人风险测评</div><h2>{question["title"]}</h2><p class="muted">{question["hint"]}</p></div>',
            unsafe_allow_html=True,
        )
        current_answer = answers.get(question["key"])
        for option_index, option in enumerate(question["options"]):
            if st.button(
                question_option_label(str(option), current_answer),
                key=f"answer_{index}_{option_index}",
                width="stretch",
                type="primary" if option == current_answer else "secondary",
            ):
                updated = dict(answers)
                updated[str(question["key"])] = str(option)
                next_index = index + 1
                persist_draft(updated, next_index)
                st.session_state.question_answers = updated
                st.session_state.question_index = next_index
                st.rerun()
        st.divider()
        if st.button("← 返回上一页", disabled=index == 0, key=f"question_back_{index}"):
            st.session_state.question_index = max(0, index - 1)
            st.rerun()
        return

    st.progress(1.0, text=f"已完成 {total} / {total} 题")
    st.subheader("确认并提交风险测评")
    if not answers_complete(answers):
        st.warning("仍有问题未完成，系统将返回第一道未完成的问题。")
        if st.button("返回继续填写", type="primary"):
            st.session_state.question_index = first_unanswered_index(answers)
            st.rerun()
        return
    review = pd.DataFrame(
        [[item["title"], answers[item["key"]]] for item in QUESTIONS],
        columns=["问题", "你的选择"],
    )
    st.dataframe(review, hide_index=True, width="stretch")
    back, submit = st.columns([1, 2])
    if back.button("← 返回上一页", width="stretch"):
        st.session_state.question_index = total - 1
        st.rerun()
    if submit.button("确认提交并生成风险等级", type="primary", width="stretch"):
        try:
            profile = answers_to_profile(answers)
            risk_score, _, risk_level, _, _ = score_investor(profile)
            version = int((st.session_state.profile_record or {}).get("version") or 0)
            record = persist_profile(profile, risk_score, risk_level, version)
            remove_draft()
            st.session_state.profile_record = record
            st.session_state.saved_profile = profile
            st.session_state.draft_record = None
            st.session_state.question_answers = {}
            st.session_state.question_index = 0
            st.session_state.view = "analysis"
            st.session_state.confirm_profile_change = False
            st.rerun()
        except ValueError as exc:
            st.error(f"提交失败：{exc}")


def v5_market_changed() -> None:
    st.session_state.v5_stock_code = ""
    st.session_state.v5_share_count = 0.0
    st.session_state.v5_cost_price = 0.0
    st.session_state.v5_current_value = 0.0
    st.session_state.v5_total_cost = 0.0
    st.session_state.v5_additional_amount = 0.0


def analysis_home() -> None:
    profile = st.session_state.saved_profile
    record = st.session_state.profile_record or {}
    render_brand("风险资料已在本次网页会话中生效；日常只需更换股票并填写本次投资信息")
    risk_cols = st.columns([1, 1, 1.4])
    risk_cols[0].metric("个人风险等级", record.get("risk_level", "—"), f"{record.get('risk_score', '—')}/100")
    risk_cols[1].metric("测评版本", f"第 {record.get('version', 1)} 版")
    risk_cols[2].metric("资产范围", profile.get("asset_band", "—"))
    st.markdown('<div class="hero-card"><b>选择本次要分析的股票</b><br><span class="muted">Agent统一使用最近五年行情，识别当前波动状态并检索历史相似周期；上市不足五年时使用全部可得数据并降低结论置信度。</span></div>', unsafe_allow_html=True)
    st.caption("提示：本版不注册账号。关闭网页、清除浏览器数据或更换设备后，需要重新完成风险测评。")

    stock_left, stock_right = st.columns([1, 1])
    with stock_left:
        st.radio("市场", ["A股", "美股"], horizontal=True, key="v5_market", on_change=v5_market_changed)
        placeholder = "例如：600519、300750" if st.session_state.v5_market == "A股" else "例如：AAPL、MSFT、NVDA"
        st.text_input("股票代码", key="v5_stock_code", placeholder=placeholder)
    with stock_right:
        st.radio("目前是否已经持有？", ["尚未持有", "已经持有"], horizontal=True, key="v5_holding_state")
        st.radio("本次是否计划使用融资或其他杠杆？", ["否", "是"], horizontal=True, key="v5_leverage")

    st.markdown("#### 本次持仓／投资信息")
    if st.session_state.v5_holding_state == "尚未持有":
        st.number_input(
            "本次计划买入金额（人民币元）",
            min_value=1000.0,
            step=1000.0,
            format="%.3f",
            key="v5_planned_amount",
            help="尚未持有时，只需填写本次拟投入金额。",
        )
    else:
        st.radio(
            "持仓信息填写方式",
            ["按持股数量填写", "按持仓金额填写"],
            horizontal=True,
            key="v5_holding_method",
        )
        holding_left, holding_right = st.columns(2)
        if st.session_state.v5_holding_method == "按持股数量填写":
            unit = "元/股" if st.session_state.v5_market == "A股" else "美元/股"
            share_step = 100.0 if st.session_state.v5_market == "A股" else 1.0
            with holding_left:
                st.number_input(
                    "持股数量（股）",
                    min_value=0.0,
                    step=share_step,
                    format="%.0f",
                    key="v5_share_count",
                )
            with holding_right:
                st.number_input(
                    f"平均持仓成本价（{unit}，可填0表示未知）",
                    min_value=0.0,
                    step=0.001,
                    format="%.3f",
                    key="v5_cost_price",
                )
            st.info("Agent会自动获取最新公开价格，并计算当前持仓市值；填写成本价后还会计算总成本和浮动盈亏。")
        else:
            with holding_left:
                st.number_input(
                    "当前持仓市值（折合人民币元）",
                    min_value=0.0,
                    step=1000.0,
                    format="%.3f",
                    key="v5_current_value",
                )
            with holding_right:
                st.number_input(
                    "累计投入成本（人民币元，可填0表示未知）",
                    min_value=0.0,
                    step=1000.0,
                    format="%.3f",
                    key="v5_total_cost",
                )
        st.radio(
            "本次计划",
            ["仅分析现有持仓", "还计划加仓"],
            horizontal=True,
            key="v5_planned_action",
        )
        if st.session_state.v5_planned_action == "还计划加仓":
            st.number_input(
                "计划新增金额（人民币元）",
                min_value=0.0,
                step=1000.0,
                format="%.3f",
                key="v5_additional_amount",
            )

    if st.button("获取近五年真实数据并分析", type="primary", width="stretch"):
        try:
            market = st.session_state.v5_market
            holding_state = st.session_state.v5_holding_state
            holding_method = st.session_state.v5_holding_method
            code = validate_code(market, st.session_state.v5_stock_code)
            share_count = float(st.session_state.get("v5_share_count", 0.0))
            cost_price = float(st.session_state.get("v5_cost_price", 0.0))
            current_value = float(st.session_state.get("v5_current_value", 0.0))
            total_cost = float(st.session_state.get("v5_total_cost", 0.0))
            additional = (
                float(st.session_state.get("v5_additional_amount", 0.0))
                if holding_state == "已经持有" and st.session_state.get("v5_planned_action") == "还计划加仓"
                else 0.0
            )
            if holding_state == "尚未持有":
                planned = float(st.session_state.v5_planned_amount)
                if planned <= 0:
                    raise ValueError("本次计划买入金额必须大于0。")
            elif holding_method == "按持股数量填写":
                if share_count <= 0:
                    raise ValueError("你选择了已经持有，请填写大于0的持股数量。")
                planned = max(additional, 1.0)
            else:
                if current_value <= 0:
                    raise ValueError("你选择了已经持有，请填写大于0的当前持仓市值。")
                planned = current_value + additional
            if holding_state == "已经持有" and st.session_state.get("v5_planned_action") == "还计划加仓" and additional <= 0:
                raise ValueError("你选择了还计划加仓，请填写大于0的计划新增金额。")
            asset_upper = profile.get("asset_upper")
            if holding_state == "尚未持有" and asset_upper is not None and planned > float(asset_upper):
                raise ValueError("本次计划金额高于风险问卷所选资产范围上限，请先到个人中心更新资料。")
            st.session_state.confirmed_market = market
            st.session_state.confirmed_stock_code = code
            st.session_state.confirmed_holding_state = holding_state
            st.session_state.confirmed_holding_method = holding_method
            st.session_state.confirmed_share_count = share_count
            st.session_state.confirmed_cost_price = cost_price
            st.session_state.confirmed_current_market_value = current_value
            st.session_state.confirmed_total_cost = total_cost
            st.session_state.confirmed_additional_amount = additional
            st.session_state.holding_snapshot = None
            st.session_state.profile = compose_analysis_profile(profile, planned, st.session_state.v5_leverage)
            st.session_state.analysis_request_token += 1
            st.session_state.view = "result"
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def personal_center() -> None:
    profile = st.session_state.saved_profile
    record = st.session_state.profile_record or {}
    render_brand("个人中心")
    cols = st.columns(4)
    cols[0].metric("当前风险等级", record.get("risk_level", "—"))
    cols[1].metric("风险分", f"{record.get('risk_score', '—')}/100")
    cols[2].metric("资料版本", f"第 {record.get('version', 1)} 版")
    updated = str(record.get("updated_at") or record.get("completed_at") or "")[:10]
    cols[3].metric("最近更新", updated or "—")
    st.subheader("本次网页会话的风险资料")
    st.dataframe(pd.DataFrame(public_profile_rows(profile), columns=["项目", "当前选择"]), hide_index=True, width="stretch")

    if st.session_state.draft_record:
        st.info("你有一份尚未提交的新测评草稿。旧资料仍然有效。")
        continue_col, discard_col = st.columns(2)
        if continue_col.button("继续填写草稿", width="stretch"):
            st.session_state.question_answers = dict(st.session_state.draft_record.get("answers") or {})
            st.session_state.question_index = int(st.session_state.draft_record.get("current_index") or 0)
            st.session_state.questionnaire_mode = "update"
            st.session_state.view = "questionnaire"
            st.rerun()
        if discard_col.button("放弃草稿", width="stretch"):
            remove_draft()
            st.session_state.draft_record = None
            st.rerun()

    st.divider()
    if st.button("更改个人信息", type="primary"):
        st.session_state.confirm_profile_change = True
    if st.session_state.confirm_profile_change:
        st.warning("重新填写期间旧资料继续有效；只有完整提交新问卷后才会替换当前资料。")
        cancel, confirm = st.columns(2)
        if cancel.button("取消", width="stretch"):
            st.session_state.confirm_profile_change = False
            st.rerun()
        if confirm.button("确认重置并重新测评", type="primary", width="stretch"):
            persist_draft({}, 0)
            st.session_state.question_answers = {}
            st.session_state.question_index = 0
            st.session_state.questionnaire_mode = "update"
            st.session_state.view = "questionnaire"
            st.session_state.confirm_profile_change = False
            st.rerun()


def clear_stock_widgets() -> None:
    for key in [
        "v5_market",
        "v5_stock_code",
        "v5_holding_state",
        "v5_holding_method",
        "v5_share_count",
        "v5_cost_price",
        "v5_current_value",
        "v5_total_cost",
        "v5_planned_action",
        "v5_additional_amount",
        "v5_planned_amount",
        "v5_leverage",
    ]:
        st.session_state.pop(key, None)
    st.session_state.view = "analysis"


def app_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📊 股票决策辅助 Agent")
        record = st.session_state.profile_record or {}
        st.caption(record.get("risk_level", "风险资料未完成"))
        if st.button("股票分析", width="stretch"):
            st.session_state.view = "analysis"
            st.rerun()
        if st.button("个人中心", width="stretch"):
            st.session_state.view = "profile"
            st.rerun()
        st.divider()
        st.caption("行情统一使用最近五年；相似样本不足时拒绝预测；新股标注低置信度。")


DISPLAY_DECIMALS = 3


def pct(value: float | None, digits: int = DISPLAY_DECIMALS) -> str:
    if value is None or pd.isna(value):
        return "数据不足"
    return f"{value:.{digits}%}"


def number(value: float | None, digits: int = DISPLAY_DECIMALS) -> str:
    if value is None or pd.isna(value):
        return "数据不足"
    return f"{value:.{digits}f}"


def money(value: float, unit: str = "元", signed: bool = False) -> str:
    parsed = float(value)
    formatted = f"{parsed:+,.3f}" if signed else f"{parsed:,.3f}"
    return f"{formatted} {unit}"


def format_field(name: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "未取得"
    if name in {"净资产收益率", "净利率", "净利润同比", "营收同比", "资产负债率", "经营现金流／净利润", "债务／权益"}:
        parsed = float(value)
        parsed = parsed / 100 if abs(parsed) > 5 else parsed
        return f"{parsed:.3%}"
    if name in {"市盈率TTM", "市净率"}:
        return f"{float(value):.3f}"
    if name == "总市值":
        return f"{float(value):,.3f}"
    return str(value)


def conclusion_box(conclusion: str, reason: str) -> None:
    text = f"### {conclusion}\n{reason}"
    if conclusion.startswith("不适合") or conclusion.startswith("证据不足"):
        st.error(text)
    elif "暂缓" in conclusion:
        st.warning(text)
    elif "观察" in conclusion:
        st.info(text)
    else:
        st.success(text)


def render_price_charts(bundle, analysis) -> None:
    metrics = analysis["metrics"]
    close = metrics["close"]
    chart_start = close.index.max() - pd.DateOffset(years=5)
    display_close = close[close.index >= chart_start]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=display_close.index,
            y=display_close,
            name="收盘价",
            line=dict(width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价 %{y:,.3f}<extra></extra>",
        )
    )
    for window, color in [(20, "#f59e0b"), (60, "#16a34a"), (250, "#7c3aed")]:
        if len(close) >= window:
            moving = close.rolling(window).mean()
            moving = moving[moving.index >= chart_start]
            fig.add_trace(
                go.Scatter(
                    x=moving.index,
                    y=moving,
                    name=f"MA{window}",
                    line=dict(width=1.2, color=color),
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>MA{window} %{{y:,.3f}}<extra></extra>",
                )
            )
    fig.update_layout(
        title="最近五年或全部可得价格",
        height=390,
        margin=dict(l=20, r=20, t=55, b=20),
        yaxis_title=bundle.price_unit,
        yaxis_tickformat=",.3f",
        legend_orientation="h",
    )
    st.plotly_chart(fig, width="stretch")

    drawdown = metrics["drawdown"]
    drawdown_fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", line=dict(color="#dc2626"), name="回撤"))
    drawdown_fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>回撤 %{y:.3%}<extra></extra>")
    drawdown_fig.update_layout(title="近五年或上市以来回撤", height=340, margin=dict(l=20, r=20, t=55, b=20), yaxis_tickformat=".3%")
    st.plotly_chart(drawdown_fig, width="stretch")


def render_summary(bundle, analysis, profile) -> None:
    selected = analysis["selected_horizon"]
    conclusion_box(analysis["conclusion"], analysis["conclusion_reason"])
    columns = st.columns(5)
    columns[0].metric("用户风险等级", analysis["investor_level"], f"{analysis['investor_score']}/100")
    columns[1].metric("用户类型", analysis["style"])
    columns[2].metric("股票风险等级", analysis["stock_risk_level"], f"{analysis['stock_risk_score']}/100")
    columns[3].metric("个人适配", analysis["suitability"]["fit"])
    columns[4].metric("数据完整度", f"{float(analysis['data_confidence']):.3f}%")

    analog = analysis.get("analog_forecast") or {}
    st.markdown("#### 历史相似状态情景")
    analog_cols = st.columns([1.6, 1, 1])
    analog_cols[0].metric("当前状态", analog.get("state", {}).get("summary", "无法识别"))
    analog_cols[1].metric(
        "相似周期可信度",
        analog.get("confidence_label", "样本不足"),
        f"{analog.get('confidence_score', 0)}/100",
    )
    best_similarity = analog.get("best_similarity")
    analog_cols[2].metric("最高相似度", f"{best_similarity:.3f}/100" if best_similarity is not None else "无有效样本")
    selected_analog = None
    if selected:
        selected_analog = next(
            (
                item
                for item in analog.get("horizons", [])
                if item.get("available") and int(item.get("days", -1)) == int(selected["days"])
            ),
            None,
        )
    if selected_analog:
        st.info(
            f"与当前状态相似的{selected_analog['sample_count']}个历史样本中，随后"
            f"{selected_analog['days']}个交易日上涨样本占比为{selected_analog['positive_ratio']:.3%}，"
            f"收益中位数{selected_analog['median_return']:.3%}，"
            f"中间50%区间{selected_analog['q25_return']:.3%}至{selected_analog['q75_return']:.3%}。"
        )
    elif not analog.get("available"):
        st.warning("有效相似样本不足，Agent没有生成方向预测；其他风险分析仍可继续查看。")
    st.caption("这里展示的是历史样本频率和情景分布，不是确定上涨概率，也不是收益承诺。")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Agent判断的持有／复核周期")
        if selected:
            st.markdown(f"**{selected['name']}** · {selected['review']}")
            st.write(f"当前时点评分：**{selected['score']}/100（{selected['label']}）**。该分数不是上涨概率。")
            for note in analysis["horizon_notes"]:
                st.caption(f"• {note}")
        else:
            st.write("历史数据不足，无法选择周期。")
    with right:
        position = analysis["position"]
        if st.session_state.confirmed_holding_state == "已经持有":
            st.markdown("#### 现有仓位与新增风险预算")
            st.write(f"模型建议的该股票**总仓位上限**约为 **{money(position['upper_amount'])}**。")
            if position["remaining_upper_amount"] > 0:
                st.markdown(f"扣除现有持仓后，**新增风险预算参考上限约 {money(position['remaining_upper_amount'])}**。")
            else:
                st.markdown("**当前新增风险预算：0.000 元**")
                st.write("这不表示你的实际仓位是0，而是模型当前不建议继续增加该股票的风险敞口。")
        else:
            st.markdown("#### 风险预算参考上限")
            if position["upper_pct"] > 0:
                st.markdown(f"**可投资金融资产的 {position['lower_pct']:.3%}—{position['upper_pct']:.3%}**")
                st.write(f"按你填写的资产口径，对应约 **{position['lower_amount']:,.3f}—{position['upper_amount']:,.3f} 元**。")
                if profile["planned_amount"] > position["upper_amount"]:
                    st.warning(f"你计划的 {money(profile['planned_amount'])}高于模型风险预算上限，主要问题是集中度，而不是股票一定不会上涨。")
            else:
                st.markdown("**当前新增风险预算：0.000%**")
                st.write("原因来自个人适配、安全限制、数据不足或当前时点偏弱；详情见下方。")

    positives = []
    risks = list(analysis["risk_reasons"])
    if selected:
        positives.extend(selected["reasons"])
    positives.extend(analysis["fundamental"].positives)
    risks.extend(analysis["fundamental"].risks)
    risks.extend(analysis["profile_flags"])
    pos_col, risk_col = st.columns(2)
    with pos_col:
        st.markdown("#### 支持因素")
        if positives:
            for item in list(dict.fromkeys(positives))[:6]:
                st.write(f"- {item}")
        else:
            st.write("目前没有形成明确支持因素。")
    with risk_col:
        st.markdown("#### 风险与限制")
        for item in list(dict.fromkeys(risks))[:7]:
            st.write(f"- {item}")

    if st.session_state.confirmed_holding_state == "已经持有":
        st.markdown("#### 已有持仓检查")
        holding = analysis.get("holding_snapshot") or {}
        current_value = float(holding.get("current_rmb") or 0.0)
        current_ratio = current_value / profile["investable_assets"] if profile["investable_assets"] else 0
        current_return = holding.get("return_rate")
        total_cost_rmb = holding.get("cost_total_rmb")
        profit_rmb = holding.get("profit_rmb")
        held_cols = st.columns(4)
        held_cols[0].metric("当前仓位占比", f"{current_ratio:.3%}")
        held_cols[1].metric("当前持仓市值", money(current_value))
        held_cols[2].metric("累计投入成本", money(total_cost_rmb) if total_cost_rmb is not None else "成本未知")
        held_cols[3].metric("持仓收益率", pct(current_return) if current_return is not None else "成本未知")

        if holding.get("method") == "按持股数量填写":
            shares = float(holding["shares"])
            native_value = float(holding["current_native"])
            st.write(
                f"市值计算：**{shares:,.0f} 股 × {float(holding['latest_price']):,.3f} {bundle.price_unit}"
                f" = {native_value:,.3f} {holding['native_currency']}**。"
            )
            if holding.get("cost_price") is not None:
                st.caption(
                    f"成本计算：{shares:,.0f} 股 × {float(holding['cost_price']):,.3f} {bundle.price_unit}"
                    f" = {float(holding['cost_total_native']):,.3f} {holding['native_currency']}。"
                )
            if holding.get("usd_cny_rate") is not None:
                st.caption(
                    f"人民币市值按 1 美元 = {float(holding['usd_cny_rate']):.3f} 元换算；"
                    f"汇率日期 {holding.get('fx_date', '—')}，来源：{holding.get('fx_provider', '公开汇率接口')}。"
                )
        else:
            st.caption("当前持仓市值由用户按人民币金额填写；股票最新公开价格仍用于行情与风险分析。")
        if profit_rmb is not None:
            st.write(f"按所填成本估算的浮动盈亏：**{money(profit_rmb, signed=True)}**。")
        else:
            st.info("你没有填写成本信息，因此Agent不会编造收益率或盈亏数值。")
        st.caption("当前仓位是根据你提供的持仓计算出的事实数据；模型新增风险预算是风险控制参考，两者不是同一个数值。")
        if current_ratio > analysis["position"]["upper_pct"] and analysis["position"]["upper_pct"] > 0:
            st.warning("当前仓位已经高于模型风险预算上限。这里提示的是集中度风险，不等同于要求立即卖出。")
        elif analysis["position"]["upper_pct"] == 0 and current_value > 0:
            st.warning("模型当前给出的新增风险预算为0；你的实际持仓仍按上方市值和仓位显示，并没有被当成0。")


def render_risk_budget(analysis, profile) -> None:
    st.subheader("个人适配与风险预算")
    st.write(f"**适配结论：{analysis['suitability']['fit']}。** {analysis['suitability']['fit_reason']}。")
    if analysis["suitability"]["hard_reasons"]:
        for reason in analysis["suitability"]["hard_reasons"]:
            st.error(reason)
    if st.session_state.confirmed_holding_state == "已经持有":
        amount_rows = [
            ["当前持仓市值", money(profile.get("current_holding_value", 0))],
            ["本次计划新增", money(profile.get("additional_amount", 0))],
            ["分析后的总风险敞口", money(profile["planned_amount"])],
        ]
    else:
        amount_rows = [["本次计划买入金额", money(profile["planned_amount"])]]
    table = pd.DataFrame(
        amount_rows
        + [
            ["可投资金融资产", profile.get("asset_band", f"约 {money(profile['investable_assets'])}")],
            ["总风险敞口占比（按资产区间代表值估算）", f"{profile['planned_amount'] / profile['investable_assets']:.3%}"],
            ["资金来源", profile["fund_source"]],
            ["最早用款时间", profile["earliest_need"]],
            ["用户风险等级", analysis["investor_level"]],
            ["股票风险等级", analysis["stock_risk_level"]],
        ],
        columns=["检查项目", "结果"],
    )
    st.dataframe(table, hide_index=True, width="stretch")
    selected = analysis["selected_horizon"]
    if selected:
        stress = analysis["position"]["stress_loss"]
        st.markdown("#### 历史压力参考")
        stress_cols = st.columns(3)
        stress_cols[0].metric("相同周期较差情景（5%分位）", pct(-stress) if stress is not None else "数据不足")
        stress_cols[1].metric("相同周期历史最差一次", pct(selected.get("historical_worst")))
        stress_cols[2].metric("全部历史最大回撤", pct(analysis["metrics"]["max_drawdown"]))
        st.info("近五年或上市以来的较差情景用于压力提示，不做一票否决。仓位上限由压力情景、用户风险预算和集中度共同计算。")


def render_horizons(analysis) -> None:
    st.subheader("Agent如何选择持有周期")
    rows = []
    selected_name = analysis["selected_horizon"]["name"] if analysis["selected_horizon"] else ""
    for item in analysis["horizon_scores"]:
        rows.append(
            {
                "周期": item["name"],
                "当前时点评分": f"{item['score']}/100" if item["score"] is not None else "—",
                "相似周期修正": (
                    f"{item.get('analog_adjustment', 0):+.1f}分"
                    if item.get("analog_used")
                    else "可信度不足" if item.get("analog_evidence") is not None else "无有效样本"
                ),
                "状态": item["label"],
                "是否选中": "✓" if item["name"] == selected_name else "",
            }
        )
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, width="stretch")
    available = [item for item in analysis["horizon_scores"] if item["score"] is not None]
    if available:
        colors = ["#2563eb" if item["name"] == selected_name else "#94a3b8" for item in available]
        fig = go.Figure(go.Bar(x=[item["name"] for item in available], y=[item["score"] for item in available], marker_color=colors))
        fig.add_hline(y=42, line_dash="dot", annotation_text="偏弱/观察分界")
        fig.add_hline(y=56, line_dash="dot", annotation_text="中性偏积极分界")
        fig.update_layout(height=360, yaxis_range=[0, 100], yaxis_title="当前时点评分", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, width="stretch")
    st.warning("1个交易日需要分钟级或实时行情、交易成本与盘口信息。本版只有公开日线，因此会显示该周期但不会假装给出可靠次日预测。")


def analog_horizon_rows(forecast: dict) -> list[dict]:
    rows: list[dict] = []
    for item in forecast.get("horizons", []):
        rows.append(
            {
                "观察期限": f"后续{item['days']}个交易日",
                "有效样本": item.get("sample_count", 0),
                "历史上涨样本占比": pct(item.get("positive_ratio")) if item.get("available") else "样本不足",
                "收益中位数": pct(item.get("median_return")) if item.get("available") else "—",
                "中间50%区间": (
                    f"{pct(item.get('q25_return'))} 至 {pct(item.get('q75_return'))}"
                    if item.get("available")
                    else "—"
                ),
                "10%较差分位": pct(item.get("q10_return")) if item.get("available") else "—",
                "期间最深浮亏中位数": pct(item.get("median_worst_loss")) if item.get("available") else "—",
                "情景判断": item.get("direction", "—"),
                "可信度": f"{item.get('confidence_score', 0)}/100" if item.get("available") else "—",
            }
        )
    return rows


def render_analog_forecast(analysis) -> None:
    forecast = analysis.get("analog_forecast") or {}
    st.subheader("历史相似周期与未来情景预测")
    st.info(
        "Agent先识别当前趋势、波动、回撤、成交量、相对基准和市场状态，再从最近五年中寻找相似且尽量分散的历史窗口，统计这些窗口之后的实际表现。"
    )
    state_cols = st.columns(4)
    state = forecast.get("state", {})
    state_cols[0].metric("趋势状态", state.get("trend", "数据不足"))
    state_cols[1].metric("波动状态", state.get("volatility", "数据不足"))
    state_cols[2].metric("回撤状态", state.get("drawdown", "数据不足"))
    state_cols[3].metric("市场状态", state.get("market", "数据不足"))

    current_features = forecast.get("current_features") or {}
    if current_features:
        feature_rows = []
        for name, value in current_features.items():
            display = number(value) if "成交量比" in name else pct(value)
            feature_rows.append([name, display])
        with st.expander("查看Agent用于匹配的当前状态特征"):
            st.dataframe(pd.DataFrame(feature_rows, columns=["当前特征", "数值"]), hide_index=True, width="stretch")

    rows = analog_horizon_rows(forecast)
    if rows:
        st.markdown("#### 不同期限的历史后续表现")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if not forecast.get("available"):
        st.warning("有效相似周期少于最低样本要求，Agent拒绝形成方向预测。")

    available_horizons = [item for item in forecast.get("horizons", []) if item.get("available")]
    if available_horizons:
        distribution = go.Figure()
        for item in available_horizons:
            distribution.add_trace(
                go.Box(
                    y=item["outcomes"],
                    name=f"{item['days']}日",
                    boxpoints="outliers",
                    marker_color="#2563eb",
                    hovertemplate="收益 %{y:.3%}<extra>%{fullData.name}</extra>",
                )
            )
        distribution.add_hline(y=0, line_dash="dot", line_color="#667085")
        distribution.update_layout(
            title="相似周期后续收益分布",
            height=390,
            yaxis_tickformat=".3%",
            yaxis_title="从相似时点开始计算的后续收益",
            margin=dict(l=20, r=20, t=55, b=20),
            showlegend=False,
        )
        st.plotly_chart(distribution, width="stretch")

        selected_days = analysis.get("selected_horizon", {}).get("days") if analysis.get("selected_horizon") else None
        path_horizon = next((item for item in available_horizons if item["days"] == selected_days), available_horizons[0])
        path_figure = go.Figure()
        sorted_paths = sorted(path_horizon.get("paths", []), key=lambda item: item["similarity"], reverse=True)[:6]
        for path in sorted_paths:
            path_figure.add_trace(
                go.Scatter(
                    x=list(range(len(path["values"]))),
                    y=path["values"],
                    mode="lines",
                    line=dict(width=1.2),
                    opacity=0.42,
                    name=f"{pd.Timestamp(path['anchor_date']).date()} · {path['similarity']:.3f}",
                    hovertemplate="后续第%{x}日<br>收益 %{y:.3%}<extra>%{fullData.name}</extra>",
                )
            )
        if sorted_paths:
            median_path = pd.DataFrame([path["values"] for path in sorted_paths]).median(axis=0)
            path_figure.add_trace(
                go.Scatter(
                    x=list(range(len(median_path))),
                    y=median_path,
                    mode="lines",
                    line=dict(width=4, color="#dc2626"),
                    name="相似路径中位数",
                )
            )
        path_figure.add_hline(y=0, line_dash="dot", line_color="#667085")
        path_figure.update_layout(
            title=f"最高相似样本之后{path_horizon['days']}个交易日的标准化路径",
            height=410,
            xaxis_title="后续交易日",
            yaxis_title="相对相似时点的收益",
            yaxis_tickformat=".3%",
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(path_figure, width="stretch")

    matches = forecast.get("matches") or []
    if matches:
        st.markdown("#### 最相似的历史窗口")
        match_rows = []
        for item in matches:
            match_rows.append(
                {
                    "状态观察起点": str(pd.Timestamp(item["start_date"]).date()),
                    "相似时点": str(pd.Timestamp(item["anchor_date"]).date()),
                    "相似度": f"{item['similarity']:.3f}/100",
                    "随后5日": pct(item.get("return_5")),
                    "随后20日": pct(item.get("return_20")),
                    "随后60日": pct(item.get("return_60")),
                    "随后120日": pct(item.get("return_120")),
                }
            )
        st.dataframe(pd.DataFrame(match_rows), hide_index=True, width="stretch")

    backtest = forecast.get("backtest") or {}
    st.markdown("#### 滚动回测检查")
    if backtest.get("available"):
        backtest_cols = st.columns(3)
        backtest_cols[0].metric("历史验证时点", f"{backtest['cases']}个")
        backtest_cols[1].metric("20日方向一致率", pct(backtest["direction_accuracy"]))
        backtest_cols[2].metric("预测中位数绝对误差", pct(backtest["median_absolute_error"]))
        st.caption(
            f"同期简单动量方向一致率：{pct(backtest['momentum_accuracy'])}。{backtest['note']}"
            "回测结果仅用于检验规则，不能保证未来表现。"
        )
    else:
        st.warning(backtest.get("note", "可回测时点不足。"))

    market_forecast = analysis.get("market_analog_forecast") or {}
    with st.expander(f"查看市场基准相似周期｜{market_forecast.get('source_label', '市场基准')}"):
        market_rows = analog_horizon_rows(market_forecast)
        if market_rows:
            st.dataframe(pd.DataFrame(market_rows), hide_index=True, width="stretch")
        else:
            st.info("市场基准没有足够的相似周期样本。")

    for note in forecast.get("notes", []):
        st.caption(f"• {note}")
    st.warning("相似周期预测是一种历史情景分析，不等于未来会复制历史路径，也不能代替对公司基本面和重大事件的判断。")


def render_evidence(bundle, analysis) -> None:
    st.subheader("行情、基本面与宏观证据")
    render_price_charts(bundle, analysis)
    fundamental = analysis["fundamental"]
    st.markdown("#### 公司基本面与估值")
    if fundamental.fields:
        rows = [[key, format_field(key, value)] for key, value in fundamental.fields.items() if key not in {"公司名称"}]
        st.dataframe(pd.DataFrame(rows, columns=["指标", "最近可得值"]), hide_index=True, width="stretch")
        if fundamental.score is not None:
            st.caption(f"基本面辅助分：{fundamental.score:.0f}/100。该分数未做完整行业横向比较，不能单独使用。")
    else:
        st.info("未取得可用的公司财务数据；该部分未参与评分。")
    for note in fundamental.notes:
        st.caption(f"• {note}")

    macro = analysis["macro"]
    st.markdown("#### 市场与宏观环境")
    macro_rows = []
    for key, value in macro.fields.items():
        if isinstance(value, float) and ("收益" in key or "波动" in key):
            display = pct(value)
        else:
            display = format_field(key, value)
        macro_rows.append([key, display])
    st.dataframe(pd.DataFrame(macro_rows, columns=["指标", "最近可得值"]), hide_index=True, width="stretch")
    for note in macro.notes:
        st.caption(f"• {note}")
    st.info("宏观环境只作为修正因素。短期个股波动不能仅靠宏观数据预测，长期判断仍需结合公司基本面和估值。")


def render_professional(bundle, analysis) -> None:
    metrics = analysis["metrics"]
    st.subheader("专业指标与模型边界")
    metric_rows = [
        ["行情首日", str(metrics["first_date"].date())],
        ["行情末日", str(metrics["last_date"].date())],
        ["交易日数量", f"{metrics['rows']}"],
        ["最新价格", f"{metrics['latest_price']:.3f} {bundle.price_unit}"],
        ["近一年年化波动率", pct(metrics["annual_volatility"])],
        ["下行波动率", pct(metrics["downside_volatility"])],
        ["近五年或上市以来最大回撤", pct(metrics["max_drawdown"])],
        ["单日95%历史VaR", pct(metrics["var95_daily"])],
        ["Beta", number(metrics["beta"])],
        ["与市场相关性", number(metrics["correlation"])],
        ["20日／60日平均成交量比", number(metrics["volume_ratio"])],
        ["极端单日变动数量", str(metrics["abnormal_days"])],
    ]
    st.dataframe(pd.DataFrame(metric_rows, columns=["专业指标", "结果"]), hide_index=True, width="stretch")
    with st.expander("查看评分逻辑"):
        st.markdown(
            """
            - 用户风险等级与用户类型分开：经验丰富不自动等于风险承受能力高。
            - 股票风险分综合近一年波动、下行波动、全部历史回撤和Beta；最大回撤不会单独否决。
            - 当前时点分按不同周期分别计算，使用均线结构、动量、相对基准、成交量、基本面、市场环境和相似周期后续分布。
            - 相似周期检索使用收益、波动、回撤、均线位置、成交量和市场基准特征；有效样本少于10个时拒绝形成预测。
            - 相似周期最多只对当前时点评分进行有限修正，不会覆盖个人适配、安全限制或数据不足结论。
            - 20日滚动回测的每个验证时点只使用当时已经可见的数据，用于检查规则是否存在明显失效。
            - Agent在资金最早使用时间允许的范围内，结合投资目标、看盘条件和退出纪律选择周期。
            - 仓位是风险预算参考值，不是收益承诺，也不等于下单指令。
            - 历史上涨样本占比不等于经过校准的真实上涨概率；所有数值由可检查的量化规则计算。
            """
        )


def page_three() -> None:
    confirmed_market, confirmed_code = ensure_confirmed_stock()
    profile = dict(st.session_state.profile)
    render_brand(f"正在分析 {confirmed_market}｜{confirmed_code}")
    st.subheader(f"Agent自动获取 {confirmed_code} 近五年数据、检索相似周期并分析")
    with st.status("正在完成自动分析……", expanded=True) as status:
        try:
            status.write("1/4 获取股票最近五年或上市以来全部可得行情")
            bundle = cached_price_bundle(confirmed_market, confirmed_code, st.session_state.analysis_request_token)
            if str(bundle.code).upper() != confirmed_code.upper():
                raise RuntimeError(f"股票代码校验失败：请求 {confirmed_code}，数据源返回 {bundle.code}。已停止分析，避免使用错误股票数据。")
            status.write("2/4 获取基准、公司财务和估值信息")
            last_price = float(bundle.stock["收盘"].iloc[-1])
            holding_snapshot = None
            if st.session_state.confirmed_holding_state == "已经持有":
                if st.session_state.confirmed_holding_method == "按持股数量填写":
                    fx_snapshot = cached_usd_cny_rate() if confirmed_market == "美股" else None
                    holding_snapshot = calculate_holding_values(
                        confirmed_market,
                        st.session_state.confirmed_share_count,
                        last_price,
                        st.session_state.confirmed_cost_price,
                        float(fx_snapshot["rate"]) if fx_snapshot else None,
                    )
                    if fx_snapshot:
                        holding_snapshot["fx_provider"] = fx_snapshot["provider"]
                        holding_snapshot["fx_date"] = pd.Timestamp(fx_snapshot["date"]).date().isoformat()
                else:
                    holding_snapshot = calculate_amount_holding_values(
                        st.session_state.confirmed_current_market_value,
                        st.session_state.confirmed_total_cost,
                    )
                current_value = float(holding_snapshot["current_rmb"])
                total_exposure = current_value + float(st.session_state.confirmed_additional_amount)
                profile["planned_amount"] = total_exposure
                profile["current_holding_value"] = current_value
                profile["additional_amount"] = float(st.session_state.confirmed_additional_amount)
                profile["holding_state"] = "已经持有"
                st.session_state.confirmed_current_market_value = current_value
            else:
                profile["current_holding_value"] = 0.0
                profile["additional_amount"] = float(profile["planned_amount"])
                profile["holding_state"] = "尚未持有"
            fundamental = cached_fundamentals(confirmed_market, bundle.code, round(last_price, 6), bundle.asset_type)
            company_name = fundamental.fields.get("公司名称") if fundamental.fields else None
            if company_name:
                bundle.name = str(company_name)
            status.write("3/4 识别市场与宏观环境")
            macro = cached_macro(confirmed_market, bundle.benchmark)
            status.write("4/4 检索历史相似周期，计算多周期信号、个人适配和风险预算")
            analysis = analyze_all(bundle, profile, fundamental, macro)
            current_holding = float(profile.get("current_holding_value") or 0.0)
            assets = float(profile.get("investable_assets") or 0.0)
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
            st.session_state.holding_snapshot = holding_snapshot
            st.session_state.profile = profile
            status.update(label="分析完成", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="真实数据获取或分析失败", state="error", expanded=True)
            st.error("Agent没有取得足够的真实公开数据，因此没有生成一个看似精确但不可靠的结果。")
            st.code(str(exc))
            st.markdown(
                "请先确认代码正确和网络可用，然后在项目文件夹运行：\n\n"
                "`python -m pip install --upgrade -r requirements.txt`\n\n"
                "再运行 `python 测试真实行情接口.py`。如果测试仍失败，把黑色窗口的完整截图发给我。"
            )
            cols = st.columns(2)
            if cols[0].button("返回修改股票", width="stretch"):
                st.session_state.view = "analysis"
                st.rerun()
            if cols[1].button("前往个人中心", width="stretch"):
                st.session_state.view = "profile"
                st.rerun()
            st.stop()

    full_first = bundle.stock["日期"].min().date()
    full_last = bundle.stock["日期"].max().date()
    st.success(f"{bundle.code}｜{bundle.name}｜{bundle.asset_type}｜分析区间 {full_first} 至 {full_last}，共 {len(bundle.stock)} 个交易日。")
    st.caption(f"行情来源：{bundle.provider}；基准：{bundle.benchmark_name}；数据以最近公开返回为准，可能延迟、缺失或调整。")
    if not bundle.history_complete:
        st.warning("该股票可得历史不足五年。数据不足，无法准确判断，分析结果仅作低置信度参考。")
    for warning in bundle.warnings:
        if "不足五年" not in warning:
            st.info(warning)

    view_mode = st.radio("结果显示", ["简明模式", "专业模式"], horizontal=True, help="两种模式使用同一套计算结果，只改变解释深度。")
    tab_names = ["结论", "相似周期预测", "风险与仓位", "持有周期", "数据证据"]
    if view_mode == "专业模式":
        tab_names.append("专业指标")
    tabs = st.tabs(tab_names)
    with tabs[0]:
        render_summary(bundle, analysis, profile)
    with tabs[1]:
        render_analog_forecast(analysis)
    with tabs[2]:
        render_risk_budget(analysis, profile)
    with tabs[3]:
        render_horizons(analysis)
    with tabs[4]:
        render_evidence(bundle, analysis)
    if view_mode == "专业模式":
        with tabs[5]:
            render_professional(bundle, analysis)

    st.divider()
    st.warning("本程序是学习和内部研究原型，不是中国建设银行正式产品，不构成投资建议，不预测确定收益，也不会代替持牌机构的适当性管理和人工审核。")
    left, middle, right = st.columns(3)
    if left.button("换一只股票", width="stretch"):
        clear_stock_widgets()
        st.rerun()
    if middle.button("个人中心", width="stretch"):
        st.session_state.view = "profile"
        st.rerun()
    if right.button("刷新公开数据", width="stretch"):
        st.cache_data.clear()
        st.session_state.analysis_request_token += 1
        st.rerun()


initialize_v5_state()
load_current_user_data()

if st.session_state.saved_profile:
    app_sidebar()
if not st.session_state.saved_profile:
    st.session_state.view = "questionnaire"

if st.session_state.view == "questionnaire":
    questionnaire_page()
elif st.session_state.view == "profile":
    personal_center()
elif st.session_state.view == "result":
    page_three()
else:
    analysis_home()
