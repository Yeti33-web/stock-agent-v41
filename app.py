from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent_core import (
    EvidenceSnapshot,
    analyze_all,
    fetch_a_fundamentals,
    fetch_macro_snapshot,
    fetch_price_bundle,
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
    page_title="个人投资者股票决策辅助 Agent 简化版",
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
        "confirmed_cost_price": 0.0,
        "confirmed_current_market_value": 0.0,
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
            "v5_cost_price": 0.0,
            "v5_current_value": 0.0,
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
    st.title("个人投资者股票决策辅助 Agent｜简化部署版")
    st.caption(subtitle or "近五年真实公开行情 · 当前会话风险测评 · 多周期风险判断 · 教学研究原型")


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


def analysis_home() -> None:
    profile = st.session_state.saved_profile
    record = st.session_state.profile_record or {}
    render_brand("风险资料已在本次网页会话中生效；日常只需更换股票并填写本次投资信息")
    risk_cols = st.columns([1, 1, 1.4])
    risk_cols[0].metric("个人风险等级", record.get("risk_level", "—"), f"{record.get('risk_score', '—')}/100")
    risk_cols[1].metric("测评版本", f"第 {record.get('version', 1)} 版")
    risk_cols[2].metric("资产范围", profile.get("asset_band", "—"))
    st.markdown('<div class="hero-card"><b>选择本次要分析的股票</b><br><span class="muted">Agent统一使用最近五年行情；上市不足五年时使用全部可得数据并降低结论置信度。</span></div>', unsafe_allow_html=True)
    st.caption("提示：本版不注册账号。关闭网页、清除浏览器数据或更换设备后，需要重新完成风险测评。")

    left, right = st.columns([1, 1])
    with left:
        st.radio("市场", ["A股", "美股"], horizontal=True, key="v5_market", on_change=v5_market_changed)
        placeholder = "例如：600519、300750" if st.session_state.v5_market == "A股" else "例如：AAPL、MSFT、NVDA"
        st.text_input("股票代码", key="v5_stock_code", placeholder=placeholder)
        st.number_input("本次计划总投资／持仓金额（折合人民币元）", min_value=1000.0, step=1000.0, key="v5_planned_amount")
    with right:
        st.radio("目前是否已经持有？", ["尚未持有", "已经持有"], horizontal=True, key="v5_holding_state")
        if st.session_state.v5_holding_state == "已经持有":
            unit = "元/股" if st.session_state.v5_market == "A股" else "美元/股"
            st.number_input(f"持仓成本价（{unit}，可填0表示未知）", min_value=0.0, step=0.01, key="v5_cost_price")
            st.number_input("当前持仓市值（折合人民币元）", min_value=0.0, step=1000.0, key="v5_current_value")
        st.radio("本次是否计划使用融资或其他杠杆？", ["否", "是"], horizontal=True, key="v5_leverage")

    if st.button("获取近五年真实数据并分析", type="primary", width="stretch"):
        try:
            code = validate_code(st.session_state.v5_market, st.session_state.v5_stock_code)
            planned = float(st.session_state.v5_planned_amount)
            asset_upper = profile.get("asset_upper")
            if asset_upper is not None and planned > float(asset_upper):
                raise ValueError("本次计划金额高于风险问卷所选资产范围上限，请先到个人中心更新资料。")
            st.session_state.confirmed_market = st.session_state.v5_market
            st.session_state.confirmed_stock_code = code
            st.session_state.confirmed_holding_state = st.session_state.v5_holding_state
            st.session_state.confirmed_cost_price = float(st.session_state.get("v5_cost_price", 0.0))
            st.session_state.confirmed_current_market_value = float(st.session_state.get("v5_current_value", 0.0))
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
    for key in ["v5_market", "v5_stock_code", "v5_holding_state", "v5_cost_price", "v5_current_value", "v5_planned_amount", "v5_leverage"]:
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
        st.caption("行情统一使用最近五年；新股使用上市以来全部可得数据并标注低置信度。")


def pct(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "数据不足"
    return f"{value:.{digits}%}"


def number(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "数据不足"
    return f"{value:.{digits}f}"


def format_field(name: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "未取得"
    if name in {"净资产收益率", "净利率", "净利润同比", "营收同比", "资产负债率", "经营现金流／净利润", "债务／权益"}:
        parsed = float(value)
        parsed = parsed / 100 if abs(parsed) > 5 else parsed
        return f"{parsed:.1%}"
    if name in {"市盈率TTM", "市净率"}:
        return f"{float(value):.2f}"
    if name == "总市值":
        return f"{float(value):,.0f}"
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
    fig.add_trace(go.Scatter(x=display_close.index, y=display_close, name="收盘价", line=dict(width=2)))
    for window, color in [(20, "#f59e0b"), (60, "#16a34a"), (250, "#7c3aed")]:
        if len(close) >= window:
            moving = close.rolling(window).mean()
            moving = moving[moving.index >= chart_start]
            fig.add_trace(go.Scatter(x=moving.index, y=moving, name=f"MA{window}", line=dict(width=1.2, color=color)))
    fig.update_layout(title="最近五年或全部可得价格", height=390, margin=dict(l=20, r=20, t=55, b=20), yaxis_title=bundle.price_unit, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")

    drawdown = metrics["drawdown"]
    drawdown_fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", line=dict(color="#dc2626"), name="回撤"))
    drawdown_fig.update_layout(title="近五年或上市以来回撤", height=340, margin=dict(l=20, r=20, t=55, b=20), yaxis_tickformat=".0%")
    st.plotly_chart(drawdown_fig, width="stretch")


def render_summary(bundle, analysis, profile) -> None:
    selected = analysis["selected_horizon"]
    conclusion_box(analysis["conclusion"], analysis["conclusion_reason"])
    columns = st.columns(5)
    columns[0].metric("用户风险等级", analysis["investor_level"], f"{analysis['investor_score']}/100")
    columns[1].metric("用户类型", analysis["style"])
    columns[2].metric("股票风险等级", analysis["stock_risk_level"], f"{analysis['stock_risk_score']}/100")
    columns[3].metric("个人适配", analysis["suitability"]["fit"])
    columns[4].metric("数据完整度", f"{analysis['data_confidence']}%")

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
        st.markdown("#### 风险预算参考上限")
        if position["upper_pct"] > 0:
            st.markdown(f"**可投资金融资产的 {position['lower_pct']:.1%}—{position['upper_pct']:.1%}**")
            st.write(f"按你填写的资产口径，对应约 **{position['lower_amount']:,.0f}—{position['upper_amount']:,.0f} 元**。")
            if profile["planned_amount"] > position["upper_amount"]:
                st.warning(f"你计划的 {profile['planned_amount']:,.0f} 元高于模型风险预算上限，主要问题是集中度，而不是股票一定不会上涨。")
        else:
            st.markdown("**当前新增风险预算：0%**")
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
        current_value = float(st.session_state.confirmed_current_market_value)
        current_ratio = current_value / profile["investable_assets"] if profile["investable_assets"] else 0
        cost = float(st.session_state.confirmed_cost_price)
        current_return = analysis["metrics"]["latest_price"] / cost - 1 if cost > 0 else None
        held_cols = st.columns(3)
        held_cols[0].metric("当前仓位占比", f"{current_ratio:.1%}")
        held_cols[1].metric("最新公开价格", f"{analysis['metrics']['latest_price']:.2f} {bundle.price_unit}")
        held_cols[2].metric("相对成本变化", pct(current_return) if current_return is not None else "成本未知")
        if current_ratio > analysis["position"]["upper_pct"] and analysis["position"]["upper_pct"] > 0:
            st.warning("当前仓位已经高于模型风险预算上限。这里提示的是集中度风险，不等同于要求立即卖出。")


def render_risk_budget(analysis, profile) -> None:
    st.subheader("个人适配与风险预算")
    st.write(f"**适配结论：{analysis['suitability']['fit']}。** {analysis['suitability']['fit_reason']}。")
    if analysis["suitability"]["hard_reasons"]:
        for reason in analysis["suitability"]["hard_reasons"]:
            st.error(reason)
    table = pd.DataFrame(
        [
            ["计划投资金额", f"{profile['planned_amount']:,.0f} 元"],
            ["可投资金融资产", profile.get("asset_band", f"约 {profile['investable_assets']:,.0f} 元")],
            ["计划集中度（按区间代表值估算）", f"{profile['planned_amount'] / profile['investable_assets']:.1%}"],
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
        ["最新价格", f"{metrics['latest_price']:.2f} {bundle.price_unit}"],
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
            - 当前时点分按不同周期分别计算，使用均线结构、动量、相对基准、成交量、基本面和市场环境。
            - Agent在资金最早使用时间允许的范围内，结合投资目标、看盘条件和退出纪律选择周期。
            - 仓位是风险预算参考值，不是收益承诺，也不等于下单指令。
            - 语言模型没有直接编造价格或评分；所有数值由可检查的量化规则计算。
            """
        )


def page_three() -> None:
    confirmed_market, confirmed_code = ensure_confirmed_stock()
    profile = st.session_state.profile
    render_brand(f"正在分析 {confirmed_market}｜{confirmed_code}")
    st.subheader(f"Agent自动获取 {confirmed_code} 近五年数据并分析")
    with st.status("正在完成自动分析……", expanded=True) as status:
        try:
            status.write("1/4 获取股票最近五年或上市以来全部可得行情")
            bundle = cached_price_bundle(confirmed_market, confirmed_code, st.session_state.analysis_request_token)
            if str(bundle.code).upper() != confirmed_code.upper():
                raise RuntimeError(f"股票代码校验失败：请求 {confirmed_code}，数据源返回 {bundle.code}。已停止分析，避免使用错误股票数据。")
            status.write("2/4 获取基准、公司财务和估值信息")
            last_price = float(bundle.stock["收盘"].iloc[-1])
            fundamental = cached_fundamentals(confirmed_market, bundle.code, round(last_price, 6), bundle.asset_type)
            company_name = fundamental.fields.get("公司名称") if fundamental.fields else None
            if company_name:
                bundle.name = str(company_name)
            status.write("3/4 识别市场与宏观环境")
            macro = cached_macro(confirmed_market, bundle.benchmark)
            status.write("4/4 计算多周期信号、适配程度和风险预算")
            analysis = analyze_all(bundle, profile, fundamental, macro)
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
    tab_names = ["结论", "风险与仓位", "持有周期", "数据证据"]
    if view_mode == "专业模式":
        tab_names.append("专业指标")
    tabs = st.tabs(tab_names)
    with tabs[0]:
        render_summary(bundle, analysis, profile)
    with tabs[1]:
        render_risk_budget(analysis, profile)
    with tabs[2]:
        render_horizons(analysis)
    with tabs[3]:
        render_evidence(bundle, analysis)
    if view_mode == "专业模式":
        with tabs[4]:
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
