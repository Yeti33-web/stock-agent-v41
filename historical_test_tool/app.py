from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


TOOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_ROOT.parent
if not (PROJECT_ROOT / "agent_core.py").exists() and (PROJECT_ROOT / "model_v64" / "agent_core.py").exists():
    PROJECT_ROOT = PROJECT_ROOT / "model_v64"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_core import MODEL_VERSION, normalize_a_code, normalize_hk_code, normalize_us_code, score_investor
from questionnaire import (
    QUESTIONS,
    answers_complete,
    answers_to_profile,
    compose_analysis_profile,
    first_unanswered_index,
    public_profile_rows,
)
from historical_test_tool.original_ui import load_original_ui
from historical_test_tool.runner import run_full_historical_agent


st.set_page_config(
    page_title="个人投资者股票决策辅助 Agent｜独立历史时点测试",
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

original_ui = load_original_ui()


def initialize_state() -> None:
    defaults = {
        "historical_view": "questionnaire",
        "historical_saved_profile": None,
        "historical_profile_record": None,
        "historical_question_answers": {},
        "historical_question_index": 0,
        "historical_result": None,
        "historical_request_token": 0,
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
        "profile": None,
        "historical_requested_date": date(2024, 6, 1),
    }
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
    for key, value in {**defaults, **widget_defaults}.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_test_notice() -> None:
    st.info(
        "这是与正式Agent分开的历史时点测试页面。除历史日期T和不写入正式账号／会话外，"
        f"风险测评、投资输入、分析规则和结果页面均沿用{MODEL_VERSION}。"
    )


def validate_code(market: str, code: str) -> str:
    if market == "A股":
        return normalize_a_code(code)
    if market == "美股":
        return normalize_us_code(code)
    if market == "港股":
        return normalize_hk_code(code)
    raise ValueError("市场仅支持A股、美股或港股。")


def question_option_label(option: str, current: str | None) -> str:
    return f"✓ 当前答案　{option}" if option == current else option


def questionnaire_page() -> None:
    answers = dict(st.session_state.historical_question_answers)
    index = int(st.session_state.historical_question_index)
    total = len(QUESTIONS)
    original_ui.render_brand(f"独立测试使用{MODEL_VERSION}完整个人风险测评")
    render_test_notice()
    if index < total:
        question = QUESTIONS[index]
        st.progress((index + 1) / total, text=f"第 {index + 1} / {total} 题")
        st.markdown(
            f'<div class="question-card"><div class="muted">个人风险测评</div>'
            f'<h2>{question["title"]}</h2><p class="muted">{question["hint"]}</p></div>',
            unsafe_allow_html=True,
        )
        current_answer = answers.get(question["key"])
        for option_index, option in enumerate(question["options"]):
            if st.button(
                question_option_label(str(option), current_answer),
                key=f"historical_answer_{index}_{option_index}",
                width="stretch",
                type="primary" if option == current_answer else "secondary",
            ):
                answers[str(question["key"])] = str(option)
                st.session_state.historical_question_answers = answers
                st.session_state.historical_question_index = index + 1
                st.rerun()
        st.divider()
        if st.button("← 返回上一页", disabled=index == 0, key=f"historical_back_{index}"):
            st.session_state.historical_question_index = max(0, index - 1)
            st.rerun()
        return

    st.progress(1.0, text=f"已完成 {total} / {total} 题")
    st.subheader("确认并提交风险测评")
    if not answers_complete(answers):
        st.warning("仍有问题未完成。")
        if st.button("返回继续填写", type="primary"):
            st.session_state.historical_question_index = first_unanswered_index(answers)
            st.rerun()
        return
    review = pd.DataFrame(
        [[item["title"], answers[item["key"]]] for item in QUESTIONS],
        columns=["问题", "你的选择"],
    )
    st.dataframe(review, hide_index=True, width="stretch")
    back, submit = st.columns([1, 2])
    if back.button("← 返回上一页", width="stretch"):
        st.session_state.historical_question_index = total - 1
        st.rerun()
    if submit.button("确认提交并生成风险等级", type="primary", width="stretch"):
        profile = answers_to_profile(answers)
        risk_score, _, risk_level, _, _ = score_investor(profile)
        st.session_state.historical_saved_profile = profile
        st.session_state.historical_profile_record = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "version": 1,
            "updated_at": datetime.now().isoformat(),
        }
        st.session_state.historical_view = "analysis"
        st.rerun()


def market_changed() -> None:
    st.session_state.v5_stock_code = ""
    st.session_state.v5_share_count = 0.0
    st.session_state.v5_cost_price = 0.0
    st.session_state.v5_current_value = 0.0
    st.session_state.v5_total_cost = 0.0
    st.session_state.v5_additional_amount = 0.0


def analysis_page() -> None:
    profile = dict(st.session_state.historical_saved_profile)
    record = dict(st.session_state.historical_profile_record or {})
    original_ui.render_brand("完整复现原Agent；唯一新增输入为历史分析日期T")
    render_test_notice()
    risk_cols = st.columns([1, 1, 1.4])
    risk_cols[0].metric("个人风险等级", record.get("risk_level", "—"), f"{record.get('risk_score', '—')}/100")
    risk_cols[1].metric("测评版本", "本次独立测试")
    risk_cols[2].metric("资产范围", profile.get("asset_band", "—"))
    st.markdown(
        '<div class="hero-card"><b>选择历史时间点和本次要分析的股票</b><br>'
        '<span class="muted">系统会假设当前时间就是T，并阻止T之后的数据进入Agent。</span></div>',
        unsafe_allow_html=True,
    )
    st.date_input(
        "历史分析日期T",
        key="historical_requested_date",
        max_value=date.today(),
        help="如果T不是交易日，自动采用T之前最近一个交易日。",
    )
    stock_left, stock_right = st.columns([1, 1])
    with stock_left:
        st.radio("市场", ["A股", "美股", "港股"], horizontal=True, key="v5_market", on_change=market_changed)
        placeholders = {
            "A股": "例如：600519、300750",
            "美股": "例如：AAPL、MSFT、NVDA",
            "港股": "例如：00700、09988，也可输入700或0700.HK",
        }
        st.text_input("股票代码", key="v5_stock_code", placeholder=placeholders[st.session_state.v5_market])
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
            unit = {"A股": "元/股", "美股": "美元/股", "港股": "港元/股"}[st.session_state.v5_market]
            share_step = 100.0 if st.session_state.v5_market == "A股" else 1.0
            with holding_left:
                st.number_input("持股数量（股）", min_value=0.0, step=share_step, format="%.0f", key="v5_share_count")
            with holding_right:
                st.number_input(
                    f"平均持仓成本价（{unit}，可填0表示未知）",
                    min_value=0.0,
                    step=0.001,
                    format="%.3f",
                    key="v5_cost_price",
                )
            st.info("Agent使用T日收盘价计算当时持仓市值；美股和港股只使用T日及以前的公开汇率。")
        else:
            with holding_left:
                st.number_input("当前持仓市值（折合人民币元）", min_value=0.0, step=1000.0, format="%.3f", key="v5_current_value")
            with holding_right:
                st.number_input("累计投入成本（人民币元，可填0表示未知）", min_value=0.0, step=1000.0, format="%.3f", key="v5_total_cost")
        st.radio("本次计划", ["仅分析现有持仓", "还计划加仓"], horizontal=True, key="v5_planned_action")
        if st.session_state.v5_planned_action == "还计划加仓":
            st.number_input("计划新增金额（人民币元）", min_value=0.0, step=1000.0, format="%.3f", key="v5_additional_amount")
    if st.button("按历史日期获取近五年数据并分析", type="primary", width="stretch"):
        try:
            market = st.session_state.v5_market
            code = validate_code(market, st.session_state.v5_stock_code)
            holding_state = st.session_state.v5_holding_state
            holding_method = st.session_state.v5_holding_method
            share_count = float(st.session_state.v5_share_count)
            current_value = float(st.session_state.v5_current_value)
            additional = (
                float(st.session_state.v5_additional_amount)
                if holding_state == "已经持有" and st.session_state.v5_planned_action == "还计划加仓"
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
            if holding_state == "已经持有" and st.session_state.v5_planned_action == "还计划加仓" and additional <= 0:
                raise ValueError("你选择了还计划加仓，请填写大于0的计划新增金额。")
            asset_upper = profile.get("asset_upper")
            if holding_state == "尚未持有" and asset_upper is not None and planned > float(asset_upper):
                raise ValueError("本次计划金额高于风险问卷所选资产范围上限，请重新测评或调整金额。")
            st.session_state.confirmed_market = market
            st.session_state.confirmed_stock_code = code
            st.session_state.confirmed_holding_state = holding_state
            st.session_state.confirmed_holding_method = holding_method
            st.session_state.confirmed_share_count = share_count
            st.session_state.confirmed_cost_price = float(st.session_state.v5_cost_price)
            st.session_state.confirmed_current_market_value = current_value
            st.session_state.confirmed_total_cost = float(st.session_state.v5_total_cost)
            st.session_state.confirmed_additional_amount = additional
            st.session_state.profile = compose_analysis_profile(profile, planned, st.session_state.v5_leverage)
            st.session_state.historical_result = None
            st.session_state.historical_request_token += 1
            st.session_state.historical_view = "result"
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def render_result_tabs(bundle, analysis, profile) -> None:
    view_mode = st.radio(
        "结果显示",
        ["简明模式", "专业模式"],
        horizontal=True,
        help="两种模式使用同一套计算结果，只改变解释深度。",
    )
    tab_names = ["结论"]
    if st.session_state.confirmed_holding_state == "已经持有":
        tab_names.append("卖出信号")
    tab_names.extend(["相似周期预测", "最新资讯", "风险与仓位", "持有周期", "因子解释与验证", "数据证据"])
    if view_mode == "专业模式":
        tab_names.append("专业指标")
    tabs = st.tabs(tab_names)
    tab_map = dict(zip(tab_names, tabs))
    with tab_map["结论"]:
        original_ui.render_summary(bundle, analysis, profile)
    if "卖出信号" in tab_map:
        with tab_map["卖出信号"]:
            original_ui.render_sell_signals(bundle, analysis, profile)
    with tab_map["相似周期预测"]:
        original_ui.render_analog_forecast(analysis)
    with tab_map["最新资讯"]:
        original_ui.render_news_analysis(dict(analysis.get("news_analysis") or {}))
    with tab_map["风险与仓位"]:
        original_ui.render_risk_budget(analysis, profile)
    with tab_map["持有周期"]:
        original_ui.render_horizons(analysis)
    with tab_map["因子解释与验证"]:
        original_ui.render_factor_analysis(bundle, analysis, profile)
    with tab_map["数据证据"]:
        original_ui.render_evidence(bundle, analysis)
    if view_mode == "专业模式":
        with tab_map["专业指标"]:
            original_ui.render_professional(bundle, analysis)


def result_page() -> None:
    market = str(st.session_state.confirmed_market)
    code = str(st.session_state.confirmed_stock_code)
    requested_date = st.session_state.historical_requested_date
    original_ui.render_brand(f"历史时点测试｜假设当前日期为 {requested_date}｜{market}｜{code}")
    render_test_notice()
    if st.session_state.historical_result is None:
        with st.status("正在按历史时点运行完整Agent……", expanded=True) as status:
            try:
                status.write("1/5 获取T日以前约五年的股票与基准行情")
                status.write("2/5 应用当次填写的风险资料、资金和持仓信息")
                status.write("3/5 排除T日后财务、利率和资讯")
                status.write(f"4/5 调用{MODEL_VERSION}评分、风险、方向验证、周期和仓位逻辑")
                result = run_full_historical_agent(
                    market=market,
                    raw_code=code,
                    requested_date=requested_date,
                    profile=dict(st.session_state.profile),
                    holding_state=st.session_state.confirmed_holding_state,
                    holding_method=st.session_state.confirmed_holding_method,
                    share_count=float(st.session_state.confirmed_share_count),
                    cost_price=float(st.session_state.confirmed_cost_price),
                    current_market_value=float(st.session_state.confirmed_current_market_value),
                    total_cost=float(st.session_state.confirmed_total_cost),
                    additional_amount=float(st.session_state.confirmed_additional_amount),
                )
                status.write("5/5 生成与正式Agent一致的完整结果标签页")
                st.session_state.historical_result = result
                st.session_state.profile = result.profile
                st.session_state.holding_snapshot = result.holding_snapshot
                status.update(label="历史时点分析完成", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="历史时点分析失败", state="error", expanded=True)
                st.error("这不是正式Agent的投资结论，而是历史数据通道或回测程序运行失败。")
                st.code(f"{type(exc).__name__}: {exc}")
                if st.button("返回修改输入"):
                    st.session_state.historical_view = "analysis"
                    st.rerun()
                st.stop()
    result = st.session_state.historical_result
    bundle = result.bundle
    analysis = result.analysis
    profile = result.profile
    requested = result.historical.requested_date
    actual = result.historical.actual_trading_date
    st.success(
        f"请求历史日期T：{requested}｜实际采用交易日：{actual}｜"
        f"{bundle.code}｜{bundle.name}｜分析区间 {bundle.stock['日期'].min().date()} 至 {bundle.stock['日期'].max().date()}｜"
        f"共 {len(bundle.stock)} 个交易日。"
    )
    st.caption(f"行情来源：{bundle.provider}；基准：{bundle.benchmark_name}。Agent未读取实际采用交易日之后的数据。")
    if requested != actual:
        st.info(f"{requested}不是该股票的有效交易日，已自动采用此前最近的交易日{actual}。")
    if not bundle.history_complete:
        st.warning("该股票在T日以前的可得历史不足五年，已降低可信度；只要周期窗口足够，仍会继续复现当时判断。")
    for warning in bundle.warnings:
        if "不足五年" not in warning:
            st.info(warning)
    render_result_tabs(bundle, analysis, profile)
    st.divider()
    st.warning("该分析结果仅供参考，本模型仅用于学习与研究。")
    st.caption("独立测试结果不会写入正式邮箱账号、股票会话、持仓记录或数据库。")
    left, right = st.columns(2)
    if left.button("换一只股票／历史日期", width="stretch"):
        st.session_state.historical_result = None
        st.session_state.historical_view = "analysis"
        st.rerun()
    if right.button("重新填写个人风险测评", width="stretch"):
        reset_questionnaire()


def reset_questionnaire() -> None:
    st.session_state.historical_saved_profile = None
    st.session_state.historical_profile_record = None
    st.session_state.historical_question_answers = {}
    st.session_state.historical_question_index = 0
    st.session_state.historical_result = None
    st.session_state.historical_view = "questionnaire"
    st.rerun()


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📊 历史时点测试Agent")
        st.caption("独立运行，不连接正式账号和数据库")
        record = st.session_state.historical_profile_record or {}
        if record:
            st.caption(f"当前测试风险等级：{record.get('risk_level', '—')}")
            if st.button("股票分析", width="stretch"):
                st.session_state.historical_view = "analysis"
                st.session_state.historical_result = None
                st.rerun()
            if st.button("查看风险资料", width="stretch"):
                profile = st.session_state.historical_saved_profile or {}
                st.dataframe(
                    pd.DataFrame(public_profile_rows(profile), columns=["项目", "当前选择"]),
                    hide_index=True,
                    width="stretch",
                )
            if st.button("重新风险测评", width="stretch"):
                reset_questionnaire()


initialize_state()
sidebar()
if not st.session_state.historical_saved_profile:
    st.session_state.historical_view = "questionnaire"
if st.session_state.historical_view == "questionnaire":
    questionnaire_page()
elif st.session_state.historical_view == "result":
    result_page()
else:
    analysis_page()
