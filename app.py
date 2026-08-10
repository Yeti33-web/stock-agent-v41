from __future__ import annotations

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
)


st.set_page_config(
    page_title="个人投资者股票决策辅助 Agent V4.1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}
    .hero-card {padding: 1.2rem 1.35rem; border: 1px solid #e4e7ec; border-radius: 16px; background: #f8fafc; margin-bottom: 1rem;}
    .result-card {padding: 1rem 1.15rem; border-radius: 14px; background: #f7f9fc; border-left: 5px solid #2563eb;}
    .muted {color: #667085; font-size: .92rem;}
    div[data-testid="stMetric"] {border: 1px solid #e7eaf0; padding: .8rem; border-radius: 12px; background: white;}
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    persistent_defaults = {
        "step": 1,
        "confirmed_market": None,
        "confirmed_stock_code": None,
        "confirmed_holding_state": "尚未持有",
        "confirmed_cost_price": 0.0,
        "confirmed_current_market_value": 0.0,
        "analysis_request_token": 0,
    }
    for key, value in persistent_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Streamlit会清理离开页面后不再显示的控件状态，因此输入值与确认值必须使用不同的key。
    if st.session_state.step == 1:
        input_defaults = {
            "market_input": "A股",
            "stock_code_input": "",
            "holding_state_input": "尚未持有",
            "cost_price_input": 0.0,
            "current_market_value_input": 0.0,
        }
        for key, value in input_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value


def market_changed() -> None:
    st.session_state.stock_code_input = ""


def restore_confirmed_inputs() -> None:
    st.session_state.market_input = st.session_state.confirmed_market or "A股"
    st.session_state.stock_code_input = st.session_state.confirmed_stock_code or ""
    st.session_state.holding_state_input = st.session_state.confirmed_holding_state
    st.session_state.cost_price_input = float(st.session_state.confirmed_cost_price)
    st.session_state.current_market_value_input = float(st.session_state.confirmed_current_market_value)
    st.session_state.step = 1


def start_new_stock() -> None:
    for key in ["market_input", "stock_code_input", "holding_state_input", "cost_price_input", "current_market_value_input"]:
        st.session_state.pop(key, None)
    st.session_state.step = 1


def ensure_confirmed_stock() -> tuple[str, str]:
    market = st.session_state.confirmed_market
    code = st.session_state.confirmed_stock_code
    if not market or not code:
        st.error("尚未确认本次要分析的股票，请返回第一步重新选择。")
        if st.button("返回选择股票"):
            start_new_stock()
            st.rerun()
        st.stop()
    return str(market), str(code)


def render_header() -> None:
    st.title("个人投资者股票决策辅助 Agent V4.1")
    st.caption("真实公开数据 · 自动选择历史样本 · 用户适配与当前时点分开判断 · 教学研究原型")
    step = st.session_state.step
    labels = ["① 选择股票", "② 个人情况", "③ 自动分析与结果"]
    columns = st.columns(3)
    for index, (column, label) in enumerate(zip(columns, labels), start=1):
        prefix = "🟦" if index == step else "✅" if index < step else "⬜"
        column.markdown(f"{prefix} **{label}**")
    st.divider()


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


def page_one() -> None:
    st.subheader("第一步：选择要判断的真实股票")
    st.markdown(
        '<div class="hero-card">只输入市场和股票代码。Agent会自行获取最大可得历史，不需要上传Excel，也不能手工挑选一段有利行情。</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1])
    with left:
        st.radio("市场", ["A股", "美股"], horizontal=True, key="market_input", on_change=market_changed)
        help_text = "例如：000001、600519、300750" if st.session_state.market_input == "A股" else "例如：AAPL、MSFT、NVDA、BRK-B"
        placeholder = "请输入6位A股代码" if st.session_state.market_input == "A股" else "请输入美股代码"
        st.text_input("股票代码", key="stock_code_input", placeholder=placeholder, help=help_text)
    with right:
        st.radio("目前是否已经持有这只股票？", ["尚未持有", "已经持有"], horizontal=True, key="holding_state_input")
        if st.session_state.holding_state_input == "已经持有":
            unit = "元/股" if st.session_state.market_input == "A股" else "美元/股"
            st.number_input(f"持仓成本价（{unit}，可填0表示未知）", min_value=0.0, step=0.01, key="cost_price_input")
            st.number_input("当前持仓市值（折合人民币元）", min_value=0.0, step=1000.0, key="current_market_value_input")

    st.info("V4.1不会要求选择历史开始日、结束日或计划持有期；持有周期由Agent根据个人约束和多周期信号判断。")
    if st.button("下一步：填写个人情况", type="primary", width="stretch"):
        try:
            confirmed_code = validate_code(st.session_state.market_input, st.session_state.stock_code_input)
            st.session_state.confirmed_market = st.session_state.market_input
            st.session_state.confirmed_stock_code = confirmed_code
            st.session_state.confirmed_holding_state = st.session_state.holding_state_input
            st.session_state.confirmed_cost_price = float(st.session_state.get("cost_price_input", 0.0))
            st.session_state.confirmed_current_market_value = float(st.session_state.get("current_market_value_input", 0.0))
            st.session_state.analysis_request_token += 1
            st.cache_data.clear()
            st.session_state.step = 2
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def build_profile() -> dict:
    return {
        "planned_amount": float(st.session_state.planned_amount),
        "investable_assets": float(st.session_state.investable_assets),
        "fund_source": st.session_state.fund_source,
        "emergency_reserve": st.session_state.emergency_reserve,
        "earliest_need": st.session_state.earliest_need,
        "income_stability": st.session_state.income_stability,
        "loss_response": st.session_state.loss_response,
        "goal": st.session_state.goal,
        "experience": st.session_state.experience,
        "trade_frequency": st.session_state.trade_frequency,
        "monitor_time": st.session_state.monitor_time,
        "existing_concentration": st.session_state.existing_concentration,
        "stop_loss": st.session_state.get("stop_loss", "不适用"),
        "leverage": st.session_state.get("leverage", "否"),
        "fundamental_action": st.session_state.get("fundamental_action", "会重新评估"),
        "fx_acceptance": st.session_state.get("fx_acceptance", "不确定"),
    }


def page_two() -> None:
    confirmed_market, confirmed_code = ensure_confirmed_stock()
    st.subheader("第二步：填写个人情况")
    st.info(f"本次确认分析：{confirmed_market}｜{confirmed_code}")
    st.caption("不需要填写姓名、身份证、银行卡或联系方式，页面不会主动保存这些信息。")
    left, right = st.columns(2)
    with left:
        amount_label = "计划投资金额（人民币元）" if st.session_state.confirmed_holding_state == "尚未持有" else "计划总持仓金额（含已有持仓，折合人民币元）"
        st.number_input(amount_label, min_value=1000.0, value=50000.0, step=1000.0, key="planned_amount")
        st.number_input("可用于投资的金融资产（人民币元）", min_value=1000.0, value=200000.0, step=10000.0, key="investable_assets")
        st.selectbox("这笔钱主要来自哪里？", ["闲置自有资金", "未来有明确用途的资金", "应急资金", "借款／融资资金"], key="fund_source")
        st.selectbox("目前预留的生活应急资金", ["不足3个月", "3—6个月", "6个月以上"], index=1, key="emergency_reserve")
        st.selectbox("这笔资金最早什么时候可能需要使用？", ["1周内", "1个月内", "3个月内", "1年内", "3年内", "没有明确时间"], index=3, key="earliest_need")
        st.selectbox("收入稳定性", ["不稳定", "较稳定", "稳定"], index=1, key="income_stability")
    with right:
        st.selectbox(
            "假设10万元下跌到8万元，你更可能怎么做？",
            ["立即全部卖出", "大部分减仓", "先复核原因再决定", "继续按原计划持有", "在条件允许时分批增加"],
            index=2,
            key="loss_response",
        )
        st.selectbox("主要投资目标", ["保值为主", "股息／稳健收益", "长期增值", "波段操作", "短线交易"], index=2, key="goal")
        st.selectbox("股票投资经验", ["没有经验", "不足1年", "1—3年", "3年以上"], index=1, key="experience")
        st.selectbox("通常交易频率", ["几乎不交易", "每月1—3次", "每周1—3次", "几乎每天"], index=1, key="trade_frequency")
        st.selectbox("每天可用于查看行情的时间", ["不足15分钟", "15—30分钟", "30—60分钟", "1小时以上"], index=1, key="monitor_time")
        st.selectbox("现有投资中最高的单只股票仓位", ["目前没有股票持仓", "不足10%", "10%—30%", "30%—50%", "超过50%"], key="existing_concentration")

    st.markdown("#### Agent追加问题")
    adaptive_left, adaptive_right = st.columns(2)
    if st.session_state.goal in {"短线交易", "波段操作"} or st.session_state.trade_frequency in {"每周1—3次", "几乎每天"}:
        with adaptive_left:
            st.selectbox("是否有明确并能执行的止损／退出规则？", ["有明确规则并能执行", "有规则但经常改变", "没有明确规则"], key="stop_loss")
        with adaptive_right:
            st.radio("是否计划使用融资或其他杠杆？", ["否", "是"], horizontal=True, key="leverage")
    else:
        with adaptive_left:
            st.selectbox("如果公司基本面发生明显恶化，你会怎么做？", ["会重新评估", "仍长期持有而不复核", "不确定如何判断"], key="fundamental_action")
        with adaptive_right:
            st.info("长期持有不等于永不复核。Agent会给出财报和投资逻辑复查节点。")
    if confirmed_market == "美股":
        st.selectbox("能否接受人民币兑美元波动影响最终收益？", ["能够接受", "只能接受较小波动", "不确定"], key="fx_acceptance")

    planned_ratio = st.session_state.planned_amount / max(st.session_state.investable_assets, 1)
    st.caption(f"目前填写的计划金额约占可投资金融资产的 {planned_ratio:.1%}。这只是风险集中度计算，不代表建议仓位。")
    back, submit = st.columns([1, 2])
    if back.button("返回选择股票", width="stretch"):
        restore_confirmed_inputs()
        st.rerun()
    if submit.button(f"获取 {confirmed_code} 真实数据并分析", type="primary", width="stretch"):
        if st.session_state.planned_amount > st.session_state.investable_assets:
            st.error("计划投入金额不能大于可投资金融资产。请先检查这两个数字。")
        elif st.session_state.confirmed_holding_state == "已经持有" and st.session_state.confirmed_current_market_value > st.session_state.investable_assets:
            st.error("当前持仓市值不能大于填写的可投资金融资产，请检查口径是否一致。")
        else:
            st.session_state.profile = build_profile()
            st.session_state.step = 3
            st.rerun()


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
    drawdown_fig.update_layout(title="全部可得历史回撤", height=340, margin=dict(l=20, r=20, t=55, b=20), yaxis_tickformat=".0%")
    st.plotly_chart(drawdown_fig, width="stretch")


def render_summary(bundle, analysis, profile) -> None:
    selected = analysis["selected_horizon"]
    conclusion_box(analysis["conclusion"], analysis["conclusion_reason"])
    columns = st.columns(5)
    columns[0].metric("用户风险等级", analysis["investor_level"], f"{analysis['investor_score']}/100")
    columns[1].metric("用户类型", analysis["style"])
    columns[2].metric("股票风险等级", analysis["stock_risk_level"], f"{analysis['stock_risk_score']}/100")
    columns[3].metric("个人适配", analysis["suitability"]["fit"])
    columns[4].metric("数据可信度", f"{analysis['data_confidence']}%")

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
            ["可投资金融资产", f"{profile['investable_assets']:,.0f} 元"],
            ["计划集中度", f"{profile['planned_amount'] / profile['investable_assets']:.1%}"],
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
        st.info("历史最差值用于压力提示，不再做一票否决。仓位上限主要使用较差情景、用户风险预算和集中度共同计算。")


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
        ["全部历史最大回撤", pct(metrics["max_drawdown"])],
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
    st.subheader(f"第三步：Agent自动获取 {confirmed_code} 数据并分析")
    with st.status("正在完成自动分析……", expanded=True) as status:
        try:
            status.write("1/4 获取股票的最大可得公开历史行情")
            bundle = cached_price_bundle(confirmed_market, confirmed_code, st.session_state.analysis_request_token)
            if str(bundle.code).upper() != confirmed_code.upper():
                raise RuntimeError(f"股票代码校验失败：请求 {confirmed_code}，数据源返回 {bundle.code}。已停止分析，避免使用错误股票数据。")
            if len(bundle.stock) < 40:
                raise RuntimeError(f"该证券只取得{len(bundle.stock)}个交易日，暂不足以进行基本风险分析。")
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
                restore_confirmed_inputs()
                st.rerun()
            if cols[1].button("返回修改个人情况", width="stretch"):
                st.session_state.step = 2
                st.rerun()
            st.stop()

    full_first = bundle.stock["日期"].min().date()
    full_last = bundle.stock["日期"].max().date()
    st.success(f"{bundle.code}｜{bundle.name}｜{bundle.asset_type}｜已取得 {full_first} 至 {full_last}，共 {len(bundle.stock)} 个交易日。")
    st.caption(f"行情来源：{bundle.provider}；基准：{bundle.benchmark_name}；数据以最近公开返回为准，可能延迟、缺失或调整。")

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
        start_new_stock()
        st.rerun()
    if middle.button("修改个人情况", width="stretch"):
        st.session_state.step = 2
        st.rerun()
    if right.button("刷新公开数据", width="stretch"):
        st.cache_data.clear()
        st.session_state.analysis_request_token += 1
        st.rerun()


initialize_state()
render_header()
with st.sidebar:
    st.header("V4.1的关键变化")
    st.write("- 面向所有个人股民，按用户自动分层")
    st.write("- 用户不选择历史区间或持有周期")
    st.write("- 个人适配与当前时点分别判断")
    st.write("- 历史最差值不再一票否决")
    st.write("- 免费数据失败时明确说明，不生成假结果")
    st.write("- 输入代码与确认代码分离，防止切换后恢复默认股票")
    st.divider()
    st.caption("当前步骤中的回答只在本次页面会话中用于计算。")

if st.session_state.step == 1:
    page_one()
elif st.session_state.step == 2:
    page_two()
else:
    page_three()
