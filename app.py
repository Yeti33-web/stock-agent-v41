from __future__ import annotations

from datetime import date, datetime
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cloud_store import CloudConfig, CloudStore, CloudStoreError

from agent_core import (
    EvidenceSnapshot,
    analyze_all,
    analyze_sell_signals,
    calculate_amount_holding_values,
    calculate_holding_values,
    fetch_a_fundamentals,
    fetch_hk_fundamentals,
    fetch_hkd_cny_rate,
    fetch_macro_snapshot,
    fetch_price_bundle,
    fetch_usd_cny_rate,
    fetch_us_fundamentals,
    normalize_a_code,
    normalize_hk_code,
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
from session_memory import (
    append_note,
    build_position_messages,
    calculate_position_transaction,
    delete_session,
    portfolio_from_sessions,
    session_key,
    set_invested_principal,
    upsert_analysis_session,
)
from snapshot_codec import build_analysis_snapshot, restore_analysis_snapshot


st.set_page_config(
    page_title="个人投资者股票决策辅助 Agent V6.1",
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
    div[data-testid="stChatMessage"] {border:1px solid #e4eaf2; border-radius:16px; background:white; padding:.25rem .55rem;}
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
    if market == "A股":
        return normalize_a_code(code)
    if market == "美股":
        return normalize_us_code(code)
    if market == "港股":
        return normalize_hk_code(code)
    raise ValueError("市场仅支持A股、美股或港股。")


@st.cache_data(ttl=1800, show_spinner=False)
def cached_price_bundle(market: str, code: str, request_token: int):
    del request_token
    return fetch_price_bundle(market, code)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamentals(market: str, code: str, last_price: float, asset_type: str) -> EvidenceSnapshot:
    if market == "A股":
        return fetch_a_fundamentals(code, last_price, asset_type)
    if market == "美股":
        return fetch_us_fundamentals(code, last_price)
    return fetch_hk_fundamentals(code, last_price)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_macro(market: str, benchmark: pd.DataFrame) -> EvidenceSnapshot:
    return fetch_macro_snapshot(market, benchmark)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_usd_cny_rate() -> dict:
    return fetch_usd_cny_rate()


@st.cache_data(ttl=21600, show_spinner=False)
def cached_hkd_cny_rate() -> dict:
    return fetch_hkd_cny_rate()


def initialize_v5_state() -> None:
    defaults = {
        "auth_user": None,
        "auth_notice": "",
        "registration_otp_pending": False,
        "registration_email": "",
        "registration_otp_sent_at": 0.0,
        "registration_notice": "",
        "cloud_store": None,
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
        "confirmed_analysis_id": "",
        "stock_sessions": {},
        "selected_session_key": None,
        "pending_delete_session": None,
        "pending_delete_transaction": None,
        "session_detail_mode": "完整分析",
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


def configured_cloud_store() -> CloudStore:
    current = st.session_state.get("cloud_store")
    if isinstance(current, CloudStore):
        return current
    try:
        secrets = st.secrets
        config = CloudConfig.from_mapping(secrets)
    except Exception as exc:
        if isinstance(exc, CloudStoreError):
            raise
        raise CloudStoreError("尚未读取到 Streamlit Secrets。") from exc
    store = CloudStore(config)
    st.session_state.cloud_store = store
    return store


def current_user_id() -> str:
    user = st.session_state.get("auth_user") or {}
    user_id = str(user.get("user_id") or "")
    if not user_id:
        raise CloudStoreError("登录状态已失效，请重新登录。")
    return user_id


def set_logged_in(identity: dict, store: CloudStore) -> None:
    if not identity.get("user_id") or not identity.get("access_token"):
        raise CloudStoreError("登录未返回有效会话，请确认邮箱后重新登录。")
    st.session_state.auth_user = {
        "user_id": str(identity["user_id"]),
        "email": str(identity.get("email") or ""),
    }
    st.session_state.cloud_store = store
    st.session_state.user_data_loaded = False
    st.session_state.view = "analysis"
    st.session_state.auth_notice = "登录成功。你的资料、会话和历史分析将永久保存到当前账号。"


def reset_registration_otp() -> None:
    st.session_state.registration_otp_pending = False
    st.session_state.registration_email = ""
    st.session_state.registration_otp_sent_at = 0.0
    st.session_state.registration_notice = ""


def begin_registration_otp(email: str, notice: str) -> None:
    # This function runs before the OTP widget is rendered, so clearing an older
    # value here avoids mutating widget state after instantiation.
    st.session_state.pop("register_otp", None)
    st.session_state.registration_otp_pending = True
    st.session_state.registration_email = email
    st.session_state.registration_otp_sent_at = time.time()
    st.session_state.registration_notice = notice


def friendly_registration_error(exc: Exception) -> str:
    raw = str(exc).strip()
    lowered = raw.lower()
    if "email rate limit exceeded" in lowered or "rate limit" in lowered:
        return "验证码邮件发送次数已达到临时上限。请不要连续点击，稍后再试。"
    if "expired" in lowered or "invalid" in lowered or "otp" in lowered and "verify" in lowered:
        return "验证码错误或已经过期，请核对后重试；必要时重新发送验证码。"
    if "already registered" in lowered or "already been registered" in lowered:
        return "该邮箱可能已经注册。已完成验证请直接登录；尚未验证请使用下方“继续未完成注册”。"
    return raw


@st.fragment(run_every=1)
def registration_otp_controls(store: CloudStore) -> None:
    email = str(st.session_state.get("registration_email") or "")
    sent_at = float(st.session_state.get("registration_otp_sent_at") or 0.0)
    remaining = max(0, 60 - int(time.time() - sent_at)) if sent_at else 0
    left, right = st.columns(2)
    if left.button("修改邮箱", width="stretch", key="change_registration_email"):
        reset_registration_otp()
        st.rerun()
    resend_label = f"重新发送验证码（{remaining}秒）" if remaining else "重新发送验证码"
    if right.button(
        resend_label,
        width="stretch",
        disabled=remaining > 0,
        key="resend_registration_otp",
    ):
        try:
            store.resend_signup_otp(email)
            st.session_state.registration_otp_sent_at = time.time()
            st.session_state.registration_notice = "新的验证码已经发送，请以最新一封邮件为准。"
            st.rerun()
        except Exception as exc:
            st.error(f"重新发送失败：{friendly_registration_error(exc)}")


def registration_otp_page(store: CloudStore) -> None:
    email = str(st.session_state.get("registration_email") or "")
    st.subheader("输入注册验证码")
    notice = str(st.session_state.get("registration_notice") or "")
    if notice:
        st.success(notice)
    st.write(f"验证码已发送至：**{email}**")
    st.caption("请返回本页面输入邮件中的数字验证码，不需要点击邮件中的任何网址。")
    with st.form("email_register_otp_form"):
        token = st.text_input(
            "邮箱验证码",
            key="register_otp",
            placeholder="请输入邮件中的验证码",
            max_chars=8,
        )
        submitted = st.form_submit_button("验证并创建账号", type="primary", width="stretch")
    if submitted:
        normalized_token = "".join(token.split())
        if not normalized_token.isdigit() or not 6 <= len(normalized_token) <= 8:
            st.error("请输入邮件中的6位验证码；如果邮件显示8位，也可以直接输入。")
        else:
            try:
                identity = store.verify_signup_otp(email, normalized_token)
                reset_registration_otp()
                set_logged_in(identity, store)
                st.session_state.auth_notice = "注册验证成功。你的资料、会话和历史分析将永久保存到当前账号。"
                st.rerun()
            except Exception as exc:
                st.error(f"验证失败：{friendly_registration_error(exc)}")
    registration_otp_controls(store)
    st.caption("验证码具有时效性且只能使用一次。请勿把验证码或登录密码告诉他人。")


def auth_page() -> None:
    render_brand("邮箱账号用于永久保存风险资料、股票会话、完整分析快照和加仓记录")
    try:
        store = configured_cloud_store()
    except CloudStoreError as exc:
        st.error("永久保存尚未完成最后一步配置。")
        st.write(str(exc))
        st.markdown(
            "请先在 Streamlit Community Cloud 的 **App settings → Secrets** 中加入：\n\n"
            "```toml\nSUPABASE_URL = \"你的 Project URL\"\n"
            "SUPABASE_PUBLISHABLE_KEY = \"你的 Publishable key\"\n```"
        )
        st.caption("不要填写 secret key 或 service_role key，也不要把真实密钥上传到 GitHub。")
        return

    if st.session_state.get("registration_otp_pending"):
        registration_otp_page(store)
        st.warning("该分析结果仅供参考，本模型仅用于学习与研究。")
        return

    login_tab, register_tab = st.tabs(["邮箱登录", "首次注册"])
    with login_tab:
        with st.form("email_login_form"):
            email = st.text_input("邮箱", key="login_email", placeholder="name@example.com")
            password = st.text_input("密码", type="password", key="login_password")
            submitted = st.form_submit_button("登录", type="primary", width="stretch")
        if submitted:
            if "@" not in email or len(password) < 8:
                st.error("请输入有效邮箱；密码至少8位。")
            else:
                try:
                    identity = store.sign_in(email.strip().lower(), password)
                    set_logged_in(identity, store)
                    st.rerun()
                except Exception as exc:
                    st.error(f"登录失败：{exc}")
    with register_tab:
        st.caption("同一邮箱在手机、平板或另一台电脑登录后，可以恢复该账号保存的数据。")
        with st.form("email_register_form"):
            email = st.text_input("注册邮箱", key="register_email", placeholder="name@example.com")
            password = st.text_input("设置密码（至少8位）", type="password", key="register_password")
            password_again = st.text_input("再次输入密码", type="password", key="register_password_again")
            submitted = st.form_submit_button("发送注册验证码", type="primary", width="stretch")
        if submitted:
            if "@" not in email:
                st.error("请输入有效邮箱。")
            elif len(password) < 8:
                st.error("密码至少8位。")
            elif password != password_again:
                st.error("两次输入的密码不一致。")
            else:
                try:
                    normalized_email = email.strip().lower()
                    identity = store.sign_up(normalized_email, password)
                    if identity.get("email_confirmation_required"):
                        begin_registration_otp(
                            normalized_email,
                            "注册验证码已经发送，请查看收件箱和垃圾邮件。",
                        )
                        st.rerun()
                    else:
                        set_logged_in(identity, store)
                        st.rerun()
                except Exception as exc:
                    st.error(f"发送失败：{friendly_registration_error(exc)}")

        with st.expander("昨天已经创建账号，但没有完成邮箱验证？"):
            with st.form("resume_unverified_signup_form"):
                pending_email = st.text_input(
                    "未验证的注册邮箱",
                    key="pending_registration_email",
                    placeholder="name@example.com",
                )
                resume_submitted = st.form_submit_button(
                    "继续未完成注册并发送验证码",
                    width="stretch",
                )
            if resume_submitted:
                if "@" not in pending_email:
                    st.error("请输入之前注册时使用的有效邮箱。")
                else:
                    normalized_email = pending_email.strip().lower()
                    try:
                        store.resend_signup_otp(normalized_email)
                        begin_registration_otp(
                            normalized_email,
                            "新的注册验证码已经发送，请以最新一封邮件为准。",
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"发送失败：{friendly_registration_error(exc)}")
    st.warning("该分析结果仅供参考，本模型仅用于学习与研究。")


def _session_load_profile() -> dict | None:
    return configured_cloud_store().load_profile(current_user_id())


def _session_load_draft() -> dict | None:
    return configured_cloud_store().load_draft(current_user_id())


def persist_draft(answers: dict, current_index: int) -> None:
    record = configured_cloud_store().save_draft(current_user_id(), answers, current_index)
    st.session_state.draft_record = record


def remove_draft() -> None:
    configured_cloud_store().delete_draft(current_user_id())
    st.session_state.draft_record = None


def persist_profile(profile: dict, risk_score: int, risk_level: str, version: int) -> dict:
    now = datetime.now().isoformat()
    record = {
        "profile_data": dict(profile),
        "risk_score": int(risk_score),
        "risk_level": risk_level,
        "version": int(version) + 1,
        "completed_at": now,
        "updated_at": now,
    }
    return configured_cloud_store().save_profile(current_user_id(), record)


def reload_cloud_sessions() -> None:
    st.session_state.stock_sessions = configured_cloud_store().load_sessions(current_user_id())


def load_current_user_data() -> None:
    if st.session_state.user_data_loaded:
        return
    try:
        profile_record = _session_load_profile()
        draft_record = _session_load_draft()
        sessions = configured_cloud_store().load_sessions(current_user_id())
    except Exception as exc:
        st.error("账号已登录，但云端资料读取失败。请确认已运行 V6.0 一键建表 SQL。")
        st.code(str(exc))
        st.stop()
    st.session_state.profile_record = profile_record
    st.session_state.saved_profile = profile_record.get("profile_data") if profile_record else None
    st.session_state.draft_record = draft_record
    st.session_state.stock_sessions = sessions
    st.session_state.question_answers = dict((draft_record or {}).get("answers") or {})
    st.session_state.question_index = int((draft_record or {}).get("current_index") or first_unanswered_index(st.session_state.question_answers))
    if not profile_record:
        st.session_state.questionnaire_mode = "first"
        st.session_state.view = "questionnaire"
    st.session_state.user_data_loaded = True


def render_brand(subtitle: str = "") -> None:
    st.markdown('<div class="app-brand">Five-year evidence · Personal suitability</div>', unsafe_allow_html=True)
    st.title("个人投资者股票决策辅助 Agent｜永久记忆版 V6.1")
    st.caption(subtitle or "近五年真实公开行情 · 历史相似状态检索 · 个人风险适配 · 教学研究原型")


def question_option_label(option: str, current: str | None) -> str:
    return f"✓ 当前答案　{option}" if option == current else option


def questionnaire_page() -> None:
    answers = dict(st.session_state.question_answers)
    index = int(st.session_state.question_index)
    total = len(QUESTIONS)
    render_brand("本账号首次测评提交后永久保存；日常换股不会再次要求填写")
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
                try:
                    persist_draft(updated, next_index)
                except Exception as exc:
                    st.error(f"本题暂未保存，请检查网络后重试：{exc}")
                    return
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
        except Exception as exc:
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
    render_brand("风险资料已永久保存到当前邮箱账号；日常只需更换股票并填写本次投资信息")
    if st.session_state.get("auth_notice"):
        st.info(str(st.session_state.auth_notice))
        st.session_state.auth_notice = ""
    risk_cols = st.columns([1, 1, 1.4])
    risk_cols[0].metric("个人风险等级", record.get("risk_level", "—"), f"{record.get('risk_score', '—')}/100")
    risk_cols[1].metric("测评版本", f"第 {record.get('version', 1)} 版")
    risk_cols[2].metric("资产范围", profile.get("asset_band", "—"))
    st.markdown('<div class="hero-card"><b>选择本次要分析的股票</b><br><span class="muted">Agent统一使用最近五年行情，识别当前波动状态并检索历史相似周期；上市不足五年时使用全部可得数据并降低结论置信度。</span></div>', unsafe_allow_html=True)
    st.caption("提示：在任何联网设备使用同一邮箱登录，即可恢复风险资料、股票会话、分析快照和买入记录。")

    stock_left, stock_right = st.columns([1, 1])
    with stock_left:
        st.radio("市场", ["A股", "美股", "港股"], horizontal=True, key="v5_market", on_change=v5_market_changed)
        placeholders = {
            "A股": "例如：600519、300750",
            "美股": "例如：AAPL、MSFT、NVDA",
            "港股": "例如：00700、09988，也可输入700或0700.HK",
        }
        placeholder = placeholders[st.session_state.v5_market]
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
            unit = {"A股": "元/股", "美股": "美元/股", "港股": "港元/股"}[st.session_state.v5_market]
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
            st.session_state.confirmed_analysis_id = datetime.now().isoformat(timespec="microseconds")
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
    st.subheader("当前邮箱账号的风险资料")
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
            try:
                remove_draft()
                st.session_state.draft_record = None
                st.rerun()
            except Exception as exc:
                st.error(f"草稿删除失败：{exc}")

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
            try:
                persist_draft({}, 0)
                st.session_state.question_answers = {}
                st.session_state.question_index = 0
                st.session_state.questionnaire_mode = "update"
                st.session_state.view = "questionnaire"
                st.session_state.confirm_profile_change = False
                st.rerun()
            except Exception as exc:
                st.error(f"新测评草稿创建失败：{exc}")


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


def logout_current_user() -> None:
    try:
        configured_cloud_store().sign_out()
    except Exception:
        pass
    st.session_state.clear()
    st.rerun()


def app_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📊 股票决策辅助 Agent")
        auth_user = st.session_state.get("auth_user") or {}
        st.caption(f"已登录：{auth_user.get('email', '—')}")
        record = st.session_state.profile_record or {}
        st.caption(record.get("risk_level", "风险资料未完成"))
        if st.button("股票分析", width="stretch"):
            st.session_state.view = "analysis"
            st.rerun()
        if st.button("个人中心", width="stretch"):
            st.session_state.view = "profile"
            st.rerun()
        st.divider()
        if st.button("💬 股票会话与组合", width="stretch", type="primary"):
            st.session_state.selected_session_key = None
            st.session_state.pending_delete_session = None
            st.session_state.view = "sessions"
            st.rerun()
        sessions = st.session_state.get("stock_sessions") or {}
        if sessions:
            st.caption("已保存的股票会话")
            ordered_sessions = sorted(
                sessions.items(),
                key=lambda item: str(item[1].get("updated_at") or ""),
                reverse=True,
            )
            for index, (key, session) in enumerate(ordered_sessions):
                code = session.get("code", "—")
                name = session.get("name", "—")
                principal = float(session.get("principal_rmb") or 0.0)
                label = f"{code} · {name}"
                if st.button(label, key=f"sidebar_session_{index}", width="stretch"):
                    st.session_state.pop(f"session_mode_{session.get('db_id') or key}", None)
                    st.session_state.pop(f"snapshot_choice_{session.get('db_id') or ''}", None)
                    st.session_state.selected_session_key = key
                    st.session_state.pending_delete_session = None
                    st.session_state.session_detail_mode = "完整分析"
                    st.session_state.view = "sessions"
                    st.rerun()
                st.caption(f"已记录本金：{principal:,.3f} 元")
        else:
            st.caption("完成一次股票分析后，会自动建立独立会话。")
        st.divider()
        st.caption("行情统一使用最近五年；相似样本不足时拒绝预测；新股标注低置信度。")
        if st.button("退出当前邮箱账号", width="stretch"):
            logout_current_user()


def render_session_portfolio() -> None:
    sessions = st.session_state.get("stock_sessions") or {}
    portfolio = portfolio_from_sessions(sessions)
    st.subheader("会话组合持仓")
    st.caption("此处是多只股票在会话组合内部的占比，不会改变原有单只股票风险预算和仓位判断。")
    metric_cols = st.columns(3)
    metric_cols[0].metric("保留的股票会话", str(portfolio["conversation_count"]))
    metric_cols[1].metric("计入组合的股票", str(portfolio["position_count"]))
    metric_cols[2].metric("已记录投入本金", money(portfolio["total_principal_rmb"]))

    rows = portfolio["rows"]
    if rows:
        table_rows = [
            {
                "市场": item["market"],
                "股票代码": item["code"],
                "股票名称": item["name"],
                "投入本金（人民币元）": f"{item['principal_rmb']:,.3f}",
                "会话组合占比": f"{item['weight']:.3%}",
            }
            for item in rows
        ]
        left, right = st.columns([1.25, 1])
        with left:
            st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")
        with right:
            figure = go.Figure(
                data=[
                    go.Pie(
                        labels=[f"{item['code']} · {item['name']}" for item in rows],
                        values=[item["principal_rmb"] for item in rows],
                        hole=0.58,
                        textinfo="label+percent",
                        hovertemplate="%{label}<br>本金：%{value:,.3f} 元<br>占比：%{percent:.3%}<extra></extra>",
                    )
                ]
            )
            figure.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340, showlegend=False)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    else:
        st.info("当前没有已记录实际投入本金的股票。仅查询但尚未买入的会话仍会保留，但不会计入持仓占比。")

    st.markdown("#### 全部股票会话")
    if not sessions:
        st.caption("暂无会话。请先进行一次股票分析。")
        return
    ordered_sessions = sorted(
        sessions.items(),
        key=lambda item: str(item[1].get("updated_at") or ""),
        reverse=True,
    )
    weights = {item["key"]: item["weight"] for item in rows}
    for index, (key, session) in enumerate(ordered_sessions):
        cols = st.columns([4, 1.2, 1.2])
        title = f"{session.get('code', '—')} · {session.get('name', '—')}（{session.get('market', '—')}）"
        if cols[0].button(title, key=f"portfolio_session_{index}", width="stretch"):
            st.session_state.pop(f"session_mode_{session.get('db_id') or key}", None)
            st.session_state.pop(f"snapshot_choice_{session.get('db_id') or ''}", None)
            st.session_state.selected_session_key = key
            st.session_state.pending_delete_session = None
            st.session_state.session_detail_mode = "完整分析"
            st.rerun()
        cols[1].metric("投入本金", f"{float(session.get('principal_rmb') or 0.0):,.3f} 元")
        cols[2].metric("组合占比", f"{weights.get(key, 0.0):.3%}" if key in weights else "未计入")


def render_saved_analysis(session: dict) -> None:
    session_id = str(session.get("db_id") or "")
    if not session_id:
        st.info("该会话还没有云端编号，请重新分析一次后再查看完整历史结果。")
        return
    try:
        store = configured_cloud_store()
        snapshots = store.list_snapshots(current_user_id(), session_id)
    except Exception as exc:
        st.error(f"历史分析列表读取失败：{exc}")
        return
    if not snapshots:
        st.info("该会话尚无完整分析快照。请点击下方“按最新持仓重新分析”，完成后会永久保存详细结果。")
        return

    labels: list[str] = []
    for index, item in enumerate(snapshots):
        created = str(item.get("created_at") or "").replace("T", " ")[:19]
        conclusion = str((item.get("summary") or {}).get("conclusion") or "已完成分析")
        prefix = "第一次分析" if index == 0 else f"第{index + 1}次分析"
        labels.append(f"{prefix}｜{created or '时间未知'}｜{conclusion}")
    chosen_label = st.selectbox(
        "选择要恢复的分析版本",
        labels,
        index=0,
        key=f"snapshot_choice_{session_id}",
        help="默认显示第一次分析；重新分析只会新增版本，不会覆盖旧结果。",
    )
    selected = snapshots[labels.index(chosen_label)]
    try:
        row = store.load_snapshot(current_user_id(), str(selected.get("id") or ""))
        if not row:
            raise ValueError("未找到该分析快照。")
        restored = restore_analysis_snapshot(row.get("snapshot_data") or {})
    except Exception as exc:
        st.error(f"完整分析恢复失败：{exc}")
        return

    bundle = restored["bundle"]
    analysis = restored["analysis"]
    profile = restored["profile"]
    holding_state = restored["holding_state"]
    first_date = bundle.stock["日期"].min().date()
    last_date = bundle.stock["日期"].max().date()
    st.success(
        f"已恢复：{bundle.code}｜{bundle.name}｜分析区间 {first_date} 至 {last_date}｜"
        f"共 {len(bundle.stock)} 个交易日。"
    )
    st.caption(f"当次行情来源：{bundle.provider}；这是已保存的历史快照，不会用今天的数据改写。")
    if not bundle.history_complete:
        st.warning("该次分析可得历史不足五年。数据不足，无法准确判断，结果仅作低置信度参考。")

    previous_holding_state = st.session_state.get("confirmed_holding_state", "尚未持有")
    previous_holding_method = st.session_state.get("confirmed_holding_method", "按持股数量填写")
    st.session_state.confirmed_holding_state = holding_state
    st.session_state.confirmed_holding_method = restored["holding_method"]
    try:
        view_mode = st.radio(
            "结果显示",
            ["简明模式", "专业模式"],
            horizontal=True,
            key=f"saved_view_mode_{selected.get('id')}",
        )
        tab_names = ["结论"]
        if holding_state == "已经持有":
            tab_names.append("卖出信号")
        tab_names.extend(["相似周期预测", "风险与仓位", "持有周期", "数据证据"])
        if view_mode == "专业模式":
            tab_names.append("专业指标")
        tabs = st.tabs(tab_names)
        tab_map = dict(zip(tab_names, tabs))
        with tab_map["结论"]:
            render_summary(bundle, analysis, profile)
        if "卖出信号" in tab_map:
            with tab_map["卖出信号"]:
                render_sell_signals(bundle, analysis, profile)
        with tab_map["相似周期预测"]:
            render_analog_forecast(analysis)
        with tab_map["风险与仓位"]:
            render_risk_budget(analysis, profile)
        with tab_map["持有周期"]:
            render_horizons(analysis)
        with tab_map["数据证据"]:
            render_evidence(bundle, analysis)
        if view_mode == "专业模式":
            with tab_map["专业指标"]:
                render_professional(bundle, analysis)
    finally:
        st.session_state.confirmed_holding_state = previous_holding_state
        st.session_state.confirmed_holding_method = previous_holding_method


def render_position_entry(session: dict) -> None:
    session_id = str(session.get("db_id") or "")
    if not session_id:
        st.error("该股票会话尚未同步到云端，暂时不能登记买入。")
        return
    principal = float(session.get("principal_rmb") or 0.0)
    shares = session.get("total_shares")
    shares_known = bool(session.get("shares_complete")) and shares is not None
    metrics = st.columns(4)
    metrics[0].metric("累计投入本金", f"{principal:,.3f} 元")
    metrics[1].metric("已记录股数", f"{float(shares):,.3f} 股" if shares_known else "不完整／未知")
    metrics[2].metric(
        "人民币口径平均成本／股",
        f"{principal / float(shares):,.3f} 元" if shares_known and float(shares) > 0 else "无法计算",
    )
    portfolio = portfolio_from_sessions(st.session_state.get("stock_sessions") or {})
    weight = next((item["weight"] for item in portfolio["rows"] if item["key"] == session.get("key")), 0.0)
    metrics[3].metric("会话组合占比", f"{weight:.3%}" if principal > 0 else "未计入")
    st.info("此处仅记录你已经实际完成的买入，不会连接券商，也不会替你执行真实交易。")

    method = st.radio(
        "录入方式",
        ["按实际支付人民币总额", "按股数和成交单价"],
        horizontal=True,
        key=f"position_method_{session_id}",
        help="如果软件只显示股数和成交价，选择第二项，Agent会自动折算本次投入本金。",
    )
    trade_date = st.date_input("成交日期", value=date.today(), key=f"position_date_{session_id}")
    amount_rmb = shares_bought = trade_price = fees_rmb = 0.0
    fx_rate = 1.0
    market = str(session.get("market") or "A股")
    if method == "按实际支付人民币总额":
        amount_rmb = st.number_input(
            "本次实际支付总额（人民币元，包含交易费用）",
            min_value=0.0,
            step=1000.0,
            format="%.3f",
            key=f"position_amount_{session_id}",
        )
    else:
        entry_cols = st.columns(3)
        shares_bought = entry_cols[0].number_input(
            "本次买入股数",
            min_value=0.0,
            step=100.0 if market == "A股" else 1.0,
            format="%.3f",
            key=f"position_shares_{session_id}",
        )
        native_unit = {"A股": "人民币元/股", "美股": "美元/股", "港股": "港元/股"}.get(market, "元/股")
        trade_price = entry_cols[1].number_input(
            f"成交单价（{native_unit}）",
            min_value=0.0,
            step=0.001,
            format="%.3f",
            key=f"position_price_{session_id}",
        )
        fees_rmb = entry_cols[2].number_input(
            "本次交易费用（人民币元）",
            min_value=0.0,
            step=1.0,
            format="%.3f",
            key=f"position_fees_{session_id}",
        )
        if market in {"美股", "港股"}:
            default_fx = 0.0
            try:
                snapshot = cached_usd_cny_rate() if market == "美股" else cached_hkd_cny_rate()
                default_fx = float(snapshot.get("rate") or 0.0)
                st.caption(f"已带入最近公开的{snapshot.get('provider', '参考')}汇率；请按你的实际换汇口径修改。")
            except Exception:
                st.caption("未能自动取得汇率，请填写成交时使用的外币兑人民币汇率。")
            fx_rate = st.number_input(
                "1单位外币折合人民币",
                min_value=0.0,
                value=default_fx,
                step=0.001,
                format="%.3f",
                key=f"position_fx_{session_id}",
            )
    note = st.text_input("本次记录备注（可不填）", key=f"position_note_{session_id}")
    preview = None
    try:
        preview = calculate_position_transaction(
            market=market,
            input_method=method,
            amount_rmb=amount_rmb,
            shares=shares_bought,
            price_native=trade_price,
            fx_rate=fx_rate,
            fees_rmb=fees_rmb,
        )
    except ValueError:
        preview = None
    if preview:
        st.success(f"本次将计入投入本金：{float(preview['principal_rmb']):,.3f} 元。保存后组合占比自动重算。")
    else:
        st.caption("填写完整后，这里会先显示本次计入的人民币本金。")

    if st.button("永久保存本次首次买入／加仓", type="primary", width="stretch", key=f"save_position_{session_id}"):
        try:
            transaction = calculate_position_transaction(
                market=market,
                input_method=method,
                amount_rmb=amount_rmb,
                shares=shares_bought,
                price_native=trade_price,
                fx_rate=fx_rate,
                fees_rmb=fees_rmb,
            )
            transaction_type = "initial" if principal <= 0 else "add"
            transaction.update(
                {
                    "transaction_type": transaction_type,
                    "trade_date": trade_date.isoformat(),
                    "note": note.strip(),
                }
            )
            store = configured_cloud_store()
            user_id = current_user_id()
            store.insert_transaction(user_id, session_id, transaction)
            refreshed = store.get_session(user_id, session_id)
            if not refreshed:
                raise CloudStoreError("买入记录已提交，但未能重新读取股票会话。")
            messages = list(refreshed.get("messages") or []) + build_position_messages(
                transaction_type=transaction_type,
                transaction=transaction,
                trade_date=trade_date.isoformat(),
                note=note,
            )
            store.update_stock_session(user_id, session_id, messages=messages)
            reload_cloud_sessions()
            st.success("已永久保存。累计投入本金和全部股票持仓占比已更新。")
            st.rerun()
        except Exception as exc:
            st.error(f"保存失败：{exc}")

    st.markdown("#### 已保存的买入／加仓记录")
    try:
        transactions = configured_cloud_store().list_transactions(current_user_id(), session_id)
    except Exception as exc:
        st.error(f"买入记录读取失败：{exc}")
        transactions = []
    if not transactions:
        st.caption("暂无记录。")
    for index, item in enumerate(transactions):
        action = "首次买入" if item.get("transaction_type") == "initial" else "加仓"
        label = f"{item.get('trade_date', '—')}｜{action}｜{float(item.get('principal_rmb') or 0.0):,.3f} 元"
        with st.expander(label):
            if item.get("shares") is not None:
                st.write(
                    f"股数：{float(item['shares']):,.3f}；成交价：{float(item.get('price_native') or 0.0):,.3f} "
                    f"{item.get('currency', '')}/股；汇率：{float(item.get('fx_rate') or 1.0):,.3f}；"
                    f"费用：{float(item.get('fees_rmb') or 0.0):,.3f} 元。"
                )
            if item.get("note"):
                st.caption(f"备注：{item['note']}")
            transaction_id = str(item.get("id") or "")
            if st.button("撤销这条误录记录", key=f"request_delete_tx_{index}_{transaction_id}"):
                st.session_state.pending_delete_transaction = transaction_id
                st.rerun()
            if st.session_state.get("pending_delete_transaction") == transaction_id:
                st.warning("仅在确属误录时撤销；撤销后，本金和持仓占比会自动回退。")
                cancel, confirm = st.columns(2)
                if cancel.button("取消", key=f"cancel_delete_tx_{transaction_id}", width="stretch"):
                    st.session_state.pending_delete_transaction = None
                    st.rerun()
                if confirm.button("确认撤销", key=f"confirm_delete_tx_{transaction_id}", type="primary", width="stretch"):
                    try:
                        store = configured_cloud_store()
                        user_id = current_user_id()
                        store.delete_transaction(user_id, transaction_id)
                        refreshed = store.get_session(user_id, session_id)
                        if refreshed:
                            messages = list(refreshed.get("messages") or []) + [
                                {
                                    "role": "assistant",
                                    "content": f"已撤销{item.get('trade_date', '')}的误录记录，组合本金和占比已重新计算。",
                                    "created_at": datetime.now().isoformat(timespec="seconds"),
                                    "kind": "position_undo",
                                }
                            ]
                            store.update_stock_session(user_id, session_id, messages=messages)
                        reload_cloud_sessions()
                        st.session_state.pending_delete_transaction = None
                        st.rerun()
                    except Exception as exc:
                        st.error(f"撤销失败：{exc}")


def render_session_messages(session: dict, selected_key: str) -> None:
    session_id = str(session.get("db_id") or "")
    with st.expander("校正累计实际投入本金"):
        st.caption("一般应通过“首次买入／加仓”逐笔登记；仅在迁移旧持仓或发现汇总数有误时使用本项。")
        principal_value = st.number_input(
            "累计实际投入本金（人民币元）",
            min_value=0.0,
            value=float(session.get("principal_rmb") or 0.0),
            step=1000.0,
            format="%.3f",
            key=f"session_principal_{session_id}",
        )
        if st.button("保存校正值", key=f"save_principal_{session_id}", type="primary"):
            try:
                updated = set_invested_principal(st.session_state.stock_sessions, selected_key, principal_value)
                record = updated[selected_key]
                configured_cloud_store().update_stock_session(
                    current_user_id(),
                    session_id,
                    principal_rmb=float(record["principal_rmb"]),
                    principal_source=str(record["principal_source"]),
                )
                reload_cloud_sessions()
                st.rerun()
            except Exception as exc:
                st.error(f"保存失败：{exc}")

    for message in session.get("messages") or []:
        role = message.get("role") if message.get("role") in {"user", "assistant"} else "assistant"
        with st.chat_message(role):
            st.markdown(str(message.get("content") or ""))
            st.caption(str(message.get("created_at") or "").replace("T", " "))
    note = st.chat_input("为这只股票添加一条永久备注", key=f"session_note_{session_id}")
    if note:
        try:
            updated = append_note(st.session_state.stock_sessions, selected_key, note)
            messages = updated[selected_key].get("messages") or []
            configured_cloud_store().update_stock_session(current_user_id(), session_id, messages=messages)
            reload_cloud_sessions()
            st.rerun()
        except Exception as exc:
            st.error(f"备注保存失败：{exc}")


def open_reanalysis_form(session: dict) -> None:
    market = str(session.get("market") or "A股")
    code = str(session.get("code") or "")
    principal = float(session.get("principal_rmb") or 0.0)
    shares = session.get("total_shares")
    shares_known = bool(session.get("shares_complete")) and shares is not None and float(shares) > 0
    clear_stock_widgets()
    st.session_state.v5_market = market
    st.session_state.v5_stock_code = code
    if principal > 0:
        st.session_state.v5_holding_state = "已经持有"
        if shares_known:
            st.session_state.v5_holding_method = "按持股数量填写"
            st.session_state.v5_share_count = float(shares)
            st.session_state.v5_cost_price = 0.0
        else:
            st.session_state.v5_holding_method = "按持仓金额填写"
            st.session_state.v5_current_value = 0.0
            st.session_state.v5_total_cost = principal
    st.session_state.auth_notice = "已带入股票和已知持仓。请核对当前持仓市值／成本后，再生成一个新的永久分析快照。"


def stock_session_page() -> None:
    render_brand("同一邮箱跨设备永久恢复；每只股票独立保存完整分析、买入记录与会话")
    navigation = st.columns(2)
    if navigation[0].button("会话组合", width="stretch"):
        st.session_state.selected_session_key = None
        st.session_state.pending_delete_session = None
        st.rerun()
    if navigation[1].button("＋ 分析另一只股票", type="primary", width="stretch"):
        clear_stock_widgets()
        st.rerun()

    sessions = st.session_state.get("stock_sessions") or {}
    selected_key = st.session_state.get("selected_session_key")
    if not selected_key or selected_key not in sessions:
        render_session_portfolio()
        st.divider()
        st.warning("该分析结果仅供参考，本模型仅用于学习与研究。")
        return

    session = sessions[selected_key]
    st.subheader(f"{session.get('code', '—')} · {session.get('name', '—')}")
    st.caption(f"{session.get('market', '—')}｜建立于 {str(session.get('created_at') or '—').replace('T', ' ')[:19]}")
    latest = session.get("latest_summary") or {}
    if latest:
        summary_cols = st.columns(4)
        summary_cols[0].metric("最近结论", latest.get("conclusion", "—"))
        summary_cols[1].metric("个人适配", latest.get("suitability", "—"))
        summary_cols[2].metric("股票风险", latest.get("stock_risk_level", "—"))
        summary_cols[3].metric("复核／持有周期", latest.get("selected_horizon", "—"))

    modes = ["完整分析", "首次买入／加仓", "会话记录"]
    preferred = st.session_state.get("session_detail_mode", "完整分析")
    mode = st.radio(
        "会话功能",
        modes,
        index=modes.index(preferred) if preferred in modes else 0,
        horizontal=True,
        key=f"session_mode_{session.get('db_id') or selected_key}",
    )
    st.session_state.session_detail_mode = mode
    if mode == "完整分析":
        render_saved_analysis(session)
    elif mode == "首次买入／加仓":
        render_position_entry(session)
    else:
        render_session_messages(session, selected_key)

    st.divider()
    action_cols = st.columns(2)
    if action_cols[0].button("按最新持仓重新分析", width="stretch"):
        open_reanalysis_form(session)
        st.rerun()
    if action_cols[1].button("删除该股票会话", width="stretch"):
        st.session_state.pending_delete_session = selected_key
        st.rerun()

    if st.session_state.get("pending_delete_session") == selected_key:
        st.error("确认删除吗？删除后业务上视为该股票已全部卖出；完整分析、买入记录、本金和组合占比都会永久剔除。")
        delete_cols = st.columns(2)
        if delete_cols[0].button("取消删除", width="stretch"):
            st.session_state.pending_delete_session = None
            st.rerun()
        if delete_cols[1].button("确认删除并视为已卖出", type="primary", width="stretch"):
            try:
                configured_cloud_store().delete_stock_session(current_user_id(), str(session.get("db_id") or ""))
                st.session_state.stock_sessions = delete_session(sessions, selected_key)
                st.session_state.selected_session_key = None
                st.session_state.pending_delete_session = None
                st.rerun()
            except Exception as exc:
                st.error(f"删除失败：{exc}")

    st.warning("该分析结果仅供参考，本模型仅用于学习与研究。")


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
    analog_cols[2].metric("最高相似度", f"{best_similarity:.3f}/100" if best_similarity is not None else "暂无可比候选")
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
        selection_mode = selected_analog.get("selection_mode", "同股历史样本")
        st.info(
            f"本期限采用{selection_mode}{selected_analog['sample_count']}个；随后"
            f"{selected_analog['days']}个交易日上涨样本占比为{selected_analog['positive_ratio']:.3%}，"
            f"收益中位数{selected_analog['median_return']:.3%}，"
            f"中间50%区间{selected_analog['q25_return']:.3%}至{selected_analog['q75_return']:.3%}。"
        )
    elif not analog.get("available"):
        st.warning("本股在近五年窗口内没有形成达到最低要求的相似样本；请在相似周期页查看逐期限原因。")
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
            if holding.get("fx_rate") is not None:
                st.caption(
                    f"人民币市值按 1 {holding.get('native_currency', '外币')} = {float(holding['fx_rate']):.3f} 元换算；"
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


def render_sell_signals(bundle, analysis, profile) -> None:
    sell = analysis.get("sell_signals") or {}
    st.subheader("已有持仓的卖出信号")
    st.caption("该模块只管理已经持有的股票，不改变原有买入分析、风险等级、持有周期或仓位预算。")
    if not sell.get("available"):
        st.warning(sell.get("summary", "当前数据不足，无法形成卖出信号。"))
        return

    status = str(sell.get("status") or "证据不足")
    message = f"### 当前状态：{status}\n{sell.get('summary', '')}"
    if status == "退出复核":
        st.error(message)
    elif status == "考虑分批减仓":
        st.warning(message)
    elif status == "警戒观察":
        st.info(message)
    else:
        st.success(message)

    metrics = st.columns(5)
    metrics[0].metric("最新公开收盘价", f"{float(sell['latest_price']):,.3f} {bundle.price_unit}")
    metrics[1].metric(
        "持仓收益率",
        pct(sell.get("current_return")) if sell.get("current_return") is not None else "成本未知",
    )
    metrics[2].metric(
        f"近{int(sell['peak_window'])}日高点回撤",
        pct(sell.get("peak_drawdown")),
    )
    metrics[3].metric("核心信号", f"{int(sell.get('hard_count', 0))}/2")
    metrics[4].metric("辅助信号", f"{int(sell.get('auxiliary_count', 0))}/4")
    st.caption(
        f"数据日期：{pd.Timestamp(sell['latest_date']).date().isoformat()}；"
        f"Agent所选周期：{sell.get('selected_horizon', '—')}；{sell.get('next_review', '按期复核')}。"
    )

    rows = []
    for signal in sell.get("signals", []):
        rows.append(
            {
                "信号": signal["name"],
                "级别": signal["level"],
                "当前状态": signal["state"],
                "当前证据": signal["current_text"],
                "触发规则": signal["threshold_text"],
                "解释": signal["detail"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("#### 动态价格参考")
    price_cols = st.columns(3)
    price_cols[0].metric(
        "成本风险参考价",
        (
            f"{float(sell['cost_protection_price']):,.3f} {bundle.price_unit}"
            if sell.get("cost_protection_price") is not None
            else "成本价未知"
        ),
    )
    price_cols[1].metric(
        f"MA{int(sell['fast_window'])}趋势参考",
        f"{float(sell['fast_ma']):,.3f} {bundle.price_unit}",
    )
    price_cols[2].metric(
        "盈利回撤保护价",
        (
            f"{float(sell['trailing_protection_price']):,.3f} {bundle.price_unit}"
            if sell.get("trailing_protection_price") is not None
            else "尚未启用"
        ),
    )
    st.caption(
        "这些价格会随最新行情、股票波动率和Agent选择的持有周期变化；它们是收盘后复核参考，不是保证成交的自动订单。"
    )

    close = analysis["metrics"]["close"]
    display_close = close.tail(min(252, len(close)))
    fast_series = close.rolling(int(sell["fast_window"])).mean().reindex(display_close.index)
    slow_series = close.rolling(int(sell["slow_window"])).mean().reindex(display_close.index)
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=display_close.index,
            y=display_close,
            name="收盘价",
            line=dict(color="#2563eb", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价 %{y:,.3f}<extra></extra>",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=fast_series.index,
            y=fast_series,
            name=f"MA{int(sell['fast_window'])}",
            line=dict(color="#f59e0b", width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>快线 %{y:,.3f}<extra></extra>",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=slow_series.index,
            y=slow_series,
            name=f"MA{int(sell['slow_window'])}",
            line=dict(color="#7c3aed", width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>慢线 %{y:,.3f}<extra></extra>",
        )
    )
    if sell.get("cost_protection_price") is not None:
        chart.add_hline(
            y=float(sell["cost_protection_price"]),
            line_dash="dot",
            line_color="#dc2626",
            annotation_text="成本风险参考",
        )
    if sell.get("trailing_protection_price") is not None:
        chart.add_hline(
            y=float(sell["trailing_protection_price"]),
            line_dash="dot",
            line_color="#16a34a",
            annotation_text="盈利回撤保护",
        )
    chart.update_layout(
        title="最近一年价格与动态复核线",
        height=410,
        margin=dict(l=20, r=20, t=55, b=20),
        yaxis_title=bundle.price_unit,
        yaxis_tickformat=",.3f",
        legend_orientation="h",
    )
    st.plotly_chart(chart, width="stretch")

    st.markdown("#### 下一步观察条件")
    for condition in sell.get("reference_conditions", []):
        st.write(f"- {condition}")
    st.info(
        "分级规则：没有触发信号为“继续持有”；出现核心或辅助信号为“警戒观察”；"
        "一个核心信号与至少一个辅助信号同时出现为“考虑分批减仓”；两个核心信号同时出现为“退出复核”。"
    )
    with st.expander("使用限制与执行风险"):
        for limitation in sell.get("limitations", []):
            st.write(f"- {limitation}")
        st.write("- 实际操作仍需结合公告、停牌、流动性、税费以及个人交易纪律人工确认。")


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
                "相似周期修正": item.get("analog_status") or "未进行相似周期修正",
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
                "严格样本": item.get("strict_sample_count", 0),
                "选样方式": item.get("selection_mode", "未形成样本"),
                "相似度门槛": f"≥{float(item.get('selection_threshold', 0)):.3f}/100",
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
                "样本说明": item.get("reason", "—"),
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

    sample_cols = st.columns(4)
    sample_cols[0].metric("可比较历史时点", f"{forecast.get('candidate_count', 0)}个")
    sample_cols[1].metric("严格门槛候选", f"{forecast.get('eligible_candidate_count', 0)}个")
    best_similarity = forecast.get("best_similarity")
    sample_cols[2].metric(
        "最高相似度",
        f"{float(best_similarity):.3f}/100" if best_similarity is not None else "暂无候选",
    )
    sample_cols[3].metric(
        "总体可信度",
        f"{forecast.get('confidence_score', 0)}/100 · {forecast.get('confidence_label', '样本不足')}",
    )

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
        st.warning("各期限没有同时达到最低样本数与可信度要求；具体门槛、样本数和原因已列在表格中。")

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
        st.caption(f"窗口选样方式：{forecast.get('matches_selection_mode', '同股历史样本')}。")
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
            st.info("市场基准未返回可比较数据；不会用其他股票冒充基准样本。")

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
            - 相似周期检索使用收益、波动、回撤、均线位置、成交量和市场基准特征；先使用相似度不低于72分的严格同股样本。
            - 严格样本不足10个时，系统自动尝试相似度不低于60分的同股样本，并下调可信度；仍不足10个时不形成该期限预测。
            - 本股样本不可靠时，仅允许高可信度市场基准作不超过正负3分的小幅修正，并清楚标注来源；不会用其他个股冒充样本。
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
                    if confirmed_market == "美股":
                        fx_snapshot = cached_usd_cny_rate()
                    elif confirmed_market == "港股":
                        fx_snapshot = cached_hkd_cny_rate()
                    else:
                        fx_snapshot = None
                    holding_snapshot = calculate_holding_values(
                        confirmed_market,
                        st.session_state.confirmed_share_count,
                        last_price,
                        st.session_state.confirmed_cost_price,
                        usd_cny_rate=float(fx_snapshot["rate"]) if fx_snapshot and confirmed_market == "美股" else None,
                        hkd_cny_rate=float(fx_snapshot["rate"]) if fx_snapshot and confirmed_market == "港股" else None,
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
            analysis["sell_signals"] = (
                analyze_sell_signals(bundle, analysis, profile, holding_snapshot)
                if st.session_state.confirmed_holding_state == "已经持有"
                else None
            )
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

    try:
        event_id = str(st.session_state.get("confirmed_analysis_id") or "")
        if not event_id:
            event_id = f"{confirmed_market}:{confirmed_code}:{st.session_state.analysis_request_token}"
        sessions_before = st.session_state.get("stock_sessions") or {}
        current_key = session_key(confirmed_market, bundle.code)
        is_new_session = current_key not in sessions_before
        updated_sessions, _ = upsert_analysis_session(
            sessions_before,
            event_id=event_id,
            market=confirmed_market,
            code=bundle.code,
            name=bundle.name,
            analysis=analysis,
            holding_state=st.session_state.confirmed_holding_state,
            holding_snapshot=holding_snapshot,
        )
        local_record = updated_sessions[current_key]
        store = configured_cloud_store()
        user_id = current_user_id()
        cloud_record = store.upsert_stock_session(
            user_id,
            local_record,
            include_position=is_new_session,
        )
        updated_sessions[current_key] = cloud_record
        st.session_state.stock_sessions = updated_sessions
        snapshot_data = build_analysis_snapshot(
            bundle=bundle,
            analysis=analysis,
            profile=profile,
            holding_state=st.session_state.confirmed_holding_state,
            holding_method=st.session_state.confirmed_holding_method,
            holding_snapshot=holding_snapshot,
        )
        store.save_snapshot(
            user_id,
            str(cloud_record.get("db_id") or ""),
            event_id,
            cloud_record.get("latest_summary") or {},
            snapshot_data,
        )
    except Exception as exc:
        st.info("本次原有分析已经完成，但永久会话或完整快照暂未保存；分析计算本身不受影响。")
        st.caption(f"保存失败原因：{exc}")

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
    tab_names = ["结论"]
    if st.session_state.confirmed_holding_state == "已经持有":
        tab_names.append("卖出信号")
    tab_names.extend(["相似周期预测", "风险与仓位", "持有周期", "数据证据"])
    if view_mode == "专业模式":
        tab_names.append("专业指标")
    tabs = st.tabs(tab_names)
    tab_map = dict(zip(tab_names, tabs))
    with tab_map["结论"]:
        render_summary(bundle, analysis, profile)
    if "卖出信号" in tab_map:
        with tab_map["卖出信号"]:
            render_sell_signals(bundle, analysis, profile)
    with tab_map["相似周期预测"]:
        render_analog_forecast(analysis)
    with tab_map["风险与仓位"]:
        render_risk_budget(analysis, profile)
    with tab_map["持有周期"]:
        render_horizons(analysis)
    with tab_map["数据证据"]:
        render_evidence(bundle, analysis)
    if view_mode == "专业模式":
        with tab_map["专业指标"]:
            render_professional(bundle, analysis)

    st.divider()
    st.warning("该分析结果仅供参考，本模型仅用于学习与研究。")
    if st.button("打开本股票会话／登记实际投入本金", type="primary", width="stretch"):
        current_session = (st.session_state.get("stock_sessions") or {}).get(session_key(confirmed_market, bundle.code), {})
        st.session_state.pop(
            f"session_mode_{current_session.get('db_id') or session_key(confirmed_market, bundle.code)}",
            None,
        )
        st.session_state.pop(f"snapshot_choice_{current_session.get('db_id') or ''}", None)
        st.session_state.selected_session_key = session_key(confirmed_market, bundle.code)
        st.session_state.pending_delete_session = None
        st.session_state.session_detail_mode = "完整分析"
        st.session_state.view = "sessions"
        st.rerun()
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
        st.session_state.confirmed_analysis_id = datetime.now().isoformat(timespec="microseconds")
        st.rerun()


initialize_v5_state()
if not st.session_state.get("auth_user"):
    auth_page()
    st.stop()
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
elif st.session_state.view == "sessions":
    stock_session_page()
else:
    analysis_home()
