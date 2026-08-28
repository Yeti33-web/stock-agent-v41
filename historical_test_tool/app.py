from __future__ import annotations

from datetime import date
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

from historical_test_tool.runner import load_test_profile, run_historical_replay, snapshot_json


st.set_page_config(page_title="独立历史时点复现工具", page_icon="🧪", layout="wide")


def _percent(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "—"


def _display_value(value: object) -> object:
    if value is None:
        return "—"
    if isinstance(value, float):
        return round(value, 4)
    return value


def render_snapshot(snapshot: dict) -> None:
    conditions = snapshot["测试条件"]
    history = snapshot["历史数据"]
    decision = snapshot["当时Agent判断"]

    st.divider()
    st.subheader(f"{conditions['股票名称']}（{conditions['股票代码']}）历史复现结果")
    st.caption("这里只展示当时的Agent判断。程序没有读取或评价T日之后的走势。")

    st.dataframe(
        pd.DataFrame(
            [
                ["用户输入日期T", conditions["用户输入日期T"]],
                ["实际采用交易日", conditions["实际采用交易日"]],
                ["基准指数", conditions["基准指数"]],
                ["历史范围", f"{history['个股首日']} 至 {history['个股末日']}（{history['个股行数']}行）"],
                ["测试画像", conditions["测试画像"]],
            ],
            columns=["项目", "结果"],
        ),
        hide_index=True,
        width="stretch",
    )

    conclusion = str(decision.get("结论") or "证据不足")
    reason = str(decision.get("结论理由") or "—")
    if conclusion.startswith("不适合") or conclusion.startswith("证据不足"):
        st.error(f"Agent当时结论：{conclusion}\n\n{reason}")
    elif "暂缓" in conclusion or "观察" in conclusion:
        st.warning(f"Agent当时结论：{conclusion}\n\n{reason}")
    else:
        st.success(f"Agent当时结论：{conclusion}\n\n{reason}")

    columns = st.columns(5)
    columns[0].metric("自动周期评分", f"{decision.get('综合观察分_自动周期') or '—'}/100")
    columns[1].metric("自动选择周期", decision.get("自动选择周期") or "—")
    columns[2].metric("股票风险", decision.get("股票风险等级") or "—", f"{decision.get('股票风险分') or '—'}/100")
    columns[3].metric("个人适配", decision.get("个人适配") or "—")
    columns[4].metric("数据完整度", f"{decision.get('数据完整度') or 0:.0f}/100")

    if decision.get("自动选择周期"):
        st.info(
            f"现有Agent根据T日以前的数据，自动给出的持有／复核周期是“{decision['自动选择周期']}”。"
            "你可以按这个周期自行查看T日之后的真实行情。"
        )

    tab_summary, tab_factors, tab_evidence = st.tabs(["判断理由", "因子贡献", "证据与防泄漏"])
    with tab_summary:
        left, right = st.columns(2)
        with left:
            st.markdown("#### 主要时点理由")
            reasons = decision.get("主要时点理由") or []
            st.write("\n".join(f"- {item}" for item in reasons) if reasons else "没有形成明确支持理由。")
        with right:
            st.markdown("#### 主要风险与数据限制")
            limits = (decision.get("主要风险理由") or []) + (decision.get("数据限制") or [])
            st.write("\n".join(f"- {item}" for item in limits) if limits else "没有额外提示。")

        horizon_rows = snapshot.get("全部现有周期评分") or []
        st.markdown("#### 现有Agent全部周期评分")
        st.dataframe(pd.DataFrame(horizon_rows), hide_index=True, width="stretch")

    with tab_factors:
        for title, rows in (snapshot.get("因子贡献") or {}).items():
            st.markdown(f"#### {title}")
            frame = pd.DataFrame(rows)
            if not frame.empty:
                frame = frame.map(_display_value)
                st.dataframe(frame, hide_index=True, width="stretch")
            else:
                st.info("本次没有可展示的因子贡献。")

    with tab_evidence:
        st.markdown("#### 历史证据使用情况")
        evidence = snapshot.get("历史证据状态") or {}
        st.dataframe(
            pd.DataFrame([[key, value] for key, value in evidence.items()], columns=["证据", "处理"]),
            hide_index=True,
            width="stretch",
        )
        st.markdown("#### 防未来数据检查")
        guard = snapshot.get("防未来数据检查") or {}
        st.dataframe(
            pd.DataFrame([[key, value] for key, value in guard.items()], columns=["检查项", "结果"]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "当前免费行情可能包含数据商事后修订的复权序列；工具能保证日期行不晚于T，"
            "但如需机构级严格点时数据，还需要接入带版本记录的历史行情与财务数据库。"
        )

    filename = f"历史复现_{conditions['市场']}_{conditions['股票代码']}_{conditions['实际采用交易日']}.json"
    st.download_button(
        "下载本次历史判断JSON",
        data=snapshot_json(snapshot),
        file_name=filename,
        mime="application/json",
        width="stretch",
    )


st.title("独立历史时点复现工具")
st.write("输入股票和历史日期T，复现现有V6.4 Agent在当时会给出的评分、风险、持有周期和结论。")
st.info("本工具与正式Agent分开运行，不连接正式账号或数据库，也不读取T日之后行情。")

profile = load_test_profile()
with st.expander("查看固定的独立测试画像"):
    st.write("为保证不同日期可以公平比较，每次都使用同一份C3平衡型测试画像。")
    profile_view = {key: str(value) for key, value in profile.items() if key != "profile_name"}
    st.dataframe(
        pd.DataFrame([[key, value] for key, value in profile_view.items()], columns=["字段", "固定值"]),
        hide_index=True,
        width="stretch",
    )

with st.form("historical_replay_form"):
    input_columns = st.columns([1, 1.3, 1.2])
    market = input_columns[0].selectbox("市场", ["A股", "美股", "港股"])
    raw_code = input_columns[1].text_input("股票代码", placeholder="例如：600519 / AAPL / 00700")
    requested_date = input_columns[2].date_input(
        "历史分析日期T",
        value=date(2024, 6, 1),
        max_value=date.today(),
    )
    submitted = st.form_submit_button("复现T日Agent判断", type="primary", width="stretch")

if submitted:
    if not raw_code.strip():
        st.error("请先输入股票代码。")
    else:
        try:
            with st.spinner("正在获取截至T日的历史数据，并用原V6.4逻辑复现当时判断……"):
                st.session_state["historical_snapshot"] = run_historical_replay(
                    market=market,
                    raw_code=raw_code.strip(),
                    requested_date=requested_date,
                    profile=profile,
                )
        except Exception as exc:
            st.session_state.pop("historical_snapshot", None)
            st.error(f"本次复现失败：{exc}")

if st.session_state.get("historical_snapshot"):
    render_snapshot(st.session_state["historical_snapshot"])
