from __future__ import annotations

import sys

from agent_core import analyze_historical_analogs, fetch_hkd_cny_rate, fetch_price_bundle, fetch_usd_cny_rate


def check(market: str, code: str) -> bool:
    print(f"\n测试 {market}：{code}")
    try:
        bundle = fetch_price_bundle(market, code)
    except Exception as exc:
        print(f"失败：{exc}")
        return False
    first_date = bundle.stock["日期"].min().date()
    last_date = bundle.stock["日期"].max().date()
    print(f"成功：{bundle.code}，{len(bundle.stock)} 个交易日")
    print(f"区间：{first_date} 至 {last_date}")
    print(f"来源：{bundle.provider}")
    print(f"基准：{bundle.benchmark_name}，{len(bundle.benchmark)} 个交易日")
    print(f"是否覆盖近五年：{'是' if bundle.history_complete else '否'}")
    analog = analyze_historical_analogs(
        bundle.stock,
        bundle.benchmark,
        history_complete=bundle.history_complete,
        source_label="目标股票（结合市场状态）",
    )
    print(f"相似周期状态：{analog.get('state', {}).get('summary', '数据不足')}")
    print(
        f"相似周期：{'可用' if analog['available'] else '样本不足'}，"
        f"可信度 {analog['confidence_label']}（{analog['confidence_score']}/100）"
    )
    for horizon in analog.get("horizons", []):
        if horizon["available"]:
            print(
                f"  后续{horizon['days']}日：{horizon['sample_count']}个样本，"
                f"{horizon.get('selection_mode', '同股样本')}，"
                f"上涨样本占比{horizon['positive_ratio']:.3%}，中位收益{horizon['median_return']:.3%}"
            )
        else:
            print(
                f"  后续{horizon['days']}日：仅{horizon['sample_count']}个样本，"
                f"不形成预测；{horizon.get('reason', '样本原因未知')}"
            )
    for warning in bundle.warnings:
        print(f"提示：{warning}")
    return True


def main() -> int:
    print("开始测试V5.7 A股、美股、港股真实行情、相似周期及汇率通道……")
    a_ok = check("A股", "600519")
    us_ok = check("美股", "AAPL")
    hk_ok = check("港股", "00700")
    usd_fx_ok = False
    hkd_fx_ok = False
    if us_ok:
        print("\n测试美股持仓人民币换算汇率")
        try:
            fx = fetch_usd_cny_rate()
            print(f"成功：1美元 = {float(fx['rate']):.3f}元，日期 {fx['date'].date()}，来源：{fx['provider']}")
            usd_fx_ok = True
        except Exception as exc:
            print(f"失败：{exc}")
            print("美股行情仍可分析；已有持仓请暂时选择“按持仓金额填写”。")
    if hk_ok:
        print("\n测试港股持仓人民币换算汇率")
        try:
            fx = fetch_hkd_cny_rate()
            print(f"成功：1港元 = {float(fx['rate']):.3f}元，日期 {fx['date'].date()}，来源：{fx['provider']}")
            hkd_fx_ok = True
        except Exception as exc:
            print(f"失败：{exc}")
            print("港股行情仍可分析；已有持仓请暂时选择“按持仓金额填写”。")
    if a_ok and us_ok and hk_ok and usd_fx_ok and hkd_fx_ok:
        print("\nA股、美股、港股真实行情及外币持仓汇率通道均可用，可以启动网页。")
        return 0
    print("\n至少一个市场测试失败。请保留本窗口完整内容用于排查。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
