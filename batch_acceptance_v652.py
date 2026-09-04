from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd

import agent_core


UNIVERSE: list[tuple[str, str]] = [
    *(('A股', code) for code in [
        '600519', '000001', '000333', '000858', '600036', '601318', '601398', '600030',
        '600276', '601888', '300750', '002594', '002415', '000651', '600900', '601012',
        '600887', '603288', '600309', '601166',
    ]),
    *(('美股', code) for code in [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JPM', 'V',
        'MA', 'UNH', 'XOM', 'JNJ', 'WMT', 'PG', 'HD', 'COST', 'AVGO', 'NFLX',
    ]),
    *(('港股', code) for code in [
        '00700', '09988', '03690', '00941', '01299', '02318', '00005', '00388', '01810',
        '09888', '09618', '00883', '00857', '02628', '01177', '02020', '06618', '02382',
        '01088', '00027',
    ]),
]


PROFILE: dict[str, Any] = {
    'asset_band': '20万—50万元',
    'investable_assets': 350_000.0,
    'fund_source': '闲置自有资金',
    'emergency_reserve': '6个月以上',
    'earliest_need': '3年内',
    'income_stability': '稳定',
    'max_loss': '10%—20%',
    'loss_response': '先复核原因再决定',
    'goal': '长期增值',
    'experience': '1—3年',
    'trade_frequency': '每月1—3次',
    'monitor_time': '15—30分钟',
    'existing_concentration': '10%—30%',
    'stop_loss': '有明确规则并能执行',
    'fx_acceptance': '只能接受较小波动',
    'fundamental_action': '会重新评估',
    'planned_amount': 50_000.0,
    'leverage': '否',
}


def fetch_benchmark(market: str, start_text: str, end_text: str) -> tuple[pd.DataFrame, str]:
    if market == 'A股':
        return agent_core.fetch_a_benchmark(start_text, end_text), '沪深300'
    if market == '美股':
        return agent_core.fetch_us_benchmark(start_text, end_text), 'SPY'
    return agent_core.fetch_hk_benchmark(start_text, end_text), '恒生指数'


def fetch_stock(market: str, code: str, start_text: str, end_text: str):
    if market == 'A股':
        return agent_core.fetch_a_security(code, start_text, end_text)
    if market == '美股':
        return agent_core.fetch_us_security(code, start_text, end_text)
    return agent_core.fetch_hk_security(code, start_text, end_text)


def main() -> int:
    end_text = date.today().isoformat()
    start_text = (pd.Timestamp(end_text) - pd.DateOffset(years=5)).date().isoformat()
    benchmarks: dict[str, tuple[pd.DataFrame, str]] = {}
    benchmark_errors: dict[str, str] = {}
    for market in ('A股', '美股', '港股'):
        try:
            benchmarks[market] = fetch_benchmark(market, start_text, end_text)
        except Exception as exc:
            benchmarks[market] = (pd.DataFrame(), '基准暂不可用')
            benchmark_errors[market] = f'{type(exc).__name__}: {exc}'

    fetched: dict[tuple[str, str], tuple[pd.DataFrame, str, str] | Exception] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fetch_stock, market, code, start_text, end_text): (market, code)
            for market, code in UNIVERSE
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                fetched[key] = future.result()
            except Exception as exc:
                fetched[key] = exc

    missing = agent_core.EvidenceSnapshot(
        False,
        '压力测试：辅助证据全部关闭',
        score=None,
        notes=['本测试故意关闭财务、宏观和资讯，用于验证它们缺失时核心判断仍能运行。'],
    )
    rows: list[dict[str, Any]] = []
    for market, code in UNIVERSE:
        item = fetched[(market, code)]
        if isinstance(item, Exception):
            rows.append({
                '市场': market,
                '股票': code,
                '交易日数': 0,
                '行情来源': '',
                '持有周期': '',
                '评分': None,
                '股票风险': '',
                '个人适配': '',
                '结论': '运行失败',
                '证据不足': True,
                '错误': f'{type(item).__name__}: {item}',
            })
            continue
        stock, name, provider = item
        benchmark, benchmark_name = benchmarks[market]
        bundle = agent_core.PriceBundle(
            stock=stock,
            benchmark=benchmark,
            code=code,
            name=name,
            provider=provider,
            benchmark_name=benchmark_name,
            asset_type='场内基金' if market == 'A股' and agent_core.is_exchange_traded_fund_code(code) else f'{market}个股',
            price_unit='人民币元' if market == 'A股' else '美元' if market == '美股' else '港元',
        )
        first_date = pd.Timestamp(stock['日期'].min())
        bundle.history_complete = first_date <= pd.Timestamp(start_text) + pd.Timedelta(days=45)
        bundle.coverage_ratio = min(1.0, max(0.0, (pd.Timestamp(end_text) - first_date).days / max(1, (pd.Timestamp(end_text) - pd.Timestamp(start_text)).days)))
        try:
            analysis = agent_core.analyze_all(bundle, PROFILE, missing, missing)
            selected = analysis.get('selected_horizon') or {}
            conclusion = str(analysis.get('conclusion') or '')
            insufficient = not selected or conclusion.startswith('证据不足')
            rows.append({
                '市场': market,
                '股票': code,
                '交易日数': len(stock),
                '行情来源': provider,
                '持有周期': selected.get('name', ''),
                '评分': selected.get('score'),
                '股票风险': analysis.get('stock_risk_level', ''),
                '个人适配': (analysis.get('suitability') or {}).get('fit', ''),
                '结论': conclusion,
                '证据不足': insufficient,
                '错误': '',
            })
        except Exception as exc:
            rows.append({
                '市场': market,
                '股票': code,
                '交易日数': len(stock),
                '行情来源': provider,
                '持有周期': '',
                '评分': None,
                '股票风险': '',
                '个人适配': '',
                '结论': '运行失败',
                '证据不足': True,
                '错误': f'{type(exc).__name__}: {exc}',
            })

    result = pd.DataFrame(rows)
    total = len(result)
    insufficient_count = int(result['证据不足'].sum())
    rate = insufficient_count / total if total else 1.0
    output_dir = Path('validation_output')
    output_dir.mkdir(exist_ok=True)
    result.to_csv(output_dir / '60只股票逐项测试结果_V6.5.2.csv', index=False, encoding='utf-8-sig')
    summary = {
        'model_version': agent_core.MODEL_VERSION,
        'test_date': end_text,
        'total': total,
        'insufficient_count': insufficient_count,
        'insufficient_rate': rate,
        'acceptance_threshold': 0.05,
        'passed': rate < 0.05,
        'markets': result.groupby('市场')['股票'].count().to_dict(),
        'conclusions': result['结论'].value_counts().to_dict(),
        'benchmark_errors': benchmark_errors,
    }
    (output_dir / '60只股票验收摘要_V6.5.2.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if insufficient_count:
        print(result[result['证据不足']][['市场', '股票', '结论', '错误']].to_string(index=False))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

