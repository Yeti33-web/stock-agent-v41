from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "agent_core.py").exists() and (PROJECT_ROOT / "model_v64" / "agent_core.py").exists():
    PROJECT_ROOT = PROJECT_ROOT / "model_v64"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agent_core
from historical_test_tool.historical_data import assert_bundle_cutoff, fetch_historical_bundle
from historical_test_tool.runner import frozen_agent_date, run_historical_replay


def prices(dates: list[str]) -> pd.DataFrame:
    count = len(dates)
    return pd.DataFrame(
        {
            "日期": pd.to_datetime(dates),
            "开盘": range(10, 10 + count),
            "最高": range(11, 11 + count),
            "最低": range(9, 9 + count),
            "收盘": range(10, 10 + count),
            "成交量": [1000] * count,
        }
    )


class FakeCore:
    PriceBundle = agent_core.PriceBundle

    def __init__(self) -> None:
        self.security_end = None
        self.benchmark_end = None

    @staticmethod
    def normalize_a_code(raw: str) -> str:
        return raw

    @staticmethod
    def is_exchange_traded_fund_code(code: str) -> bool:
        return False

    def fetch_a_security(self, code: str, start: str, end: str):
        self.security_end = end
        return prices(["2024-05-30", "2024-05-31", "2024-06-03"]), "测试股票", "假数据源"

    def fetch_a_benchmark(self, start: str, end: str):
        self.benchmark_end = end
        return prices(["2024-05-30", "2024-05-31", "2024-06-03"])


class HistoricalToolTests(unittest.TestCase):
    def test_non_trading_day_uses_previous_trading_day_and_never_requests_future(self):
        fake = FakeCore()
        result = fetch_historical_bundle("A股", "600000", date(2024, 6, 1), fake)
        self.assertEqual(fake.security_end, "2024-06-01")
        self.assertEqual(fake.benchmark_end, "2024-06-01")
        self.assertEqual(result.actual_trading_date, date(2024, 5, 31))
        self.assertEqual(result.bundle.stock["日期"].max().date(), date(2024, 5, 31))
        self.assertEqual(result.bundle.benchmark["日期"].max().date(), date(2024, 5, 31))

    def test_guard_rejects_future_row(self):
        bundle = agent_core.PriceBundle(
            stock=prices(["2024-05-31", "2024-06-03"]),
            benchmark=prices(["2024-05-31"]),
            code="600000",
            name="测试",
            provider="测试",
            benchmark_name="沪深300",
            asset_type="A股个股",
            price_unit="人民币元",
        )
        with self.assertRaisesRegex(RuntimeError, "个股行情包含"):
            assert_bundle_cutoff(bundle, date(2024, 5, 31))

    def test_date_freeze_is_runtime_only_and_restored(self):
        original = agent_core.date
        with frozen_agent_date(agent_core, date(2020, 1, 2)):
            self.assertEqual(agent_core.date.today(), date(2020, 1, 2))
        self.assertIs(agent_core.date, original)

    def test_full_replay_returns_agent_result_without_post_t_outcome(self):
        dates = pd.bdate_range("2019-06-03", "2024-06-03")
        steps = np.arange(len(dates), dtype=float)
        close = 100.0 * np.exp(0.00025 * steps + 0.025 * np.sin(steps / 18.0))
        stock = pd.DataFrame(
            {
                "日期": dates,
                "开盘": close * 0.998,
                "最高": close * 1.01,
                "最低": close * 0.99,
                "收盘": close,
                "成交量": 1_000_000 + (steps % 30) * 10_000,
            }
        )
        benchmark = stock.assign(收盘=100.0 * np.exp(0.00018 * steps)).copy()
        original_date = agent_core.date
        with (
            patch.object(agent_core, "fetch_a_security", return_value=(stock, "测试股票", "合成日线")),
            patch.object(agent_core, "fetch_a_benchmark", return_value=benchmark),
        ):
            snapshot = run_historical_replay("A股", "600000", date(2024, 6, 1))

        self.assertEqual(snapshot["测试条件"]["实际采用交易日"], "2024-05-31")
        self.assertEqual(snapshot["防未来数据检查"]["T后行情传入Agent行数"], 0)
        self.assertFalse(snapshot["防未来数据检查"]["T后财务_宏观利率_资讯传入"])
        self.assertIn("结论", snapshot["当时Agent判断"])
        self.assertIn("自动选择周期", snapshot["当时Agent判断"])
        self.assertNotIn("人工验证期限H_交易日", snapshot["测试条件"])
        self.assertNotIn("用户指定H参考", snapshot)
        self.assertNotIn("未来收益", snapshot)
        self.assertNotIn("有效性评价", snapshot)
        self.assertIs(agent_core.date, original_date)


if __name__ == "__main__":
    unittest.main()
