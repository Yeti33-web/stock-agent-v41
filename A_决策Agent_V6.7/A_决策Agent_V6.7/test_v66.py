from __future__ import annotations

import unittest

import agent_core
import factor_analysis


class FundamentalFactorV67Tests(unittest.TestCase):
    """Cover the four V6.7 fundamental factors: ROIC, FCF yield, margin trend
    and valuation percentile, plus their integration into the fundamental score."""

    def _score(self, fields: dict) -> tuple[float | None, list, list]:
        return agent_core._score_fundamentals(fields)

    def test_roic_scoring_tiers(self):
        base = {"净资产收益率": 0.20, "净利率": 0.15, "净利润同比": 0.2, "营收同比": 0.1}
        high, pos, _ = self._score({**base, "投入资本回报率ROIC": 0.18})
        mid, _, _ = self._score({**base, "投入资本回报率ROIC": 0.10})
        low, _, risk = self._score({**base, "投入资本回报率ROIC": -0.03})
        self.assertIsNotNone(high)
        self.assertGreater(high, mid)
        self.assertGreater(mid, low)
        self.assertIn("投入资本回报率相对较高", pos)
        self.assertTrue(any("为负" in item for item in risk))

    def test_fcf_yield_scoring(self):
        base = {"净资产收益率": 0.20, "净利率": 0.15, "净利润同比": 0.2, "营收同比": 0.1}
        high, pos, _ = self._score({**base, "自由现金流收益率": 0.07})
        zero, _, _ = self._score({**base, "自由现金流收益率": 0.02})
        neg, _, risk = self._score({**base, "自由现金流收益率": -0.03})
        self.assertIsNotNone(high)
        self.assertGreater(high, zero)
        self.assertGreater(zero, neg)
        self.assertIn("自由现金流收益率相对较高", pos)
        self.assertTrue(any("自由现金流为负" in item for item in risk))

    def test_margin_and_trend_scoring(self):
        base = {"净资产收益率": 0.20, "净利率": 0.15, "净利润同比": 0.2, "营收同比": 0.1}
        strong, pos, _ = self._score({**base, "毛利率": 0.55, "营业利润率": 0.25, "利润率趋势": 1})
        weak, _, risk = self._score({**base, "毛利率": 0.10, "营业利润率": 0.02, "利润率趋势": -1})
        self.assertIsNotNone(strong)
        self.assertGreater(strong, weak)
        self.assertIn("毛利率相对较高", pos)
        self.assertIn("营业利润率相对较高", pos)
        self.assertIn("毛利率或营业利润率近期改善", pos)
        self.assertTrue(any("近期走弱" in item for item in risk))

    def test_valuation_percentile_scoring(self):
        base = {"净资产收益率": 0.20, "净利率": 0.15, "净利润同比": 0.2, "营收同比": 0.1}
        cheap, pos, _ = self._score({**base, "估值历史分位": 0.10, "市盈率TTM": 12.0})
        expensive, _, risk = self._score({**base, "估值历史分位": 0.95, "市盈率TTM": 60.0})
        self.assertIsNotNone(cheap)
        self.assertGreater(cheap, expensive)
        self.assertIn("估值处于自身历史较低分位", pos)
        self.assertTrue(any("较高分位" in item for item in risk))

    def test_loss_making_company_is_penalized_in_valuation(self):
        base = {"净资产收益率": -0.05, "净利率": -0.1, "净利润同比": -0.3, "营收同比": -0.1}
        loss, _, risk = self._score({**base, "估值历史分位": 0.10, "市盈率TTM": -8.0})
        self.assertTrue(any("亏损" in item or "不可用" in item for item in risk))

    def test_missing_new_factors_do_not_break_score(self):
        base = {"净资产收益率": 0.20, "净利率": 0.15}
        score, _, _ = self._score(base)
        self.assertIsNotNone(score)

    def test_catalog_includes_v67_fundamental_factors(self):
        rows = factor_analysis.factor_catalog()
        factor_names = {item["因子"] for item in rows}
        self.assertTrue(
            {
                "投入资本回报率ROIC",
                "自由现金流收益率",
                "毛利率／营业利润率趋势",
                "估值历史分位",
            }.issubset(factor_names)
        )
        module_counts = {}
        for row in rows:
            module_counts[row["模块"]] = module_counts.get(row["模块"], 0) + 1
        self.assertEqual(module_counts["基本面评分"], 11)

    def test_model_version_bumped(self):
        self.assertEqual(agent_core.MODEL_VERSION, "V6.7")


if __name__ == "__main__":
    unittest.main()
