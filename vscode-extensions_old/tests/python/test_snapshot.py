from __future__ import annotations

import sys
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from relife_engine.snapshot import build_market_snapshot


class SnapshotTests(unittest.TestCase):
    def test_summary_uses_current_value_fifo_profit_and_previous_close(self) -> None:
        transactions = [
            {"timestamp": datetime(2026, 7, 1), "kind": "deposit", "symbol": "", "quantity": Decimal("1000"), "price": Decimal(), "fee": Decimal(), "cashDelta": Decimal("1000"), "externalFlow": Decimal("1000"), "rawAction": "Deposit"},
            {"timestamp": datetime(2026, 7, 1), "kind": "buy", "symbol": "DEMO", "name": "示例科技", "quantity": Decimal("10"), "price": Decimal("10"), "fee": Decimal(), "cashDelta": Decimal("-100"), "externalFlow": Decimal(), "rawAction": "Buy"},
            {"timestamp": datetime(2026, 7, 2), "kind": "dividend", "symbol": "DEMO", "name": "示例科技", "quantity": Decimal("5"), "price": Decimal(), "fee": Decimal(), "cashDelta": Decimal("5"), "externalFlow": Decimal(), "rawAction": "Dividend"},
        ]
        closes = {"DEMO": {"2026-07-01": Decimal("11"), "2026-07-02": Decimal("12")}}
        benchmark = {"2026-07-01": Decimal("100"), "2026-07-02": Decimal("101")}
        snapshot = build_market_snapshot("us", transactions, closes, benchmark, [], [])
        summary = snapshot["summary"]
        self.assertEqual(summary["portfolioValue"], 1025.0)
        self.assertEqual(summary["cash"], 905.0)
        self.assertEqual(summary["unrealized"], 20.0)
        self.assertEqual(summary["lastDayAmount"], 10.0)
        self.assertAlmostEqual(summary["lastDayPercent"], 10 / 110)
        self.assertEqual(summary["netDividends"], 5.0)
        self.assertEqual(summary["totalReturn"], 25.0)
        self.assertEqual(snapshot["holdings"][0]["name"], "示例科技")
        self.assertEqual(snapshot["distribution"][0]["value"], 120.0)

    def test_missing_one_symbol_marks_incomplete_but_keeps_other_positions(self) -> None:
        transactions = [
            {"timestamp": datetime(2026, 7, 1), "kind": "buy", "symbol": "OK", "name": "可用", "quantity": Decimal("1"), "price": Decimal("10"), "fee": Decimal(), "cashDelta": Decimal("-10"), "externalFlow": Decimal(), "rawAction": "Buy"},
            {"timestamp": datetime(2026, 7, 1), "kind": "buy", "symbol": "MISS", "name": "缺失", "quantity": Decimal("1"), "price": Decimal("20"), "fee": Decimal(), "cashDelta": Decimal("-20"), "externalFlow": Decimal(), "rawAction": "Buy"},
        ]
        snapshot = build_market_snapshot(
            "us", transactions, {"OK": {"2026-07-01": Decimal("11")}}, {}, [],
            [{"symbol": "MISS", "message": "行情不可用"}],
        )
        self.assertTrue(snapshot["incomplete"])
        self.assertEqual(len(snapshot["holdings"]), 2)
        self.assertEqual(snapshot["holdings"][1]["lastPrice"], None)
        self.assertEqual(snapshot["errors"][0]["symbol"], "MISS")

    def test_fully_sold_symbol_is_separate_from_current_holdings(self) -> None:
        transactions = [
            {"timestamp": datetime(2026, 7, 1), "kind": "buy", "symbol": "SOLD", "name": "已售示例", "quantity": Decimal("2"), "price": Decimal("10"), "fee": Decimal(), "cashDelta": Decimal("-20"), "externalFlow": Decimal(), "rawAction": "Buy"},
            {"timestamp": datetime(2026, 7, 2), "kind": "sell", "symbol": "SOLD", "name": "已售示例", "quantity": Decimal("2"), "price": Decimal("12"), "fee": Decimal(), "cashDelta": Decimal("24"), "externalFlow": Decimal(), "rawAction": "Sell"},
        ]
        snapshot = build_market_snapshot("us", transactions, {"SOLD": {"2026-07-01": Decimal("10"), "2026-07-02": Decimal("12")}}, {}, [], [])
        self.assertEqual(snapshot["holdings"], [])
        self.assertEqual(snapshot["soldHoldings"][0]["symbol"], "SOLD")
        self.assertEqual(snapshot["soldHoldings"][0]["realized"], 4.0)


if __name__ == "__main__":
    unittest.main()
