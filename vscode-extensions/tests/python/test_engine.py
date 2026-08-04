from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from relife_engine.analytics import risk_metrics, twr, xirr
from relife_engine.ledger import replay
from relife_engine.sources import discover_latest, parse_source, validate_source


US_HEADERS = ["Symbol", "Side", "Qty", "Fill Price", "Commission", "Closing Time"]
CN_HEADERS = [
    "日期", "成交日期", "证券代码", "证券名称", "操作", "资金余额", "成交数量",
    "成交均价", "备注", "成交金额", "发生金额", "手续费", "印花税", "其他杂费",
    "本次金额", "合同编号", "成交编号", "交易市场", "货币单位", "委托日期",
    "证券中文全称", "股份余额",
]


class SourceTests(unittest.TestCase):
    def test_validate_source_reports_market_range_and_bad_field(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tradingview_full_latest_2026-08-01.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=US_HEADERS)
                writer.writeheader()
                writer.writerow({
                    "Symbol": "NASDAQ:DEMO", "Side": "Buy", "Qty": "2",
                    "Fill Price": "10", "Commission": "1",
                    "Closing Time": "2026-07-01 09:30:00",
                })
                writer.writerow({
                    "Symbol": "NASDAQ:DEMO", "Side": "Sell", "Qty": "1",
                    "Fill Price": "12", "Commission": "1",
                    "Closing Time": "2026-07-02 09:30:00",
                })

            result = validate_source(path, "us")
            self.assertEqual(result["market"], "us")
            self.assertEqual(result["recordCount"], 2)
            self.assertEqual(result["dateRange"], ["2026-07-01", "2026-07-02"])

            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["Qty"] = "oops"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=US_HEADERS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "第 3 行字段 Qty"):
                validate_source(path, "us")

    def test_validate_cn_and_discover_latest_standard_name(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for stamp in ("2026-07-01", "2026-08-03"):
                path = root / f"交割单_{stamp}.csv"
                row = dict.fromkeys(CN_HEADERS, "")
                row.update({
                    "日期": "20260801", "成交日期": "20260801", "证券代码": "600000",
                    "证券名称": "示例银行", "操作": "证券买入", "资金余额": "8000",
                    "成交数量": "100", "成交均价": "10", "发生金额": "-1002",
                    "手续费": "2", "货币单位": "人民币",
                })
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CN_HEADERS)
                    writer.writeheader()
                    writer.writerow(row)
            self.assertEqual(discover_latest(root, "cn").name, "交割单_2026-08-03.csv")
            self.assertEqual(validate_source(discover_latest(root, "cn"), "cn")["recordCount"], 1)

    def test_blank_cn_cash_balance_is_not_treated_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "交割单_2026-08-03.csv"
            row = dict.fromkeys(CN_HEADERS, "")
            row.update({"日期": "20260801", "成交日期": "20260801", "证券代码": "600X01", "证券名称": "示例", "操作": "红利入账", "发生金额": "12"})
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CN_HEADERS)
                writer.writeheader()
                writer.writerow(row)
            self.assertIsNone(parse_source(path, "cn")[0]["cashBalance"])


class LedgerTests(unittest.TestCase):
    def test_us_uses_fifo_including_buy_and_sell_fees(self) -> None:
        tx = [
            {"kind": "deposit", "symbol": "", "quantity": Decimal("1000"), "price": Decimal(), "fee": Decimal(), "cashDelta": Decimal("1000")},
            {"kind": "buy", "symbol": "DEMO", "name": "Demo", "quantity": Decimal("10"), "price": Decimal("10"), "fee": Decimal("1"), "cashDelta": Decimal("-101")},
            {"kind": "buy", "symbol": "DEMO", "name": "Demo", "quantity": Decimal("5"), "price": Decimal("20"), "fee": Decimal(), "cashDelta": Decimal("-100")},
            {"kind": "sell", "symbol": "DEMO", "name": "Demo", "quantity": Decimal("12"), "price": Decimal("30"), "fee": Decimal("2"), "cashDelta": Decimal("358")},
        ]
        result = replay("us", tx)
        position = result["allPositions"]["DEMO"]
        self.assertEqual(position["quantity"], Decimal("3"))
        self.assertEqual(position["remainingCost"], Decimal("60"))
        self.assertEqual(position["realized"], Decimal("217"))
        self.assertEqual(result["cash"], Decimal("1157"))

    def test_cn_uses_moving_weighted_cost(self) -> None:
        tx = [
            {"kind": "buy", "symbol": "600000", "name": "示例银行", "quantity": Decimal("10"), "price": Decimal("10"), "fee": Decimal(), "cashDelta": Decimal("-100")},
            {"kind": "buy", "symbol": "600000", "name": "示例银行", "quantity": Decimal("10"), "price": Decimal("20"), "fee": Decimal(), "cashDelta": Decimal("-200")},
            {"kind": "sell", "symbol": "600000", "name": "示例银行", "quantity": Decimal("5"), "price": Decimal("18"), "fee": Decimal(), "cashDelta": Decimal("90"), "cashBalance": Decimal("790")},
        ]
        position = replay("cn", tx)["allPositions"]["600000"]
        self.assertEqual(position["quantity"], Decimal("15"))
        self.assertEqual(position["remainingCost"], Decimal("225"))
        self.assertEqual(position["realized"], Decimal("15"))


class AnalyticsTests(unittest.TestCase):
    def test_cash_flow_adjusted_twr_and_xirr(self) -> None:
        self.assertAlmostEqual(
            twr([(Decimal("1000"), Decimal("1000")), (Decimal("1100"), Decimal("0")), (Decimal("1650"), Decimal("500"))]),
            0.15,
            places=9,
        )
        rate = xirr([(date(2025, 1, 1), Decimal("-1000")), (date(2026, 1, 1), Decimal("1200"))])
        self.assertAlmostEqual(rate or 0, 0.2, places=6)

    def test_risk_metrics_uses_common_return_days(self) -> None:
        metrics = risk_metrics([0.01, -0.01, 0.02], [0.005, -0.005, 0.01])
        self.assertAlmostEqual(metrics["beta"], 2.0, places=9)
        self.assertIsNotNone(metrics["sharpe"])
        self.assertIsNotNone(metrics["sortino"])


if __name__ == "__main__":
    unittest.main()
