import unittest
from decimal import Decimal
from pathlib import Path

from portfolio_dashboard.parsers import parse_tradingview, parse_yinhe


FIXTURES = Path(__file__).parent / "fixtures" / "portfolio"


class TradingViewParserTest(unittest.TestCase):
    def test_parses_supported_rows_and_normalizes_symbols(self) -> None:
        rows = parse_tradingview(FIXTURES / "tradingview.csv")

        self.assertEqual(["dividend", "buy", "deposit"], [row.kind for row in rows])
        self.assertEqual("XQQI", rows[0].symbol)
        self.assertEqual(Decimal("0.08"), rows[0].cash_delta)
        self.assertEqual(Decimal("3037.72"), rows[-1].external_cash_flow)


class YinheParserTest(unittest.TestCase):
    def test_preserves_source_order_for_same_day_rows(self) -> None:
        rows = parse_yinhe(FIXTURES / "yinhe.csv")
        same_day = [
            row
            for row in rows
            if row.timestamp.date().isoformat() == "2026-07-14"
        ]

        self.assertEqual([0, 1, 2], [row.source_index for row in same_day])
        self.assertEqual(["sell", "sell", "sell"], [row.kind for row in same_day])
        self.assertEqual(Decimal("3749.160"), same_day[0].cash_balance)

    def test_normalizes_cash_dividend_and_reverse_repo_actions(self) -> None:
        rows = parse_yinhe(FIXTURES / "yinhe.csv")
        kinds = {row.kind for row in rows}

        self.assertTrue(
            {"deposit", "withdrawal", "interest", "dividend", "cash"} <= kinds
        )
        dividend = next(row for row in rows if row.kind == "dividend")
        self.assertEqual(Decimal("5.000"), dividend.cash_delta)

    def test_keeps_reverse_repo_open_and_close_with_shared_trade_id(self) -> None:
        rows = parse_yinhe(FIXTURES / "yinhe.csv")
        repo_rows = [row for row in rows if row.symbol == "204001"]

        self.assertEqual(2, len(repo_rows))
        self.assertEqual(
            {"通用回购逆回购", "通用回购逆回购购"},
            {row.raw_action for row in repo_rows},
        )


if __name__ == "__main__":
    unittest.main()
