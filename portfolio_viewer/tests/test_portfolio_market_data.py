import sys
import types
import unittest
from datetime import date
from unittest.mock import patch

from decimal import Decimal

from portfolio_dashboard.market_data import (
    AksharePriceProvider,
    YahooPriceProvider,
    _repair_price_discontinuities,
)


class _EmptyYahooFrame:
    empty = True

    def __contains__(self, key: object) -> bool:
        return False


class YahooPriceProviderTest(unittest.TestCase):
    def test_repairs_integer_multiple_price_discontinuity(self) -> None:
        repaired = _repair_price_discontinuities(
            {
                date(2026, 5, 22): Decimal("1159.5"),
                date(2026, 5, 26): Decimal("62.9"),
                date(2026, 5, 27): Decimal("65.3"),
            }
        )

        self.assertEqual(Decimal("57.975"), repaired[date(2026, 5, 22)])
        self.assertEqual(Decimal("62.9"), repaired[date(2026, 5, 26)])

    def test_maps_tradingview_berkshire_symbol_to_yahoo_symbol(self) -> None:
        requested: list[str] = []

        class FakeTicker:
            def __init__(self, symbol: str) -> None:
                requested.append(symbol)

            def history(self, **kwargs: object) -> _EmptyYahooFrame:
                del kwargs
                return _EmptyYahooFrame()

        fake_module = types.SimpleNamespace(Ticker=FakeTicker)
        with patch.dict(sys.modules, {"yfinance": fake_module}):
            YahooPriceProvider().history(
                ["BRKB"],
                date(2026, 6, 1),
                date(2026, 6, 3),
            )

        self.assertEqual(["BRK-B"], requested)


class AksharePriceProviderTest(unittest.TestCase):
    def test_uses_sina_endpoints_and_exchange_prefixed_symbols(self) -> None:
        calls: list[tuple[str, str]] = []
        empty_frame = {"date": [], "close": []}

        fake_module = types.SimpleNamespace(
            fund_etf_hist_sina=lambda *, symbol: (
                calls.append(("fund", symbol)) or empty_frame
            ),
            stock_zh_a_daily=lambda *, symbol, start_date, end_date, adjust: (
                calls.append(("stock", symbol)) or empty_frame
            ),
            stock_zh_index_daily=lambda *, symbol: (
                calls.append(("index", symbol)) or empty_frame
            ),
        )
        with patch.dict(sys.modules, {"akshare": fake_module}):
            AksharePriceProvider().history(
                ["510300", "161716", "600158", "__SSE__"],
                date(2026, 1, 1),
                date(2026, 7, 17),
            )

        self.assertEqual(
            [
                ("fund", "sh510300"),
                ("fund", "sz161716"),
                ("stock", "sh600158"),
                ("index", "sh000001"),
            ],
            calls,
        )


if __name__ == "__main__":
    unittest.main()
