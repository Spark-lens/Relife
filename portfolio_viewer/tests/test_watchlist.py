from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal

from portfolio_dashboard.watchlist import (
    aggregate_closes,
    check_watchlist,
    evaluate_bollinger,
    validate_watchlist,
)


def item(
    symbol: str,
    *,
    market: str = "us",
    timeframes: list[str] | None = None,
) -> dict:
    return {
        "market": market,
        "symbol": symbol,
        "name": symbol,
        "bollinger": {
            "enabled": True,
            "timeframes": timeframes or ["daily", "weekly", "monthly"],
            "window": 20,
            "standardDeviations": 2,
        },
    }


def config(*items: dict) -> dict:
    return {"groups": [{"id": "broad", "label": "大盘", "items": list(items)}]}


class StaticProvider:
    def __init__(self, series_by_symbol: dict[str, dict[date, Decimal]]):
        self.series_by_symbol = series_by_symbol

    def history(self, symbols: list[str], start: date, end: date):
        del start, end
        symbol = symbols[0]
        value = self.series_by_symbol[symbol]
        if isinstance(value, Exception):
            raise value
        return {symbol: value}


class WatchlistConfigTest(unittest.TestCase):
    def test_rejects_duplicate_market_symbol(self) -> None:
        with self.assertRaisesRegex(ValueError, "重复标的 us:QQQ"):
            validate_watchlist(config(item("qqq"), item("QQQ")))


class BollingerTest(unittest.TestCase):
    def test_aggregates_latest_close_for_daily_weekly_and_monthly_buckets(self) -> None:
        closes = {
            date(2026, 7, 30): Decimal("10"),
            date(2026, 7, 31): Decimal("11"),
            date(2026, 8, 3): Decimal("12"),
        }

        self.assertEqual(
            [("2026-07-30", date(2026, 7, 30), Decimal("10")),
             ("2026-07-31", date(2026, 7, 31), Decimal("11")),
             ("2026-08-03", date(2026, 8, 3), Decimal("12"))],
            aggregate_closes(closes, "daily"),
        )
        self.assertEqual(
            [("2026-W31", date(2026, 7, 31), Decimal("11")),
             ("2026-W32", date(2026, 8, 3), Decimal("12"))],
            aggregate_closes(closes, "weekly"),
        )
        self.assertEqual(
            [("2026-07", date(2026, 7, 31), Decimal("11")),
             ("2026-08", date(2026, 8, 3), Decimal("12"))],
            aggregate_closes(closes, "monthly"),
        )

    def test_triggers_when_latest_close_is_below_population_lower_band(self) -> None:
        buckets = [
            (f"2026-07-{day:02d}", date(2026, 7, day), Decimal("100"))
            for day in range(1, 20)
        ] + [("2026-07-20", date(2026, 7, 20), Decimal("50"))]

        result = evaluate_bollinger(buckets, window=20, standard_deviations=Decimal("2"))

        self.assertTrue(result["triggered"])
        self.assertEqual("2026-07-20", result["periodKey"])
        self.assertEqual(Decimal("50"), result["close"])
        self.assertEqual(Decimal("75.7055"), result["lowerBand"].quantize(Decimal("0.0001")))


class WatchlistCheckTest(unittest.TestCase):
    def test_keeps_other_symbols_when_one_provider_request_fails(self) -> None:
        start = date(2026, 1, 1)
        good = {
            start + timedelta(days=offset): Decimal("100" if offset < 19 else "50")
            for offset in range(20)
        }
        provider = StaticProvider({"QQQ": good, "BAD": RuntimeError("行情不可用")})

        payload = check_watchlist(
            config(item("QQQ", timeframes=["daily"]), item("BAD", timeframes=["daily"])),
            providers={"us": provider},
            checked_at=datetime.fromisoformat("2026-08-01T09:00:00+08:00"),
        )

        self.assertEqual("2026-08-01T09:00:00+08:00", payload["checkedAt"])
        self.assertEqual({"QQQ"}, {row["symbol"] for row in payload["results"]})
        self.assertEqual({"QQQ"}, {row["symbol"] for row in payload["alerts"]})
        self.assertEqual("BAD", payload["errors"][0]["symbol"])
        self.assertIn("行情不可用", payload["errors"][0]["message"])


if __name__ == "__main__":
    unittest.main()
