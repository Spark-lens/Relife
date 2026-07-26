import json
import tempfile
import unittest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from portfolio_dashboard.generator import (
    MissingPriceError,
    ProviderBundle,
    _adjust_transactions_for_splits,
    build_performance,
    generate_dashboard,
)
from portfolio_dashboard.models import Transaction


FIXTURES = Path(__file__).parent / "fixtures" / "portfolio"


class StaticProvider:
    def __init__(self, prices: dict[str, dict[date, Decimal]]) -> None:
        self.prices = prices

    def history(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> dict[str, dict[date, Decimal]]:
        del start, end
        return {
            symbol: self.prices.get(symbol, {})
            for symbol in symbols
        }


def providers(*, omit_qqqi: bool = False) -> ProviderBundle:
    us_prices = {
        "QQQI": {} if omit_qqqi else {
            date(2026, 3, 26): Decimal("49.50"),
            date(2026, 7, 9): Decimal("50.00"),
            date(2026, 7, 10): Decimal("50.50"),
        },
        "XQQI": {
            date(2026, 7, 9): Decimal("50.00"),
            date(2026, 7, 10): Decimal("50.10"),
        },
        "__QQQ__": {
            date(2026, 3, 26): Decimal("100"),
            date(2026, 7, 10): Decimal("110"),
        },
        "__SPY__": {
            date(2026, 3, 26): Decimal("100"),
            date(2026, 7, 10): Decimal("105"),
        },
    }
    cn_prices = {
        "512100": {
            date(2026, 3, 2): Decimal("3.00"),
            date(2026, 7, 14): Decimal("3.20"),
        },
        "513530": {
            date(2026, 3, 1): Decimal("1.40"),
            date(2026, 7, 14): Decimal("1.51"),
        },
        "__SSE__": {
            date(2026, 3, 2): Decimal("100"),
            date(2026, 7, 14): Decimal("104"),
        },
        "__CSI300__": {
            date(2026, 3, 2): Decimal("100"),
            date(2026, 7, 14): Decimal("102"),
        },
    }
    return ProviderBundle(
        us=StaticProvider(us_prices),
        cn=StaticProvider(cn_prices),
    )


class PerformanceTest(unittest.TestCase):
    def test_split_adjustment_converts_trades_to_current_share_basis(self) -> None:
        original = Transaction(
            market="us",
            timestamp=datetime.fromisoformat("2026-06-18T00:00:00"),
            source_index=0,
            symbol="SOXS",
            name="SOXS",
            kind="buy",
            quantity=Decimal("4"),
            price=Decimal("3.50"),
            fee=Decimal("0"),
            cash_delta=Decimal("-14"),
            external_cash_flow=Decimal("0"),
            source_id="split-test",
            raw_action="Buy",
        )

        adjusted = _adjust_transactions_for_splits(
            [original],
            {"SOXS": {date(2026, 7, 15): Decimal("0.1")}},
        )

        self.assertEqual(Decimal("0.4"), adjusted[0].quantity)
        self.assertEqual(Decimal("35.0"), adjusted[0].price)
        self.assertEqual(Decimal("-14"), adjusted[0].cash_delta)

    def test_twr_removes_external_deposit(self) -> None:
        points = build_performance(
            values=[
                (date(2026, 1, 1), Decimal("100")),
                (date(2026, 1, 2), Decimal("210")),
            ],
            external_flows={date(2026, 1, 2): Decimal("100")},
        )

        self.assertEqual(Decimal("100"), points[0][1])
        self.assertEqual(Decimal("110.0"), points[-1][1])

    def test_full_withdrawal_then_new_deposit_restarts_measurement_period(self) -> None:
        points = build_performance(
            values=[
                (date(2026, 1, 1), Decimal("100")),
                (date(2026, 1, 2), Decimal("0")),
                (date(2026, 2, 1), Decimal("200")),
                (date(2026, 2, 2), Decimal("210")),
            ],
            external_flows={
                date(2026, 1, 2): Decimal("-100"),
                date(2026, 2, 1): Decimal("200"),
            },
        )

        self.assertEqual(
            [
                (date(2026, 1, 1), Decimal("100")),
                (date(2026, 2, 1), Decimal("100")),
                (date(2026, 2, 2), Decimal("105.00")),
            ],
            points,
        )


class GeneratorTest(unittest.TestCase):
    def test_generates_isolated_markets_and_strategy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "portfolio.json"
            payload = generate_dashboard(
                us_path=FIXTURES / "tradingview.csv",
                cn_path=FIXTURES / "yinhe.csv",
                output_path=output,
                providers=providers(),
                generated_at=datetime.fromisoformat(
                    "2026-07-17T12:00:00+08:00"
                ),
            )

        self.assertEqual({"us", "cn"}, set(payload["markets"]))
        self.assertEqual("USD", payload["markets"]["us"]["currency"])
        self.assertEqual("CNY", payload["markets"]["cn"]["currency"])
        self.assertEqual([], payload["markets"]["cn"]["groups"])
        self.assertEqual("QQQI", payload["markets"]["us"]["groups"][0]["positions"][0]["symbol"])

    def test_does_not_replace_previous_json_when_current_price_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "portfolio.json"
            output.write_text('{"sentinel": true}', encoding="utf-8")

            with self.assertRaisesRegex(MissingPriceError, "QQQI"):
                generate_dashboard(
                    us_path=FIXTURES / "tradingview.csv",
                    cn_path=FIXTURES / "yinhe.csv",
                    output_path=output,
                    providers=providers(omit_qqqi=True),
                    generated_at=datetime.fromisoformat(
                        "2026-07-17T12:00:00+08:00"
                    ),
                )

            self.assertEqual(
                {"sentinel": True},
                json.loads(output.read_text(encoding="utf-8")),
            )


if __name__ == "__main__":
    unittest.main()
