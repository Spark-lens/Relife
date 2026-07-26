import json
import unittest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from portfolio_dashboard.classification import (
    classify_symbol,
    load_classification_config,
)
from portfolio_dashboard.ledger import position_metrics, replay_ledger
from portfolio_dashboard.models import Transaction


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "templates"
    / "portfolio"
    / "strategy_groups.json"
)


def tx(
    kind: str,
    *,
    day: str,
    symbol: str = "ABC",
    quantity: str = "0",
    price: str = "0",
    fee: str = "0",
    cash_delta: str | None = None,
    source_index: int = 0,
    market: str = "us",
    raw_action: str | None = None,
    cash_balance: str | None = None,
) -> Transaction:
    qty = Decimal(quantity)
    fill = Decimal(price)
    commission = Decimal(fee)
    if cash_delta is None:
        if kind == "buy":
            cash = -(qty * fill + commission)
        elif kind == "sell":
            cash = qty * fill - commission
        else:
            cash = Decimal("0")
    else:
        cash = Decimal(cash_delta)
    return Transaction(
        market=market,  # type: ignore[arg-type]
        timestamp=datetime.fromisoformat(f"{day}T00:00:00"),
        source_index=source_index,
        symbol=symbol,
        name=symbol,
        kind=kind,  # type: ignore[arg-type]
        quantity=qty,
        price=fill,
        fee=commission,
        cash_delta=cash,
        external_cash_flow=cash if kind in {"deposit", "withdrawal"} else Decimal("0"),
        source_id=f"{day}:{source_index}:{kind}",
        raw_action=raw_action or kind,
        cash_balance=(
            None if cash_balance is None else Decimal(cash_balance)
        ),
    )


class LedgerTest(unittest.TestCase):
    def test_chooses_same_day_cash_balance_that_matches_all_cash_deltas(self) -> None:
        result = replay_ledger(
            [
                tx(
                    "deposit",
                    day="2026-01-01",
                    cash_delta="100",
                    market="cn",
                    cash_balance="100",
                ),
                tx(
                    "interest",
                    day="2026-01-02",
                    cash_delta="20",
                    market="cn",
                    cash_balance="120",
                    source_index=0,
                ),
                tx(
                    "deposit",
                    day="2026-01-02",
                    cash_delta="50",
                    market="cn",
                    cash_balance="170",
                    source_index=1,
                ),
            ],
            closes={},
        )

        self.assertEqual(Decimal("170"), result.cash)

    def test_reverse_repo_principal_remains_in_total_assets_until_redemption(self) -> None:
        result = replay_ledger(
            [
                tx(
                    "deposit",
                    day="2026-01-01",
                    cash_delta="30000",
                    market="cn",
                    cash_balance="30000",
                ),
                tx(
                    "cash",
                    day="2026-01-02",
                    symbol="132001",
                    cash_delta="-30000",
                    market="cn",
                    raw_action="报价融券回购",
                    cash_balance="0",
                ),
                tx(
                    "cash",
                    day="2026-01-03",
                    symbol="132001",
                    cash_delta="30188.25",
                    market="cn",
                    raw_action="报价融券购回",
                    cash_balance="30188.25",
                ),
            ],
            closes={},
        )

        self.assertEqual(
            [
                Decimal("30000"),
                Decimal("30000"),
                Decimal("30188.25"),
            ],
            [item.total_assets for item in result.daily_values],
        )

    def test_partial_sell_uses_weighted_average_and_net_proceeds(self) -> None:
        result = replay_ledger(
            [
                tx("deposit", day="2026-07-01", cash_delta="100"),
                tx("buy", day="2026-07-02", quantity="2", price="10", fee="1"),
                tx("buy", day="2026-07-03", quantity="2", price="14", fee="1"),
                tx("sell", day="2026-07-04", quantity="1", price="15", fee="0.5"),
            ],
            closes={
                "ABC": {
                    date(2026, 7, 3): Decimal("16"),
                    date(2026, 7, 4): Decimal("17"),
                }
            },
        )

        position = result.positions["ABC"]
        self.assertEqual(Decimal("3"), position.quantity)
        self.assertEqual(Decimal("37.5"), position.total_cost)
        self.assertEqual(Decimal("2"), position.realized_pnl)
        self.assertEqual(Decimal("13.5"), position.unrealized_pnl)
        self.assertEqual(Decimal("64.5"), result.cash)

    def test_position_metrics_match_approved_column_formulas(self) -> None:
        result = replay_ledger(
            [
                tx("deposit", day="2026-07-01", cash_delta="200"),
                tx("buy", day="2026-07-02", quantity="10", price="8"),
            ],
            closes={
                "ABC": {
                    date(2026, 7, 2): Decimal("9"),
                    date(2026, 7, 3): Decimal("10"),
                }
            },
        )
        metrics = position_metrics(
            result.positions["ABC"],
            last_close=Decimal("10"),
            previous_close=Decimal("9"),
            total_assets=Decimal("200"),
        )

        self.assertEqual(Decimal("0.5"), metrics.weight)
        self.assertEqual(Decimal("10"), metrics.daily_pnl)
        self.assertEqual(Decimal("20"), metrics.unrealized_pnl)
        self.assertEqual(Decimal("0.25"), metrics.unrealized_pnl_pct)
        self.assertEqual(Decimal("0.10"), metrics.portfolio_contribution_pct)

    def test_dividend_tax_adjustments_are_grouped_by_day_and_symbol(self) -> None:
        result = replay_ledger(
            [
                tx("dividend", day="2026-07-10", symbol="XQQI", cash_delta="0.09"),
                tx(
                    "dividend",
                    day="2026-07-10",
                    symbol="XQQI",
                    cash_delta="-0.01",
                    source_index=1,
                ),
            ],
            closes={},
        )

        self.assertEqual(1, len(result.dividends))
        self.assertEqual(Decimal("0.09"), result.dividends[0].gross)
        self.assertEqual(Decimal("-0.01"), result.dividends[0].tax_adjustment)
        self.assertEqual(Decimal("0.08"), result.dividends[0].net)


class ClassificationTest(unittest.TestCase):
    def test_strategy_groups_and_fallback_follow_the_strategy_document(self) -> None:
        config = load_classification_config(CONFIG_PATH)

        self.assertEqual("cashflow", classify_symbol("us", "BOXX", "BOXX", config).id)
        self.assertEqual("dividend", classify_symbol("us", "QQQI", "QQQI", config).id)
        self.assertEqual("leverage", classify_symbol("us", "SOXS", "SOXS", config).id)
        fallback = classify_symbol("us", "XBI", "XBI", config)
        self.assertEqual("other", fallback.id)
        self.assertEqual("策略外", fallback.badge)

    def test_classification_config_is_valid_json(self) -> None:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertIn("groups", data)
        self.assertIn("fallback", data)


if __name__ == "__main__":
    unittest.main()
