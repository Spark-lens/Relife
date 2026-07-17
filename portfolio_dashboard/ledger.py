from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Iterable, Mapping, Sequence

from .models import Transaction


PriceMatrix = Mapping[str, Mapping[date, Decimal]]


@dataclass
class PositionState:
    symbol: str
    name: str
    quantity: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")
    cumulative_buy_cost: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    net_dividends: Decimal = Decimal("0")
    last_close: Decimal = Decimal("0")
    previous_close: Decimal = Decimal("0")

    @property
    def average_cost(self) -> Decimal:
        return (
            Decimal("0")
            if self.quantity == 0
            else self.total_cost / self.quantity
        )

    @property
    def market_value(self) -> Decimal:
        return self.quantity * self.last_close

    @property
    def unrealized_pnl(self) -> Decimal:
        return self.market_value - self.total_cost

    @property
    def total_pnl(self) -> Decimal:
        return self.realized_pnl + self.unrealized_pnl + self.net_dividends


@dataclass(frozen=True)
class PositionMetrics:
    weight: Decimal | None
    market_value: Decimal
    daily_pnl: Decimal
    daily_pnl_pct: Decimal | None
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal | None
    portfolio_contribution_pct: Decimal | None
    total_pnl: Decimal
    total_pnl_pct: Decimal | None


@dataclass(frozen=True)
class DividendRecord:
    day: date
    symbol: str
    name: str
    gross: Decimal
    tax_adjustment: Decimal
    net: Decimal


@dataclass(frozen=True)
class DailyValue:
    day: date
    total_assets: Decimal
    cash: Decimal


@dataclass
class LedgerResult:
    positions: dict[str, PositionState]
    all_positions: dict[str, PositionState]
    cash: Decimal
    dividends: list[DividendRecord]
    daily_values: list[DailyValue]
    external_flows: dict[date, Decimal]
    transactions: list[Transaction]


@dataclass
class _DividendAccumulator:
    name: str
    gross: Decimal = Decimal("0")
    tax_adjustment: Decimal = Decimal("0")

    @property
    def net(self) -> Decimal:
        return self.gross + self.tax_adjustment


def safe_ratio(
    numerator: Decimal,
    denominator: Decimal,
) -> Decimal | None:
    return None if denominator == 0 else numerator / denominator


def position_metrics(
    position: PositionState,
    *,
    last_close: Decimal,
    previous_close: Decimal,
    total_assets: Decimal,
) -> PositionMetrics:
    market_value = position.quantity * last_close
    previous_market_value = position.quantity * previous_close
    daily_pnl = position.quantity * (last_close - previous_close)
    unrealized_pnl = market_value - position.total_cost
    total_pnl = position.realized_pnl + unrealized_pnl + position.net_dividends
    return PositionMetrics(
        weight=safe_ratio(market_value, total_assets),
        market_value=market_value,
        daily_pnl=daily_pnl,
        daily_pnl_pct=safe_ratio(daily_pnl, previous_market_value),
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=safe_ratio(unrealized_pnl, position.total_cost),
        portfolio_contribution_pct=safe_ratio(unrealized_pnl, total_assets),
        total_pnl=total_pnl,
        total_pnl_pct=safe_ratio(total_pnl, position.cumulative_buy_cost),
    )


def _chronological(transactions: Sequence[Transaction]) -> list[Transaction]:
    return sorted(
        transactions,
        key=lambda item: (item.timestamp, -item.source_index),
    )


def _price_on_or_before(
    series: Mapping[date, Decimal],
    day: date,
) -> Decimal | None:
    eligible = [price_day for price_day in series if price_day <= day]
    return None if not eligible else series[max(eligible)]


def _update_position(
    states: dict[str, PositionState],
    transaction: Transaction,
) -> None:
    if not transaction.symbol:
        return
    state = states.setdefault(
        transaction.symbol,
        PositionState(symbol=transaction.symbol, name=transaction.name),
    )
    if transaction.kind == "buy":
        purchase_cost = transaction.quantity * transaction.price + transaction.fee
        state.quantity += transaction.quantity
        state.total_cost += purchase_cost
        state.cumulative_buy_cost += purchase_cost
    elif transaction.kind == "sell":
        if transaction.quantity > state.quantity:
            raise ValueError(
                f"{transaction.symbol} 卖出数量 {transaction.quantity} "
                f"超过持仓 {state.quantity}"
            )
        allocated_cost = state.average_cost * transaction.quantity
        net_proceeds = transaction.quantity * transaction.price - transaction.fee
        state.realized_pnl += net_proceeds - allocated_cost
        state.quantity -= transaction.quantity
        state.total_cost -= allocated_cost
        if state.quantity == 0:
            state.total_cost = Decimal("0")
    elif transaction.kind == "dividend":
        state.net_dividends += transaction.cash_delta


def _build_dividends(
    transactions: Iterable[Transaction],
) -> list[DividendRecord]:
    grouped: dict[tuple[date, str], _DividendAccumulator] = {}
    for transaction in transactions:
        if transaction.kind != "dividend":
            continue
        key = (transaction.timestamp.date(), transaction.symbol)
        accumulator = grouped.setdefault(
            key,
            _DividendAccumulator(name=transaction.name),
        )
        if transaction.cash_delta >= 0:
            accumulator.gross += transaction.cash_delta
        else:
            accumulator.tax_adjustment += transaction.cash_delta
    return [
        DividendRecord(
            day=day,
            symbol=symbol,
            name=accumulator.name,
            gross=accumulator.gross,
            tax_adjustment=accumulator.tax_adjustment,
            net=accumulator.net,
        )
        for (day, symbol), accumulator in sorted(grouped.items(), reverse=True)
    ]


def replay_ledger(
    transactions: Sequence[Transaction],
    closes: PriceMatrix,
) -> LedgerResult:
    ordered = _chronological(transactions)
    transactions_by_day: dict[date, list[Transaction]] = defaultdict(list)
    for transaction in ordered:
        transactions_by_day[transaction.timestamp.date()].append(transaction)

    all_days = set(transactions_by_day)
    for series in closes.values():
        all_days.update(series)

    states: dict[str, PositionState] = {}
    cash = Decimal("0")
    daily_values: list[DailyValue] = []
    external_flows: dict[date, Decimal] = defaultdict(Decimal)

    for day in sorted(all_days):
        day_transactions = transactions_by_day.get(day, [])
        for transaction in day_transactions:
            cash += transaction.cash_delta
            external_flows[day] += transaction.external_cash_flow
            _update_position(states, transaction)

        authoritative = [
            transaction
            for transaction in day_transactions
            if transaction.market == "cn" and transaction.cash_balance is not None
        ]
        if authoritative:
            cash = min(
                authoritative,
                key=lambda item: item.source_index,
            ).cash_balance or Decimal("0")

        securities_value = Decimal("0")
        for symbol, state in states.items():
            if state.quantity == 0:
                continue
            price = _price_on_or_before(closes.get(symbol, {}), day)
            if price is not None:
                securities_value += state.quantity * price
        daily_values.append(
            DailyValue(
                day=day,
                total_assets=cash + securities_value,
                cash=cash,
            )
        )

    for symbol, state in states.items():
        series = closes.get(symbol, {})
        if not series:
            continue
        ordered_prices = sorted(series.items())
        state.last_close = ordered_prices[-1][1]
        state.previous_close = (
            ordered_prices[-2][1] if len(ordered_prices) > 1 else state.last_close
        )

    current_positions = {
        symbol: state
        for symbol, state in states.items()
        if state.quantity != 0
    }
    return LedgerResult(
        positions=current_positions,
        all_positions=states,
        cash=cash,
        dividends=_build_dividends(ordered),
        daily_values=daily_values,
        external_flows=dict(external_flows),
        transactions=list(transactions),
    )

