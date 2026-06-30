from __future__ import annotations

import hashlib
import uuid
from datetime import date
from decimal import Decimal, ROUND_FLOOR

from .brokers import BrokerAdapter
from .config import StrategyConfig
from .models import IndicatorSnapshot, PaperOrder, RunResult, Signal, utc_now
from .providers import MarketDataProvider
from .storage import StrategyStorage


class StrategyEngine:
    def __init__(
        self,
        config: StrategyConfig,
        market_data: MarketDataProvider,
        broker: BrokerAdapter,
        storage: StrategyStorage,
    ):
        self.config = config
        self.market_data = market_data
        self.broker = broker
        self.storage = storage

    def run_once(self, *, dry_run: bool = False, run_date: date | None = None) -> RunResult:
        if self.config.mode != "paper":
            raise RuntimeError("Only paper mode is implemented. Real trading must be enabled separately.")

        started_at = utc_now()
        effective_run_date = run_date or started_at.date()
        result = RunResult(strategy_id=self.config.strategy_id, dry_run=dry_run)
        snapshots_by_symbol: dict[str, dict[str, IndicatorSnapshot]] = {}

        for symbol in self.config.symbols:
            snapshots_by_symbol[symbol] = {}
            for timeframe in self.config.timeframes:
                try:
                    snapshot = self.market_data.fetch_indicator(symbol, timeframe)
                    snapshots_by_symbol[symbol][timeframe] = snapshot
                    result.indicators.append(snapshot)
                    if not dry_run:
                        self.storage.save_indicator(snapshot)
                except Exception as exc:
                    result.provider_errors.append(f"{symbol} {timeframe}: {exc}")

        account = self.broker.account_snapshot()
        positions = self.broker.positions(list(self.config.symbols))
        if not dry_run:
            self.storage.save_account_snapshot(account)

        for symbol in self.config.symbols:
            symbol_snapshots = snapshots_by_symbol.get(symbol, {})
            buy_triggers = [
                snapshot
                for timeframe in self.config.timeframes
                if (snapshot := symbol_snapshots.get(timeframe)) and snapshot.close < snapshot.lower_band
            ]
            sell_trigger = symbol_snapshots.get("monthly")
            if buy_triggers:
                signals = [
                    build_signal(self.config.strategy_id, symbol, "buy", snapshot, "close_below_lower_band")
                    for snapshot in buy_triggers
                ]
                result.signals.extend(signals)
                if not dry_run:
                    for signal in signals:
                        self.storage.save_signal(signal)
                order = self._build_buy_order(symbol, signals, account, effective_run_date)
                if isinstance(order, PaperOrder):
                    result.orders.append(order)
                    if not dry_run:
                        self.storage.save_order(order)
                        self.storage.save_notification(
                            self.config.strategy_id,
                            f"[Relife] paper BUY {symbol}",
                            order.reason,
                        )
                elif order:
                    result.blocked_actions.append(order)

            if sell_trigger and sell_trigger.close > sell_trigger.upper_band:
                signal = build_signal(self.config.strategy_id, symbol, "sell", sell_trigger, "monthly_close_above_upper_band")
                result.signals.append(signal)
                if not dry_run:
                    self.storage.save_signal(signal)
                order = self._build_sell_order(symbol, signal, positions.get(symbol).shares if positions.get(symbol) else 0)
                if isinstance(order, PaperOrder):
                    result.orders.append(order)
                    if not dry_run:
                        self.storage.save_order(order)
                        self.storage.save_notification(
                            self.config.strategy_id,
                            f"[Relife] paper SELL {symbol}",
                            order.reason,
                        )
                elif order:
                    result.blocked_actions.append(order)

        if not dry_run:
            self.storage.save_run(
                self.config.strategy_id,
                started_at,
                dry_run,
                result.status,
                summarize_result(result),
            )
        return result

    def _build_buy_order(
        self,
        symbol: str,
        signals: list[Signal],
        account,
        run_date: date,
    ) -> PaperOrder | str | None:
        if self.storage.has_buy_order_on_date(self.config.strategy_id, symbol, run_date):
            return f"{symbol}: buy blocked because a buy order already exists on {run_date.isoformat()}."
        if account.net_liquidation is None:
            return f"{symbol}: buy blocked because net liquidation is unavailable."
        if account.cash_available_without_margin is None:
            return f"{symbol}: buy blocked because cash_available_without_margin is unavailable."

        trigger = select_trigger_signal(signals, self.config.timeframes)
        amount = account.net_liquidation * self.config.buy_percent_of_net_liquidation
        shares = floor_decimal(amount / trigger.close)
        if shares < 1:
            return f"{symbol}: buy blocked because 1% net liquidation buys fewer than 1 share."

        notional = Decimal(shares) * trigger.close
        required_cash = notional + self.config.estimated_fee_per_order
        if account.cash_available_without_margin < required_cash:
            return (
                f"{symbol}: buy blocked because non-margin cash {account.cash_available_without_margin} "
                f"is less than required {required_cash}."
            )

        return PaperOrder(
            id=f"paper-{uuid.uuid4()}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            side="buy",
            shares=shares,
            limit_price=trigger.close,
            fill_price=trigger.close,
            notional=notional,
            broker=account.broker,
            source_signal_id=trigger.id,
            status="draft",
            reason=(
                f"paper buy: {symbol} {shares} shares at close {trigger.close}; "
                f"trigger={trigger.timeframe}; bar={trigger.bar_date.isoformat()}; "
                f"basis=net_liquidation*{self.config.buy_percent_of_net_liquidation}"
            ),
        )

    def _build_sell_order(self, symbol: str, signal: Signal, position_shares: int) -> PaperOrder | str | None:
        if self.storage.has_sell_order_for_monthly_bar(self.config.strategy_id, symbol, signal.bar_date):
            return f"{symbol}: sell blocked because monthly bar {signal.bar_date.isoformat()} already has a sell order."
        if position_shares <= 0:
            return f"{symbol}: sell blocked because there is no current position."

        shares = max(1, floor_decimal(Decimal(position_shares) * self.config.sell_percent_of_position))
        shares = min(shares, position_shares)
        notional = Decimal(shares) * signal.close
        return PaperOrder(
            id=f"paper-{uuid.uuid4()}",
            strategy_id=self.config.strategy_id,
            symbol=symbol,
            side="sell",
            shares=shares,
            limit_price=signal.close,
            fill_price=signal.close,
            notional=notional,
            broker=self.broker.broker,
            source_signal_id=signal.id,
            status="draft",
            reason=(
                f"paper sell: {symbol} {shares}/{position_shares} shares at close {signal.close}; "
                f"monthly_bar={signal.bar_date.isoformat()}; "
                f"basis=position*{self.config.sell_percent_of_position}"
            ),
        )


def build_signal(strategy_id: str, symbol: str, side: str, snapshot: IndicatorSnapshot, reason: str) -> Signal:
    band_value = snapshot.lower_band if side == "buy" else snapshot.upper_band
    signal_id = stable_id(strategy_id, symbol, side, snapshot.timeframe, snapshot.bar_date.isoformat())
    return Signal(
        id=signal_id,
        strategy_id=strategy_id,
        symbol=symbol,
        side=side,
        timeframe=snapshot.timeframe,
        bar_date=snapshot.bar_date,
        close=snapshot.close,
        band_value=band_value,
        reason=reason,
    )


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"sig-{digest}"


def floor_decimal(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def select_trigger_signal(signals: list[Signal], timeframe_priority: tuple[str, ...]) -> Signal:
    by_timeframe = {signal.timeframe: signal for signal in signals}
    for timeframe in timeframe_priority:
        if timeframe in by_timeframe:
            return by_timeframe[timeframe]
    return signals[0]


def summarize_result(result: RunResult) -> str:
    return (
        f"indicators={len(result.indicators)} signals={len(result.signals)} "
        f"orders={len(result.orders)} blocked={len(result.blocked_actions)} "
        f"provider_errors={len(result.provider_errors)}"
    )
