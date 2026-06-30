from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any


TIMEFRAMES = ("daily", "weekly", "monthly")


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def to_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if value is None:
        raise ValueError("Cannot convert None to Decimal.")
    return Decimal(str(value))


@dataclass(frozen=True)
class IndicatorSnapshot:
    symbol: str
    timeframe: str
    bar_date: date
    close: Decimal
    lower_band: Decimal
    upper_band: Decimal
    source: str


@dataclass(frozen=True)
class AccountSnapshot:
    broker: str
    currency: str
    net_liquidation: Decimal | None
    cash_available_without_margin: Decimal | None
    as_of: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class Position:
    symbol: str
    shares: int


@dataclass(frozen=True)
class Signal:
    id: str
    strategy_id: str
    symbol: str
    side: str
    timeframe: str
    bar_date: date
    close: Decimal
    band_value: Decimal
    reason: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class PaperOrder:
    id: str
    strategy_id: str
    symbol: str
    side: str
    shares: int
    limit_price: Decimal
    fill_price: Decimal
    notional: Decimal
    broker: str
    source_signal_id: str
    status: str
    reason: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class RunResult:
    strategy_id: str
    dry_run: bool
    indicators: list[IndicatorSnapshot] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    orders: list[PaperOrder] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    provider_errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.orders or self.signals:
            return "completed"
        if self.provider_errors:
            return "completed_with_provider_errors"
        return "completed_no_signals"
