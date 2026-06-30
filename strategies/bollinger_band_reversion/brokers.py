from __future__ import annotations

import os
import re
from decimal import Decimal, InvalidOperation
from typing import Protocol

from .config import StrategyConfig
from .models import AccountSnapshot, Position


class BrokerAdapter(Protocol):
    broker: str

    def account_snapshot(self) -> AccountSnapshot:
        ...

    def positions(self, symbols: list[str]) -> dict[str, Position]:
        ...


class StaticPaperBrokerAdapter:
    """Safe paper adapter backed by env vars and optional config defaults.

    It never reads margin buying power. Missing non-margin cash is represented
    as None so buy sizing fails closed.
    """

    broker = "schwab"

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.broker_config = ((config.raw.get("brokers") or {}).get("schwab") or {})

    def account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            broker=self.broker,
            currency=str(self.broker_config.get("currency") or "USD"),
            net_liquidation=read_decimal_env_or_config(
                self.broker_config.get("net_liquidation_env"),
                self.broker_config.get("net_liquidation"),
            ),
            cash_available_without_margin=read_decimal_env_or_config(
                self.broker_config.get("cash_available_without_margin_env"),
                self.broker_config.get("cash_available_without_margin"),
            ),
        )

    def positions(self, symbols: list[str]) -> dict[str, Position]:
        configured = self.broker_config.get("positions") or {}
        prefix = str(self.broker_config.get("positions_env_prefix") or "RELIFE_POSITION_")
        result: dict[str, Position] = {}
        for symbol in symbols:
            value = os.environ.get(prefix + env_symbol(symbol), configured.get(symbol, 0))
            result[symbol] = Position(symbol=symbol, shares=max(0, int(Decimal(str(value)))))
        return result


class ReservedBrokerAdapter:
    broker = "ibkr"

    def account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            broker=self.broker,
            currency="USD",
            net_liquidation=None,
            cash_available_without_margin=None,
        )

    def positions(self, symbols: list[str]) -> dict[str, Position]:
        return {symbol: Position(symbol=symbol, shares=0) for symbol in symbols}


def build_broker_adapter(config: StrategyConfig) -> BrokerAdapter:
    # First version is intentionally paper-only and Schwab-first.
    if config.broker_priority and config.broker_priority[0] == "schwab":
        return StaticPaperBrokerAdapter(config)
    return ReservedBrokerAdapter()


def read_decimal_env_or_config(env_name: str | None, configured: object) -> Decimal | None:
    raw = os.environ.get(str(env_name)) if env_name else None
    if raw is None:
        raw = configured
    if raw in (None, ""):
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid decimal value for {env_name or 'broker config'}: {raw!r}") from exc


def env_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", symbol.upper()).strip("_")
