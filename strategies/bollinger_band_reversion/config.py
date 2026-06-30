from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from .models import TIMEFRAMES


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class StrategyConfig:
    strategy_id: str
    mode: str
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    buy_percent_of_net_liquidation: Decimal
    sell_percent_of_position: Decimal
    estimated_fee_per_order: Decimal
    broker_priority: tuple[str, ...]
    data_source_priority: tuple[str, ...]
    provider_symbol_map: dict[str, dict[str, str]]
    raw: dict[str, Any]
    config_path: Path
    sqlite_path: Path

    def provider_symbol(self, provider: str, symbol: str) -> str | None:
        return self.provider_symbol_map.get(provider, {}).get(symbol)


def load_config(path: str | Path | None = None) -> StrategyConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    strategy = raw.get("strategy") or {}
    buy = strategy.get("buy") or {}
    sell = strategy.get("sell") or {}
    risk = strategy.get("risk") or {}
    brokers = raw.get("brokers") or {}
    data_sources = raw.get("data_sources") or {}
    storage = raw.get("storage") or {}

    symbols = tuple(str(item) for item in required_list(strategy, "symbols"))
    timeframes = tuple(str(item) for item in required_list(strategy, "timeframes"))
    unknown_timeframes = [item for item in timeframes if item not in TIMEFRAMES]
    if unknown_timeframes:
        raise ValueError(f"Unsupported timeframes: {unknown_timeframes}")

    sqlite_raw = storage.get("sqlite_path") or "strategies/bollinger_band_reversion/state.sqlite"
    sqlite_path = Path(sqlite_raw)
    if not sqlite_path.is_absolute():
        sqlite_path = config_path.parents[2] / sqlite_path

    return StrategyConfig(
        strategy_id=str(strategy.get("id") or "bollinger_band_reversion"),
        mode=str(strategy.get("mode") or "paper"),
        symbols=symbols,
        timeframes=timeframes,
        buy_percent_of_net_liquidation=Decimal(str(buy.get("percent_of_net_liquidation", "0.01"))),
        sell_percent_of_position=Decimal(str(sell.get("percent_of_position", "0.20"))),
        estimated_fee_per_order=Decimal(str(risk.get("estimated_fee_per_order", "0"))),
        broker_priority=tuple(str(item) for item in brokers.get("priority", ("schwab", "ibkr"))),
        data_source_priority=tuple(str(item) for item in data_sources.get("priority", ("alpha_vantage",))),
        provider_symbol_map={
            str(provider): {str(symbol): str(mapped) for symbol, mapped in mapping.items()}
            for provider, mapping in (raw.get("provider_symbol_map") or {}).items()
        },
        raw=raw,
        config_path=config_path,
        sqlite_path=sqlite_path.resolve(),
    )


def required_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"config.yaml requires a non-empty strategy.{key} list.")
    return value
