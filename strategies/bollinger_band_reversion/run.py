#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategies.bollinger_band_reversion.brokers import build_broker_adapter
from strategies.bollinger_band_reversion.config import DEFAULT_CONFIG_PATH, load_config
from strategies.bollinger_band_reversion.engine import StrategyEngine, summarize_result
from strategies.bollinger_band_reversion.providers import build_market_data_provider
from strategies.bollinger_band_reversion.storage import StrategyStorage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Relife multi-symbol Bollinger band paper strategy.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to strategy config.yaml.")
    parser.add_argument("--once", action="store_true", help="Run one strategy check.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write SQLite state or notifications.")
    parser.add_argument("--mode", default="paper", choices=["paper"], help="Only paper mode is implemented.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.once:
        print("Nothing to do: pass --once to run one strategy check.")
        return 2

    config = load_config(args.config)
    if args.mode != config.mode:
        raise RuntimeError(f"CLI mode {args.mode!r} does not match config mode {config.mode!r}.")

    storage = StrategyStorage(":memory:" if args.dry_run else config.sqlite_path)
    try:
        storage.initialize()
        engine = StrategyEngine(
            config=config,
            market_data=build_market_data_provider(config),
            broker=build_broker_adapter(config),
            storage=storage,
        )
        result = engine.run_once(dry_run=args.dry_run)
    finally:
        storage.close()

    print("Bollinger band reversion summary")
    print(f"- Strategy: {config.strategy_id}")
    print(f"- Mode: {config.mode}")
    print(f"- Dry run: {args.dry_run}")
    print(f"- {summarize_result(result)}")
    for error in result.provider_errors:
        print(f"- Provider error: {error}")
    for blocked in result.blocked_actions:
        print(f"- Blocked: {blocked}")
    for order in result.orders:
        print(f"- Order: {order.side.upper()} {order.symbol} {order.shares} @ {order.limit_price} ({order.status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
