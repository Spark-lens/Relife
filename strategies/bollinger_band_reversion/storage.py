from __future__ import annotations

import sqlite3
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from .models import AccountSnapshot, IndicatorSnapshot, PaperOrder, Signal, utc_now


class StrategyStorage:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) == ":memory:":
            self.connection = sqlite3.connect(":memory:")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS indicator_snapshots (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                bar_date TEXT NOT NULL,
                close TEXT NOT NULL,
                lower_band TEXT NOT NULL,
                upper_band TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (symbol, timeframe, bar_date, source)
            );

            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                broker TEXT NOT NULL,
                currency TEXT NOT NULL,
                net_liquidation TEXT,
                cash_available_without_margin TEXT,
                as_of TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                bar_date TEXT NOT NULL,
                close TEXT NOT NULL,
                band_value TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (strategy_id, symbol, side, timeframe, bar_date)
            );

            CREATE TABLE IF NOT EXISTS paper_orders (
                id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                shares INTEGER NOT NULL,
                limit_price TEXT NOT NULL,
                fill_price TEXT NOT NULL,
                notional TEXT NOT NULL,
                broker TEXT NOT NULL,
                source_signal_id TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                provider TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS strategy_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                dry_run INTEGER NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def save_indicator(self, snapshot: IndicatorSnapshot) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO indicator_snapshots
            (symbol, timeframe, bar_date, close, lower_band, upper_band, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.symbol,
                snapshot.timeframe,
                snapshot.bar_date.isoformat(),
                str(snapshot.close),
                str(snapshot.lower_band),
                str(snapshot.upper_band),
                snapshot.source,
                utc_now().isoformat(),
            ),
        )
        self.connection.commit()

    def save_account_snapshot(self, snapshot: AccountSnapshot) -> None:
        self.connection.execute(
            """
            INSERT INTO account_snapshots
            (broker, currency, net_liquidation, cash_available_without_margin, as_of)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot.broker,
                snapshot.currency,
                maybe_decimal(snapshot.net_liquidation),
                maybe_decimal(snapshot.cash_available_without_margin),
                snapshot.as_of.isoformat(),
            ),
        )
        self.connection.commit()

    def save_signal(self, signal: Signal) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO signals
            (id, strategy_id, symbol, side, timeframe, bar_date, close, band_value, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.id,
                signal.strategy_id,
                signal.symbol,
                signal.side,
                signal.timeframe,
                signal.bar_date.isoformat(),
                str(signal.close),
                str(signal.band_value),
                signal.reason,
                signal.created_at.isoformat(),
            ),
        )
        self.connection.commit()

    def save_order(self, order: PaperOrder) -> None:
        self.connection.execute(
            """
            INSERT INTO paper_orders
            (id, strategy_id, symbol, side, shares, limit_price, fill_price, notional, broker,
             source_signal_id, status, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.id,
                order.strategy_id,
                order.symbol,
                order.side,
                order.shares,
                str(order.limit_price),
                str(order.fill_price),
                str(order.notional),
                order.broker,
                order.source_signal_id,
                order.status,
                order.reason,
                order.created_at.isoformat(),
            ),
        )
        self.connection.commit()

    def save_notification(self, strategy_id: str, subject: str, body: str, provider: str = "mock") -> None:
        self.connection.execute(
            """
            INSERT INTO notifications (strategy_id, subject, body, provider, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (strategy_id, subject, body, provider, utc_now().isoformat()),
        )
        self.connection.commit()

    def save_run(self, strategy_id: str, started_at: datetime, dry_run: bool, status: str, message: str) -> None:
        self.connection.execute(
            """
            INSERT INTO strategy_runs (strategy_id, started_at, finished_at, dry_run, status, message)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (strategy_id, started_at.isoformat(), utc_now().isoformat(), int(dry_run), status, message),
        )
        self.connection.commit()

    def has_buy_order_on_date(self, strategy_id: str, symbol: str, run_date: date) -> bool:
        row = self.connection.execute(
            """
            SELECT 1 FROM paper_orders
            WHERE strategy_id = ? AND symbol = ? AND side = 'buy' AND date(created_at) = ?
            LIMIT 1
            """,
            (strategy_id, symbol, run_date.isoformat()),
        ).fetchone()
        return row is not None

    def has_sell_order_for_monthly_bar(self, strategy_id: str, symbol: str, monthly_bar_date: date) -> bool:
        reason_fragment = f"monthly_bar={monthly_bar_date.isoformat()}"
        row = self.connection.execute(
            """
            SELECT 1 FROM paper_orders
            WHERE strategy_id = ? AND symbol = ? AND side = 'sell' AND reason LIKE ?
            LIMIT 1
            """,
            (strategy_id, symbol, f"%{reason_fragment}%"),
        ).fetchone()
        return row is not None


def maybe_decimal(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None
