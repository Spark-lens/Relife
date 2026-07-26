from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal


Market = Literal["us", "cn"]
TransactionKind = Literal[
    "buy",
    "sell",
    "dividend",
    "deposit",
    "withdrawal",
    "interest",
    "cash",
]


@dataclass(frozen=True)
class Transaction:
    market: Market
    timestamp: datetime
    source_index: int
    symbol: str
    name: str
    kind: TransactionKind
    quantity: Decimal
    price: Decimal
    fee: Decimal
    cash_delta: Decimal
    external_cash_flow: Decimal
    source_id: str
    raw_action: str
    cash_balance: Decimal | None = None
