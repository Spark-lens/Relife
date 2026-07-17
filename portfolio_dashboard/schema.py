from __future__ import annotations

from typing import Any


REQUIRED_MARKET_KEYS = {
    "currency",
    "asOf",
    "summary",
    "performance",
    "benchmarks",
    "groups",
    "transactions",
    "dividends",
    "dividendMonths",
}


def validate_dashboard_payload(payload: dict[str, Any]) -> None:
    generated_at = payload.get("generatedAt")
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("generatedAt 必须是非空字符串")

    markets = payload.get("markets")
    if not isinstance(markets, dict):
        raise ValueError("markets 必须是对象")

    for market in ("us", "cn"):
        value = markets.get(market)
        if not isinstance(value, dict):
            raise ValueError(f"缺少 markets.{market}")
        missing = REQUIRED_MARKET_KEYS - value.keys()
        if missing:
            raise ValueError(f"markets.{market} 缺少字段：{sorted(missing)}")

