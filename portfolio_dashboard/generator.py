from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .classification import (
    Classification,
    classify_symbol,
    load_classification_config,
)
from .ledger import DailyValue, LedgerResult, PriceMatrix, position_metrics, replay_ledger
from .market_data import PriceProvider
from .parsers import parse_tradingview, parse_yinhe
from .schema import validate_dashboard_payload


ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_CONFIG = (
    ROOT / "data" / "templates" / "portfolio" / "strategy_groups.json"
)
BENCHMARKS = {
    "us": [
        {"id": "__QQQ__", "label": "QQQ", "field": "qqq"},
        {"id": "__SPY__", "label": "SPY", "field": "spy"},
    ],
    "cn": [
        {"id": "__SSE__", "label": "上证指数", "field": "sse"},
        {"id": "__CSI300__", "label": "沪深300", "field": "csi300"},
    ],
}


class MissingPriceError(ValueError):
    pass


@dataclass(frozen=True)
class ProviderBundle:
    us: PriceProvider
    cn: PriceProvider


def _number(value: Decimal | None) -> float | None:
    return None if value is None else float(round(value, 10))


def build_performance(
    values: list[tuple[date, Decimal]] | list[DailyValue],
    external_flows: Mapping[date, Decimal],
) -> list[tuple[date, Decimal]]:
    normalized = [
        (
            item.day,
            item.total_assets,
        )
        if isinstance(item, DailyValue)
        else item
        for item in values
    ]
    normalized = sorted(normalized)
    if not normalized:
        return []

    result: list[tuple[date, Decimal]] = []
    previous_value: Decimal | None = None
    index_value = Decimal("100")
    for value_day, total_assets in normalized:
        if total_assets <= 0:
            continue
        if previous_value is None:
            result.append((value_day, index_value))
            previous_value = total_assets
            continue
        if previous_value != 0:
            external_flow = external_flows.get(value_day, Decimal("0"))
            factor = (total_assets - external_flow) / previous_value
            index_value *= factor
            result.append((value_day, index_value))
        previous_value = total_assets
    return result


def _price_on_or_before(
    series: Mapping[date, Decimal],
    target: date,
) -> Decimal | None:
    eligible = [price_day for price_day in series if price_day <= target]
    return None if not eligible else series[max(eligible)]


def _aligned_performance(
    ledger: LedgerResult,
    prices: PriceMatrix,
    benchmarks: list[dict[str, str]],
) -> list[dict[str, Any]]:
    portfolio = build_performance(ledger.daily_values, ledger.external_flows)
    common: list[tuple[date, Decimal, list[Decimal]]] = []
    for point_day, portfolio_value in portfolio:
        benchmark_values = [
            _price_on_or_before(prices.get(item["id"], {}), point_day)
            for item in benchmarks
        ]
        if all(value is not None for value in benchmark_values):
            common.append(
                (
                    point_day,
                    portfolio_value,
                    [value for value in benchmark_values if value is not None],
                )
            )
    if not common:
        return []

    portfolio_base = common[0][1]
    benchmark_bases = common[0][2]
    result: list[dict[str, Any]] = []
    for point_day, portfolio_value, benchmark_values in common:
        item: dict[str, Any] = {
            "date": point_day.isoformat(),
            "portfolio": _number(portfolio_value / portfolio_base * Decimal("100")),
        }
        for benchmark, value, base in zip(
            benchmarks,
            benchmark_values,
            benchmark_bases,
        ):
            item[benchmark["field"]] = _number(value / base * Decimal("100"))
        result.append(item)
    return result


def _transaction_symbols(transactions: list[Any]) -> list[str]:
    return sorted(
        {
            transaction.symbol
            for transaction in transactions
            if transaction.kind in {"buy", "sell"} and transaction.symbol
        }
    )


def _serialize_transactions(transactions: list[Any]) -> list[dict[str, Any]]:
    ordered = sorted(
        transactions,
        key=lambda item: (item.timestamp, -item.source_index),
        reverse=True,
    )
    return [
        {
            "id": transaction.source_id,
            "timestamp": transaction.timestamp.isoformat(),
            "symbol": transaction.symbol,
            "name": transaction.name,
            "kind": transaction.kind,
            "action": transaction.raw_action,
            "quantity": _number(transaction.quantity),
            "price": _number(transaction.price),
            "amount": _number(transaction.cash_delta),
            "fee": _number(transaction.fee),
        }
        for transaction in ordered
    ]


def _serialize_dividends(ledger: LedgerResult) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        {
            "date": item.day.isoformat(),
            "symbol": item.symbol,
            "name": item.name,
            "gross": _number(item.gross),
            "taxAdjustment": _number(item.tax_adjustment),
            "net": _number(item.net),
        }
        for item in ledger.dividends
    ]
    months: dict[str, Decimal] = defaultdict(Decimal)
    for item in ledger.dividends:
        months[item.day.strftime("%Y-%m")] += item.net
    month_rows = [
        {"month": month, "net": _number(value)}
        for month, value in sorted(months.items())
    ]
    return rows, month_rows


def _serialize_groups(
    market: str,
    ledger: LedgerResult,
    total_assets: Decimal,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, tuple[Classification, list[dict[str, Any]]]] = {}
    for state in ledger.positions.values():
        classification = classify_symbol(
            market,
            state.symbol,
            state.name,
            config,
        )
        metrics = position_metrics(
            state,
            last_close=state.last_close,
            previous_close=state.previous_close,
            total_assets=total_assets,
        )
        position = {
            "symbol": state.symbol,
            "name": state.name,
            "quantity": _number(state.quantity),
            "weight": _number(metrics.weight),
            "totalCost": _number(state.total_cost),
            "averageCost": _number(state.average_cost),
            "lastClose": _number(state.last_close),
            "marketValue": _number(metrics.market_value),
            "dailyPnl": _number(metrics.daily_pnl),
            "dailyPnlPct": _number(metrics.daily_pnl_pct),
            "unrealizedPnl": _number(metrics.unrealized_pnl),
            "unrealizedPnlPct": _number(metrics.unrealized_pnl_pct),
            "portfolioContributionPct": _number(
                metrics.portfolio_contribution_pct
            ),
            "totalPnl": _number(metrics.total_pnl),
            "totalPnlPct": _number(metrics.total_pnl_pct),
        }
        grouped.setdefault(classification.id, (classification, []))[1].append(position)

    config_order = [item["id"] for item in config["groups"]] + [
        config["fallback"]["id"]
    ]
    result: list[dict[str, Any]] = []
    for group_id in config_order:
        if group_id not in grouped:
            continue
        classification, positions = grouped[group_id]
        positions.sort(key=lambda item: item["marketValue"] or 0, reverse=True)
        result.append(
            {
                "id": classification.id,
                "label": classification.label,
                "subgroup": classification.subgroup,
                "badge": classification.badge,
                "marketValue": sum(
                    (item["marketValue"] or 0) for item in positions
                ),
                "weight": (
                    sum((item["marketValue"] or 0) for item in positions)
                    / float(total_assets)
                    if total_assets
                    else None
                ),
                "positions": positions,
            }
        )
    return result


def _build_market(
    market: str,
    transactions: list[Any],
    provider: PriceProvider,
    generated_at: datetime,
    config: dict[str, Any],
) -> dict[str, Any]:
    benchmark_config = BENCHMARKS[market]
    symbols = _transaction_symbols(transactions)
    requested = symbols + [item["id"] for item in benchmark_config]
    start = min(transaction.timestamp.date() for transaction in transactions)
    prices = provider.history(requested, start, generated_at.date())
    ledger = replay_ledger(transactions, prices)

    missing_current = [
        symbol
        for symbol, state in ledger.positions.items()
        if state.quantity != 0 and not prices.get(symbol)
    ]
    if missing_current:
        raise MissingPriceError(
            f"当前持仓缺少行情：{', '.join(sorted(missing_current))}"
        )

    total_assets = (
        ledger.daily_values[-1].total_assets
        if ledger.daily_values
        else ledger.cash
    )
    performance = _aligned_performance(
        ledger,
        prices,
        benchmark_config,
    )
    dividends, dividend_months = _serialize_dividends(ledger)
    as_of_days = [
        max(series)
        for series in prices.values()
        if series
    ]
    as_of = max(as_of_days) if as_of_days else generated_at.date()
    unrealized = sum(
        (state.unrealized_pnl for state in ledger.positions.values()),
        Decimal("0"),
    )
    cumulative_return = (
        Decimal(str(performance[-1]["portfolio"])) / Decimal("100") - Decimal("1")
        if performance
        else Decimal("0")
    )
    return {
        "currency": "USD" if market == "us" else "CNY",
        "asOf": as_of.isoformat(),
        "summary": {
            "totalAssets": _number(total_assets),
            "cumulativeReturn": _number(cumulative_return),
            "unrealizedPnl": _number(unrealized),
            "cash": _number(ledger.cash),
        },
        "performance": performance,
        "benchmarks": [
            {"id": item["field"], "label": item["label"]}
            for item in benchmark_config
        ],
        "groups": _serialize_groups(
            market,
            ledger,
            total_assets,
            config,
        ),
        "transactions": _serialize_transactions(transactions),
        "dividends": dividends,
        "dividendMonths": dividend_months,
    }


def build_dashboard(
    *,
    us_path: Path,
    cn_path: Path,
    providers: ProviderBundle,
    generated_at: datetime,
) -> dict[str, Any]:
    config = load_classification_config(CLASSIFICATION_CONFIG)
    payload = {
        "generatedAt": generated_at.isoformat(),
        "markets": {
            "us": _build_market(
                "us",
                parse_tradingview(us_path),
                providers.us,
                generated_at,
                config,
            ),
            "cn": _build_market(
                "cn",
                parse_yinhe(cn_path),
                providers.cn,
                generated_at,
                config,
            ),
        },
    }
    validate_dashboard_payload(payload)
    return payload


def generate_dashboard(
    *,
    us_path: Path,
    cn_path: Path,
    output_path: Path,
    providers: ProviderBundle,
    generated_at: datetime,
) -> dict[str, Any]:
    payload = build_dashboard(
        us_path=us_path,
        cn_path=cn_path,
        providers=providers,
        generated_at=generated_at,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return payload

