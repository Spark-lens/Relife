from __future__ import annotations

from datetime import date
from decimal import Decimal

from .analytics import risk_metrics, twr, xirr
from .ledger import replay


def _float(value):
    return None if value is None else float(value)


def _last_two(series: dict[str, Decimal]) -> tuple[Decimal | None, Decimal | None]:
    values = [value for _, value in sorted(series.items())]
    if not values:
        return None, None
    return values[-1], values[-2] if len(values) > 1 else values[-1]


def _returns(series: dict[str, float]) -> dict[str, float]:
    result = {}
    previous = None
    for day, value in sorted(series.items()):
        if previous not in {None, 0}:
            result[day] = value / previous - 1
        previous = value
    return result


def build_market_snapshot(
    market: str,
    transactions: list[dict],
    closes: dict[str, dict[str, Decimal]],
    benchmark_closes: dict[str, Decimal],
    dividend_calendar: list[dict],
    errors: list[dict],
    *,
    source: dict | None = None,
) -> dict:
    ledger = replay(market, transactions, closes)
    holdings = []
    prices = []
    distribution = []
    unrealized = Decimal()
    last_day = Decimal()
    previous_value = Decimal()
    for symbol, position in ledger["positions"].items():
        latest, previous = _last_two(closes.get(symbol, {}))
        market_value = position["quantity"] * latest if latest is not None else None
        item_unrealized = market_value - position["remainingCost"] if market_value is not None else None
        day_amount = position["quantity"] * (latest - previous) if latest is not None and previous is not None else None
        if item_unrealized is not None:
            unrealized += item_unrealized
        if day_amount is not None:
            last_day += day_amount
            previous_value += position["quantity"] * previous
        holdings.append({
            "symbol": symbol, "name": position["name"], "quantity": _float(position["quantity"]),
            "averageCost": _float(position["remainingCost"] / position["quantity"] if position["quantity"] else None),
            "remainingCost": _float(position["remainingCost"]), "lastPrice": _float(latest),
            "previousClose": _float(previous), "marketValue": _float(market_value),
            "unrealized": _float(item_unrealized),
            "unrealizedPercent": _float(item_unrealized / position["remainingCost"] if item_unrealized is not None and position["remainingCost"] else None),
            "lastDayAmount": _float(day_amount),
            "lastDayPercent": _float((latest - previous) / previous if latest is not None and previous else None),
            "realized": _float(position["realized"]), "netDividends": _float(position["netDividends"]),
        })
        prices.append({"symbol": symbol, "name": position["name"], "latest": _float(latest), "change": _float(latest - previous if latest is not None and previous is not None else None), "changePercent": _float((latest - previous) / previous if latest is not None and previous else None)})
        if market_value is not None:
            distribution.append({"symbol": symbol, "name": position["name"], "value": _float(market_value)})
    holdings.sort(key=lambda item: item["marketValue"] or 0, reverse=True)
    sold_holdings = [{
        "symbol": symbol, "name": position["name"], "quantity": 0.0,
        "averageCost": None, "remainingCost": 0.0, "lastPrice": None,
        "previousClose": None, "marketValue": 0.0, "unrealized": 0.0,
        "unrealizedPercent": None, "lastDayAmount": 0.0, "lastDayPercent": None,
        "realized": _float(position["realized"]), "netDividends": _float(position["netDividends"]),
    } for symbol, position in ledger["allPositions"].items() if position["quantity"] == 0 and (position["realized"] or position["netDividends"])]
    distribution.sort(key=lambda item: item["value"], reverse=True)
    realized = sum((position["realized"] for position in ledger["allPositions"].values()), Decimal())
    net_dividends = sum((position["netDividends"] for position in ledger["allPositions"].values()), Decimal())
    portfolio_value = ledger["cash"] + sum((Decimal(str(item["value"])) for item in distribution), Decimal())
    total_return = unrealized + realized + net_dividends
    daily = ledger["dailyValues"]
    total_return_rate = twr([(item["value"], item["flow"]) for item in daily])
    external_flows = [(item["timestamp"].date(), -item.get("externalFlow", Decimal())) for item in transactions if item.get("externalFlow")]
    annualized = xirr(external_flows + ([(date.fromisoformat(daily[-1]["date"]), portfolio_value)] if daily else []))
    portfolio_curve = {item["date"]: float(item["value"]) for item in daily}
    common_days = sorted(set(portfolio_curve) & set(benchmark_closes))[-253:]
    portfolio_returns = _returns({day: portfolio_curve[day] for day in common_days})
    benchmark_returns = _returns({day: float(benchmark_closes[day]) for day in common_days})
    common_return_days = sorted(set(portfolio_returns) & set(benchmark_returns))
    risk = risk_metrics([portfolio_returns[day] for day in common_return_days], [benchmark_returns[day] for day in common_return_days])
    base_portfolio = next((value for value in portfolio_curve.values() if value), None)
    base_benchmark = next((float(value) for value in benchmark_closes.values() if value), None)
    curve_days = sorted(set(portfolio_curve) | set(benchmark_closes))
    curve = [{
        "date": day,
        "portfolio": portfolio_curve.get(day) / base_portfolio * 100 if base_portfolio and day in portfolio_curve else None,
        "benchmark": float(benchmark_closes[day]) / base_benchmark * 100 if base_benchmark and day in benchmark_closes else None,
    } for day in curve_days]
    all_dates = [item["timestamp"].date().isoformat() for item in transactions if item.get("timestamp")]
    all_dates.extend(day for series in closes.values() for day in series)
    serialized_transactions = [{
        "date": item["timestamp"].date().isoformat() if item.get("timestamp") else None,
        "symbol": item.get("symbol"), "name": item.get("name"), "kind": item["kind"],
        "action": item.get("rawAction"), "quantity": _float(item.get("quantity")),
        "price": _float(item.get("price")), "fee": _float(item.get("fee")),
        "amount": _float(item.get("cashDelta")), "balance": _float(item.get("cashBalance")),
    } for item in reversed(transactions)]
    return {
        "market": market, "currency": "USD" if market == "us" else "CNY",
        "benchmark": {"symbol": "^GSPC" if market == "us" else "sh000300", "name": "SPX" if market == "us" else "沪深300"},
        "source": source or {"mode": "directory", "label": "真实数据"},
        "asOf": max(all_dates) if all_dates else None, "incomplete": bool(errors) or any(item["lastPrice"] is None for item in holdings),
        "summary": {
            "portfolioValue": _float(portfolio_value), "cash": _float(ledger["cash"]),
            "unrealized": _float(unrealized), "lastDayAmount": _float(last_day),
            "lastDayPercent": _float(last_day / previous_value if previous_value else None),
            "realized": _float(realized + net_dividends), "tradingRealized": _float(realized),
            "netDividends": _float(net_dividends), "totalReturn": _float(total_return),
            "totalReturnRate": total_return_rate, "annualizedReturn": annualized,
        },
        "holdings": holdings, "soldHoldings": sold_holdings, "prices": prices, "transactions": serialized_transactions,
        "cashTransactions": [item for item in serialized_transactions if item["kind"] in {"deposit", "withdrawal", "interest", "cash"}],
        "dividends": [item for item in serialized_transactions if item["kind"] == "dividend"],
        "dividendCalendar": dividend_calendar, "curve": curve, "distribution": distribution,
        "risk": {**risk, "sampleDays": len(common_return_days)}, "errors": errors,
    }
