from __future__ import annotations

import math
from datetime import date
from decimal import Decimal
from statistics import mean, stdev


def twr(values_and_flows: list[tuple[Decimal, Decimal]]) -> float | None:
    if len(values_and_flows) < 2:
        return None
    growth = 1.0
    previous = float(values_and_flows[0][0])
    for value, flow in values_and_flows[1:]:
        if previous == 0:
            continue
        growth *= (float(value) - float(flow)) / previous
        previous = float(value)
    return growth - 1.0


def xirr(flows: list[tuple[date, Decimal]]) -> float | None:
    if not flows or not any(value < 0 for _, value in flows) or not any(value > 0 for _, value in flows):
        return None
    origin = min(day for day, _ in flows)

    def present_value(rate: float) -> float:
        return sum(float(value) / ((1 + rate) ** ((day - origin).days / 365.0)) for day, value in flows)

    low, high = -0.999999, 1000.0
    if present_value(low) * present_value(high) > 0:
        return None
    for _ in range(160):
        middle = (low + high) / 2
        if present_value(low) * present_value(middle) <= 0:
            high = middle
        else:
            low = middle
    return (low + high) / 2


def risk_metrics(portfolio_returns: list[float], benchmark_returns: list[float]) -> dict[str, float | None]:
    count = min(len(portfolio_returns), len(benchmark_returns))
    portfolio = portfolio_returns[-count:]
    benchmark = benchmark_returns[-count:]
    if count < 2:
        return {"beta": None, "sharpe": None, "sortino": None}
    benchmark_mean = mean(benchmark)
    portfolio_mean = mean(portfolio)
    variance = sum((value - benchmark_mean) ** 2 for value in benchmark) / (count - 1)
    covariance = sum((portfolio[index] - portfolio_mean) * (benchmark[index] - benchmark_mean) for index in range(count)) / (count - 1)
    volatility = stdev(portfolio)
    downside = math.sqrt(sum(min(value, 0) ** 2 for value in portfolio) / count)
    return {
        "beta": covariance / variance if variance else None,
        "sharpe": portfolio_mean / volatility * math.sqrt(252) if volatility else None,
        "sortino": portfolio_mean / downside * math.sqrt(252) if downside else None,
    }
