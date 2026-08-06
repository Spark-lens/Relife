from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Mapping

from .market_data import PriceProvider


TIMEFRAMES = {"daily", "weekly", "monthly"}


def validate_watchlist(data: dict[str, Any]) -> dict[str, Any]:
    groups = data.get("groups")
    if not isinstance(groups, list):
        raise ValueError("观察配置缺少 groups 数组")

    group_ids: set[str] = set()
    symbols: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("观察分组必须是对象")
        group_id = _text(group.get("id"), "分组 id")
        _text(group.get("label"), "分组名称")
        if group_id in group_ids:
            raise ValueError(f"重复分组 id {group_id}")
        group_ids.add(group_id)
        items = group.get("items")
        if not isinstance(items, list):
            raise ValueError(f"分组 {group_id} 缺少 items 数组")
        for item in items:
            _validate_item(item, symbols)
    return data


def _validate_item(item: Any, symbols: set[str]) -> None:
    if not isinstance(item, dict):
        raise ValueError("观察标的必须是对象")
    market = _text(item.get("market"), "标的 market").lower()
    symbol = _text(item.get("symbol"), "标的 symbol").upper()
    _text(item.get("name"), "标的 name")
    if market not in {"us", "cn"}:
        raise ValueError(f"非法市场 {market}")
    key = f"{market}:{symbol}"
    if key in symbols:
        raise ValueError(f"重复标的 {key}")
    symbols.add(key)

    rule = item.get("bollinger")
    if not isinstance(rule, dict):
        raise ValueError(f"{key} 缺少 bollinger 配置")
    if not isinstance(rule.get("enabled"), bool):
        raise ValueError(f"{key} 的 enabled 必须是布尔值")
    timeframes = rule.get("timeframes")
    if not isinstance(timeframes, list) or not timeframes:
        raise ValueError(f"{key} 至少需要一个观察周期")
    unknown = set(timeframes) - TIMEFRAMES
    if unknown:
        raise ValueError(f"{key} 包含非法周期 {sorted(unknown)}")
    window = rule.get("window")
    if not isinstance(window, int) or isinstance(window, bool) or window < 2:
        raise ValueError(f"{key} 的 window 必须是不小于 2 的整数")
    try:
        deviations = Decimal(str(rule.get("standardDeviations")))
    except Exception as exc:
        raise ValueError(f"{key} 的 standardDeviations 必须是正数") from exc
    if not deviations.is_finite() or deviations <= 0:
        raise ValueError(f"{key} 的 standardDeviations 必须是正数")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}不能为空")
    return value.strip()


def aggregate_closes(
    closes: Mapping[date, Decimal],
    timeframe: str,
) -> list[tuple[str, date, Decimal]]:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"非法周期 {timeframe}")
    buckets: dict[str, tuple[date, Decimal]] = {}
    for day, close in sorted(closes.items()):
        if timeframe == "daily":
            key = day.isoformat()
        elif timeframe == "weekly":
            iso_year, iso_week, _ = day.isocalendar()
            key = f"{iso_year}-W{iso_week:02d}"
        else:
            key = day.strftime("%Y-%m")
        buckets[key] = (day, close)
    return [(key, day, close) for key, (day, close) in buckets.items()]


def evaluate_bollinger(
    buckets: list[tuple[str, date, Decimal]],
    *,
    window: int,
    standard_deviations: Decimal,
) -> dict[str, Any]:
    if len(buckets) < window:
        raise ValueError(f"只有 {len(buckets)} 个周期，少于所需 {window} 个")
    sample = buckets[-window:]
    values = [close for _, _, close in sample]
    mean = sum(values) / Decimal(window)
    variance = sum((value - mean) ** 2 for value in values) / Decimal(window)
    lower = mean - standard_deviations * variance.sqrt()
    period_key, bar_date, close = sample[-1]
    return {
        "periodKey": period_key,
        "barDate": bar_date,
        "close": close,
        "lowerBand": lower,
        "triggered": close < lower,
    }


def check_watchlist(
    config: dict[str, Any],
    *,
    providers: Mapping[str, PriceProvider],
    checked_at: datetime,
) -> dict[str, Any]:
    validate_watchlist(config)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    end = checked_at.date()
    start = end - timedelta(days=3 * 366)

    for group in config["groups"]:
        for item in group["items"]:
            rule = item["bollinger"]
            if not rule["enabled"]:
                continue
            market = item["market"].lower()
            symbol = item["symbol"].upper()
            try:
                provider = providers[market]
                closes = provider.history([symbol], start, end).get(symbol, {})
                if not closes:
                    raise ValueError("未返回可用收盘价")
            except Exception as exc:
                errors.append(_error(group, item, None, exc))
                continue

            for timeframe in rule["timeframes"]:
                try:
                    evaluated = evaluate_bollinger(
                        aggregate_closes(closes, timeframe),
                        window=rule["window"],
                        standard_deviations=Decimal(str(rule["standardDeviations"])),
                    )
                    results.append(
                        {
                            "groupId": group["id"],
                            "groupLabel": group["label"],
                            "market": market,
                            "symbol": symbol,
                            "name": item["name"],
                            "timeframe": timeframe,
                            "window": rule["window"],
                            "standardDeviations": rule["standardDeviations"],
                            **evaluated,
                        }
                    )
                except Exception as exc:
                    errors.append(_error(group, item, timeframe, exc))

    alerts = [row for row in results if row["triggered"]]
    return {
        "checkedAt": checked_at.isoformat(),
        "results": [_json_row(row) for row in results],
        "alerts": [_json_row(row) for row in alerts],
        "errors": errors,
    }


def _error(group: dict[str, Any], item: dict[str, Any], timeframe: str | None, exc: Exception) -> dict[str, Any]:
    return {
        "groupId": group["id"],
        "market": item["market"].lower(),
        "symbol": item["symbol"].upper(),
        "timeframe": timeframe,
        "message": str(exc),
    }


def _json_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.isoformat() if isinstance(value, date) else float(value) if isinstance(value, Decimal) else value
        for key, value in row.items()
    }
