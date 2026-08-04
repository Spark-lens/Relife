from __future__ import annotations

from collections import defaultdict, deque
from decimal import Decimal


def _replay_core(market: str, transactions: list[dict]) -> dict:
    states: dict[str, dict] = {}
    lots: dict[str, deque[list[Decimal]]] = defaultdict(deque)
    cash = Decimal()
    dividends = []
    for transaction in transactions:
        cash += transaction.get("cashDelta", Decimal())
        if market == "cn" and transaction.get("cashBalance") is not None:
            cash = transaction["cashBalance"]
        symbol = transaction.get("symbol", "")
        kind = transaction["kind"]
        if kind == "dividend":
            dividends.append(transaction)
        if not symbol or kind not in {"buy", "sell", "dividend"}:
            continue
        state = states.setdefault(symbol, {
            "symbol": symbol, "name": transaction.get("name") or symbol,
            "quantity": Decimal(), "remainingCost": Decimal(), "realized": Decimal(),
            "netDividends": Decimal(), "cumulativeBuyCost": Decimal(),
        })
        if kind == "dividend":
            state["netDividends"] += transaction.get("cashDelta", Decimal())
            continue
        quantity = transaction["quantity"]
        if kind == "buy":
            cost = quantity * transaction["price"] + transaction.get("fee", Decimal())
            state["quantity"] += quantity
            state["remainingCost"] += cost
            state["cumulativeBuyCost"] += cost
            if market == "us":
                lots[symbol].append([quantity, cost / quantity if quantity else Decimal()])
            continue
        if quantity > state["quantity"]:
            raise ValueError(f"{symbol} 卖出数量 {quantity} 超过持仓 {state['quantity']}")
        if market == "us":
            unsold = quantity
            sold_cost = Decimal()
            while unsold:
                lot_quantity, unit_cost = lots[symbol][0]
                used = min(unsold, lot_quantity)
                sold_cost += used * unit_cost
                unsold -= used
                lot_quantity -= used
                if lot_quantity == 0:
                    lots[symbol].popleft()
                else:
                    lots[symbol][0][0] = lot_quantity
        else:
            sold_cost = state["remainingCost"] / state["quantity"] * quantity
        proceeds = quantity * transaction["price"] - transaction.get("fee", Decimal())
        state["realized"] += proceeds - sold_cost
        state["quantity"] -= quantity
        state["remainingCost"] -= sold_cost
        if state["quantity"] == 0:
            state["remainingCost"] = Decimal()
    return {
        "cash": cash,
        "positions": {key: value for key, value in states.items() if value["quantity"] != 0},
        "allPositions": states,
        "dividends": dividends,
    }


def replay(market: str, transactions: list[dict], closes: dict[str, dict[str, Decimal]] | None = None) -> dict:
    ordered = sorted(
        transactions,
        key=lambda item: (item.get("timestamp"), -item.get("line", 0)) if item.get("timestamp") else (0, 0),
    )
    result = _replay_core(market, ordered)
    if not closes or not any(item.get("timestamp") for item in ordered):
        result["dailyValues"] = []
        return result
    days = {item["timestamp"].date().isoformat() for item in ordered if item.get("timestamp")}
    days.update(day for series in closes.values() for day in series)
    daily_values = []
    # ponytail: 日级回放为 O(日数×交易数)；交易达到数万条时再改为单遍状态机。
    for day in sorted(days):
        subset = [item for item in ordered if not item.get("timestamp") or item["timestamp"].date().isoformat() <= day]
        state = _replay_core(market, subset)
        securities = Decimal()
        for symbol, position in state["positions"].items():
            eligible = [(price_day, price) for price_day, price in closes.get(symbol, {}).items() if price_day <= day]
            if eligible:
                securities += position["quantity"] * max(eligible)[1]
        flow = sum((item.get("externalFlow", Decimal()) for item in ordered if item.get("timestamp") and item["timestamp"].date().isoformat() == day), Decimal())
        daily_values.append({"date": day, "value": state["cash"] + securities, "cash": state["cash"], "flow": flow})
    result["dailyValues"] = daily_values
    return result
