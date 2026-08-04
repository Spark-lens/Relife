from __future__ import annotations

import csv
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

US_HEADERS = ["Symbol", "Side", "Qty", "Fill Price", "Commission", "Closing Time"]
CN_HEADERS = [
    "日期", "成交日期", "证券代码", "证券名称", "操作", "资金余额", "成交数量",
    "成交均价", "备注", "成交金额", "发生金额", "手续费", "印花税", "其他杂费",
    "本次金额", "合同编号", "成交编号", "交易市场", "货币单位", "委托日期",
    "证券中文全称", "股份余额",
]
CN_KINDS = {
    "证券买入": "buy", "证券卖出": "sell", "银行转证券": "deposit",
    "证券转银行": "withdrawal", "红利入账": "dividend", "利息归本": "interest",
}


def _number(value: str, field: str, line: int) -> Decimal:
    raw = (value or "").strip().replace(",", "")
    if not raw:
        return Decimal()
    try:
        return Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"第 {line} 行字段 {field} 不是有效数值：{value!r}") from exc


def _read(path: Path, headers: list[str]) -> list[dict[str, str]]:
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"无法读取文件：{path}：{exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != headers:
            raise ValueError(f"表头不匹配：期望 {headers}，实际为 {reader.fieldnames}")
        return list(reader)


def parse_source(path: Path, market: str) -> list[dict]:
    if market == "us":
        rows = _read(path, US_HEADERS)
        result = []
        for index, row in enumerate(rows, 2):
            side = row["Side"].strip()
            if side not in {"Buy", "Sell", "Dividend", "Deposit", "Withdraw", "Withdrawal"}:
                raise ValueError(f"第 {index} 行字段 Side 不支持：{side!r}")
            try:
                timestamp = datetime.strptime(row["Closing Time"].strip(), "%Y-%m-%d %H:%M:%S")
            except ValueError as exc:
                raise ValueError(f"第 {index} 行字段 Closing Time 格式错误：{row['Closing Time']!r}") from exc
            quantity = _number(row["Qty"], "Qty", index)
            price = _number(row["Fill Price"], "Fill Price", index)
            fee = _number(row["Commission"], "Commission", index)
            kind = {"Buy": "buy", "Sell": "sell", "Dividend": "dividend", "Deposit": "deposit", "Withdraw": "withdrawal", "Withdrawal": "withdrawal"}[side]
            cash_delta = {
                "buy": -(quantity * price + fee), "sell": quantity * price - fee,
                "dividend": quantity, "deposit": quantity, "withdrawal": -abs(quantity),
            }[kind]
            symbol = row["Symbol"].strip().split(":")[-1]
            result.append({
                "market": "us", "timestamp": timestamp, "kind": kind, "symbol": symbol,
                "name": symbol, "quantity": quantity, "price": price, "fee": fee,
                "cashDelta": cash_delta,
                "externalFlow": cash_delta if kind in {"deposit", "withdrawal"} else Decimal(),
                "cashBalance": None, "rawAction": side, "line": index,
            })
        return result
    if market != "cn":
        raise ValueError(f"不支持的市场：{market}")
    rows = _read(path, CN_HEADERS)
    result = []
    for index, row in enumerate(rows, 2):
        raw_date = row["成交日期"].strip() or row["日期"].strip()
        try:
            timestamp = datetime.strptime(raw_date, "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"第 {index} 行字段 成交日期 格式错误：{raw_date!r}") from exc
        quantity = _number(row["成交数量"], "成交数量", index)
        price = _number(row["成交均价"], "成交均价", index)
        fee = sum((_number(row[name], name, index) for name in ("手续费", "印花税", "其他杂费")), Decimal())
        cash_delta = _number(row["发生金额"], "发生金额", index)
        kind = CN_KINDS.get(row["操作"].strip(), "cash")
        result.append({
            "market": "cn", "timestamp": timestamp, "kind": kind,
            "symbol": row["证券代码"].strip(),
            "name": row["证券名称"].strip() or row["证券中文全称"].strip() or row["证券代码"].strip() or row["操作"].strip(),
            "quantity": quantity, "price": price, "fee": fee, "cashDelta": cash_delta,
            "externalFlow": cash_delta if kind in {"deposit", "withdrawal"} else Decimal(),
            "cashBalance": _number(row["资金余额"], "资金余额", index) if row["资金余额"].strip() else None,
            "rawAction": row["操作"].strip(), "line": index,
        })
    return result


def validate_source(path: Path, market: str) -> dict:
    transactions = parse_source(path, market)
    dates = [item["timestamp"].date().isoformat() for item in transactions]
    return {
        "market": market,
        "format": "tradingview" if market == "us" else "银河交割单",
        "recordCount": len(transactions),
        "dateRange": [min(dates), max(dates)] if dates else [None, None],
        "file": str(path),
    }


def discover_latest(directory: Path, market: str) -> Path:
    expression = re.compile(
        r"^tradingview_full_latest_(\d{4}-\d{2}-\d{2})\.csv$"
        if market == "us" else r"^交割单_(\d{4}-\d{2}-\d{2})\.csv$"
    )
    candidates = [(match.group(1), path) for path in directory.iterdir() if path.is_file() and (match := expression.match(path.name))]
    if not candidates:
        raise FileNotFoundError(f"{directory} 下没有 {market} 标准命名 CSV")
    return max(candidates)[1]
