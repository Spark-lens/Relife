from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import Transaction, TransactionKind


TRADINGVIEW_HEADERS = [
    "Symbol",
    "Side",
    "Qty",
    "Fill Price",
    "Commission",
    "Closing Time",
]

YINHE_HEADERS = [
    "日期",
    "成交日期",
    "证券代码",
    "证券名称",
    "操作",
    "资金余额",
    "成交数量",
    "成交均价",
    "备注",
    "成交金额",
    "发生金额",
    "手续费",
    "印花税",
    "其他杂费",
    "本次金额",
    "合同编号",
    "成交编号",
    "交易市场",
    "货币单位",
    "委托日期",
    "证券中文全称",
    "股份余额",
]

YINHE_KIND_MAP: dict[str, TransactionKind] = {
    "证券买入": "buy",
    "证券卖出": "sell",
    "银行转证券": "deposit",
    "证券转银行": "withdrawal",
    "利息归本": "interest",
    "红利入账": "dividend",
}


def _decimal(value: str, *, field: str, row_number: int) -> Decimal:
    raw = (value or "").strip().replace(",", "")
    if not raw:
        return Decimal("0")
    try:
        return Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"第 {row_number} 行字段 {field} 不是有效数值：{value!r}") from exc


def _fingerprint(row: dict[str, str]) -> str:
    body = "\x1f".join(f"{key}={row.get(key, '')}" for key in sorted(row))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _read_rows(path: Path, expected_headers: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_headers:
            raise ValueError(
                f"{path} 表头不匹配：期望 {expected_headers}，实际为 {reader.fieldnames}"
            )
        return list(reader)


def _bare_symbol(value: str) -> str:
    symbol = value.strip()
    return symbol.split(":", 1)[-1] if ":" in symbol else symbol


def parse_tradingview(path: Path) -> list[Transaction]:
    rows = _read_rows(path, TRADINGVIEW_HEADERS)
    parsed: list[Transaction] = []
    for source_index, row in enumerate(rows):
        row_number = source_index + 2
        side = row["Side"].strip()
        symbol = _bare_symbol(row["Symbol"])
        quantity = _decimal(row["Qty"], field="Qty", row_number=row_number)
        price = _decimal(row["Fill Price"], field="Fill Price", row_number=row_number)
        fee = _decimal(row["Commission"], field="Commission", row_number=row_number)
        timestamp = datetime.strptime(row["Closing Time"], "%Y-%m-%d %H:%M:%S")

        if side == "Buy":
            kind: TransactionKind = "buy"
            cash_delta = -(quantity * price + fee)
            external_cash_flow = Decimal("0")
        elif side == "Sell":
            kind = "sell"
            cash_delta = quantity * price - fee
            external_cash_flow = Decimal("0")
        elif side == "Dividend":
            kind = "dividend"
            cash_delta = quantity
            external_cash_flow = Decimal("0")
        elif side == "Deposit":
            kind = "deposit"
            cash_delta = quantity
            external_cash_flow = quantity
        elif side in {"Withdraw", "Withdrawal"}:
            kind = "withdrawal"
            cash_delta = -abs(quantity)
            external_cash_flow = -abs(quantity)
        else:
            raise ValueError(f"{path} 第 {row_number} 行不支持的 Side：{side!r}")

        parsed.append(
            Transaction(
                market="us",
                timestamp=timestamp,
                source_index=source_index,
                symbol=symbol,
                name=symbol,
                kind=kind,
                quantity=quantity,
                price=price,
                fee=fee,
                cash_delta=cash_delta,
                external_cash_flow=external_cash_flow,
                source_id=f"us:{_fingerprint(row)}",
                raw_action=side,
                cash_balance=None,
            )
        )
    return parsed


def parse_yinhe(path: Path) -> list[Transaction]:
    rows = _read_rows(path, YINHE_HEADERS)
    parsed: list[Transaction] = []
    seen: set[str] = set()

    for source_index, row in enumerate(rows):
        row_number = source_index + 2
        action = row["操作"].strip()
        contract_id = row["合同编号"].strip()
        trade_id = row["成交编号"].strip()
        row_fingerprint = _fingerprint(row)
        if trade_id and trade_id != "0":
            source_id = f"cn:{trade_id}:{contract_id}:{row_fingerprint}"
        elif contract_id:
            source_id = f"cn:contract:{contract_id}:{row_fingerprint}"
        else:
            source_id = f"cn:row:{row_fingerprint}"
        if source_id in seen:
            continue
        seen.add(source_id)

        kind = YINHE_KIND_MAP.get(action, "cash")
        quantity = _decimal(
            row["成交数量"], field="成交数量", row_number=row_number
        )
        price = _decimal(row["成交均价"], field="成交均价", row_number=row_number)
        fee = sum(
            (
                _decimal(row["手续费"], field="手续费", row_number=row_number),
                _decimal(row["印花税"], field="印花税", row_number=row_number),
                _decimal(row["其他杂费"], field="其他杂费", row_number=row_number),
            ),
            Decimal("0"),
        )
        cash_delta = _decimal(
            row["发生金额"], field="发生金额", row_number=row_number
        )
        external_cash_flow = (
            cash_delta if kind in {"deposit", "withdrawal"} else Decimal("0")
        )
        raw_date = row["成交日期"].strip() or row["日期"].strip()
        timestamp = datetime.strptime(raw_date, "%Y%m%d")
        symbol = row["证券代码"].strip()
        name = (
            row["证券名称"].strip()
            or row["证券中文全称"].strip()
            or symbol
            or action
        )

        parsed.append(
            Transaction(
                market="cn",
                timestamp=timestamp,
                source_index=source_index,
                symbol=symbol,
                name=name,
                kind=kind,
                quantity=quantity,
                price=price,
                fee=fee,
                cash_delta=cash_delta,
                external_cash_flow=external_cash_flow,
                source_id=source_id,
                raw_action=action,
                cash_balance=_decimal(
                    row["资金余额"], field="资金余额", row_number=row_number
                ),
            )
        )
    return parsed


def latest_source(pattern: str, root: Path) -> Path:
    candidates = sorted(root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"{root} 下未找到匹配文件：{pattern}")
    return candidates[-1]
