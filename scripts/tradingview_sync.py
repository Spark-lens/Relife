#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable

TV_HEADERS = ["Symbol", "Side", "Qty", "Fill Price", "Commission", "Closing Time"]
SCHWAB_HEADERS = [
    "Date",
    "Action",
    "Symbol",
    "Description",
    "Quantity",
    "Price",
    "Fees & Comm",
    "Amount",
]
BROKER_DEFAULT = "charles_schwab"
DEFAULT_UNKNOWN_SYMBOL_EXCHANGE = "NASDAQ"
BROKER_DIRS = {
    "charles_schwab": "charles_schwab",
    "ibkr": "ibkr",
    "yinhe": "yinhe",
}
LATEST_PATTERN = re.compile(r"^tradingview_full_latest_(\d{4}-\d{2}-\d{2})\.csv$")
SNAPSHOT_PATTERN = re.compile(r"^tradingview_full_(\d{4}-\d{2}-\d{2})\.csv$")
LEGACY_PATTERN = re.compile(r"^tradingview_.*?(\d{4}-\d{2}-\d{2})\.csv$")
GENERIC_DATE_PATTERN = re.compile(r"(20\d{2}-\d{2}-\d{2}|20\d{6})(?:-\d{6})?")
DIVIDEND_ACTIONS = {"Cash Dividend", "Non-Qualified Div", "NRA Tax Adj"}


class ChineseArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("add_help", False)
        super().__init__(*args, **kwargs)
        positionals = getattr(self, "_positionals", None)
        optionals = getattr(self, "_optionals", None)
        if positionals is not None:
            positionals.title = "位置参数"
        if optionals is not None:
            optionals.title = "可选参数"
        self.add_argument("-h", "--help", action="help", help="显示此帮助信息并退出")

    def format_help(self) -> str:
        text = super().format_help()
        text = text.replace("usage:", "用法：", 1)
        text = text.replace("options:", "可选参数：", 1)
        text = text.replace("可选参数:\n", "可选参数：\n", 1)
        return text


@dataclass(frozen=True)
class NormalizedTransaction:
    trade_date: date
    action: str
    symbol_raw: str
    description: str
    quantity: str
    price: str
    commission: str
    amount: str
    broker: str


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(description="将券商交易流水增量同步为 TradingView 可导入的 CSV 文件。")
    parser.add_argument(
        "--broker",
        default=BROKER_DEFAULT,
        choices=sorted(BROKER_DIRS),
        help="券商类型，默认使用 charles_schwab。",
    )
    parser.add_argument(
        "--template-dir",
        default="data/templates/tradingview",
        help="TradingView 模板目录，默认 data/templates/tradingview。",
    )
    parser.add_argument(
        "--template-name",
        default="tradingview_template.csv",
        help="模板文件名，默认 tradingview_template.csv。",
    )
    parser.add_argument(
        "--symbol-map-name",
        default="symbol_map.json",
        help="symbol 映射配置文件名，默认 symbol_map.json。",
    )
    parser.add_argument(
        "--unknown-symbol-exchange",
        default=DEFAULT_UNKNOWN_SYMBOL_EXCHANGE,
        help=f"自动补新标的映射时使用的 TradingView 交易所前缀，默认 {DEFAULT_UNKNOWN_SYMBOL_EXCHANGE}。",
    )
    parser.add_argument(
        "--transactions-dir",
        default="data/transactions",
        help="交易原始文件根目录，默认 data/transactions。",
    )
    parser.add_argument(
        "--transactions",
        help="交易文件路径；可传绝对路径、相对券商子目录路径，未传时自动选择该券商目录下最新 CSV。",
    )
    parser.add_argument(
        "--tradingview-dir",
        default="data/tradingview",
        help="TradingView 数据目录，默认 data/tradingview。",
    )
    parser.add_argument(
        "--output-dir",
        help="输出目录，默认与 tradingview-dir 相同。",
    )
    parser.add_argument(
        "--skip-full-maintenance",
        action="store_true",
        help="跳过全量文件自动维护；不传时默认自动维护全量文件。",
    )
    return parser.parse_args()


def ensure_headers(fieldnames: list[str] | None, expected: list[str], label: str) -> None:
    if fieldnames != expected:
        raise ValueError(f"{label} 表头不匹配：期望 {expected}，实际为 {fieldnames}")


def read_csv_rows(path: Path, expected_headers: list[str], label: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        ensure_headers(reader.fieldnames, expected_headers, label)
        return list(reader)


def write_csv_rows(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def load_symbol_overrides(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"symbol_map 文件必须是 JSON 对象：{path}")

    overrides: dict[str, str] = {}
    for raw_symbol, tv_symbol in data.items():
        if not isinstance(raw_symbol, str) or not isinstance(tv_symbol, str):
            raise ValueError(f"symbol_map 的键和值都必须是字符串：{path}")
        overrides[raw_symbol.strip()] = tv_symbol.strip()
    return overrides


def write_symbol_overrides(path: Path, overrides: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(overrides.items())), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def clean_number(value: str, *, default: str = "") -> str:
    raw = (value or "").strip()
    if not raw:
        return default
    raw = raw.replace("$", "").replace(",", "")
    if raw.startswith("(") and raw.endswith(")"):
        raw = f"-{raw[1:-1]}"
    return raw


def normalize_number(value: str, *, default: str = "") -> str:
    cleaned = clean_number(value, default=default)
    if cleaned == "":
        return ""
    try:
        dec = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"无效的数值字段：{value!r}") from exc
    return format(dec.normalize(), "f") if dec != dec.to_integral() else str(dec.quantize(Decimal("1")))


def parse_tv_datetime(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def format_tv_datetime(value: date) -> str:
    return f"{value.isoformat()} 00:00:00"


def parse_schwab_row(row: dict[str, str], broker: str) -> NormalizedTransaction:
    return NormalizedTransaction(
        trade_date=datetime.strptime(row["Date"], "%m/%d/%Y").date(),
        action=row["Action"].strip(),
        symbol_raw=row["Symbol"].strip(),
        description=row["Description"].strip(),
        quantity=clean_number(row["Quantity"]),
        price=clean_number(row["Price"]),
        commission=clean_number(row["Fees & Comm"], default="0"),
        amount=clean_number(row["Amount"]),
        broker=broker,
    )


def resolve_transactions_path(root: Path, broker: str, supplied: str | None) -> Path:
    broker_dir = root / BROKER_DIRS[broker]
    if supplied:
        supplied_path = Path(supplied)
        candidates = [supplied_path]
        if not supplied_path.is_absolute():
            candidates.append(broker_dir / supplied_path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f"未找到交易流水文件：{supplied}")

    csv_files = sorted((path for path in broker_dir.glob("*.csv")), key=transaction_sort_key, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"券商目录下未找到 CSV 文件：{broker_dir}")
    return csv_files[0].resolve()


def transaction_sort_key(path: Path) -> tuple[int, str]:
    maybe_date = extract_date_from_name(path.name)
    if maybe_date:
        return (int(maybe_date.strftime("%Y%m%d")), path.name)
    return (0, path.name)


def extract_date_from_name(name: str) -> date | None:
    match = GENERIC_DATE_PATTERN.search(name)
    if not match:
        return None
    raw = match.group(1)
    fmt = "%Y-%m-%d" if "-" in raw else "%Y%m%d"
    return datetime.strptime(raw, fmt).date()


def resolve_current_full_path(tradingview_dir: Path) -> Path:
    latest_candidates = collect_dated_files(tradingview_dir, LATEST_PATTERN)
    if latest_candidates:
        return latest_candidates[-1][1].resolve()

    snapshot_candidates = collect_dated_files(tradingview_dir, SNAPSHOT_PATTERN)
    if snapshot_candidates:
        return snapshot_candidates[-1][1].resolve()

    legacy_candidates = collect_dated_files(tradingview_dir, LEGACY_PATTERN, exclude_prefixes=("tradingview_increment_",))
    if legacy_candidates:
        return legacy_candidates[-1][1].resolve()

    raise FileNotFoundError(f"无法在 {tradingview_dir} 中找到当前 TradingView 全量文件。")


def collect_dated_files(
    directory: Path,
    pattern: re.Pattern[str],
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> list[tuple[date, Path]]:
    matches: list[tuple[date, Path]] = []
    for path in directory.glob("*.csv"):
        if any(path.name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        matches.append((datetime.strptime(match.group(1), "%Y-%m-%d").date(), path))
    matches.sort(key=lambda item: (item[0], item[1].name))
    return matches


def build_symbol_map(current_rows: list[dict[str, str]], symbol_overrides: dict[str, str]) -> dict[str, str]:
    symbol_map: dict[str, str] = {}
    for row in current_rows:
        symbol = row["Symbol"]
        bare = symbol.split(":", 1)[-1]
        symbol_map.setdefault(bare, symbol)
    symbol_map.update(symbol_overrides)
    return symbol_map


def infer_tradingview_symbol(raw_symbol: str, default_exchange: str) -> str:
    symbol = raw_symbol.strip()
    exchange = default_exchange.strip().upper()
    if not symbol:
        raise ValueError("无法为空标的自动生成 TradingView 映射。")
    if ":" in symbol or not exchange:
        return symbol
    return f"{exchange}:{symbol}"


def ensure_symbol_map_for_transactions(
    current_rows: list[dict[str, str]],
    symbol_overrides: dict[str, str],
    symbol_map_path: Path,
    transactions: Iterable[NormalizedTransaction],
    *,
    default_exchange: str,
) -> tuple[dict[str, str], dict[str, str]]:
    symbol_map = build_symbol_map(current_rows, symbol_overrides)
    additions: dict[str, str] = {}
    for tx in transactions:
        if tx.action == "Journal" or not tx.symbol_raw or tx.symbol_raw in symbol_map:
            continue
        tv_symbol = infer_tradingview_symbol(tx.symbol_raw, default_exchange)
        symbol_map[tx.symbol_raw] = tv_symbol
        additions[tx.symbol_raw] = tv_symbol

    if additions:
        write_symbol_overrides(symbol_map_path, {**symbol_overrides, **additions})

    return symbol_map, additions


def tv_row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    trade_date = row["Closing Time"].split(" ", 1)[0]
    bare_symbol = row["Symbol"].split(":", 1)[-1]
    return (
        trade_date,
        row["Side"],
        bare_symbol,
        normalize_number(row["Qty"]),
        normalize_number(row["Fill Price"]),
        normalize_number(row["Commission"]),
    )


def convert_transaction(
    tx: NormalizedTransaction,
    symbol_map: dict[str, str],
) -> tuple[dict[str, str] | None, str | None]:
    if tx.action == "Journal":
        return None, "journal"
    if tx.symbol_raw not in symbol_map:
        raise ValueError(f"未知的 TradingView 标的映射：{tx.symbol_raw}。请更新 symbol_map.json。")

    symbol = symbol_map[tx.symbol_raw]
    if tx.action in {"Buy", "Sell"}:
        if not tx.quantity or not tx.price:
            raise ValueError(f"交易记录缺少数量或价格：{tx}")
        return {
            "Symbol": symbol,
            "Side": tx.action,
            "Qty": tx.quantity,
            "Fill Price": tx.price,
            "Commission": tx.commission or "0",
            "Closing Time": format_tv_datetime(tx.trade_date),
        }, None
    if tx.action in DIVIDEND_ACTIONS:
        if not tx.amount:
            raise ValueError(f"分红记录缺少金额：{tx}")
        return {
            "Symbol": symbol,
            "Side": "Dividend",
            "Qty": tx.amount,
            "Fill Price": "",
            "Commission": "",
            "Closing Time": format_tv_datetime(tx.trade_date),
        }, None

    raise ValueError(f"暂不支持的交易动作：{tx.action}")


def previous_month_end(today: date) -> date:
    return today.replace(day=1) - timedelta(days=1)


def sort_tv_rows_desc(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: parse_tv_datetime(row["Closing Time"]), reverse=True)


def prune_old_files(directory: Path, keep_path: Path, matcher: Callable[[str], bool]) -> None:
    for path in directory.glob("*.csv"):
        if path.resolve() == keep_path.resolve():
            continue
        if matcher(path.name):
            path.unlink()


def infer_reference_date(current_full_path: Path, current_rows: list[dict[str, str]], added_rows: list[dict[str, str]]) -> date:
    file_date = extract_date_from_name(current_full_path.name)
    if file_date:
        return file_date
    if added_rows:
        return max(datetime.strptime(row["Closing Time"][:10], "%Y-%m-%d").date() for row in added_rows)
    return max(parse_tv_datetime(row["Closing Time"]).date() for row in current_rows)


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    template_dir = (root / args.template_dir).resolve()
    transactions_dir = (root / args.transactions_dir).resolve()
    tradingview_dir = (root / args.tradingview_dir).resolve()
    output_dir = (root / (args.output_dir or args.tradingview_dir)).resolve()

    template_path = (template_dir / args.template_name).resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"未找到模板文件：{template_path}")
    symbol_map_path = (template_dir / args.symbol_map_name).resolve()
    if not symbol_map_path.exists():
        raise FileNotFoundError(f"未找到 symbol_map 配置文件：{symbol_map_path}")

    transactions_path = resolve_transactions_path(transactions_dir, args.broker, args.transactions)
    current_full_path = resolve_current_full_path(tradingview_dir)

    _ = read_csv_rows(template_path, TV_HEADERS, "TradingView template")
    symbol_overrides = load_symbol_overrides(symbol_map_path)
    current_rows = read_csv_rows(current_full_path, TV_HEADERS, "Current TradingView full file")
    schwab_rows = read_csv_rows(transactions_path, SCHWAB_HEADERS, "Charles Schwab transactions")

    current_max_dt = max(parse_tv_datetime(row["Closing Time"]) for row in current_rows)
    cutoff = current_max_dt.date()
    normalized_txs = [parse_schwab_row(row, args.broker) for row in schwab_rows]
    candidate_txs = [tx for tx in normalized_txs if tx.trade_date > cutoff]
    symbol_map, added_symbol_mappings = ensure_symbol_map_for_transactions(
        current_rows,
        symbol_overrides,
        symbol_map_path,
        candidate_txs,
        default_exchange=args.unknown_symbol_exchange,
    )

    existing_counts = Counter(tv_row_key(row) for row in current_rows)
    added_rows: list[dict[str, str]] = []
    skipped_journal = 0
    skipped_existing = 0

    for tx in candidate_txs:
        converted_row, skip_reason = convert_transaction(tx, symbol_map)
        if skip_reason == "journal":
            skipped_journal += 1
            continue
        assert converted_row is not None
        key = tv_row_key(converted_row)
        if existing_counts[key] > 0:
            existing_counts[key] -= 1
            skipped_existing += 1
            continue
        added_rows.append(converted_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    merged_rows = current_rows
    month_snapshot_path: Path | None = None
    latest_full_path: Path | None = None

    if added_rows:
        start_date = min(datetime.strptime(row["Closing Time"][:10], "%Y-%m-%d").date() for row in added_rows)
        end_date = max(datetime.strptime(row["Closing Time"][:10], "%Y-%m-%d").date() for row in added_rows)
        incremental_name = f"tradingview_increment_{start_date.isoformat()}_{end_date.isoformat()}.csv"
        incremental_path = output_dir / incremental_name
        write_csv_rows(incremental_path, added_rows)
        written_files.append(incremental_path)

        if not args.skip_full_maintenance:
            merged_rows = sort_tv_rows_desc(current_rows + added_rows)
            latest_full_path = output_dir / f"tradingview_full_latest_{end_date.isoformat()}.csv"
            write_csv_rows(latest_full_path, merged_rows)
            written_files.append(latest_full_path)

            month_end = previous_month_end(infer_reference_date(current_full_path, current_rows, added_rows))
            month_snapshot_path = output_dir / f"tradingview_full_{month_end.isoformat()}.csv"
            if not month_snapshot_path.exists():
                month_rows = [
                    row
                    for row in merged_rows
                    if datetime.strptime(row["Closing Time"][:10], "%Y-%m-%d").date() <= month_end
                ]
                if month_rows:
                    write_csv_rows(month_snapshot_path, month_rows)
                    written_files.append(month_snapshot_path)

            prune_old_files(output_dir, latest_full_path, lambda name: bool(LATEST_PATTERN.match(name)))
            if month_snapshot_path.exists():
                prune_old_files(output_dir, month_snapshot_path, lambda name: bool(SNAPSHOT_PATTERN.match(name)))

    print("TradingView 同步摘要")
    print(f"- 券商：{args.broker}")
    print(f"- 模板文件：{template_path}")
    print(f"- Symbol 映射：{symbol_map_path}")
    print(f"- 交易流水：{transactions_path}")
    print(f"- 当前全量文件：{current_full_path}")
    print(f"- 截止日期：{cutoff.isoformat()}")
    print(f"- 原始记录数：{len(normalized_txs)}")
    print(f"- 截止日期后的候选记录数：{len(candidate_txs)}")
    print(f"- 新增记录数：{len(added_rows)}")
    print(f"- 跳过的 Journal 记录数：{skipped_journal}")
    print(f"- 跳过的已存在记录数：{skipped_existing}")
    if added_symbol_mappings:
        print("- 自动补充的 Symbol 映射：")
        for raw_symbol, tv_symbol in sorted(added_symbol_mappings.items()):
            print(f"  - {raw_symbol}: {tv_symbol}")
    else:
        print("- 自动补充的 Symbol 映射：无")
    if written_files:
        print("- 已写入文件：")
        for path in written_files:
            print(f"  - {path}")
    else:
        print("- 已写入文件：无")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
