#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Iterable

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
BROKER_DIRS = {
    "charles_schwab": "charles_schwab",
    "ibkr": "ibkr",
    "yinhe": "yinhe",
}
SYMBOL_OVERRIDES = {
    "QQQM": "NASDAQ:QQQM",
    "GGLL": "NASDAQ:GGLL",
    "SSPC": "CBOE:SSPC",
}
LATEST_PATTERN = re.compile(r"^tradingview_full_latest_(\d{4}-\d{2}-\d{2})\.csv$")
SNAPSHOT_PATTERN = re.compile(r"^tradingview_full_(\d{4}-\d{2}-\d{2})\.csv$")
LEGACY_PATTERN = re.compile(r"^tradingview_.*?(\d{4}-\d{2}-\d{2})\.csv$")
GENERIC_DATE_PATTERN = re.compile(r"(20\d{2}-\d{2}-\d{2}|20\d{6})(?:-\d{6})?")
DIVIDEND_ACTIONS = {"Cash Dividend", "Non-Qualified Div", "NRA Tax Adj"}


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
    parser = argparse.ArgumentParser(description="将券商交易流水增量同步为 TradingView 可导入的 CSV 文件。")
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
        raise ValueError(f"{label} headers mismatch: expected {expected}, got {fieldnames}")


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
        raise ValueError(f"Invalid numeric value: {value!r}") from exc
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
        raise FileNotFoundError(f"Transactions file not found: {supplied}")

    csv_files = sorted((path for path in broker_dir.glob("*.csv")), key=transaction_sort_key, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in broker directory: {broker_dir}")
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

    raise FileNotFoundError(f"Unable to find current full TradingView file in {tradingview_dir}.")


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


def build_symbol_map(current_rows: list[dict[str, str]]) -> dict[str, str]:
    symbol_map: dict[str, str] = {}
    for row in current_rows:
        symbol = row["Symbol"]
        bare = symbol.split(":", 1)[-1]
        symbol_map.setdefault(bare, symbol)
    symbol_map.update({k: v for k, v in SYMBOL_OVERRIDES.items() if k not in symbol_map})
    return symbol_map


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
        raise ValueError(f"Unknown TradingView symbol mapping for {tx.symbol_raw}. Update SYMBOL_OVERRIDES.")

    symbol = symbol_map[tx.symbol_raw]
    if tx.action in {"Buy", "Sell"}:
        if not tx.quantity or not tx.price:
            raise ValueError(f"Missing quantity or price for trade: {tx}")
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
            raise ValueError(f"Missing amount for dividend action: {tx}")
        return {
            "Symbol": symbol,
            "Side": "Dividend",
            "Qty": tx.amount,
            "Fill Price": "",
            "Commission": "",
            "Closing Time": format_tv_datetime(tx.trade_date),
        }, None

    raise ValueError(f"Unsupported transaction action: {tx.action}")


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
        raise FileNotFoundError(f"Template file not found: {template_path}")

    transactions_path = resolve_transactions_path(transactions_dir, args.broker, args.transactions)
    current_full_path = resolve_current_full_path(tradingview_dir)

    _ = read_csv_rows(template_path, TV_HEADERS, "TradingView template")
    current_rows = read_csv_rows(current_full_path, TV_HEADERS, "Current TradingView full file")
    schwab_rows = read_csv_rows(transactions_path, SCHWAB_HEADERS, "Charles Schwab transactions")

    current_max_dt = max(parse_tv_datetime(row["Closing Time"]) for row in current_rows)
    cutoff = current_max_dt.date()
    symbol_map = build_symbol_map(current_rows)

    normalized_txs = [parse_schwab_row(row, args.broker) for row in schwab_rows]
    candidate_txs = [tx for tx in normalized_txs if tx.trade_date > cutoff]

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

    print("TradingView sync summary")
    print(f"- Broker: {args.broker}")
    print(f"- Template: {template_path}")
    print(f"- Transactions: {transactions_path}")
    print(f"- Current full: {current_full_path}")
    print(f"- Cutoff date: {cutoff.isoformat()}")
    print(f"- Source rows: {len(normalized_txs)}")
    print(f"- Candidate rows after cutoff: {len(candidate_txs)}")
    print(f"- Added rows: {len(added_rows)}")
    print(f"- Skipped journal rows: {skipped_journal}")
    print(f"- Skipped already-existing rows: {skipped_existing}")
    if written_files:
        print("- Written files:")
        for path in written_files:
            print(f"  - {path}")
    else:
        print("- Written files: none")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
