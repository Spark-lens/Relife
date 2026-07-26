from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Protocol

from .ledger import PriceMatrix


class PriceProvider(Protocol):
    def history(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> PriceMatrix: ...


def _decimal_close(value: object) -> Decimal:
    return Decimal(str(value))


def _repair_price_discontinuities(
    series: dict[date, Decimal],
) -> dict[date, Decimal]:
    repaired = dict(series)
    candidates = [
        Decimal("5"),
        Decimal("10"),
        Decimal("20"),
        Decimal("25"),
        Decimal("50"),
        Decimal("100"),
        Decimal("200"),
    ]
    ordered_days = sorted(repaired)
    for index in range(1, len(ordered_days)):
        previous_day = ordered_days[index - 1]
        current_day = ordered_days[index]
        previous = repaired[previous_day]
        current = repaired[current_day]
        if previous <= 0 or current <= 0:
            continue
        ratio = max(previous, current) / min(previous, current)
        candidate = min(
            candidates,
            key=lambda value: abs(float(ratio / value) - 1),
        )
        if abs(float(ratio / candidate) - 1) > 0.35:
            continue
        scale = (
            Decimal("1") / candidate
            if previous > current
            else candidate
        )
        for earlier_day in ordered_days[:index]:
            repaired[earlier_day] *= scale
    return repaired


class YahooPriceProvider:
    SYMBOL_MAP = {
        "BRKB": "BRK-B",
        "__QQQ__": "QQQ",
        "__SPY__": "SPY",
    }

    def __init__(self) -> None:
        self.splits: dict[str, dict[date, Decimal]] = {}

    def history(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> PriceMatrix:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "缺少 yfinance，请在项目 Python 环境中安装 requirements.txt"
            ) from exc

        result: dict[str, dict[date, Decimal]] = {}
        for symbol in symbols:
            remote_symbol = self.SYMBOL_MAP.get(symbol, symbol)
            frame = yf.Ticker(remote_symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=True,
            )
            series: dict[date, Decimal] = {}
            if not frame.empty and "Close" in frame:
                for timestamp, value in frame["Close"].dropna().items():
                    price_day = timestamp.date()
                    if start <= price_day <= end:
                        series[price_day] = _decimal_close(value)
            result[symbol] = _repair_price_discontinuities(series)
            split_series: dict[date, Decimal] = {}
            if not frame.empty and "Stock Splits" in frame:
                for timestamp, value in frame["Stock Splits"].dropna().items():
                    ratio = _decimal_close(value)
                    split_day = timestamp.date()
                    if ratio > 0 and ratio != 1 and start <= split_day <= end:
                        split_series[split_day] = ratio
            self.splits[symbol] = split_series
        return result


class AksharePriceProvider:
    INDEX_MAP = {
        "__SSE__": "sh000001",
        "__CSI300__": "sh000300",
    }

    @staticmethod
    def _exchange_symbol(symbol: str) -> str:
        exchange = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
        return f"{exchange}{symbol}"

    def history(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> PriceMatrix:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError(
                "缺少 akshare，请在项目 Python 环境中安装 requirements.txt"
            ) from exc

        result: dict[str, dict[date, Decimal]] = {}
        for symbol in symbols:
            if symbol in self.INDEX_MAP:
                frame = ak.stock_zh_index_daily(symbol=self.INDEX_MAP[symbol])
            elif symbol.startswith(("15", "16", "51", "56", "58")):
                frame = ak.fund_etf_hist_sina(
                    symbol=self._exchange_symbol(symbol),
                )
            else:
                frame = ak.stock_zh_a_daily(
                    symbol=self._exchange_symbol(symbol),
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust="",
                )

            date_column = "日期" if "日期" in frame else "date"
            close_column = "收盘" if "收盘" in frame else "close"
            series: dict[date, Decimal] = {}
            if date_column in frame and close_column in frame:
                for raw_day, raw_close in zip(
                    frame[date_column],
                    frame[close_column],
                ):
                    price_day = date.fromisoformat(str(raw_day)[:10])
                    if start <= price_day <= end:
                        series[price_day] = _decimal_close(raw_close)
            result[symbol] = series
        return result
