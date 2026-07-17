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


class YahooPriceProvider:
    SYMBOL_MAP = {
        "__QQQ__": "QQQ",
        "__SPY__": "SPY",
    }

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
                actions=False,
            )
            series: dict[date, Decimal] = {}
            if not frame.empty and "Close" in frame:
                for timestamp, value in frame["Close"].dropna().items():
                    price_day = timestamp.date()
                    if start <= price_day <= end:
                        series[price_day] = _decimal_close(value)
            result[symbol] = series
        return result


class AksharePriceProvider:
    INDEX_MAP = {
        "__SSE__": "sh000001",
        "__CSI300__": "sh000300",
    }

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
            elif symbol.startswith(("15", "51", "56", "58")):
                frame = ak.fund_etf_hist_em(
                    symbol=symbol,
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust="",
                )
            elif symbol.startswith("16"):
                frame = ak.fund_lof_hist_em(
                    symbol=symbol,
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust="",
                )
            else:
                frame = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
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

