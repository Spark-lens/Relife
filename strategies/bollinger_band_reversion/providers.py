from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Protocol

import requests

from .config import StrategyConfig
from .models import IndicatorSnapshot


ALPHA_BBANDS_KEY = "Technical Analysis: BBANDS"
ALPHA_TIME_SERIES = {
    "daily": ("TIME_SERIES_DAILY", "Time Series (Daily)"),
    "weekly": ("TIME_SERIES_WEEKLY", "Weekly Time Series"),
    "monthly": ("TIME_SERIES_MONTHLY", "Monthly Time Series"),
}
YFINANCE_INTERVALS = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
AKSHARE_PERIODS = {"daily": "daily", "weekly": "weekly", "monthly": "monthly"}


class MarketDataProvider(Protocol):
    name: str

    def fetch_indicator(self, symbol: str, timeframe: str) -> IndicatorSnapshot:
        ...


class AlphaVantageProvider:
    name = "alpha_vantage"

    def __init__(self, config: StrategyConfig):
        self.config = config
        alpha_config = (config.raw.get("data_sources") or {}).get("alpha_vantage") or {}
        self.api_key_env = str(alpha_config.get("api_key_env") or "ALPHA_VANTAGE_API_KEY")
        self.api_key = read_env(self.api_key_env)
        self.time_period = str(alpha_config.get("time_period") or 20)
        self.nbdevup = str(alpha_config.get("nbdevup") or 2)
        self.nbdevdn = str(alpha_config.get("nbdevdn") or 2)
        self.series_type = str(alpha_config.get("series_type") or "close")

    def fetch_indicator(self, symbol: str, timeframe: str) -> IndicatorSnapshot:
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env} is not set.")
        provider_symbol = require_provider_symbol(self.config, self.name, symbol)
        indicator_data = self._get_json(
            {
                "function": "BBANDS",
                "symbol": provider_symbol,
                "interval": timeframe,
                "time_period": self.time_period,
                "series_type": self.series_type,
                "nbdevup": self.nbdevup,
                "nbdevdn": self.nbdevdn,
                "apikey": self.api_key,
            }
        )
        time_function, series_key = ALPHA_TIME_SERIES[timeframe]
        price_data = self._get_json(
            {
                "function": time_function,
                "symbol": provider_symbol,
                "apikey": self.api_key,
            }
        )
        bbands = indicator_data.get(ALPHA_BBANDS_KEY)
        prices = price_data.get(series_key)
        if not isinstance(bbands, dict):
            raise RuntimeError(f"Alpha Vantage BBANDS response missing {ALPHA_BBANDS_KEY!r}: {indicator_data}")
        if not isinstance(prices, dict):
            raise RuntimeError(f"Alpha Vantage price response missing {series_key!r}: {price_data}")

        latest = latest_common_date(bbands, prices)
        band_row = bbands[latest.isoformat()]
        price_row = prices[latest.isoformat()]
        return IndicatorSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            bar_date=latest,
            close=Decimal(str(price_row["4. close"])),
            lower_band=Decimal(str(band_row["Real Lower Band"])),
            upper_band=Decimal(str(band_row["Real Upper Band"])),
            source=self.name,
        )

    def _get_json(self, params: dict[str, str]) -> dict[str, Any]:
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "Error Message" in data or "Note" in data or "Information" in data:
            raise RuntimeError(f"Alpha Vantage returned an error/limit response: {data}")
        return data


class YFinanceFallbackProvider:
    name = "yfinance"

    def __init__(self, config: StrategyConfig):
        self.config = config

    def fetch_indicator(self, symbol: str, timeframe: str) -> IndicatorSnapshot:
        try:
            import pandas as pd
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance fallback requires installing yfinance and pandas.") from exc

        provider_symbol = require_provider_symbol(self.config, self.name, symbol)
        frame = yf.download(
            provider_symbol,
            period="3y",
            interval=YFINANCE_INTERVALS[timeframe],
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if frame.empty:
            raise RuntimeError(f"yfinance returned no bars for {symbol}.")
        close = frame["Close"].dropna()
        if len(close) < 20:
            raise RuntimeError(f"yfinance returned fewer than 20 closes for {symbol} {timeframe}.")
        rolling = close.rolling(window=20)
        middle = rolling.mean()
        std = rolling.std(ddof=0)
        lower = (middle - 2 * std).dropna()
        upper = (middle + 2 * std).dropna()
        latest_index = lower.index[-1]
        latest_date = pd.Timestamp(latest_index).date()
        return IndicatorSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            bar_date=latest_date,
            close=Decimal(str(close.loc[latest_index])),
            lower_band=Decimal(str(lower.loc[latest_index])),
            upper_band=Decimal(str(upper.loc[latest_index])),
            source=self.name,
        )


class AkshareFallbackProvider:
    name = "akshare"

    def __init__(self, config: StrategyConfig):
        self.config = config

    def fetch_indicator(self, symbol: str, timeframe: str) -> IndicatorSnapshot:
        try:
            import akshare as ak
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("AKShare fallback requires installing akshare and pandas.") from exc

        provider_symbol = require_provider_symbol(self.config, self.name, symbol)
        frame = ak.stock_zh_a_hist(symbol=provider_symbol, period=AKSHARE_PERIODS[timeframe], adjust="")
        if frame.empty or "收盘" not in frame:
            raise RuntimeError(f"AKShare returned no usable bars for {symbol}.")
        close = pd.to_numeric(frame["收盘"], errors="coerce").dropna()
        if len(close) < 20:
            raise RuntimeError(f"AKShare returned fewer than 20 closes for {symbol} {timeframe}.")
        rolling = close.rolling(window=20)
        middle = rolling.mean()
        std = rolling.std(ddof=0)
        lower = (middle - 2 * std).dropna()
        upper = (middle + 2 * std).dropna()
        latest_date = pd.to_datetime(frame.loc[lower.index[-1], "日期"]).date()
        return IndicatorSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            bar_date=latest_date,
            close=Decimal(str(close.loc[lower.index[-1]])),
            lower_band=Decimal(str(lower.iloc[-1])),
            upper_band=Decimal(str(upper.iloc[-1])),
            source=self.name,
        )


class CompositeMarketDataProvider:
    name = "composite"

    def __init__(self, providers: list[MarketDataProvider]):
        self.providers = providers

    def fetch_indicator(self, symbol: str, timeframe: str) -> IndicatorSnapshot:
        errors: list[str] = []
        for provider in self.providers:
            try:
                return provider.fetch_indicator(symbol, timeframe)
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
        raise RuntimeError(f"No market data provider succeeded for {symbol} {timeframe}: {'; '.join(errors)}")


def build_market_data_provider(config: StrategyConfig) -> CompositeMarketDataProvider:
    provider_map: dict[str, MarketDataProvider] = {
        "alpha_vantage": AlphaVantageProvider(config),
        "yfinance": YFinanceFallbackProvider(config),
        "akshare": AkshareFallbackProvider(config),
    }
    return CompositeMarketDataProvider(
        [provider_map[name] for name in config.data_source_priority if name in provider_map]
    )


def require_provider_symbol(config: StrategyConfig, provider: str, symbol: str) -> str:
    mapped = config.provider_symbol(provider, symbol)
    if not mapped:
        raise RuntimeError(f"Missing provider_symbol_map.{provider}.{symbol}; failing closed.")
    return mapped


def read_env(name: str) -> str | None:
    import os

    value = os.environ.get(name)
    return value if value else None


def latest_common_date(first: dict[str, Any], second: dict[str, Any]) -> date:
    first_dates = {date.fromisoformat(value) for value in first}
    second_dates = {date.fromisoformat(value) for value in second}
    common = sorted(first_dates & second_dates)
    if not common:
        raise RuntimeError("Provider responses had no common bar date.")
    return common[-1]
