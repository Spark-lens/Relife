from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal


def _decimal(value) -> Decimal:
    return Decimal(str(value))


def load_market_data(market: str, symbols: list[str]) -> tuple[dict, dict, list, list]:
    closes: dict[str, dict[str, Decimal]] = {}
    calendar: list[dict] = []
    errors: list[dict] = []
    if market == "us":
        import yfinance as yf

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                frame = ticker.history(period="2y", auto_adjust=False, actions=True)
                if frame.empty:
                    raise ValueError("无历史行情")
                closes[symbol] = {index.date().isoformat(): _decimal(row["Close"]) for index, row in frame.iterrows() if row.get("Close") == row.get("Close")}
                for index, row in frame.iterrows():
                    if row.get("Dividends", 0):
                        calendar.append({"date": index.date().isoformat(), "symbol": symbol, "amount": float(row["Dividends"]), "status": "已派息"})
                try:
                    upcoming = ticker.calendar or {}
                    ex_day = upcoming.get("Ex-Dividend Date") if isinstance(upcoming, dict) else None
                    if ex_day:
                        calendar.append({"date": str(ex_day)[:10], "symbol": symbol, "amount": None, "status": "预计除息"})
                except Exception:
                    pass
            except Exception as exc:
                errors.append({"symbol": symbol, "message": str(exc)})
        try:
            frame = yf.Ticker("^GSPC").history(period="2y", auto_adjust=False)
            benchmark = {index.date().isoformat(): _decimal(row["Close"]) for index, row in frame.iterrows() if row.get("Close") == row.get("Close")}
        except Exception as exc:
            benchmark = {}
            errors.append({"symbol": "^GSPC", "message": str(exc)})
        return closes, benchmark, calendar, errors

    import akshare as ak

    start = (date.today() - timedelta(days=760)).strftime("%Y%m%d")
    end = date.today().strftime("%Y%m%d")
    for symbol in symbols:
        try:
            frame = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
            if frame.empty:
                raise ValueError("无历史行情")
            closes[symbol] = {str(row["日期"])[:10]: _decimal(row["收盘"]) for _, row in frame.iterrows()}
            try:
                actions = ak.stock_history_dividend_detail(symbol=symbol, indicator="分红")
                cutoff = date.today() - timedelta(days=180)
                for _, row in actions.iterrows():
                    action_day = row.get("除权除息日") or row.get("公告日期")
                    if action_day and action_day >= cutoff:
                        calendar.append({"date": str(action_day)[:10], "symbol": symbol, "amount": None if row.get("派息") != row.get("派息") else float(row["派息"]), "status": "公司行动"})
            except Exception:
                pass
        except Exception as exc:
            errors.append({"symbol": symbol, "message": str(exc)})
    try:
        frame = ak.index_zh_a_hist(symbol="000300", period="daily", start_date=start, end_date=end)
        benchmark = {str(row["日期"])[:10]: _decimal(row["收盘"]) for _, row in frame.iterrows()}
    except Exception as exc:
        benchmark = {}
        errors.append({"symbol": "sh000300", "message": str(exc)})
    return closes, benchmark, calendar, errors
