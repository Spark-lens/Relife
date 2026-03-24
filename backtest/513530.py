"""
回测策略：
    港股红利ETF(513530) / 红利低波ETF(159307)

买入条件（任意一个满足即可）：
    1. A股上涨家数 ≤ 500
    2. 上证 / 创业板 / 沪深300 任一当日跌幅 ≥ 2.5%
    3. 标的当日跌幅 ≥ 2.5%
    4. 标的收盘价 < MA120

分红：
    到账后留作现金，等待买入信号

卖出条件：
    不卖出，长期持有

资金：
    初始 10000 元，每次买入 5 手（500 份）

PS：
    图示保存
    表格化展示 买入、资金变化、持仓、盈亏等信息
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import akshare as ak


@dataclass
class BacktestConfig:
    symbol: str = "513530"
    start_date: str = "20210101"
    end_date: str = "20261231"
    initial_cash: float = 10000.0
    lot_size: int = 500
    ma_window: int = 120
    up_count_threshold: int = 500
    index_drop_threshold: float = -2.5
    target_drop_threshold: float = -2.5


def _normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    date_col = "日期" if "日期" in df.columns else df.columns[0]
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.rename(columns={date_col: "date"})


def _call_with_retry(func, *args, retries: int = 3, delay: float = 1.5, name: str = "接口", **kwargs):
    last_error = None
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if i < retries - 1:
                time.sleep(delay * (i + 1))
    raise RuntimeError(f"{name} 请求失败，已重试 {retries} 次，最后错误: {last_error}")


def fetch_etf_history(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    raw = _call_with_retry(
        ak.fund_etf_hist_em,
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="",
        name=f"ETF {symbol} 日线",
    )
    if raw.empty:
        raise ValueError(f"未获取到ETF {symbol} 的历史行情")

    df = _normalize_date_col(raw)
    required = ["开盘", "收盘", "涨跌幅"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"ETF行情缺少字段: {col}")

    out = pd.DataFrame(
        {
            "date": df["date"],
            "open": pd.to_numeric(df["开盘"], errors="coerce"),
            "close": pd.to_numeric(df["收盘"], errors="coerce"),
            "pct_chg": pd.to_numeric(df["涨跌幅"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return out


def fetch_index_pct_change(symbol: str, start_date: str, end_date: str, out_col: str) -> pd.DataFrame:
    raw = _call_with_retry(
        ak.index_zh_a_hist,
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        name=f"指数 {symbol} 日线",
    )
    if raw.empty:
        raise ValueError(f"未获取到指数 {symbol} 行情")

    df = _normalize_date_col(raw)
    if "涨跌幅" not in df.columns:
        raise ValueError(f"指数 {symbol} 行情缺少字段: 涨跌幅")

    return pd.DataFrame(
        {
            "date": df["date"],
            out_col: pd.to_numeric(df["涨跌幅"], errors="coerce"),
        }
    ).dropna(subset=[out_col])


def fetch_up_count_series(start_date: str, end_date: str) -> pd.DataFrame:
    """
    历史上涨家数没有统一且稳定的免费接口，这里默认返回空列。
    保留字段可便于后续接入你自己的上涨家数数据。
    """
    dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="B")
    return pd.DataFrame({"date": dates, "up_count": pd.NA})


def _pick_first(columns: Iterable[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def fetch_dividend_events(symbol: str) -> pd.DataFrame:
    """
    尝试从AKShare读取ETF分红记录。
    若接口结构变更或无数据，返回空表并继续回测。
    """
    try:
        raw = _call_with_retry(
            ak.fund_etf_fund_info_em,
            symbol=symbol,
            name=f"ETF {symbol} 分红",
        )
    except Exception:
        return pd.DataFrame(columns=["date", "dividend_per_share"])

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "dividend_per_share"])

    date_col = _pick_first(raw.columns, ["权益登记日", "除息日", "公告日期", "日期"])
    div_col = _pick_first(raw.columns, ["每份分红", "分红", "分红金额", "每份派息"])
    if not date_col or not div_col:
        return pd.DataFrame(columns=["date", "dividend_per_share"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "dividend_per_share": pd.to_numeric(raw[div_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "dividend_per_share"])
    out = out[out["dividend_per_share"] > 0]
    if out.empty:
        return pd.DataFrame(columns=["date", "dividend_per_share"])

    out = out.groupby("date", as_index=False)["dividend_per_share"].sum()
    return out.sort_values("date").reset_index(drop=True)


def run_backtest(cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    etf = fetch_etf_history(cfg.symbol, cfg.start_date, cfg.end_date)
    sh = fetch_index_pct_change("000001", cfg.start_date, cfg.end_date, "sh_pct")
    cyb = fetch_index_pct_change("399006", cfg.start_date, cfg.end_date, "cyb_pct")
    hs300 = fetch_index_pct_change("000300", cfg.start_date, cfg.end_date, "hs300_pct")
    up_count = fetch_up_count_series(cfg.start_date, cfg.end_date)
    dividend = fetch_dividend_events(cfg.symbol)

    df = (
        etf.merge(sh, on="date", how="left")
        .merge(cyb, on="date", how="left")
        .merge(hs300, on="date", how="left")
        .merge(up_count, on="date", how="left")
    )
    df["ma120"] = df["close"].rolling(cfg.ma_window).mean()

    cond_1 = df["up_count"].notna() & (pd.to_numeric(df["up_count"], errors="coerce") <= cfg.up_count_threshold)
    cond_2 = (df[["sh_pct", "cyb_pct", "hs300_pct"]] <= cfg.index_drop_threshold).any(axis=1)
    cond_3 = df["pct_chg"] <= cfg.target_drop_threshold
    cond_4 = df["close"] < df["ma120"]

    df["buy_signal"] = cond_1 | cond_2 | cond_3 | cond_4
    df["signal_reason"] = ""
    df.loc[cond_1, "signal_reason"] += "上涨家数<=500;"
    df.loc[cond_2, "signal_reason"] += "大盘单日跌幅>=2.5%;"
    df.loc[cond_3, "signal_reason"] += "标的单日跌幅>=2.5%;"
    df.loc[cond_4, "signal_reason"] += "收盘价<MA120;"

    dividend_map = {
        row.date.normalize(): float(row.dividend_per_share)
        for row in dividend.itertuples(index=False)
    }

    cash = cfg.initial_cash
    shares = 0
    trades: list[dict] = []
    records: list[dict] = []

    for row in df.itertuples(index=False):
        day = row.date.normalize()
        close = float(row.close)

        dividend_per_share = dividend_map.get(day, 0.0)
        dividend_cash = shares * dividend_per_share if shares > 0 else 0.0
        if dividend_cash > 0:
            cash += dividend_cash

        bought = 0
        cost = 0.0
        if bool(row.buy_signal):
            lot_cost = close * cfg.lot_size
            if cash >= lot_cost:
                bought = cfg.lot_size
                cost = lot_cost
                cash -= lot_cost
                shares += bought
                trades.append(
                    {
                        "date": row.date,
                        "action": "BUY",
                        "price": round(close, 4),
                        "shares": bought,
                        "amount": round(cost, 2),
                        "reason": str(row.signal_reason).rstrip(";"),
                    }
                )

        market_value = shares * close
        total_asset = cash + market_value
        pnl = total_asset - cfg.initial_cash

        records.append(
            {
                "date": row.date,
                "close": close,
                "ma120": row.ma120,
                "buy_signal": bool(row.buy_signal),
                "signal_reason": str(row.signal_reason).rstrip(";"),
                "bought": bought,
                "dividend_cash": round(dividend_cash, 2),
                "trade_cost": round(cost, 2),
                "cash": round(cash, 2),
                "shares": shares,
                "market_value": round(market_value, 2),
                "total_asset": round(total_asset, 2),
                "pnl": round(pnl, 2),
                "return_pct": round((total_asset / cfg.initial_cash - 1) * 100, 2),
            }
        )

    result_df = pd.DataFrame(records)
    trade_df = pd.DataFrame(trades)
    return result_df, trade_df


def save_outputs(result_df: pd.DataFrame, trade_df: pd.DataFrame, symbol: str) -> None:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    csv_path = root_dir / f"backtest_{symbol}_daily.csv"
    trade_path = root_dir / f"backtest_{symbol}_trades.csv"
    fig_path = root_dir / f"backtest_{symbol}.png"

    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    trade_df.to_csv(trade_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(13, 7))
    plt.plot(result_df["date"], result_df["close"], label=f"{symbol} Close", linewidth=1.3)
    plt.plot(result_df["date"], result_df["ma120"], label="MA120", linewidth=1.1)

    buy_points = result_df[result_df["bought"] > 0]
    if not buy_points.empty:
        plt.scatter(
            buy_points["date"],
            buy_points["close"],
            label="Buy",
            marker="^",
            s=42,
            alpha=0.9,
        )

    plt.title(f"Backtest {symbol}: Buy-on-Dip and Hold")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print("=" * 90)
    print(f"已保存逐日结果: {csv_path}")
    print(f"已保存交易记录: {trade_path}")
    print(f"已保存图表: {fig_path}")
    print("=" * 90)


def print_summary(result_df: pd.DataFrame, trade_df: pd.DataFrame) -> None:
    if result_df.empty:
        print("无回测结果")
        return

    final_row = result_df.iloc[-1]
    print("回测汇总")
    print("-" * 90)
    print(f"最后交易日: {final_row['date'].date()}")
    print(f"最终现金: {final_row['cash']:.2f}")
    print(f"最终持仓: {int(final_row['shares'])} 份")
    print(f"持仓市值: {final_row['market_value']:.2f}")
    print(f"账户总资产: {final_row['total_asset']:.2f}")
    print(f"累计收益: {final_row['pnl']:.2f}")
    print(f"收益率: {final_row['return_pct']:.2f}%")
    print(f"买入次数: {len(trade_df)}")

    print("\n最近10条逐日记录")
    print("-" * 90)
    show_cols = [
        "date",
        "close",
        "buy_signal",
        "bought",
        "dividend_cash",
        "cash",
        "shares",
        "market_value",
        "total_asset",
        "pnl",
        "return_pct",
    ]
    print(result_df[show_cols].tail(10).to_string(index=False))

    print("\n买入记录")
    print("-" * 90)
    if trade_df.empty:
        print("无买入交易")
    else:
        print(trade_df.to_string(index=False))


def main() -> None:
    cfg = BacktestConfig(symbol="513530")
    try:
        result_df, trade_df = run_backtest(cfg)
    except Exception as exc:  # noqa: BLE001
        print("回测执行失败，请检查网络是否可访问东方财富数据接口，或稍后重试。")
        print(f"错误详情: {exc}")
        return

    print_summary(result_df, trade_df)
    save_outputs(result_df, trade_df, cfg.symbol)


if __name__ == "__main__":
    main()