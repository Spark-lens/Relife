"""
回测策略：港股红利ETF(513530) / 红利低波ETF(159307)

买入条件（全部满足）：
  1. A股上涨家数 ≤ 500（akshare 暂不支持历史数据时自动跳过此条件）
  2. 上证 / 创业板 / 沪深300 任一当日跌幅 ≥ 2.5%
  3. 标的当日跌幅 ≥ 2.5%
  4. 标的收盘价 < MA120

分红：到账后留作现金，等待买入信号
卖出：不卖出，长期持有

资金：初始 10000 元，每次买入 5 手（500 份）
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
})

# ══════════════════════════ 策略参数 ═════════════════════════════════════════
START     = "20200101"
END       = "20260324"
INIT_CASH = 10_000
BUY_LOTS  = 5
LOT_SIZE  = 100        # ETF 1手 = 100份
DROP      = 0.025      # 跌幅触发阈值
RISE_MAX  = 500        # 上涨家数阈值
MA_N      = 120

SYMBOLS = ["513530", "159307"]
NAMES   = {"513530": "港股红利ETF", "159307": "红利低波ETF"}
INDICES = {"上证": "sh000001", "创业板": "sz399006", "沪深300": "sh000300"}


# ══════════════════════════ 数据获取 ═════════════════════════════════════════
def get_etf(sym: str) -> pd.Series:
    df = ak.fund_etf_hist_em(symbol=sym, period="daily",
                             start_date=START, end_date=END, adjust="")
    df["日期"] = pd.to_datetime(df["日期"])
    return df.set_index("日期")["收盘"].sort_index().astype(float)


def get_index(sym: str) -> pd.Series:
    df = ak.stock_zh_index_daily(symbol=sym)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index().astype(float)


def get_breadth() -> pd.Series | None:
    """A股每日上涨家数。若无历史数据则返回 None（跳过该条件）。"""
    try:
        df = ak.stock_market_activity_legu()
        if "date" in df.columns and len(df) > 5:
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")["上涨"].astype(int).sort_index()
    except Exception:
        pass
    print("[提示] 上涨家数历史数据不可用，条件1自动视为满足")
    return None


def get_dividend(sym: str) -> dict:
    """返回 {date: 每份分红金额} 字典。获取失败则返回空字典。"""
    for fn_name, kwargs in [
        ("fund_etf_dividend_em", {"fund": sym}),
        ("fund_etf_dividend_em", {"symbol": sym}),
    ]:
        try:
            fn = getattr(ak, fn_name)
            df = fn(**kwargs)
            if df.empty:
                continue
            date_col = next(c for c in df.columns if "日期" in c or "date" in c.lower())
            amt_col  = next(c for c in df.columns if any(k in c for k in ["每份", "分红", "派息", "金额"]))
            out = {}
            for _, row in df.iterrows():
                try:
                    out[pd.to_datetime(row[date_col])] = float(str(row[amt_col]).replace(",", ""))
                except Exception:
                    continue
            return out
        except Exception:
            continue
    print(f"[提示] {sym} 分红数据获取失败，跳过分红处理")
    return {}


# ══════════════════════════ 回测引擎 ═════════════════════════════════════════
def backtest(sym: str, index_series: dict[str, pd.Series],
             breadth: pd.Series | None) -> dict:
    price = get_etf(sym)
    ma    = price.rolling(MA_N).mean()
    ret   = price.pct_change()
    divs  = get_dividend(sym)

    # 指数任一跌幅 >= DROP
    idx_df    = pd.DataFrame(index_series).reindex(price.index).ffill()
    idx_crash = (idx_df.pct_change() <= -DROP).any(axis=1)

    # 上涨家数条件
    if breadth is not None:
        br = breadth.reindex(price.index).ffill().fillna(9999)
    else:
        br = None

    cash, shares = float(INIT_CASH), 0
    buy_log, div_log = [], []

    for date in price.index:
        p = price[date]

        # 分红到账
        if date in divs and shares > 0:
            got   = divs[date] * shares
            cash += got
            div_log.append({"日期": date, "每份(元)": divs[date],
                            "到账(元)": got, "现金余额": cash})

        if pd.isna(ma[date]) or pd.isna(ret[date]):
            continue

        c1 = (br is None) or (br[date] <= RISE_MAX)
        c2 = bool(idx_crash.get(date, False))
        c3 = ret[date] <= -DROP
        c4 = p < ma[date]

        if c1 and c2 and c3 and c4:
            cost = BUY_LOTS * LOT_SIZE * p
            if cash >= cost:
                cash   -= cost
                shares += BUY_LOTS * LOT_SIZE
                buy_log.append({"日期": date, "价格": p,
                                "购入份数": BUY_LOTS * LOT_SIZE,
                                "现金余额": cash})

    return dict(sym=sym, price=price, ma=ma,
                buy_log=buy_log, div_log=div_log,
                cash=cash, shares=shares, last_price=price.iloc[-1])


# ══════════════════════════ 文字输出 ═════════════════════════════════════════
def summarize(r: dict):
    mkt_val   = r["shares"] * r["last_price"]
    total     = r["cash"] + mkt_val
    total_div = sum(x["到账(元)"] for x in r["div_log"])
    gain      = total - INIT_CASH

    print(f"\n{'─'*50}")
    print(f"  {NAMES[r['sym']]}（{r['sym']}）  [{START} ~ {END}]")
    print(f"{'─'*50}")
    print(f"  初始资金：{INIT_CASH:>12,.2f} 元")
    print(f"  买入次数：{len(r['buy_log']):>12} 次")
    print(f"  总持仓  ：{r['shares']:>12,} 份")
    print(f"  最新价  ：{r['last_price']:>12.4f} 元/份")
    print(f"  持仓市值：{mkt_val:>12,.2f} 元")
    print(f"  剩余现金：{r['cash']:>12,.2f} 元")
    print(f"  总资产  ：{total:>12,.2f} 元")
    print(f"  累计分红：{total_div:>12,.2f} 元")
    print(f"  盈亏    ：{gain:>+12,.2f} 元（{gain/INIT_CASH*100:+.2f}%）")
    print(f"{'─'*50}")

    if r["buy_log"]:
        df = pd.DataFrame(r["buy_log"])
        df["日期"]    = df["日期"].dt.strftime("%Y-%m-%d")
        df["价格"]    = df["价格"].map("{:.4f}".format)
        df["现金余额"] = df["现金余额"].map("{:,.2f}".format)
        print("\n【买入明细】")
        print(df.to_string(index=False))

    if r["div_log"]:
        df = pd.DataFrame(r["div_log"])
        df["日期"]    = df["日期"].dt.strftime("%Y-%m-%d")
        df["每份(元)"] = df["每份(元)"].map("{:.4f}".format)
        df["到账(元)"] = df["到账(元)"].map("{:.2f}".format)
        df["现金余额"] = df["现金余额"].map("{:,.2f}".format)
        print("\n【分红明细】")
        print(df.to_string(index=False))


# ══════════════════════════ 可视化 ════════════════════════════════════════════
def plot(r: dict):
    name  = NAMES[r["sym"]]
    price = r["price"]
    ma    = r["ma"]
    buys  = r["buy_log"]
    divs  = r["div_log"]

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"{name}（{r['sym']}）回测结果  [{START} ~ {END}]",
                 fontsize=13, weight="bold")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1.4],
                  hspace=0.45, wspace=0.32)

    # ── 价格走势 + MA + 买入标记 ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(price.index, price, lw=1, color="#1f77b4", label="收盘价（不复权）")
    ax1.plot(ma.index, ma, lw=1, ls="--", color="darkorange", label=f"MA{MA_N}")
    if buys:
        bx = [x["日期"] for x in buys]
        by = [float(x["价格"]) if isinstance(x["价格"], str) else x["价格"] for x in buys]
        # 买入点从 buy_log 直接取原始浮点
        by_raw = [float(str(x["价格"]).replace(",", "")) for x in buys]
        ax1.scatter(bx, by_raw, marker="^", color="red", s=80, zorder=5, label="买入")
    ax1.set_ylabel("价格（元）")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
    ax1.grid(alpha=0.3)

    # ── 现金余额变化 ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if buys:
        raw_cash = [float(str(x["现金余额"]).replace(",", "")) for x in buys]
        bx2 = [x["日期"] for x in buys]
        ax2.step([price.index[0]] + bx2, [INIT_CASH] + raw_cash, where="post", color="steelblue")
        ax2.scatter(bx2, raw_cash, s=25, color="steelblue", zorder=5)
    ax2.axhline(INIT_CASH, ls=":", color="gray", lw=0.8, label="初始资金")
    ax2.set_title("现金余额变化")
    ax2.set_ylabel("元")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
    ax2.grid(alpha=0.3)

    # ── 每次分红到账 ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if divs:
        dx = [x["日期"] for x in divs]
        dy = [x["到账(元)"] for x in divs]
        ax3.bar(dx, dy, color="seagreen", alpha=0.75, width=8)
        ax3.set_title("每次分红到账（元）")
        ax3.set_ylabel("元")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")
        ax3.grid(alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5, "暂无分红数据", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=11, color="gray")
        ax3.set_title("分红记录")

    out = f"backtest_{r['sym']}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"图表已保存：{out}")


# ══════════════════════════ 主程序 ════════════════════════════════════════════
if __name__ == "__main__":
    print("获取指数数据...")
    index_series = {k: get_index(v) for k, v in INDICES.items()}

    print("获取上涨家数数据...")
    breadth = get_breadth()

    for sym in SYMBOLS:
        print(f"\n{'═'*50}")
        print(f"  开始回测：{NAMES[sym]}（{sym}）")
        print(f"{'═'*50}")
        result = backtest(sym, index_series, breadth)
        summarize(result)
        plot(result)