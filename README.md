# **Relife**

> 等待，等待，漫长枯燥煎熬的等待...

## 当前功能

### 嘉信交易流水同步 TradingView

读取嘉信 Charles Schwab 原始交易流水 CSV，生成 TradingView 可导入交易记录，并同步维护增量文件、最新全量文件和月末快照。

| 项目 | 内容 |
| --- | --- |
| 输入 | `data/transactions/charles_schwab/` 下的嘉信交易流水 CSV，默认使用最新文件 |
| 输出 | `data/tradingview/` 下的 TradingView CSV |
| 命令 | `python3 scripts/tradingview_sync.py` |

## 证券账户

| 市场 | 账户/券商 | 当前状态 |
| --- | --- | --- |
| A 股 | 银河证券 | A 股账户 |
| 美股 | 嘉信 Schwab | 当前美股交易记录基于此账户 |
| 美股/多币种 | 盈透 IBKR | 已入金 `10000 CNY`，当前未交易 |

## 交易标的配置建议

默认按中国税务居民口径记录配置建议，不展开具体税务申报细节。

| 标的 | 类型 | 建议存放 | 形式 |
| --- | --- | --- | --- |
| `BOXX` | 闲置美元现金管理 | 美股账户 | 现金替代 ETF |
| `XQQI`、`QQQI` | 收入/分红型 ETF | 美股账户 | 分红型配置 |
| `SOXS`、`SQQQ`、`YANG` | 反向杠杆 ETF | 美股账户 | 短线交易/对冲工具 |
| `GOOGL`、`SPCX/SpaceX`、`JD` | 成长/主题股票 | 美股账户 | 股票配置 |
| `SCHD`、`BRK.B`、`KO` | 长期核心/质量/分红类美股资产 | 美股账户 | 长期核心配置 |
