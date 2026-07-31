# **Relife**

> 等待，等待，漫长枯燥煎熬的等待...

## 当前功能

### 私有投资组合仪表盘

仪表盘从最新美股 TradingView 全量文件和最新银河证券交割单生成，A 股与美股独立核算，不做汇率换算。页面包括组合总览、策略分组持仓、倒序交易记录和股息分析，并以最后收盘价对比市场基准。

| 市场 | 基准 | 行情源 |
| --- | --- | --- |
| A 股 | 上证指数、沪深 300 | AkShare |
| 美股 | QQQ、SPY | Yahoo Finance |

更新数据、构建并启动看板：

```bash
./scripts/run_portfolio_viewer.sh
```

刷新交易数据、联网获取行情并构建站点：

```bash
npm --prefix portfolio_viewer run portfolio:update
```

只重新生成 `portfolio_viewer/public/data/portfolio.json`：

```bash
npm --prefix portfolio_viewer run portfolio:data
```

VS Code 插件开发与打包：

```bash
npm --prefix portfolio_viewer run vscode:test
npm --prefix portfolio_viewer run vscode:build
npm --prefix portfolio_viewer run vscode:package
```

安装本地 0.2.0 包：在 VS Code 扩展视图右上角选择“从 VSIX 安装…”，选择 `portfolio_viewer/relife-portfolio-0.2.0.vsix`；或在仓库根目录执行：

```bash
code --install-extension portfolio_viewer/relife-portfolio-0.2.0.vsix --force
```

安装后可点击 Activity Bar 中的 `Re` 图标打开投资组合，也可继续使用命令面板。页面顶部支持立即更新；VS Code 打开期间，插件会按上海时间在 A 股 09:15/15:15、美股 21:15/次日 05:15 自动更新。

Python 使用 `/home/clannad/miniforge3/envs/istorm_rag_gpu`。行情查询仅向服务商发送标的代码和日期范围，不发送持仓数量、成本、账户余额或原始交易文件。账本会按行情源的拆并股事件换算到当前股本口径，交易记录仍展示原始成交数量与价格。持仓列的显示与顺序保存在当前设备浏览器中。

### 嘉信交易流水同步 TradingView

读取嘉信 Charles Schwab 原始交易流水 CSV，生成 TradingView 可导入交易记录，并同步维护增量文件、最新全量文件和月末快照。

| 项目 | 内容 |
| --- | --- |
| 输入 | `data/transactions/charles_schwab/` 下的嘉信交易流水 CSV，默认使用最新文件 |
| 输出 | `data/tradingview/` 下的 TradingView CSV |
| Symbol 映射 | `data/templates/tradingview/symbol_map.json` |
| 命令 | `python3 scripts/tradingview_sync.py` |

新增标的时，脚本会默认按 `NASDAQ:<标的>` 自动补充到 `data/templates/tradingview/symbol_map.json`。如果实际交易所不是 NASDAQ，请在运行后手动调整对应值，例如：

```json
{
  "NOK": "NYSE:NOK",
  "CAMT": "NASDAQ:CAMT"
}
```

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
