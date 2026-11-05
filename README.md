# Relife

Relife 当前只保留一个功能：将嘉信证券（Charles Schwab）交易流水增量转换为 TradingView 可导入的 CSV，并维护最新全量文件和月末快照。

## 功能流程

1. 从 `data/transactions/charles_schwab/` 读取最新嘉信交易流水 CSV；也可通过 `--transactions` 指定文件。
2. 读取 `data/tradingview/` 中日期最新的 `tradingview_full_latest_*.csv`，以其中最大成交日为增量截止日期。
3. 转换日期不早于截止日期的买入、卖出和分红记录，跳过 `Journal` 记录并输出中文摘要。
4. 按交易日期、方向、标的、数量、成交价和佣金去重，避免重复运行产生重复记录。
5. 有新增记录时，写入本次增量文件、更新最新全量文件，并在跨月后补建最近一个月末快照。

## 数据目录

| 路径 | 用途 |
| --- | --- |
| `data/transactions/charles_schwab/` | 嘉信原始交易流水输入 |
| `data/templates/tradingview/tradingview_template.csv` | TradingView CSV 表头模板 |
| `data/templates/tradingview/symbol_map.json` | 嘉信标的到 TradingView 代码的映射 |
| `data/tradingview/` | 增量、最新全量和月末快照输出 |
| `tests/` | 同步逻辑测试 |

## 运行环境

脚本和测试只依赖 Python 标准库。当前仓库以 Python 3.12 为运行基准：

```bash
/home/relife/miniforge3/envs/istorm_rag_gpu/bin/python scripts/tradingview_sync.py --help
/home/relife/miniforge3/envs/istorm_rag_gpu/bin/python scripts/tradingview_sync.py
```

默认从仓库根目录执行。常用参数包括：

- `--transactions`：指定嘉信交易流水文件；相对路径会先在当前目录查找，再在券商子目录查找。
- `--tradingview-dir` / `--output-dir`：指定读取全量文件和写入结果的目录。
- `--unknown-symbol-exchange`：显式允许未知标的统一补充交易所前缀。未提供且映射缺失时会停止同步。
- `--skip-full-maintenance`：只写本次增量，不更新全量文件或月末快照。

## Symbol 映射

`symbol_map.json` 的值必须使用 `交易所:代码` 格式，例如 `NASDAQ:QQQI`、`CBOE:BOXX`。遇到未映射标的时，默认报错并列出需要补充的代码，以免生成错误的 TradingView 标的。

`--broker` 仍支持 `charles_schwab`、`ibkr` 和 `yinhe` 三个选项。当前只有 `charles_schwab` 实现了转换；`ibkr` 与 `yinhe` 仅作为后续扩展占位。

## 测试

```bash
/home/relife/miniforge3/envs/istorm_rag_gpu/bin/python -m unittest discover -s tests -v
```

`data/tradingview/` 中的文件属于生成产物。除非任务明确要求，不要批量覆盖或删除历史月末快照。
