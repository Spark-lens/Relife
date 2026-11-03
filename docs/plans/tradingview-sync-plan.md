# TradingView 增量同步

## 目标

将嘉信证券原始交易流水转换为 TradingView 可导入的 CSV，并在重复运行时保持幂等。当前只实现嘉信，`ibkr` 和 `yinhe` 保留为后续扩展入口。

## 输入与输出

- 输入：`data/transactions/charles_schwab/*.csv` 中日期最新的文件，或 `--transactions` 指定的文件。
- 当前全量：`data/tradingview/tradingview_full_latest_YYYY-MM-DD.csv` 中日期最新的一份。
- 本次增量：`tradingview_increment_起始日期_结束日期.csv`。
- 最新全量：`tradingview_full_latest_结束日期.csv`。
- 月末快照：`tradingview_full_YYYY-MM-DD.csv`。

## 转换规则

- `Buy` 转为 `Buy`。
- `Sell` 转为 `Sell`。
- `Cash Dividend`、`Non-Qualified Div`、`NRA Tax Adj` 转为 `Dividend`，金额写入 `Qty`。
- `Journal` 跳过并在摘要中计数。
- 其他交易动作报错，避免静默丢失数据。
- `Closing Time` 统一为 `YYYY-MM-DD 00:00:00`。

## 增量与去重

当前全量文件的最大 `Closing Time` 是截止日期。脚本只转换日期晚于截止日期的源记录，并用 `(日期, 方向, 裸标的, 数量, 成交价, 佣金)` 与全量已有记录去重。

重复运行同一份流水时，候选记录可以存在，但已存在记录不会再次写入。

## Symbol 映射

- 先从当前全量文件提取已有裸标的与 TradingView 代码的对应关系。
- 再使用 `data/templates/tradingview/symbol_map.json` 覆盖或补充映射。
- 未映射标的默认直接报错，不自动猜测交易所。
- 只有显式提供 `--unknown-symbol-exchange` 时才允许补充统一交易所前缀。

## 全量维护

有实际新增记录且未指定 `--skip-full-maintenance` 时：

1. 合并当前全量与新增记录，按成交时间倒序生成最新全量文件。
2. 如最近一个月末快照不存在，生成该月末快照。
3. 删除旧的最新全量文件，只保留一份。
4. 删除旧的月末快照，只保留一份。

没有新增记录时不写文件，只输出同步摘要。

## 扩展边界

多券商的目录和 CLI 参数结构已经保留；实际解析器当前只有嘉信。实现 `ibkr` 或 `yinhe` 时，必须分别补充表头校验、动作映射和测试，不能复用嘉信的解析假设。
