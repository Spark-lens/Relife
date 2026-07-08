# TradingView 增量同步脚本与计划文档

## Summary
先产出一份独立的 Markdown 计划文档，再按文档直接实现并执行脚本。

这版方案固定这些默认值：
- `broker` 默认使用嘉信 `Charles Schwab`
- 模板文件单独放在模板目录
- 交易原始文件按券商分目录存放
- 当前全量文件默认取 `data/tradingview` 中最新的全量交易文件
- 本次新增文件默认按“实际新增记录起止日期”命名
- 全量文件自动维护为 2 份：
  - 上一自然月月末快照
  - 当前最新全量

计划文档单独存放在：
- `docs/plans/tradingview-sync-plan.md`

## 目录结构
- `data/templates/tradingview`
- `data/tradingview`
- `data/transactions/charles_schwab`
- `data/transactions/ibkr`
- `data/transactions/yinhe`
- `docs/plans`
- `scripts`

当前这版实际只实现：
- `data/templates/tradingview` 的模板读取
- `data/transactions/charles_schwab` 的嘉信交易读取
- `data/tradingview` 的输出与全量维护

## 文件命名
模板文件：
- `tradingview_template.csv`

TradingView 全量文件：
- 当前最新全量：`tradingview_full_latest_YYYY-MM-DD.csv`
- 月末快照：`tradingview_full_YYYY-MM-DD.csv`

本次新增文件：
- 默认：`tradingview_increment_YYYY-MM-DD_YYYY-MM-DD.csv`
- 起止日期按“本次实际成功写入的新增记录”计算
- 文件名始终自动生成，不提供手动覆盖参数

## 脚本参数
脚本：`scripts/tradingview_sync.py`

CLI 参数：
- `--broker`：券商类型，默认 `charles_schwab`
- `--template-dir`：TradingView 模板目录，默认 `data/templates/tradingview`
- `--template-name`：模板文件名，默认 `tradingview_template.csv`
- `--transactions-dir`：交易原始文件根目录，默认 `data/transactions`
- `--transactions`：交易文件路径；可传绝对路径、相对 broker 子目录路径，或省略后自动选择该目录下最新 CSV
- `--tradingview-dir`：TradingView 数据目录，默认 `data/tradingview`
- `--output-dir`：输出目录，默认等于 `tradingview-dir`
- `--skip-full-maintenance`：是否跳过全量文件自动维护；不传时默认自动维护

默认解析规则：
- `--broker` 不传时使用 `charles_schwab`
- `--transactions` 未传时，自动从 `data/transactions/charles_schwab` 选择最新 CSV
- 当前全量文件不开放手动指定，始终自动在 `data/tradingview` 中选择最新全量文件

## 当前全量文件识别
“当前最后的全量交易文件”定义为：
- 优先匹配 `tradingview_full_latest_YYYY-MM-DD.csv`
- 取其中日期最新的一份
- 若不存在，再从 `tradingview_full_YYYY-MM-DD.csv` 中选日期最新的一份
- 若仍不存在，则兼容识别历史命名的 `tradingview_*.csv`
- 若仍找不到，直接报错

脚本读取该文件内容后：
- 取最大 `Closing Time` 作为增量 cutoff
- 只处理交易日期晚于 cutoff 的源记录

## 转换与去重逻辑
嘉信到 TradingView 的映射：
- `Buy` -> `Buy`
- `Sell` -> `Sell`
- `Cash Dividend` / `Non-Qualified Div` / `NRA Tax Adj` -> `Dividend`
- `Dividend` 的 `Qty` 使用 `Amount`
- `Fees & Comm` 为空写 `0`
- `Closing Time` 统一写成 `YYYY-MM-DD 00:00:00`

特殊交易：
- `Journal` 跳过并告警
- 其他未知交易类型直接报错

去重策略：
- 使用标准化键 `(date, side, bare_symbol, qty, fill_price, commission)`
- 合并增量前先与当前全量比对
- 避免历史重叠或重复运行造成重复写入

## 多券商扩展结构
脚本按 broker 适配器设计，但当前只实现嘉信：
- 主流程负责参数、文件发现、去重、输出、全量维护
- `charles_schwab` 适配器负责 CSV 解析和标准化
- 预留 `ibkr`、`yinhe` 适配器接口，不在这版实现具体逻辑

统一中间记录结构包含：
- `trade_date`
- `action`
- `symbol_raw`
- `quantity`
- `price`
- `commission`
- `amount`
- `description`
- `broker`

## Symbol 映射
- 优先从当前全量文件自动提取 symbol 映射
- 再补内置映射字典，至少覆盖：
  - `QQQM -> NASDAQ:QQQM`
  - `GGLL -> NASDAQ:GGLL`
  - `SSPC -> CBOE:SSPC`
- 未知 symbol 默认按配置的交易所前缀自动补充到 `symbol_map.json`，默认前缀为 `NASDAQ`

## 全量文件维护
默认自动维护：
- 每次运行先输出本次新增文件
- 若无实际新增：
  - 不更新全量
  - 只输出摘要
- 若有实际新增：
  - 合并出新的 `tradingview_full_latest_<end_date>.csv`
- 若当前已跨月且不存在上一月月末快照：
  - 自动补建 `tradingview_full_<month_end>.csv`
- 清理旧文件：
  - 只保留最近 1 份月末快照
  - 只保留最近 1 份当前最新全量

## 执行顺序
1. 创建计划文档 `docs/plans/tradingview-sync-plan.md`
2. 调整目录约定并准备模板、交易文件默认路径
3. 编写 `scripts/tradingview_sync.py`
4. 用当前样本运行一次脚本
5. 验证增量输出与全量维护结果
6. 如输出正确，保留脚本与生成文件

## Test Plan
1. `--broker` 不传时默认走 `charles_schwab`
2. `--transactions` 只给文件名时，能在 `data/transactions/charles_schwab` 找到文件
3. `--transactions` 不传时，能自动选中 `data/transactions/charles_schwab` 下最新 CSV
4. 不提供 `current` 参数时，脚本仍能自动识别 `data/tradingview` 中最新全量文件
5. 不提供 `incremental-output-name` 参数时，本次新增文件仍按实际新增起止日期自动命名，例如 `tradingview_increment_2026-06-05_2026-06-17.csv`
6. `Buy`、`Sell`、分红类交易正确转换，`Journal` 跳过并告警
7. 有新增时自动更新 `tradingview_full_latest_<date>.csv`
8. 缺少上一月月末快照时自动补建 `tradingview_full_2026-05-31.csv`
9. 同一交易文件重复执行时不重复追加
10. 模板表头错误、未知交易类型时明确报错；未知 symbol 会自动补充映射

## Assumptions
- 当前这版只实现嘉信，但入口默认值与目录结构按多券商扩展设计
- “直接执行”指在实现阶段，文档产出后立即开始脚本开发、运行与验证
- 本次新增文件日期范围始终按“实际写入的增量记录”计算
- 当前全量文件始终自动识别，不开放手动覆盖参数
- 本次新增文件名始终自动生成，不开放手动覆盖参数
