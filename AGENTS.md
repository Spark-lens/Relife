# AGENTS.md

## 项目概览

Relife 当前主要包含两类 Python 工作流：

1. `scripts/tradingview_sync.py`
   将券商原始交易流水 CSV 转换为 TradingView 可导入 CSV，并维护增量文件、最新全量文件和月末快照。
2. `strategies/bollinger_band_reversion/`
   一个基于布林带的多标的 paper trading 策略，读取配置、拉取行情、根据账户资金和持仓生成模拟订单，并将状态写入 SQLite。

## 运行环境

- 优先使用 Conda 环境：`/home/relife/miniforge3/envs/stock`
- Python 解释器：`/home/relife/miniforge3/envs/stock/bin/python`
- 当前仓库以 Python 3.12 运行为准

推荐命令：

```bash
source /home/relife/miniforge3/bin/activate stock
```

如果不走 `conda activate`，直接使用解释器也可以：

```bash
/home/relife/miniforge3/envs/stock/bin/python ...
```

## 目录说明

- `scripts/`：一次性或运维型脚本
- `strategies/`：策略代码
- `data/transactions/`：券商原始交易流水输入目录
- `data/tradingview/`：TradingView 导出结果目录
- `data/templates/tradingview/`：TradingView CSV 模板
- `data/templates/tradingview/symbol_map.json`：Schwab 标的到 TradingView 代码的手工映射
- `docs/plans/`：已有实现计划或设计文档

## 常用命令

交易流水同步：

```bash
/home/relife/miniforge3/envs/stock/bin/python scripts/tradingview_sync.py --help
/home/relife/miniforge3/envs/stock/bin/python scripts/tradingview_sync.py
```

布林带策略：

```bash
/home/relife/miniforge3/envs/stock/bin/python strategies/bollinger_band_reversion/run.py --once --mode paper --dry-run
```

## 策略运行前置环境变量

布林带策略默认依赖以下环境变量：

- `ALPHA_VANTAGE_API_KEY`
- `RELIFE_SCHWAB_NET_LIQUIDATION`
- `RELIFE_SCHWAB_CASH_AVAILABLE_WITHOUT_MARGIN`

可选持仓变量：

- `RELIFE_POSITION_QQQ`
- `RELIFE_POSITION_QQQI`
- `RELIFE_POSITION_BRK_B`
- `RELIFE_POSITION_YANG`
- `RELIFE_POSITION_SQQQ`
- `RELIFE_POSITION_SOXS`

## 依赖说明

- `scripts/tradingview_sync.py` 仅使用 Python 标准库
- 策略核心运行依赖第三方包：`PyYAML`、`requests`
- 行情回退源为可选能力，使用时需要：`pandas`、`yfinance`、`akshare`
- 新增嘉信标的时，优先更新 `data/templates/tradingview/symbol_map.json`，不要直接改脚本常量

## 协作约定

- 优先保持现有目录结构，不要随意移动 `data/`、`scripts/`、`strategies/` 下的路径
- `scripts/` 下脚本面向日常使用时，命令行提示、报错信息、摘要输出默认使用中文
- 修改策略逻辑前，先同步检查 `config.yaml`、`README.md`、运行入口 `run.py` 是否需要一起更新
- `data/tradingview/` 下包含生成产物；除非任务明确要求，不要批量覆盖历史快照文件
- 当前仓库未见正式测试套件；完成修改后，至少运行相关脚本的 `--help` 或一次 `--dry-run` 做基本验证

## 新增依赖时

如果新增了 import 到第三方包：

1. 同步更新根目录 `requirements.txt`
2. 如涉及新的环境变量或运行方式，同步更新对应 README
3. 优先保持脚本可直接从仓库根目录运行
