# AGENTS.md

## 项目概览

Relife 当前只包含嘉信交易流水到 TradingView CSV 的增量同步工作流。入口为 `scripts/tradingview_sync.py`，负责读取原始流水、转换交易、去重，并维护增量文件、最新全量文件和月末快照。

## 运行环境

- Python 3.12
- 解释器：`/home/relife/miniforge3/envs/istorm_rag_gpu/bin/python`
- 同步脚本和测试只依赖 Python 标准库
- 所有命令默认从仓库根目录执行

## 目录说明

- `scripts/`：同步脚本
- `tests/`：与同步逻辑相关的测试
- `data/transactions/charles_schwab/`：嘉信原始交易流水输入
- `data/templates/tradingview/`：TradingView 模板和 Symbol 映射
- `data/tradingview/`：增量、最新全量和月末快照
- `docs/plans/`：当前同步功能的实现说明

`ibkr` 与 `yinhe` 只在 CLI 和文档中保留扩展占位，尚未实现对应解析器。

## 常用命令

```bash
/home/relife/miniforge3/envs/istorm_rag_gpu/bin/python scripts/tradingview_sync.py --help
/home/relife/miniforge3/envs/istorm_rag_gpu/bin/python scripts/tradingview_sync.py
/home/relife/miniforge3/envs/istorm_rag_gpu/bin/python tests/test_tradingview_symbol_policy.py
```

## 数据规则

- 当前全量文件优先从 `tradingview_full_latest_*.csv` 中自动选择最新日期。
- 只处理不早于当前全量最大成交日的源记录。
- 未知 Symbol 默认阻止同步；新增映射时必须写入 `交易所:代码`。
- `data/tradingview/` 是生成产物目录，除非任务明确要求，不要覆盖历史月末快照。
- 脚本的命令行提示、错误和摘要输出使用中文。

## 协作约定

- 仓库收敛后只保留 `main` 分支。
- 当前仓库不使用隔离 worktree。
- 不主动提交 Git；只有任务明确要求时再提交，提交说明正文使用中文。
- 修改数据规则时，同步检查 `README.md`、本文件和 `docs/plans/tradingview-sync-plan.md`。
- 完成修改后至少运行相关测试、`--help`，并用当前数据执行一次无写入的同步检查。
