# Relife 0.0.1

Relife 是固定黑色主题的双市场投资组合 VS Code 插件。首次启动使用完全虚构的美股与 A 股示例；可分别选择标准命名的真实 CSV，之后自动读取同目录日期最新文件。

## 环境要求

- VS Code >= 1.95，Node.js >= 22.13
- Python 3.12+，需安装 `yfinance` `akshare`：`pip install yfinance akshare`

扩展按以下优先级自动发现 Python 解释器：

1. VS Code 设置 `relife.pythonPath`（手动指定）
2. VS Code Python 扩展当前选中的解释器
3. `CONDA_PREFIX`（Conda 环境）
4. 系统 `python3`

可通过 Output 面板（下拉选 Relife）查看实际使用的 Python 路径。若刷新报 `No module named 'yfinance'`，请在对应环境中 `pip install yfinance akshare`。

## 打包

```bash
npm run package       # 运行测试 + 构建 + 打包
npm run package:quick # 仅构建 + 打包（跳过测试）
```
```bash
# 打包脚本
cd /mnt/d/workspace-codex/Relife/vscode-extensions
npm run package:local
```

```bash
code --install-extension vscode-extensions/clannad0710.relife-0.0.1.vsix --force
```

## 数据源

- 美股：`tradingview_full_latest_YYYY-MM-DD.csv`
- A 股：`交割单_YYYY-MM-DD.csv`

扩展只在 VS Code `workspaceState` 保存目录、命名模式和最后有效文件；原始 CSV 不复制、不修改。成功快照写入扩展 `storageUri`，刷新失败继续显示最后成功快照。

## 隐私

VSIX 仅包含虚构示例数据。真实交易文件、缓存、密钥和本机路径不会打包。行情刷新由配置的 Python 环境调用 `yfinance` 与 `akshare`。
