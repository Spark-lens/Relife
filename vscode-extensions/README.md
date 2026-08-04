# Relife 0.0.1

Relife 是固定黑色主题的双市场投资组合 VS Code 插件。首次启动使用完全虚构的美股与 A 股示例；可分别选择标准命名的真实 CSV，之后自动读取同目录日期最新文件。

## 安装
```bash
code --install-extension vscode-extensions/clannad0710.relife-0.0.1.vsix --force
```

## 数据源

- 美股：`tradingview_full_latest_YYYY-MM-DD.csv`
- A 股：`交割单_YYYY-MM-DD.csv`

扩展只在 VS Code `workspaceState` 保存目录、命名模式和最后有效文件；原始 CSV 不复制、不修改。成功快照写入扩展 `storageUri`，刷新失败继续显示最后成功快照。

## 隐私

VSIX 仅包含虚构示例数据。真实交易文件、缓存、密钥和本机路径不会打包。行情刷新由配置的 Python 环境调用 `yfinance` 与 `akshare`。
