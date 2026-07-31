# Relife Portfolio

在 VS Code 中查看当前 Relife 工作区的 A 股与美股投资组合。扩展复用仓库现有交易流水、Python 账本和 React 看板，不会把账户数据打包或上传到 Marketplace。

## 使用

### 安装 0.2.0

在打开 Relife 工作区的 VS Code 窗口中：

1. 打开“扩展”视图（`Ctrl+Shift+X`）。
2. 点击扩展视图右上角的 `…`，选择“从 VSIX 安装…”。
3. 选择 `portfolio_viewer/relife-portfolio-0.2.0.vsix`。
4. 安装完成后按提示重新加载窗口，并确认 `Relife Portfolio` 的版本为 `0.2.0`。

也可以在仓库根目录执行：

```bash
code --install-extension portfolio_viewer/relife-portfolio-0.2.0.vsix --force
```

如果 Relife 工作区通过 WSL 打开，请在该 WSL 窗口中安装，使扩展运行在能够访问仓库和 Python 环境的扩展宿主中。

### 打开投资组合

打开包含以下路径的 Relife 工作区：

```text
data/transactions/
portfolio_viewer/scripts/generate_portfolio_dashboard.py
portfolio_viewer/public/data/portfolio.json
```

点击 Activity Bar 中的 `Re` 图标，再点击侧栏中的“打开投资组合”；也可以在命令面板运行：

- `Relife: 打开投资组合`
- `Relife: 立即更新数据`

投资组合页面顶部的“立即更新”按钮会在更新期间禁用；更新失败时保留当前数据并显示错误。

扩展激活时会立即更新一次。VS Code 保持打开时，还会按上海时间自动更新：

| 市场 | 时间 |
| --- | --- |
| A 股 | 周一至周五 09:15、15:15 |
| 美股 | 周一至周五 21:15；周二至周六 05:15 |

休市日不额外判断，行情源会返回最近交易日数据。

## 环境

- VS Code 1.95 或更高版本
- `/home/clannad/miniforge3/envs/istorm_rag_gpu/bin/python`
- Python 环境需要现有投资组合生成器使用的行情依赖

扩展只支持当前 Relife 仓库结构，并应安装在能够访问工作区和上述 Python 环境的 VS Code/WSL 扩展宿主中。
由于扩展会运行仓库内的 Python 生成器，VS Code 必须先信任当前工作区。

## 隐私

VSIX 只包含扩展代码和已构建的界面资源，不包含交易 CSV、`portfolio.json`、环境文件或 API 密钥。更新行情时沿用现有生成器，仅向行情服务商发送标的代码和日期范围。

## 开发

```bash
npm run vscode:test
npm run vscode:build
npm run vscode:package
```

许可证：Apache-2.0。
