# Relife Portfolio

在 VS Code 中查看当前 Relife 工作区的 A 股与美股投资组合。扩展复用仓库现有交易流水、Python 账本和 React 看板，不会把账户数据打包或上传到 Marketplace。

## 使用

### 安装 0.3.0

在打开 Relife 工作区的 VS Code 窗口中：

1. 打开“扩展”视图（`Ctrl+Shift+X`）。
2. 点击扩展视图右上角的 `…`，选择“从 VSIX 安装…”。
3. 选择 `portfolio_viewer/relife-portfolio-0.3.0.vsix`。
4. 安装完成后按提示重新加载窗口，并确认 `Relife Portfolio` 的版本为 `0.3.0`。

也可以在仓库根目录执行：

```bash
code --install-extension portfolio_viewer/relife-portfolio-0.3.0.vsix --force
```

如果 Relife 工作区通过 WSL 打开，请在该 WSL 窗口中安装，使扩展运行在能够访问仓库和 Python 环境的扩展宿主中。

### 投资组合与观察列表

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

“打开投资组合”下方是仓库共享的观察列表。右键分组或标的可以：

- 新建、重命名和删除分组；
- 添加、移动和删除标的；
- 启用或停用提醒；
- 设置日线、周线、月线周期、布林窗口和标准差倍数。

配置保存在 `portfolio_viewer/data/watchlist.json`。同一市场的同一标的只能出现一次；删除非空分组时会要求确认。

### 布林候选提醒

扩展读取约三年日线，按所选周期取当前周期的最新收盘价。最近 `window` 个周期的总体标准差下轨为：

```text
下轨 = 均值 - standardDeviations × 总体标准差
```

收盘价低于下轨时，VS Code 显示 `[策略触发]` 候选通知。同一标的、规则参数和日/周/月周期桶只提醒一次；这只表示进入候选池，不构成买入信号，也不会生成订单。

### 飞书通知

先在飞书群中添加“自定义机器人”，再从命令面板运行：

- `Relife: 配置或停用飞书通知`
- `Relife: 发送飞书测试消息`

Webhook 与可选签名密钥保存在 VS Code SecretStorage，不写入仓库。建议在飞书机器人中启用签名校验；使用关键词校验时，关键词可设置为“策略触发”。发送失败会写入 `Relife Portfolio` 输出日志，并在下次检查时重试飞书，不重复 VS Code 弹窗。

扩展激活和手动“立即更新”时会分别更新投资组合并检查观察策略。VS Code 保持打开时，还会按上海时间自动执行：

| 市场 | 时间 |
| --- | --- |
| A 股 | 周一至周五 09:15、15:15 |
| 美股 | 周一至周五 21:15；周二至周六 05:15 |

休市日不额外判断，行情源会返回最近交易日数据。
关闭 VS Code 或未打开 Relife 工作区时不会检查策略，也不会补发关闭期间错过的提醒。

## 环境

- VS Code 1.95 或更高版本
- `/home/clannad/miniforge3/envs/istorm_rag_gpu/bin/python`
- Python 环境需要现有投资组合生成器使用的行情依赖

扩展只支持当前 Relife 仓库结构，并应安装在能够访问工作区和上述 Python 环境的 VS Code/WSL 扩展宿主中。
由于扩展会运行仓库内的 Python 生成器，VS Code 必须先信任当前工作区。

## 隐私

VSIX 只包含扩展代码和已构建的界面资源，不包含交易 CSV、`portfolio.json`、Webhook、签名密钥、环境文件或 API 密钥。更新与观察行情时，仅向现有行情服务商发送标的代码和日期范围。

## 开发

```bash
npm run vscode:test
npm run vscode:build
npm run vscode:package:test
npm run vscode:package
```

许可证：Apache-2.0。
