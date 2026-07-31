# VS Code Activity Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Relife Portfolio 0.2.0 增加 VS Code Activity Bar 入口和 Webview 内立即更新交互，并生成可发布 VSIX。

**Architecture:** 使用 VS Code 原生 `viewsContainers.activitybar`、`views` 与 TreeDataProvider 提供单入口；入口继续调用现有打开命令。Webview 通过 `refresh` 消息调用现有 single-flight 更新函数，并用宿主回传消息管理按钮状态和错误提示。

**Tech Stack:** VS Code Extension API、CommonJS、React 19、Vite 8、Node.js 内置测试运行器。

## Global Constraints

- 不建立隔离 worktree，在当前 `*-xql` 分支修改。
- 不增加第三方依赖。
- 保留 `relifePortfolio.open` 与 `relifePortfolio.refresh` 命令。
- 版本号升级为 `0.2.0`，Marketplace 图标继续使用 `media/icon.png`。
- 更新失败必须保留旧组合数据，并恢复 Webview 按钮。

---

### Task 1: Activity Bar 清单与单入口 Tree View

**Files:**
- Create: `portfolio_viewer/media/activitybar.svg`
- Create: `portfolio_viewer/vscode/sidebar.cjs`
- Modify: `portfolio_viewer/package.json`
- Modify: `portfolio_viewer/package-lock.json`
- Modify: `portfolio_viewer/vscode/extension.cjs`
- Modify: `portfolio_viewer/tests/vscode-package.test.mjs`
- Create: `portfolio_viewer/tests/vscode-sidebar.test.mjs`

**Interfaces:**
- Produces: `createPortfolioTreeDataProvider(vscode)`，Tree Item command 为 `relifePortfolio.open`。
- Consumes: VS Code 的 `TreeItem`、`TreeItemCollapsibleState.None`、`window.registerTreeDataProvider`。

- [ ] **Step 1: 写清单和侧栏行为测试**

```js
assert.equal(packageJson.version, "0.2.0");
assert.equal(container.icon, "media/activitybar.svg");
assert.equal(item.command.command, "relifePortfolio.open");
```

- [ ] **Step 2: 运行测试并确认因 0.2.0 清单和 provider 尚不存在而失败**

Run: `node --test tests/vscode-package.test.mjs tests/vscode-sidebar.test.mjs`
Expected: FAIL，指出版本仍为 0.1.0 或找不到 `sidebar.cjs`。

- [ ] **Step 3: 写最小清单、SVG、provider 和注册代码**

```js
function createPortfolioTreeDataProvider(vscode) {
  const item = new vscode.TreeItem(
    "打开投资组合",
    vscode.TreeItemCollapsibleState.None,
  );
  item.command = {
    command: "relifePortfolio.open",
    title: "打开投资组合",
  };
  return { getTreeItem: (treeItem) => treeItem, getChildren: () => [item] };
}
```

- [ ] **Step 4: 运行侧栏和清单测试并确认通过**

Run: `node --test tests/vscode-package.test.mjs tests/vscode-sidebar.test.mjs`
Expected: PASS。

### Task 2: Webview 更新消息协议与按钮状态

**Files:**
- Create: `portfolio_viewer/vscode/webview-state.mjs`
- Create: `portfolio_viewer/vscode/refresh-control.mjs`
- Create: `portfolio_viewer/vscode/loading-view.mjs`
- Modify: `portfolio_viewer/vscode/webview.tsx`
- Modify: `portfolio_viewer/vscode/extension.cjs`
- Create: `portfolio_viewer/tests/vscode-webview.test.mjs`
- Create: `portfolio_viewer/tests/vscode-refresh-control.test.mjs`
- Create: `portfolio_viewer/tests/vscode-extension.test.mjs`
- Modify: `portfolio_viewer/app/components/PortfolioDashboard.tsx`
- Modify: `portfolio_viewer/app/globals.css`
- Modify: `portfolio_viewer/package.json`

**Interfaces:**
- Webview → host: `{ type: "refresh" }`。
- Host → Webview: `{ type: "refresh-start" }`、`{ type: "refresh-success" }`、`{ type: "refresh-error", message: string }`、现有 `{ type: "portfolio", data }`。
- Produces: `reduceWebviewState(state, message)` 与 `createRefreshRequest()`。
- 顶部栏通过可选 `toolbar` 节点承载可渲染测试的 `RefreshControl`，不覆盖原有状态区。

- [ ] **Step 1: 写 reducer 和出站消息测试**

```js
assert.deepEqual(requestRefresh(), { type: "refresh" });
assert.equal(reduceWebviewState(idle, { type: "refresh-start" }).refreshing, true);
assert.equal(reduceWebviewState(refreshing, { type: "refresh-error", message: "失败" }).error, "失败");
```

- [ ] **Step 2: 运行测试并确认因协议模块尚不存在而失败**

Run: `node --test tests/vscode-webview.test.mjs`
Expected: FAIL，找不到 `webview-state.mjs`。

- [ ] **Step 3: 实现最小 reducer、按钮和宿主消息处理**

```js
if (message?.type === "refresh") {
  postToWebview({ type: "refresh-start" });
  refresh().then(
    () => postToWebview({ type: "refresh-success" }),
    (error) => postToWebview({ type: "refresh-error", message: error.message }),
  );
}
```

- [ ] **Step 4: 运行 Webview、扩展核心和构建测试**

Run: `npm run vscode:test && npm run vscode:build && npm run vscode:package:test`
Expected: PASS。

### Task 3: 文档、隐私检查与 0.2.0 打包

**Files:**
- Modify: `portfolio_viewer/README.md`
- Create: `portfolio_viewer/relife-portfolio-0.2.0.vsix`

**Interfaces:**
- Produces: 可由 Publisher `clannad0710` 发布的 `relife-portfolio-0.2.0.vsix`。

- [ ] **Step 1: 更新 README 使用说明**

补充 Activity Bar 入口和 Webview 内“立即更新”按钮，不更改环境要求。

- [ ] **Step 2: 运行完整扩展验证**

Run: `npm run vscode:test && npm run vscode:build && npm run vscode:package:test`
Expected: 全部 PASS，构建目录仅含 `webview.css`、`webview.js`。

- [ ] **Step 3: 打包并验证 VSIX 隐私内容**

Run: `npm run vscode:package`
Expected: 生成 `relife-portfolio-0.2.0.vsix`，包内不含交易 CSV、`portfolio.json`、环境文件或密钥。

- [ ] **Step 4: 检查最终差异**

Run: `git diff -- portfolio_viewer docs/superpowers/plans/2026-07-31-vscode-activity-bar.md`
Expected: 仅包含本设计所需源文件、测试、文档和版本变更；既有用户改动不被覆盖。
