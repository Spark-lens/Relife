# Portfolio Viewer VS Code Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前仓库的 `portfolio_viewer` 发布为可从 VS Code Marketplace 安装的工作区扩展，并在固定上海时间更新现有交易记录生成的组合数据。

**Architecture:** 扩展宿主用 Node 标准库定位 Relife 工作区、调用现有 Python 生成器并读取 `portfolio.json`。现有 React 看板通过独立 Vite 入口打包进原生 Webview；调度器仅在 VS Code 打开时运行。

**Tech Stack:** VS Code Extension API、Node.js 22、React 19、Vite 8、Python 3.12。

## Global Constraints

- 不建立隔离 worktree，在 `0.1-xql` 分支修改。
- Python 使用 `/home/clannad/miniforge3/envs/istorm_rag_gpu/bin/python`。
- Node 使用 `/home/clannad/.nvm/versions/node/v22.19.0/bin/node`。
- 时区固定为 `Asia/Shanghai`。
- A 股在周一至周五 09:15、15:15 更新。
- 美股在周一至周五 21:15、周二至周六 05:15 更新。
- 扩展激活时立即更新一次，重叠刷新合并为同一个任务。
- 不添加交易所节假日日历；休市日允许取得最近交易日数据。
- 不启动 Vinext/Cloudflare 服务，不新增运行时依赖。
- VSIX 不包含交易 CSV、`public/data/portfolio.json`、环境文件或密钥。

---

### Task 1: 固定上海时间调度器

**Files:**
- Create: `portfolio_viewer/vscode/schedule.cjs`
- Test: `portfolio_viewer/tests/vscode-schedule.test.mjs`

**Interfaces:**
- Produces: `nextUpdateAt(now: Date): Date`
- Produces: `scheduleUpdates(runUpdate: () => Promise<void>, timers?): { dispose(): void }`

- [ ] 写入四个边界时间和周末跳转的失败测试。
- [ ] 运行 `node --test portfolio_viewer/tests/vscode-schedule.test.mjs`，确认因模块缺失失败。
- [ ] 用 `Date` 和固定 UTC+8 实现最小调度逻辑。
- [ ] 重跑测试并确认通过。

### Task 2: 工作区识别和组合数据刷新

**Files:**
- Create: `portfolio_viewer/vscode/portfolio.cjs`
- Test: `portfolio_viewer/tests/vscode-portfolio.test.mjs`

**Interfaces:**
- Produces: `findRelifeRoot(folderPaths: string[]): string | null`
- Produces: `loadPortfolio(repositoryRoot: string): Promise<object>`
- Produces: `refreshPortfolio(repositoryRoot: string, run?): Promise<object>`

- [ ] 写入仓库标志文件、Python 命令参数和失败保留旧文件的测试。
- [ ] 运行 `node --test portfolio_viewer/tests/vscode-portfolio.test.mjs`，确认因模块缺失失败。
- [ ] 用 `child_process.execFile` 调用现有生成器，成功后解析 JSON。
- [ ] 重跑测试并确认通过。

### Task 3: VS Code 扩展宿主

**Files:**
- Create: `portfolio_viewer/vscode/extension.cjs`

**Interfaces:**
- Consumes: `nextUpdateAt`, `scheduleUpdates`, `findRelifeRoot`, `loadPortfolio`, `refreshPortfolio`
- Produces: commands `relifePortfolio.open` and `relifePortfolio.refresh`

- [ ] 注册打开看板和立即更新两个命令。
- [ ] 在 `onStartupFinished` 激活后识别工作区、加载旧数据、立即刷新并启动调度器。
- [ ] 使用单个 Promise 合并重叠更新。
- [ ] 更新失败时保留旧数据，写入 OutputChannel 并显示错误通知。
- [ ] 用 `node --check` 验证三个 CJS 文件语法。

### Task 4: React Webview

**Files:**
- Create: `portfolio_viewer/vscode/webview.tsx`
- Create: `portfolio_viewer/vscode/vite.config.ts`

**Interfaces:**
- Consumes: `{ type: "portfolio", data: DashboardPayload }`
- Produces: `{ type: "ready" }`

- [ ] 复用 `PortfolioDashboard` 和 `app/globals.css`。
- [ ] Webview ready 后由扩展发送当前数据，刷新成功后再次发送。
- [ ] 添加只允许本扩展资源的 CSP。
- [ ] 用 Vite 生成固定的 `vscode/dist/webview.js` 和 `webview.css`。
- [ ] 运行 Webview 构建和现有前端测试。

### Task 5: 扩展清单、文档和 VSIX

**Files:**
- Modify: `portfolio_viewer/package.json`
- Modify: `portfolio_viewer/package-lock.json`
- Create: `portfolio_viewer/README.md`
- Modify: `README.md`

**Interfaces:**
- Produces: Marketplace extension `relife.relife-portfolio`

- [ ] 添加 Publisher、VS Code engine、workspace extensionKind、激活事件和命令清单。
- [ ] 添加仅包含扩展宿主、Webview 构建产物和 README 的 `files` 白名单。
- [ ] 添加 `vscode:build`、`vscode:test`、`vscode:package` 脚本。
- [ ] 安装 `@vscode/vsce` 开发依赖前征求用户同意。
- [ ] 运行 Node 测试、前端测试、Python 测试、lint 和 Webview 构建。
- [ ] 生成 VSIX，检查包内不存在交易数据、生成 JSON、`.env` 或密钥。
- [ ] 仅暂存本计划与插件相关文件并提交。
- [ ] 创建/登录 Marketplace Publisher，发布 VSIX 并验证 Marketplace 安装页。
