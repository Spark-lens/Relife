const { spawn } = require("node:child_process");
const { randomBytes } = require("node:crypto");
const { existsSync } = require("node:fs");
const fs = require("node:fs/promises");
const path = require("node:path");
const vscode = require("vscode");
const { webviewHtml } = require("./src/webview-html.cjs");

const SOURCE_KEY = "relife.sources.v1";
const WATCHLIST_KEY = "relife.watchlist.v1";
const STARTUP_KEY = "relife.lastStartupRefreshDay";

async function activate(context) {
  const sourceState = await import("./src/extension-state.mjs");
  const watchlistState = await import("./src/watchlist-state.mjs");
  const output = vscode.window.createOutputChannel("Relife");
  const webviews = new Set();
  let panel;
  let strategyPanel;
  let refreshing;
  let sources = context.workspaceState.get(SOURCE_KEY) || sourceState.defaultSources();
  let watchlist = context.workspaceState.get(WATCHLIST_KEY) || watchlistState.defaultWatchlist();
  let snapshot = await loadInitialSnapshot(context, output);

  const sendState = (webview) => {
    webview.postMessage({ type: "snapshot", data: snapshot });
    webview.postMessage({ type: "watchlist", data: watchlist });
    webview.postMessage({ type: "source-status", data: sources });
  };
  const broadcast = (message) => webviews.forEach((webview) => webview.postMessage(message));

  async function runEngine(payload) {
    const configuredPython = vscode.workspace.getConfiguration("relife").get("pythonPath", "").trim();
    const condaPython = process.env.CONDA_PREFIX
      ? path.join(process.env.CONDA_PREFIX, process.platform === "win32" ? "python.exe" : "bin/python")
      : "";
    const python = configuredPython || (condaPython && existsSync(condaPython)
      ? condaPython
      : process.platform === "win32" ? "python" : "python3");
    const cli = context.asAbsolutePath(path.join("python", "relife_cli.py"));
    return new Promise((resolve, reject) => {
      const process = spawn(python, [cli], { cwd: context.extensionPath, stdio: ["pipe", "pipe", "pipe"] });
      let stdout = "";
      let stderr = "";
      const timer = setTimeout(() => process.kill(), 120000);
      process.stdout.on("data", (chunk) => { stdout += chunk; });
      process.stderr.on("data", (chunk) => { stderr += chunk; });
      process.on("error", (error) => reject(new Error(error.code === "ENOENT"
        ? `找不到 Python 解释器“${python}”。请在设置 relife.pythonPath 中指定可用的 Python 3.12 路径`
        : error.message)));
      process.on("close", (code) => {
        clearTimeout(timer);
        try {
          const response = JSON.parse(stdout);
          if (code === 0 && response.ok) resolve(response.result);
          else reject(new Error(response.error?.message || stderr || `Python 退出码 ${code}`));
        } catch {
          reject(new Error(stderr || stdout || `Python 退出码 ${code}`));
        }
      });
      process.stdin.end(JSON.stringify(payload));
    });
  }

  async function refresh() {
    if (refreshing) return refreshing;
    refreshing = (async () => {
      broadcast({ type: "refresh-status", status: "loading" });
      output.appendLine(`[${new Date().toISOString()}] 开始刷新`);
      try {
        const next = await runEngine({
          command: "build-snapshot", sources, watchlist,
          samplePath: context.asAbsolutePath(path.join("resources", "sample", "portfolio-snapshot.json")),
        });
        snapshot = next;
        await writeSnapshot(context, snapshot);
        broadcast({ type: "snapshot", data: snapshot });
        broadcast({ type: "refresh-status", status: "success", generatedAt: snapshot.generatedAt });
        output.appendLine(`[${new Date().toISOString()}] 刷新完成`);
        return snapshot;
      } catch (error) {
        const message = error?.message || String(error);
        output.appendLine(`[${new Date().toISOString()}] 刷新失败：${message}`);
        broadcast({ type: "refresh-status", status: "error", message, staleAt: snapshot?.generatedAt || null });
        broadcast({ type: "error", message });
        vscode.window.showErrorMessage(`Relife 刷新失败：${message}`);
        throw error;
      } finally {
        refreshing = undefined;
      }
    })();
    return refreshing;
  }

  async function selectSource(market) {
    const picked = await vscode.window.showOpenDialog({
      canSelectFiles: true, canSelectFolders: false, canSelectMany: false,
      filters: { CSV: ["csv"] }, title: market === "us" ? "选择美股交易文件" : "选择 A 股交割单",
    });
    if (!picked?.[0]) return;
    const file = picked[0].fsPath;
    try {
      sourceState.sourceFromFile(market, file);
      const validation = await runEngine({ command: "validate-source", market, path: file });
      sources = { ...sources, [market]: sourceState.sourceFromFile(market, file) };
      await context.workspaceState.update(SOURCE_KEY, sources);
      broadcast({ type: "source-status", data: sources, validation });
      vscode.window.showInformationMessage(`Relife 已验证 ${validation.recordCount} 条记录`);
      await refresh();
    } catch (error) {
      const message = error?.message || String(error);
      broadcast({ type: "error", message });
      vscode.window.showErrorMessage(`Relife 数据源无效：${message}`);
    }
  }

  async function resetSources(market) {
    const defaults = sourceState.defaultSources();
    sources = market ? { ...sources, [market]: defaults[market] } : defaults;
    await context.workspaceState.update(SOURCE_KEY, sources);
    broadcast({ type: "source-status", data: sources });
    await refresh();
  }

  const safeText = (value, label, limit) => {
    if (typeof value !== "string" || !value.trim() || value.trim().length > limit) throw new Error(`${label}长度应为 1-${limit}`);
    return value.trim();
  };
  const safeSymbol = (value) => {
    const symbol = safeText(value, "代码", 24).toUpperCase();
    if (!/^[A-Z0-9.^:_-]+$/.test(symbol)) throw new Error("代码包含不支持的字符");
    return symbol;
  };

  async function updateWatchlist(message) {
    try {
      const { type } = message;
      if (type === "watchlist-add-category") watchlist = watchlistState.addCategory(watchlist, safeText(message.name, "分类名称", 40));
      if (type === "watchlist-rename-category") watchlist = watchlistState.renameCategory(watchlist, message.categoryId, safeText(message.name, "分类名称", 40));
      if (type === "watchlist-delete-category") watchlist = watchlistState.removeCategory(watchlist, message.categoryId);
      if (type === "watchlist-move-category") watchlist = watchlistState.moveCategory(watchlist, message.categoryId, message.offset === -1 ? -1 : 1);
      if (type === "watchlist-add-symbol") watchlist = watchlistState.addSymbol(watchlist, message.categoryId, {
        market: message.symbol?.market === "cn" ? "cn" : "us", symbol: safeSymbol(message.symbol?.symbol),
        name: safeText(message.symbol?.name || message.symbol?.symbol, "名称", 80), note: String(message.symbol?.note || "").slice(0, 120),
      });
      if (type === "watchlist-edit-symbol") watchlist = watchlistState.editSymbol(watchlist, message.key, {
        market: message.symbol?.market === "cn" ? "cn" : "us", symbol: safeSymbol(message.symbol?.symbol),
        name: safeText(message.symbol?.name || message.symbol?.symbol, "名称", 80), note: String(message.symbol?.note || "").slice(0, 120),
      });
      if (type === "watchlist-delete-symbol") watchlist = watchlistState.removeSymbol(watchlist, message.key);
      if (type === "watchlist-move-symbol") watchlist = watchlistState.moveSymbol(watchlist, message.key, message.targetCategoryId, Number(message.targetIndex) || 0);
      await context.workspaceState.update(WATCHLIST_KEY, watchlist);
      broadcast({ type: "watchlist", data: watchlist });
    } catch (error) {
      broadcast({ type: "error", message: error.message, code: error.code, existingCategoryId: error.existingCategoryId });
    }
  }

  async function openStrategyFile() {
    const folder = vscode.workspace.workspaceFolders?.[0];
    if (!folder) return vscode.window.showWarningMessage("请先打开 Relife 工作区");
    const uri = vscode.Uri.joinPath(folder.uri, "strategies", "bollinger_band_reversion", "run.py");
    try { await vscode.window.showTextDocument(uri); } catch { vscode.window.showWarningMessage("未找到布林带策略入口 run.py"); }
  }

  async function handleMessage(message, webview) {
    if (!sourceState.allowedRequest(message)) return;
    if (message.type === "ready") return sendState(webview);
    if (message.type === "refresh") return refresh().catch(() => {});
    if (message.type === "select-source") return selectSource(message.market === "cn" ? "cn" : "us");
    if (message.type === "reset-source") return resetSources(message.market);
    if (message.type === "open-portfolio") return openPortfolio();
    if (message.type === "open-strategy") return openStrategy();
    if (message.type === "open-strategy-file") return openStrategyFile();
    if (message.type.startsWith("watchlist-")) return updateWatchlist(message);
  }

  function configureWebview(webview, kind) {
    webview.options = { enableScripts: true, localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, "dist")] };
    const script = webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, "dist", "webview.js"));
    const style = webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, "dist", "webview.css"));
    webview.html = webviewHtml({ kind, script, style, cspSource: webview.cspSource, nonce: randomBytes(16).toString("base64") });
    webviews.add(webview);
    webview.onDidReceiveMessage((message) => handleMessage(message, webview), undefined, context.subscriptions);
  }

  function openPortfolio() {
    if (panel) return panel.reveal(vscode.ViewColumn.One);
    panel = vscode.window.createWebviewPanel("relife.portfolio", "Relife · 美股持仓", vscode.ViewColumn.One, { enableScripts: true, retainContextWhenHidden: true });
    configureWebview(panel.webview, "portfolio");
    panel.onDidDispose(() => { webviews.delete(panel.webview); panel = undefined; });
  }

  function openStrategy() {
    if (strategyPanel) return strategyPanel.reveal(vscode.ViewColumn.One);
    strategyPanel = vscode.window.createWebviewPanel("relife.strategy", "Relife · 布林带策略", vscode.ViewColumn.One, { enableScripts: true });
    configureWebview(strategyPanel.webview, "strategy");
    strategyPanel.onDidDispose(() => { webviews.delete(strategyPanel.webview); strategyPanel = undefined; });
  }

  const sidebarProvider = {
    resolveWebviewView(view) {
      configureWebview(view.webview, "sidebar");
      view.onDidDispose(() => webviews.delete(view.webview));
    },
  };
  context.subscriptions.push(
    output,
    vscode.window.registerWebviewViewProvider("relife.sidebar", sidebarProvider, { webviewOptions: { retainContextWhenHidden: true } }),
    vscode.commands.registerCommand("relife.openPortfolio", openPortfolio),
    vscode.commands.registerCommand("relife.openStrategy", openStrategy),
    vscode.commands.registerCommand("relife.selectUsSource", () => selectSource("us")),
    vscode.commands.registerCommand("relife.selectCnSource", () => selectSource("cn")),
    vscode.commands.registerCommand("relife.resetSources", () => resetSources()),
    vscode.commands.registerCommand("relife.refresh", () => refresh().catch(() => {})),
  );
  const today = new Date().toISOString().slice(0, 10);
  if (context.workspaceState.get(STARTUP_KEY) !== today) {
    await context.workspaceState.update(STARTUP_KEY, today);
    setTimeout(() => refresh().catch(() => {}), 1000);
  }
}

async function loadInitialSnapshot(context, output) {
  const cache = context.storageUri && vscode.Uri.joinPath(context.storageUri, "snapshot.json");
  if (cache) {
    try { return JSON.parse(await fs.readFile(cache.fsPath, "utf8")); } catch (error) { output.appendLine(`读取缓存失败，改用示例：${error.message}`); }
  }
  return JSON.parse(await fs.readFile(context.asAbsolutePath(path.join("resources", "sample", "portfolio-snapshot.json")), "utf8"));
}

async function writeSnapshot(context, snapshot) {
  if (!context.storageUri) return;
  await fs.mkdir(context.storageUri.fsPath, { recursive: true });
  const target = path.join(context.storageUri.fsPath, "snapshot.json");
  const temporary = `${target}.${process.pid}.${Date.now()}.tmp`;
  await fs.writeFile(temporary, JSON.stringify(snapshot), "utf8");
  await fs.rename(temporary, target);
}

function deactivate() {}

module.exports = { activate, deactivate, writeSnapshot };
