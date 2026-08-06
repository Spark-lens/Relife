const { spawn } = require("node:child_process");
const { randomBytes } = require("node:crypto");
const { existsSync, readdirSync } = require("node:fs");
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

  // 上海时区（Asia/Shanghai）本地时间字符串，避免 toISOString 把它转回 UTC
  const ts = () => {
    const fmt = new Intl.DateTimeFormat("zh-CN", {
      timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
    });
    const parts = Object.fromEntries(fmt.formatToParts(new Date()).filter((p) => p.type !== "literal").map((p) => [p.type, p.value]));
    return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}:${parts.second}`;
  };
  // 简单日志：中文文本行，[时间] 消息
  const info = (msg) => output.appendLine(`[${ts()}] ${msg}`);
  // 复杂数据：中文文本行 + 紧跟一行 JSON
  const json = (msg, data) => {
    output.appendLine(`[${ts()}] ${msg}`);
    output.appendLine(JSON.stringify(data));
  };

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

  function findCondaEnvs() {
    // 从多个可能的 conda 根目录动态发现 envs 下的 python 解释器
    const candidates = [];
    const roots = [
      process.env.CONDA_PREFIX ? path.dirname(process.env.CONDA_PREFIX) : "",
      process.env.HOME ? path.join(process.env.HOME, "miniforge3") : "",
      process.env.HOME ? path.join(process.env.HOME, "anaconda3") : "",
      process.env.HOME ? path.join(process.env.HOME, "miniconda3") : "",
      "/opt/conda",
      "/usr/local/miniconda3",
    ].filter(Boolean);
    const seen = new Set();
    for (const root of roots) {
      const envsDir = path.join(root, "envs");
      try {
        for (const name of readdirSync(envsDir)) {
          const py = path.join(envsDir, name, process.platform === "win32" ? "python.exe" : "bin/python");
          if (existsSync(py) && !seen.has(py)) {
            seen.add(py);
            candidates.push({ name, py });
          }
        }
      } catch { /* 目录不存在则跳过 */ }
    }
    return candidates;
  }

  function resolvePython() {
    // 用户在 VS Code 设置里显式指定的路径优先级最高
    const configured = vscode.workspace.getConfiguration("relife").get("pythonPath", "").trim();
    if (configured && existsSync(configured)) return configured;
    // 动态发现 conda envs，优先匹配 istorm_rag_gpu（用户当前主力环境）
    for (const env of findCondaEnvs()) {
      if (env.name === "istorm_rag_gpu") return env.py;
    }
    // 其次用当前激活的 conda 环境
    if (process.env.CONDA_PREFIX) {
      const condaPython = path.join(
        process.env.CONDA_PREFIX,
        process.platform === "win32" ? "python.exe" : "bin/python",
      );
      if (existsSync(condaPython)) return condaPython;
    }
    // 再借助 ms-python 扩展推断工作区选中的解释器
    try {
      const pyExt = vscode.extensions.getExtension("ms-python.python");
      if (pyExt?.isActive) {
        const execDetails = pyExt.exports?.settings?.getExecutionDetails?.();
        if (execDetails?.execCommand?.[0] && existsSync(execDetails.execCommand[0])) {
          return execDetails.execCommand[0];
        }
      }
    } catch { /* Python 扩展不可用时静默跳过 */ }
    if (configured) return configured;
    return process.platform === "win32" ? "python" : "python3";
  }

  async function runEngine(payload) {
    const python = resolvePython();
    const command = payload.command || "unknown";
    const label = command === "validate-source" ? `校验源文件(${payload.market})` : "构建快照";
    info(`${label} → 启动引擎 | python=${python.split("/").slice(-2).join("/")}`);
    const cli = context.asAbsolutePath(path.join("python", "relife_cli.py"));
    const startedAt = Date.now();
    return new Promise((resolve, reject) => {
      const proc = spawn(python, [cli], { cwd: context.extensionPath, stdio: ["pipe", "pipe", "pipe"] });
      let stdout = "";
      let stderr = "";
      const timer = setTimeout(() => proc.kill(), 120000);
      // yfinance 良性噪音：quoteSummary 404 / possibly delisted，不打印到 Output
      const isYFinanceNoise = (text) =>
        /HTTP Error 404.*quoteSummary/.test(text) ||
        /possibly delisted/.test(text);
      proc.stdout.on("data", (chunk) => {
        stdout += chunk;
        const text = chunk.toString();
        if (!isYFinanceNoise(text)) output.append(text);
      });
      proc.stderr.on("data", (chunk) => {
        stderr += chunk;
        const text = chunk.toString();
        if (!isYFinanceNoise(text)) output.append(text);
      });
      proc.on("error", (error) => reject(new Error(error.code === "ENOENT"
        ? `找不到 Python 解释器"${python}"。请在设置 relife.pythonPath 中指定可用的 Python 3.12 路径`
        : error.message)));
      proc.on("close", (code) => {
        clearTimeout(timer);
        const elapsed = Date.now() - startedAt;
        try {
          const response = JSON.parse(stdout);
          if (code === 0 && response.ok) {
            info(`${label} → 完成 (${elapsed}ms)`);
            resolve(response.result);
          } else {
            const err = new Error(response.error?.message || stderr || `Python 退出码 ${code}`);
            err.type = response.error?.type || "UnknownError";
            err.errors = response.result?.errors;
            info(`${label} → 失败 (${elapsed}ms) | ${err.type}: ${err.message}`);
            if (Array.isArray(err.errors) && err.errors.length) {
              json(`  数据获取错误明细 (${err.errors.length} 条)`, err.errors);
            }
            reject(err);
          }
        } catch {
          const message = stderr || stdout || `Python 退出码 ${code}`;
          if (/ModuleNotFoundError|No module named/i.test(message)) {
            reject(new Error(`${message}\n请在 Python 环境"${python}"中安装依赖：pip install yfinance akshare`));
          } else {
            reject(new Error(message));
          }
        }
      });
      proc.stdin.end(JSON.stringify(payload));
    });
  }

  // 按现有优先级从候选 conda 环境中选定一个作为首选解释器
  function pickPreferredEnv(envs) {
    if (!envs.length) return undefined;
    const istorm = envs.find((e) => e.name === "istorm_rag_gpu");
    if (istorm) return istorm;
    if (process.env.CONDA_PREFIX) {
      const active = envs.find((e) => process.env.CONDA_PREFIX && e.py.startsWith(process.env.CONDA_PREFIX));
      if (active) return active;
    }
    return envs[0];
  }

  // 插件激活时一次性检测 conda 环境：命中则自动回填配置，未命中则弹窗引导
  async function detectAndApplyPythonEnv() {
    const envs = findCondaEnvs();
    const chosen = pickPreferredEnv(envs);
    if (chosen) {
      const config = vscode.workspace.getConfiguration("relife");
      const current = (config.get("pythonPath", "") || "").trim();
      info(`检测到 conda 环境: ${chosen.name} (${chosen.py})`);
      if (envs.length > 1) {
        const others = envs.filter((e) => e.name !== chosen.name).map((e) => e.name).join(", ");
        info(`  其他可用环境: ${others}`);
      }
      if (current !== chosen.py) {
        await config.update("pythonPath", chosen.py, vscode.ConfigurationTarget.Workspace);
        info(`  已自动填写 relife.pythonPath = ${chosen.py}`);
      }
      return;
    }
    info("未检测到可用的 conda 环境，请在设置中指定 Python 解释器");
    const choice = await vscode.window.showWarningMessage(
      "Relife 未检测到 conda 环境，数据引擎将无法运行。请在设置中指定 Python 解释器。",
      "选择解释器",
    );
    if (choice !== "选择解释器") return;
    const quickPicks = envs.length
      ? envs.map((e) => ({ label: e.name, description: e.py, py: e.py }))
      : [{ label: "浏览文件系统...", py: null }];
    const picked = await vscode.window.showQuickPick(quickPicks, { placeHolder: "选择 Python 解释器" });
    if (!picked) return;
    let py = picked.py;
    if (!py) {
      const file = await vscode.window.showOpenDialog({
        canSelectFiles: true, canSelectFolders: false, canSelectMany: false,
        filters: { "Python": process.platform === "win32" ? ["exe"] : ["*"] },
        title: "选择 Python 解释器",
      });
      if (!file?.[0]) return;
      py = file[0].fsPath;
    }
    await vscode.workspace.getConfiguration("relife").update("pythonPath", py, vscode.ConfigurationTarget.Workspace);
    info(`已手动指定 Python 解释器: ${py}`);
  }

  async function refresh() {
    if (refreshing) return refreshing;
    refreshing = (async () => {
      const startedAt = Date.now();
      broadcast({ type: "refresh-status", status: "loading" });
      const python = resolvePython();
      const markets = Object.keys(sources);
      const parts = markets.map((m) => `${m}(${sources[m]?.mode}, ${sources[m]?.symbols?.length || 0}个标的)`).join(", ");
      info(`开始刷新 | ${parts} | python=${python.split("/").slice(-2).join("/")}`);
      try {
        for (const market of markets) {
          if (sources[market]?.mode === "directory" && sources[market]?.directory) {
            await runEngine({ command: "validate-source", market, path: sources[market].directory });
          }
        }
        const next = await runEngine({
          command: "build-snapshot", sources, watchlist,
          samplePath: context.asAbsolutePath(path.join("resources", "sample", "portfolio-snapshot.json")),
        });
        const elapsed = Date.now() - startedAt;
        const countParts = markets.map((m) => {
          const mk = next[m] || {};
          return `${m}:持仓${(mk.holdings || []).length}/行情${(mk.prices || []).length}`;
        }).join(", ");
        snapshot = next;
        await writeSnapshot(context, snapshot);
        broadcast({ type: "snapshot", data: snapshot });
        broadcast({ type: "refresh-status", status: "success", generatedAt: snapshot.generatedAt });
        info(`刷新完成 (${(elapsed / 1000).toFixed(1)}s) | ${countParts}`);
        if (Array.isArray(next.errors) && next.errors.length) {
          json(`⚠ 数据获取警告 (${next.errors.length} 条)`, next.errors.map((e) => ({ market: e.market, symbol: e.symbol, message: e.message })));
        }
        return snapshot;
      } catch (error) {
        const message = error?.message || String(error);
        const type = error?.type || "Error";
        const elapsed = Date.now() - startedAt;
        info(`刷新失败 (${(elapsed / 1000).toFixed(1)}s) | ${type}: ${message}`);
        if (Array.isArray(error?.errors) && error.errors.length) {
          json(`  数据获取错误明细 (${error.errors.length} 条)`, error.errors.map((e) => ({ market: e.market, symbol: e.symbol, message: e.message })));
        }
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
  await detectAndApplyPythonEnv();
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("relife.pythonPath")) {
        info(`配置变更: pythonPath → ${vscode.workspace.getConfiguration("relife").get("pythonPath", "")}`);
        detectAndApplyPythonEnv().catch(() => {});
      }
    }),
  );
  if (context.workspaceState.get(STARTUP_KEY) !== today) {
    await context.workspaceState.update(STARTUP_KEY, today);
    setTimeout(() => refresh().catch(() => {}), 1000);
  }
}

async function loadInitialSnapshot(context, output) {
  const cache = context.storageUri && vscode.Uri.joinPath(context.storageUri, "snapshot.json");
  if (cache) {
    try { return JSON.parse(await fs.readFile(cache.fsPath, "utf8")); } catch (error) { info(`读取缓存失败，改用示例数据: ${error.message}`); }
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
