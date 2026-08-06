/* eslint-disable @typescript-eslint/no-require-imports */
const { randomBytes } = require("node:crypto");
const vscode = require("vscode");

const {
  findRelifeRoot,
  loadPortfolio,
  refreshPortfolio,
  singleFlight,
} = require("./portfolio.cjs");
const { scheduleUpdates } = require("./schedule.cjs");
const { createPortfolioTreeDataProvider } = require("./sidebar.cjs");
const {
  deliverAlerts,
  registerFeishuCommands,
  runWatchlistCheck,
  sendFeishu,
} = require("./alerts.cjs");
const {
  loadWatchlist,
  saveWatchlist,
  watchlistPath,
} = require("./watchlist.cjs");
const { registerWatchlistCommands } = require("./watchlist-editor.cjs");

function activate(context) {
  const output = vscode.window.createOutputChannel("Relife Portfolio");
  const repositoryRoot = findRelifeRoot(
    (vscode.workspace.workspaceFolders ?? []).map((folder) => folder.uri.fsPath),
  );
  let panel;
  let portfolio;
  let watchlist;
  const sidebar = createPortfolioTreeDataProvider(vscode, { groups: [] });

  function postToWebview(message) {
    panel?.webview.postMessage(message);
  }

  function sendPortfolio() {
    if (panel && portfolio) {
      postToWebview({ type: "portfolio", data: portfolio });
    }
  }

  const refresh = singleFlight(async () => {
    if (!repositoryRoot) {
      throw new Error("当前工作区不是 Relife 仓库");
    }
    postToWebview({ type: "refresh-start" });
    output.appendLine(`[${new Date().toISOString()}] 开始更新投资组合`);
    try {
      portfolio = await refreshPortfolio(repositoryRoot);
      sendPortfolio();
      postToWebview({ type: "refresh-success" });
      output.appendLine(`[${new Date().toISOString()}] 投资组合更新完成`);
      return portfolio;
    } catch (error) {
      const detail = error?.stderr || error?.stack || String(error);
      const message = error?.message || String(error);
      output.appendLine(`[${new Date().toISOString()}] 更新失败\n${detail}`);
      vscode.window.showErrorMessage(`Relife 投资组合更新失败：${message}`);
      postToWebview({ type: "refresh-error", message });
      throw error;
    }
  });

  const checkAlerts = singleFlight(async () => {
    if (!repositoryRoot) throw new Error("当前工作区不是 Relife 仓库");
    output.appendLine(`[${new Date().toISOString()}] 开始检查观察策略`);
    try {
      const payload = await runWatchlistCheck(repositoryRoot);
      for (const error of payload.errors ?? []) {
        output.appendLine(
          `观察策略数据错误：${error.symbol ?? "配置"} ${error.timeframe ?? ""} ${error.message}`,
        );
      }
      await deliverAlerts(payload.alerts ?? [], {
        workspaceState: context.workspaceState,
        secrets: context.secrets,
        showWarningMessage: (message) => vscode.window.showWarningMessage(message),
        showErrorMessage: (message) => vscode.window.showErrorMessage(message),
        output,
        send: sendFeishu,
      });
      if ((payload.errors?.length ?? 0) > 0 && (payload.results?.length ?? 0) === 0) {
        vscode.window.showWarningMessage(
          `Relife 观察策略检查无可用结果：${payload.errors.length} 项错误，详情见输出日志`,
        );
      }
      output.appendLine(
        `[${new Date().toISOString()}] 观察策略检查完成：${payload.alerts?.length ?? 0} 个候选`,
      );
      return payload;
    } catch (error) {
      const detail = error?.stderr || error?.stack || String(error);
      const message = error?.message || String(error);
      output.appendLine(`[${new Date().toISOString()}] 观察策略检查失败\n${detail}`);
      vscode.window.showErrorMessage(`Relife 观察策略检查失败：${message}`);
      throw error;
    }
  });

  async function runUpdates() {
    return Promise.allSettled([refresh(), checkAlerts()]);
  }

  async function persistWatchlist(next) {
    if (!repositoryRoot) throw new Error("当前工作区不是 Relife 仓库");
    await saveWatchlist(watchlistPath(repositoryRoot), next);
    watchlist = next;
    sidebar.refresh(watchlist);
  }

  function openPortfolio() {
    if (!repositoryRoot) {
      vscode.window.showErrorMessage("当前工作区不是 Relife 仓库");
      return;
    }
    if (panel) {
      panel.reveal(vscode.ViewColumn.One);
      return;
    }
    panel = vscode.window.createWebviewPanel(
      "relifePortfolio",
      "Relife Portfolio",
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, "vscode", "dist")],
      },
    );
    panel.webview.html = webviewHtml(panel.webview, context.extensionUri);
    panel.webview.onDidReceiveMessage(
      (message) => {
        if (message?.type === "ready") sendPortfolio();
        if (message?.type === "refresh") runUpdates();
      },
      undefined,
      context.subscriptions,
    );
    panel.onDidDispose(() => {
      panel = undefined;
    });
  }

  context.subscriptions.push(
    output,
    vscode.window.registerTreeDataProvider(
      "relifePortfolio.actions",
      sidebar,
    ),
    vscode.commands.registerCommand("relifePortfolio.open", openPortfolio),
    vscode.commands.registerCommand("relifePortfolio.refresh", () =>
      vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: "正在更新 Relife 投资组合",
        },
        runUpdates,
      ),
    ),
    ...registerWatchlistCommands(vscode, {
      getWatchlist: () => {
        if (!watchlist) throw new Error("观察列表尚未加载，请稍后重试");
        return watchlist;
      },
      persist: persistWatchlist,
    }),
    ...registerFeishuCommands(vscode, context.secrets),
  );

  if (repositoryRoot) {
    loadWatchlist(repositoryRoot)
      .then((data) => {
        watchlist = data;
        sidebar.refresh(watchlist);
      })
      .catch((error) => {
        output.appendLine(`读取观察列表失败：${error.message}`);
        vscode.window.showErrorMessage(`Relife 观察列表读取失败：${error.message}`);
      });
    loadPortfolio(repositoryRoot)
      .then((data) => {
        portfolio = data;
        sendPortfolio();
      })
      .catch((error) => output.appendLine(`读取现有组合数据失败：${error.message}`));
    runUpdates();
    context.subscriptions.push(scheduleUpdates(runUpdates));
  }
}

function webviewHtml(webview, extensionUri) {
  const nonce = randomBytes(16).toString("base64");
  const script = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, "vscode", "dist", "webview.js"),
  );
  const style = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, "vscode", "dist", "webview.css"),
  );
  return `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
    <link rel="stylesheet" href="${style}">
    <title>Relife Portfolio</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" nonce="${nonce}" src="${script}"></script>
  </body>
</html>`;
}

module.exports = { activate };
