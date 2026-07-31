import assert from "node:assert/strict";
import Module, { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const extensionPath = require.resolve("../vscode/extension.cjs");
const { singleFlight } = require("../vscode/portfolio.cjs");

function flush() {
  return new Promise((resolve) => setImmediate(resolve));
}

test("the extension handles webview refresh success and failure messages", async () => {
  const commands = new Map();
  const messages = [];
  const notifications = [];
  let receiveMessage;
  let nextRefresh = { generatedAt: "auto" };
  const panel = {
    webview: {
      cspSource: "test-csp",
      html: "",
      asWebviewUri: (uri) => uri,
      postMessage: (message) => messages.push(message),
      onDidReceiveMessage(callback) {
        receiveMessage = callback;
        return { dispose() {} };
      },
    },
    reveal() {},
    onDidDispose() {
      return { dispose() {} };
    },
  };
  class TreeItem {
    constructor(label, collapsibleState) {
      this.label = label;
      this.collapsibleState = collapsibleState;
    }
  }
  const vscode = {
    workspace: { workspaceFolders: [{ uri: { fsPath: "/repo" } }] },
    window: {
      createOutputChannel: () => ({ appendLine() {}, dispose() {} }),
      createWebviewPanel: () => panel,
      registerTreeDataProvider: () => ({ dispose() {} }),
      showErrorMessage: (message) => notifications.push(message),
      withProgress: (_options, task) => task(),
    },
    commands: {
      registerCommand(name, callback) {
        commands.set(name, callback);
        return { dispose() {} };
      },
    },
    Uri: { joinPath: (...parts) => parts.join("/") },
    ViewColumn: { One: 1 },
    ProgressLocation: { Notification: 1 },
    TreeItem,
    TreeItemCollapsibleState: { None: 0 },
  };
  const portfolio = {
    findRelifeRoot: () => "/repo",
    loadPortfolio: async () => ({ generatedAt: "old" }),
    refreshPortfolio: async () => {
      if (nextRefresh instanceof Error) throw nextRefresh;
      return nextRefresh;
    },
    singleFlight,
  };
  const originalLoad = Module._load;
  Module._load = function load(request, parent, isMain) {
    if (request === "vscode") return vscode;
    if (parent?.filename === extensionPath && request === "./portfolio.cjs") {
      return portfolio;
    }
    if (parent?.filename === extensionPath && request === "./schedule.cjs") {
      return { scheduleUpdates: () => ({ dispose() {} }) };
    }
    return originalLoad.call(this, request, parent, isMain);
  };

  let extension;
  try {
    delete require.cache[extensionPath];
    extension = require(extensionPath);
  } finally {
    Module._load = originalLoad;
  }

  extension.activate({ extensionUri: "/extension", subscriptions: [] });
  await flush();
  commands.get("relifePortfolio.open")();
  receiveMessage({ type: "ready" });
  messages.length = 0;

  nextRefresh = { generatedAt: "new" };
  receiveMessage({ type: "refresh" });
  await flush();
  assert.deepEqual(messages, [
    { type: "refresh-start" },
    { type: "portfolio", data: { generatedAt: "new" } },
    { type: "refresh-success" },
  ]);

  messages.length = 0;
  nextRefresh = new Error("provider unavailable");
  receiveMessage({ type: "refresh" });
  await flush();
  assert.deepEqual(messages, [
    { type: "refresh-start" },
    { type: "refresh-error", message: "provider unavailable" },
  ]);
  assert.deepEqual(notifications, [
    "Relife 投资组合更新失败：provider unavailable",
  ]);
});
