import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const {
  DELIVERY_KEY,
  FEISHU_SECRET_KEY,
  FEISHU_WEBHOOK_KEY,
  alertId,
  createFeishuPayload,
  deliverAlerts,
  formatAlert,
  registerFeishuCommands,
  runWatchlistCheck,
  sendFeishu,
} = require("../vscode/alerts.cjs");

function alert(overrides = {}) {
  return {
    groupId: "broad",
    groupLabel: "大盘",
    market: "us",
    symbol: "QQQ",
    name: "纳指",
    timeframe: "daily",
    periodKey: "2026-08-01",
    barDate: "2026-08-01",
    close: 90,
    lowerBand: 95,
    triggered: true,
    window: 20,
    standardDeviations: 2,
    ...overrides,
  };
}

class MemoryState {
  values = new Map();
  get(key, fallback) { return this.values.get(key) ?? fallback; }
  async update(key, value) { this.values.set(key, value); }
}

test("builds a stable event id from symbol, timeframe, bucket and rule", () => {
  assert.equal(alertId(alert()), alertId(alert({ groupLabel: "新分组" })));
  assert.notEqual(alertId(alert()), alertId(alert({ periodKey: "2026-08-02" })));
  assert.notEqual(alertId(alert()), alertId(alert({ window: 30 })));
});

test("formats candidate details and Feishu signed payload", () => {
  const message = formatAlert(alert());
  assert.match(message, /^\[策略触发\] QQQ/);
  assert.match(message, /大盘 · 日线/);
  assert.match(message, /收盘价 90/);
  assert.match(message, /下轨 95/);
  assert.match(message, /仅进入候选池，不构成买入信号/);

  const payload = createFeishuPayload(message, "secret", 1_599_360_473_000);
  const expected = createHmac("sha256", "1599360473\nsecret")
    .update("")
    .digest("base64");
  assert.equal(payload.timestamp, "1599360473");
  assert.equal(payload.sign, expected);
  assert.deepEqual(payload.content, { text: message });
});

test("posts only to a valid Feishu bot webhook and checks response code", async () => {
  let request;
  await sendFeishu(
    "https://open.feishu.cn/open-apis/bot/v2/hook/token",
    "[策略触发] 测试",
    {
      fetchImpl: async (url, options) => {
        request = { url, options };
        return { ok: true, json: async () => ({ code: 0, msg: "success" }) };
      },
    },
  );
  assert.equal(request.url, "https://open.feishu.cn/open-apis/bot/v2/hook/token");
  assert.equal(JSON.parse(request.options.body).msg_type, "text");
  await assert.rejects(
    sendFeishu("http://example.com/hook", "test", { fetchImpl: async () => ({}) }),
    /飞书 Webhook 地址无效/,
  );
});

test("shows a popup once and retries only failed Feishu delivery", async () => {
  const state = new MemoryState();
  const popups = [];
  const errors = [];
  const secrets = {
    async get(key) {
      if (key === FEISHU_WEBHOOK_KEY) return "https://open.feishu.cn/open-apis/bot/v2/hook/token";
      if (key === FEISHU_SECRET_KEY) return "signing";
      return undefined;
    },
  };
  let attempts = 0;
  const options = {
    workspaceState: state,
    secrets,
    showWarningMessage: (message) => popups.push(message),
    showErrorMessage: (message) => errors.push(message),
    output: { appendLine() {} },
    send: async () => {
      attempts += 1;
      if (attempts === 1) throw new Error("网络失败");
    },
    now: () => 1000 + attempts,
  };

  await deliverAlerts([alert()], options);
  await deliverAlerts([], options);

  assert.equal(popups.length, 1);
  assert.equal(attempts, 2);
  assert.equal(errors.length, 1);
  const entries = state.get(DELIVERY_KEY, {}).entries;
  assert.equal(entries[alertId(alert())].popup, true);
  assert.equal(entries[alertId(alert())].feishu, "sent");
});

test("runs the repository checker with the configured Python", async () => {
  let call;
  const payload = await runWatchlistCheck("/repo", async (file, args, options) => {
    call = { file, args, options };
    return { checkedAt: "now", alerts: [], errors: [] };
  });

  assert.match(call.file, /istorm_rag_gpu\/bin\/python$/);
  assert.deepEqual(call.args, ["/repo/portfolio_viewer/scripts/check_watchlist.py"]);
  assert.equal(call.options.cwd, "/repo");
  assert.deepEqual(payload.alerts, []);
});

test("configures and tests Feishu credentials through commands", async () => {
  const commands = new Map();
  const stored = new Map();
  const notices = [];
  const inputs = [
    "https://open.feishu.cn/open-apis/bot/v2/hook/token",
    "secret",
  ];
  let sent;
  const vscode = {
    commands: {
      registerCommand(name, callback) {
        commands.set(name, callback);
        return { dispose() {} };
      },
    },
    window: {
      showQuickPick: async () => ({ value: "configure" }),
      showInputBox: async () => inputs.shift(),
      showInformationMessage: (message) => notices.push(message),
      showErrorMessage: (message) => notices.push(message),
    },
  };
  const secrets = {
    get: async (key) => stored.get(key),
    store: async (key, value) => stored.set(key, value),
    delete: async (key) => stored.delete(key),
  };

  const disposables = registerFeishuCommands(vscode, secrets, {
    send: async (...args) => { sent = args; },
  });
  await commands.get("relifePortfolio.configureFeishu")();
  await commands.get("relifePortfolio.testFeishu")();

  assert.equal(disposables.length, 2);
  assert.equal(stored.get(FEISHU_SECRET_KEY), "secret");
  assert.match(sent[1], /\[策略触发\].*测试/);
  assert.equal(sent[2].secret, "secret");
  assert.equal(notices.length, 2);
});
