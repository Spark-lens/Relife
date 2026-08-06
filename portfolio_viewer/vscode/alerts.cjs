/* eslint-disable @typescript-eslint/no-require-imports */
const { createHash, createHmac } = require("node:crypto");
const { execFile } = require("node:child_process");
const path = require("node:path");

const PYTHON = "/home/clannad/miniforge3/envs/istorm_rag_gpu/bin/python";
const CHECKER = path.join("portfolio_viewer", "scripts", "check_watchlist.py");
const DELIVERY_KEY = "relifePortfolio.alertDeliveries.v1";
const FEISHU_WEBHOOK_KEY = "relifePortfolio.feishuWebhookUrl";
const FEISHU_SECRET_KEY = "relifePortfolio.feishuSigningSecret";
const TIMEFRAME_LABELS = { daily: "日线", weekly: "周线", monthly: "月线" };

function alertId(alert) {
  const key = [
    alert.market,
    alert.symbol,
    alert.timeframe,
    alert.periodKey,
    alert.window,
    alert.standardDeviations,
  ].join("|");
  return createHash("sha256").update(key).digest("hex").slice(0, 24);
}

function formatAlert(alert) {
  const deviation = ((alert.close - alert.lowerBand) / alert.lowerBand) * 100;
  return [
    `[策略触发] ${alert.symbol} ${alert.name}`,
    `${alert.groupLabel} · ${TIMEFRAME_LABELS[alert.timeframe] ?? alert.timeframe}`,
    `收盘价 ${alert.close} · 下轨 ${alert.lowerBand} · 偏离 ${deviation.toFixed(2)}%`,
    `数据日期 ${alert.barDate}`,
    "仅进入候选池，不构成买入信号。",
  ].join("\n");
}

function createFeishuPayload(text, secret, now = Date.now()) {
  const payload = { msg_type: "text", content: { text } };
  if (!secret) return payload;
  const timestamp = String(Math.floor(now / 1000));
  return {
    timestamp,
    sign: createHmac("sha256", `${timestamp}\n${secret}`).update("").digest("base64"),
    ...payload,
  };
}

function validateFeishuWebhook(value) {
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error("飞书 Webhook 地址无效");
  }
  if (
    url.protocol !== "https:" ||
    url.hostname !== "open.feishu.cn" ||
    !/^\/open-apis\/bot\/v2\/hook\/[^/]+$/.test(url.pathname) ||
    url.username ||
    url.password
  ) {
    throw new Error("飞书 Webhook 地址无效");
  }
  return url.toString();
}

async function sendFeishu(url, text, options = {}) {
  const target = validateFeishuWebhook(url);
  const response = await (options.fetchImpl ?? fetch)(target, {
    method: "POST",
    headers: { "Content-Type": "application/json; charset=utf-8" },
    body: JSON.stringify(createFeishuPayload(text, options.secret, options.now)),
    signal: options.signal ?? AbortSignal.timeout(options.timeoutMs ?? 10_000),
  });
  if (!response.ok) throw new Error(`飞书 Webhook HTTP ${response.status}`);
  const body = await response.json();
  if (body.code !== 0) throw new Error(`飞书 Webhook 失败：${body.msg || body.code}`);
}

async function deliverAlerts(alerts, options) {
  const stored = options.workspaceState.get(DELIVERY_KEY, { entries: {} });
  const entries = structuredClone(stored.entries ?? {});
  const now = options.now ?? Date.now;

  for (const alert of alerts) {
    const id = alertId(alert);
    if (!entries[id]) {
      entries[id] = {
        alert,
        popup: false,
        feishu: "pending",
        updatedAt: now(),
      };
    }
    if (!entries[id].popup) {
      entries[id].popup = true;
      options.showWarningMessage(formatAlert(alert));
    }
  }

  const webhook = await options.secrets.get(FEISHU_WEBHOOK_KEY);
  const secret = await options.secrets.get(FEISHU_SECRET_KEY);
  const failures = [];
  for (const entry of Object.values(entries)) {
    if (entry.feishu !== "pending") continue;
    if (!webhook) {
      entry.feishu = "skipped";
      entry.updatedAt = now();
      continue;
    }
    try {
      await options.send(webhook, formatAlert(entry.alert), { secret });
      entry.feishu = "sent";
      entry.updatedAt = now();
    } catch (error) {
      failures.push(error);
      options.output.appendLine(`飞书通知失败：${error.message}`);
    }
  }

  const limited = Object.fromEntries(
    Object.entries(entries)
      .sort((left, right) => right[1].updatedAt - left[1].updatedAt)
      .slice(0, 500),
  );
  await options.workspaceState.update(DELIVERY_KEY, { entries: limited });
  if (failures.length) {
    options.showErrorMessage(`Relife 飞书通知失败：${failures.length} 条待下次重试`);
  }
}

function runWatchlistCheck(repositoryRoot, run = runChecker) {
  return run(PYTHON, [path.join(repositoryRoot, CHECKER)], { cwd: repositoryRoot });
}

function runChecker(file, args, options) {
  return new Promise((resolve, reject) => {
    execFile(file, args, { ...options, maxBuffer: 10 * 1024 * 1024 }, (error, stdout, stderr) => {
      try {
        const payload = JSON.parse(stdout);
        resolve(payload);
      } catch {
        if (error) {
          error.stdout = stdout;
          error.stderr = stderr;
          reject(error);
        } else {
          reject(new Error("观察策略检查器未返回有效 JSON"));
        }
      }
    });
  });
}

function registerFeishuCommands(vscode, secrets, options = {}) {
  const send = options.send ?? sendFeishu;
  function register(name, operation) {
    return vscode.commands.registerCommand(name, () =>
      Promise.resolve(operation()).catch((error) => {
        vscode.window.showErrorMessage(`Relife 飞书通知失败：${error.message}`);
      }),
    );
  }
  return [
    register("relifePortfolio.configureFeishu", async () => {
      const action = await vscode.window.showQuickPick(
        [
          { label: "配置飞书通知", value: "configure" },
          { label: "停用飞书通知", value: "disable" },
        ],
        { placeHolder: "飞书通知" },
      );
      if (!action) return;
      if (action.value === "disable") {
        await Promise.all([
          secrets.delete(FEISHU_WEBHOOK_KEY),
          secrets.delete(FEISHU_SECRET_KEY),
        ]);
        vscode.window.showInformationMessage("Relife 飞书通知已停用");
        return;
      }
      const webhook = await vscode.window.showInputBox({
        prompt: "飞书群自定义机器人 Webhook",
        password: true,
        validateInput(value) {
          try {
            validateFeishuWebhook(value);
            return undefined;
          } catch (error) {
            return error.message;
          }
        },
      });
      if (webhook === undefined) return;
      const secret = await vscode.window.showInputBox({
        prompt: "签名密钥（未启用签名校验可留空）",
        password: true,
      });
      if (secret === undefined) return;
      await secrets.store(FEISHU_WEBHOOK_KEY, validateFeishuWebhook(webhook));
      if (secret) await secrets.store(FEISHU_SECRET_KEY, secret);
      else await secrets.delete(FEISHU_SECRET_KEY);
      vscode.window.showInformationMessage("Relife 飞书通知配置已保存");
    }),
    register("relifePortfolio.testFeishu", async () => {
      const webhook = await secrets.get(FEISHU_WEBHOOK_KEY);
      if (!webhook) throw new Error("尚未配置飞书 Webhook");
      const secret = await secrets.get(FEISHU_SECRET_KEY);
      await send(webhook, "[策略触发] Relife 飞书通知测试", { secret });
      vscode.window.showInformationMessage("Relife 飞书测试消息已发送");
    }),
  ];
}

module.exports = {
  CHECKER,
  DELIVERY_KEY,
  FEISHU_SECRET_KEY,
  FEISHU_WEBHOOK_KEY,
  PYTHON,
  alertId,
  createFeishuPayload,
  deliverAlerts,
  formatAlert,
  registerFeishuCommands,
  runWatchlistCheck,
  sendFeishu,
  validateFeishuWebhook,
};
