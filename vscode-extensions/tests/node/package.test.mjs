import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

test("扩展清单注册新身份、侧栏和纯前端命令", () => {
  const manifest = JSON.parse(fs.readFileSync(new URL("../../package.json", import.meta.url), "utf8"));
  assert.equal(`${manifest.publisher}.${manifest.name}@${manifest.version}`, "clannad0710.relife@0.0.1");
  assert.equal(manifest.displayName, "Relife");
  assert.equal(manifest.main, "./extension.cjs");
  assert.equal(manifest.contributes.viewsContainers.activitybar[0].id, "relife");
  assert.equal(manifest.contributes.views.relife[0].id, "relife.sidebar");
  assert.equal(manifest.contributes.configuration, undefined);
  const extension = fs.readFileSync(new URL("../../extension.cjs", import.meta.url), "utf8");
  assert.match(extension, /纯前端预览/);
  assert.doesNotMatch(extension, /child_process|spawn\(|python3|yfinance|akshare/);
  const commands = manifest.contributes.commands.map((item) => item.command);
  for (const command of ["relife.openPortfolio", "relife.selectUsSource", "relife.selectCnSource", "relife.resetSources", "relife.refresh"]) assert.ok(commands.includes(command));
  assert.ok(!manifest.files.some((entry) => entry.includes("portfolio_viewer") || entry.includes("data/transactions") || entry.startsWith("python/")));
});

test("Webview 生产包不依赖 Node 全局变量", () => {
  const bundle = fs.readFileSync(new URL("../../dist/webview.js", import.meta.url), "utf8");
  assert.doesNotMatch(bundle, /\bprocess\.env\b/);
});
