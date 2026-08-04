import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { webviewHtml } = require("../../src/webview-html.cjs");

test("Webview HTML 使用白名单资源和 nonce CSP", () => {
  const html = webviewHtml({ kind: "portfolio", script: "webview.js", style: "webview.css", cspSource: "vscode-webview://relife", nonce: "fixed" });
  assert.match(html, /data-view="portfolio"/);
  assert.match(html, /script-src 'nonce-fixed'/);
  assert.doesNotMatch(html, /unsafe-inline/);
  assert.match(html, /lang="zh-CN"/);
});
