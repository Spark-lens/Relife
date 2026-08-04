import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";

import { allowedRequest, defaultSources, sourceFromFile } from "../../src/extension-state.mjs";

test("两个市场默认独立使用示例数据", () => {
  assert.deepEqual(defaultSources(), {
    us: { market: "us", mode: "sample", directory: null, pattern: "tradingview_full_latest_YYYY-MM-DD.csv", lastValidFile: null },
    cn: { market: "cn", mode: "sample", directory: null, pattern: "交割单_YYYY-MM-DD.csv", lastValidFile: null },
  });
});

test("选中文件后只保存目录、命名模式和最后有效文件", () => {
  const selected = path.join("tmp", "source", "tradingview_full_latest_2026-08-01.csv");
  assert.deepEqual(sourceFromFile("us", selected), {
    market: "us", mode: "directory", directory: path.dirname(selected),
    pattern: "tradingview_full_latest_YYYY-MM-DD.csv", lastValidFile: selected,
  });
  assert.throws(() => sourceFromFile("cn", selected), /文件名不符合/);
});

test("宿主只接受约定的 Webview 请求", () => {
  assert.equal(allowedRequest({ type: "ready" }), true);
  assert.equal(allowedRequest({ type: "watchlist-move-symbol" }), true);
  assert.equal(allowedRequest({ type: "run-shell" }), false);
  assert.equal(allowedRequest(null), false);
});
