import assert from "node:assert/strict";
import test from "node:test";

import {
  createRefreshRequest,
  initialWebviewState,
  reduceWebviewState,
} from "../vscode/webview-state.mjs";

test("the webview requests a host refresh with the public message protocol", () => {
  assert.deepEqual(createRefreshRequest(), { type: "refresh" });
});

test("refresh messages disable and restore the button without discarding old data", () => {
  const oldData = { generatedAt: "old" };
  const idle = { ...initialWebviewState, data: oldData, error: "旧错误" };
  const refreshing = reduceWebviewState(idle, { type: "refresh-start" });

  assert.deepEqual(refreshing, {
    data: oldData,
    refreshing: true,
    error: null,
  });
  assert.deepEqual(
    reduceWebviewState(refreshing, { type: "refresh-success" }),
    { data: oldData, refreshing: false, error: null },
  );
  assert.deepEqual(
    reduceWebviewState(refreshing, {
      type: "refresh-error",
      message: "行情服务不可用",
    }),
    { data: oldData, refreshing: false, error: "行情服务不可用" },
  );
});

test("portfolio messages replace the displayed data", () => {
  const data = { generatedAt: "new" };
  assert.deepEqual(
    reduceWebviewState(initialWebviewState, { type: "portfolio", data }),
    { data, refreshing: false, error: null },
  );
});
