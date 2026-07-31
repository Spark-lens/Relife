import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";

import { RefreshControl } from "../vscode/refresh-control.mjs";
import { LoadingView } from "../vscode/loading-view.mjs";

test("renders an enabled refresh button and a host error", () => {
  const html = renderToStaticMarkup(
    RefreshControl({
      refreshing: false,
      error: "行情服务不可用",
      onRefresh() {},
    }),
  );

  assert.match(html, /<button[^>]*class="refresh-button"/);
  assert.match(html, />立即更新<\/button>/);
  assert.doesNotMatch(html, / disabled=""/);
  assert.match(html, /role="alert"/);
  assert.match(html, /更新失败：行情服务不可用/);
});

test("disables the refresh button while an update is running", () => {
  const html = renderToStaticMarkup(
    RefreshControl({ refreshing: true, error: null, onRefresh() {} }),
  );

  assert.match(html, /<button[^>]*disabled=""/);
  assert.match(html, /aria-busy="true"/);
  assert.match(html, />正在更新…<\/button>/);
});

test("keeps refresh and errors reachable before portfolio data loads", () => {
  const html = renderToStaticMarkup(
    LoadingView({
      refreshing: false,
      error: "初始更新失败",
      onRefresh() {},
    }),
  );

  assert.match(html, /正在读取投资组合…/);
  assert.match(html, />立即更新<\/button>/);
  assert.match(html, /更新失败：初始更新失败/);
});
