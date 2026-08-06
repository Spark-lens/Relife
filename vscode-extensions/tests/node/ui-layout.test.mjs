import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

const webview = fs.readFileSync(new URL("../../src/webview.tsx", import.meta.url), "utf8");
const styles = fs.readFileSync(new URL("../../src/webview.css", import.meta.url), "utf8");
const snapshot = JSON.parse(fs.readFileSync(new URL("../../resources/sample/portfolio-snapshot.json", import.meta.url), "utf8"));

test("投资组合默认打开讨论稿的概览和值", () => {
  assert.match(webview, /const \[tab, setTab\] = useState\("概览"\)/);
  assert.match(webview, /const \[subtab, setSubtab\] = useState\("值"\)/);
  assert.match(webview, /chart-caption/);
  assert.doesNotMatch(webview, /投资组合更改/);
});

test("投资组合使用确认过的 v2 工作台，不再使用旧选择框顶部", () => {
  assert.match(webview, /className="portfolio-v2"/);
  assert.match(webview, /className="workbench-head"/);
  assert.match(webview, /"控股"/);
  assert.doesNotMatch(webview, /className="topbar"/);
  assert.doesNotMatch(webview, /className="market-select"/);
});

test("收益卡只保留确认过的辅助项", () => {
  assert.match(webview, /最后一天/);
  assert.match(webview, /净股息/);
  assert.match(webview, /年化收益率/);
  assert.doesNotMatch(webview, /交易收益/);
  assert.match(styles, /\.portfolio-v2[\s\S]*\.summary-subline[\s\S]*text-align:\s*right/);
});

test("侧栏不显示分类数量且背景覆盖整个 webview", () => {
  assert.doesNotMatch(webview, /category-count/);
  assert.match(styles, /body\[data-view="sidebar"\] #root/);
});

test("默认标的带有示例行情值", async () => {
  const { defaultWatchlist } = await import("../../src/watchlist-state.mjs");
  const symbols = defaultWatchlist().categories.flatMap((category) => category.symbols);
  assert.ok(symbols.length > 0);
  assert.ok(symbols.every((item) => Number.isFinite(item.latest)));
  assert.ok(symbols.every((item) => Number.isFinite(item.changePercent)));
});

test("组合概览的示例曲线包含投资组合值和自由现金", () => {
  for (const market of Object.values(snapshot.markets)) {
    assert.ok(market.curve.length > 0);
    assert.ok(market.curve.every((row) => Number.isFinite(row.portfolio) && Number.isFinite(row.flow)));
  }
});
