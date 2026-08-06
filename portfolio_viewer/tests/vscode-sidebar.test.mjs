import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { createPortfolioTreeDataProvider } = require("../vscode/sidebar.cjs");

test("sidebar has HOME and 观察列表 as top-level collapsible sections", () => {
  class TreeItem {
    constructor(label, collapsibleState) {
      this.label = label;
      this.collapsibleState = collapsibleState;
    }
  }
  const Collapsed = 1;
  const None = 0;
  const provider = createPortfolioTreeDataProvider({
    TreeItem,
    TreeItemCollapsibleState: { None, Collapsed },
    EventEmitter: class { event = () => {}; fire() {} },
  }, {
    groups: [{
      id: "broad",
      label: "大盘",
      subtitle: "核心",
      items: [{
        market: "us",
        symbol: "QQQ",
        name: "纳指",
        bollinger: { enabled: true, timeframes: ["daily"], window: 20, standardDeviations: 2 },
      }],
    }],
  });

  const top = provider.getChildren();
  assert.equal(top.length, 2);
  assert.equal(top[0].label, "HOME");
  assert.equal(top[0].collapsibleState, Collapsed);
  assert.equal(top[1].label, "观察列表");
  assert.equal(top[1].collapsibleState, Collapsed);

  // HOME -> 投资组合
  const homeChildren = provider.getChildren(top[0]);
  assert.equal(homeChildren.length, 1);
  assert.equal(homeChildren[0].label, "投资组合");
  assert.equal(homeChildren[0].collapsibleState, None);
  assert.deepEqual(homeChildren[0].command, { command: "relifePortfolio.open", title: "投资组合" });

  // 观察列表 -> groups
  const wlChildren = provider.getChildren(top[1]);
  assert.equal(wlChildren.length, 1);
  assert.equal(wlChildren[0].label, "大盘 - 核心");
  assert.equal(wlChildren[0].collapsibleState, Collapsed);

  // group -> symbols (no strategy info)
  const symbols = provider.getChildren(wlChildren[0]);
  assert.equal(symbols.length, 1);
  assert.equal(symbols[0].label, "QQQ 纳指");
  assert.equal(symbols[0].description, undefined);
  assert.equal(symbols[0].contextValue, "relifeWatchlistItemEnabled");
});
