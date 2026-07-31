import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { createPortfolioTreeDataProvider } = require("../vscode/sidebar.cjs");

test("the only sidebar entry opens the existing portfolio command", () => {
  class TreeItem {
    constructor(label, collapsibleState) {
      this.label = label;
      this.collapsibleState = collapsibleState;
    }
  }
  const provider = createPortfolioTreeDataProvider({
    TreeItem,
    TreeItemCollapsibleState: { None: 0 },
  });
  const children = provider.getChildren();

  assert.equal(children.length, 1);
  assert.equal(provider.getTreeItem(children[0]), children[0]);
  assert.equal(children[0].label, "打开投资组合");
  assert.equal(children[0].collapsibleState, 0);
  assert.deepEqual(children[0].command, {
    command: "relifePortfolio.open",
    title: "打开投资组合",
  });
});
