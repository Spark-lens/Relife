import test from "node:test";
import assert from "node:assert/strict";

import {
  addCategory,
  addSymbol,
  defaultWatchlist,
  editSymbol,
  moveCategory,
  moveSymbol,
  removeCategory,
  removeSymbol,
  renameCategory,
} from "../../src/watchlist-state.mjs";

test("默认分类顺序固定且标的为空", () => {
  const state = defaultWatchlist();
  assert.deepEqual(state.categories.map((item) => item.name), ["现金流", "大盘", "股息", "个股", "杠杆", "比特币"]);
  assert.ok(state.categories.every((item) => item.symbols.length === 0));
});

test("分类与标的支持完整增删改", () => {
  let state = addCategory(defaultWatchlist(), "自选");
  const category = state.categories.at(-1);
  state = renameCategory(state, category.id, "长期观察");
  state = addSymbol(state, category.id, { market: "us", symbol: "DEMO", name: "示例", note: "旧备注" });
  state = editSymbol(state, "us:DEMO", { name: "示例科技", note: "新备注" });
  assert.equal(state.categories.at(-1).symbols[0].note, "新备注");
  state = removeSymbol(state, "us:DEMO");
  state = removeCategory(state, category.id);
  assert.equal(state.categories.length, 6);
});

test("同市场代码全局唯一，重复添加给出可移动冲突", () => {
  let state = defaultWatchlist();
  state = addSymbol(state, state.categories[0].id, { market: "us", symbol: "SPY", name: "标普ETF", note: "" });
  assert.throws(
    () => addSymbol(state, state.categories[1].id, { market: "us", symbol: "spy", name: "重复", note: "" }),
    (error) => error.code === "DUPLICATE_SYMBOL" && error.existingCategoryId === state.categories[0].id,
  );
});

test("分类和标的可排序并跨分类移动", () => {
  let state = defaultWatchlist();
  const first = state.categories[0].id;
  const second = state.categories[1].id;
  state = addSymbol(state, first, { market: "cn", symbol: "600000", name: "示例银行", note: "观察" });
  state = moveSymbol(state, "cn:600000", second, 0);
  assert.equal(state.categories[0].symbols.length, 0);
  assert.equal(state.categories[1].symbols[0].key, "cn:600000");
  state = moveCategory(state, second, -1);
  assert.equal(state.categories[0].id, second);
});
