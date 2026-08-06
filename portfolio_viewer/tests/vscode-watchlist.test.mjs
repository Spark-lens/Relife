import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const {
  addGroup,
  addItem,
  deleteGroup,
  moveGroupUp,
  moveGroupDown,
  moveItem,
  removeItem,
  renameGroup,
  saveWatchlist,
  updateItem,
  validateWatchlist,
} = require("../vscode/watchlist.cjs");

function rule(overrides = {}) {
  return {
    enabled: true,
    timeframes: ["daily", "weekly", "monthly"],
    window: 20,
    standardDeviations: 2,
    ...overrides,
  };
}

function config() {
  return {
    groups: [
      {
        id: "broad",
        label: "大盘",
        items: [
          { market: "us", symbol: "QQQ", name: "纳指", bollinger: rule() },
        ],
      },
      { id: "stock", label: "个股", items: [] },
    ],
  };
}

test("validates duplicate symbols and invalid Bollinger settings", () => {
  const duplicate = config();
  duplicate.groups[1].items.push({
    market: "us",
    symbol: "qqq",
    name: "重复",
    bollinger: rule(),
  });
  assert.throws(() => validateWatchlist(duplicate), /重复标的 us:QQQ/);

  const invalid = config();
  invalid.groups[0].items[0].bollinger.window = 1;
  assert.throws(() => validateWatchlist(invalid), /window/);
});

test("applies group and symbol edits without mutating the input", () => {
  const original = config();
  let next = addGroup(original, { id: "leverage", label: "杠杆" });
  next = renameGroup(next, "leverage", "杠杆 ETF");
  next = addItem(next, "stock", {
    market: "us",
    symbol: "KO",
    name: "可口可乐",
    bollinger: rule({ enabled: false }),
  });
  next = moveItem(next, "us", "KO", "leverage");
  next = updateItem(next, "us", "KO", {
    bollinger: rule({ timeframes: ["daily"], window: 30 }),
  });
  next = removeItem(next, "us", "QQQ");
  next = deleteGroup(next, "stock");

  assert.equal(original.groups.length, 2);
  assert.equal(next.groups.length, 2);
  assert.equal(next.groups[1].label, "杠杆 ETF");
  assert.equal(next.groups[1].items[0].bollinger.window, 30);
  assert.deepEqual(next.groups[1].items[0].bollinger.timeframes, ["daily"]);
});

test("writes validated JSON to a temporary file before atomic rename", async () => {
  const calls = [];
  const fs = {
    async writeFile(file, content, encoding) {
      calls.push(["write", file, JSON.parse(content).groups.length, encoding]);
    },
    async rename(from, to) {
      calls.push(["rename", from, to]);
    },
    async unlink() {
      throw new Error("should not unlink on success");
    },
  };

  await saveWatchlist("/repo/portfolio_viewer/data/watchlist.json", config(), fs, 42);

  assert.deepEqual(calls, [
    ["write", "/repo/portfolio_viewer/data/watchlist.json.42.tmp", 2, "utf8"],
    ["rename", "/repo/portfolio_viewer/data/watchlist.json.42.tmp", "/repo/portfolio_viewer/data/watchlist.json"],
  ]);
});

test("moves watchlist group up and down without mutating input", () => {
  const original = config();
  const down = moveGroupDown(original, original.groups[0].id);
  assert.equal(original.groups[0].id, "broad");
  assert.equal(down.groups[1].id, "broad");
  assert.equal(down.groups[0].id, "stock");

  const up = moveGroupUp(down, "broad");
  assert.equal(up.groups[0].id, "broad");
  assert.equal(up.groups[1].id, "stock");

  // Boundary: moving first group up is a no-op
  const atTop = moveGroupUp(up, "broad");
  assert.equal(atTop.groups[0].id, "broad");
  // Boundary: moving last group down is a no-op
  const atBottom = moveGroupDown(up, "stock");
  assert.equal(atBottom.groups[1].id, "stock");
});
