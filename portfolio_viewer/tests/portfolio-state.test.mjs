import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_COLUMNS,
  filterTransactions,
  normalizeColumnPreferences,
  selectRange,
} from "../app/portfolio-state.mjs";


test("all keeps full history and shorter ranges use the requested cutoff", () => {
  const points = [
    { date: "2025-12-31" },
    { date: "2026-04-16" },
    { date: "2026-06-17" },
    { date: "2026-07-17" },
  ];

  assert.deepEqual(selectRange(points, "all", new Date("2026-07-17")), points);
  assert.deepEqual(
    selectRange(points, "1m", new Date("2026-07-17")),
    points.slice(2),
  );
  assert.deepEqual(
    selectRange(points, "ytd", new Date("2026-07-17")),
    points.slice(1),
  );
});


test("transaction filters preserve reverse chronological order", () => {
  const transactions = [
    { id: "old", kind: "buy", timestamp: "2026-01-01T00:00:00" },
    { id: "cash", kind: "deposit", timestamp: "2026-05-01T00:00:00" },
    { id: "new", kind: "dividend", timestamp: "2026-07-01T00:00:00" },
  ];

  assert.deepEqual(
    filterTransactions(transactions, "all").map((row) => row.id),
    ["new", "cash", "old"],
  );
  assert.deepEqual(
    filterTransactions(transactions, "buy").map((row) => row.id),
    ["old"],
  );
  assert.deepEqual(
    filterTransactions(transactions, "cash").map((row) => row.id),
    ["cash"],
  );
});


test("column preferences keep symbol locked and reject unknown ids", () => {
  const normalized = normalizeColumnPreferences([
    { id: "dailyPnl", visible: false },
    { id: "unknown", visible: true },
    { id: "symbol", visible: false },
  ]);

  assert.equal(DEFAULT_COLUMNS.length, 13);
  assert.equal(normalized[0].id, "symbol");
  assert.equal(normalized[0].visible, true);
  assert.equal(normalized.find((column) => column.id === "dailyPnl").visible, false);
  assert.equal(normalized.some((column) => column.id === "unknown"), false);
  assert.deepEqual(
    normalized.map((column) => column.id).sort(),
    DEFAULT_COLUMNS.map((column) => column.id).sort(),
  );
});
