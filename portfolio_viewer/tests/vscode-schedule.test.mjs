import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { nextUpdateAt, scheduleUpdates } = require("../vscode/schedule.cjs");

test("chooses the next fixed Asia/Shanghai market update", () => {
  const cases = [
    ["2026-07-31T01:14:00Z", "2026-07-31T01:15:00.000Z"],
    ["2026-07-31T07:16:00Z", "2026-07-31T13:15:00.000Z"],
    ["2026-07-31T20:00:00Z", "2026-07-31T21:15:00.000Z"],
    ["2026-07-31T21:16:00Z", "2026-08-03T01:15:00.000Z"],
  ];

  for (const [now, expected] of cases) {
    assert.equal(nextUpdateAt(new Date(now)).toISOString(), expected);
  }
});

test("runs once at the next update and schedules the following update", async () => {
  let now = new Date("2026-07-31T01:14:00Z");
  const timers = [];
  let runs = 0;
  const scheduler = scheduleUpdates(
    async () => {
      runs += 1;
      now = new Date("2026-07-31T01:16:00Z");
    },
    {
      now: () => now,
      setTimeout: (callback, delay) => {
        timers.push({ callback, delay });
        return timers.length;
      },
      clearTimeout: () => {},
    },
  );

  assert.equal(timers[0].delay, 60_000);
  await timers[0].callback();
  assert.equal(runs, 1);
  assert.equal(timers[1].delay, 21_540_000);
  scheduler.dispose();
});
