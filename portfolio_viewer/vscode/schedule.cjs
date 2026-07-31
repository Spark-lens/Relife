const SHANGHAI_OFFSET_HOURS = 8;
const UPDATE_TIMES = [
  { hour: 9, minute: 15, days: new Set([1, 2, 3, 4, 5]) },
  { hour: 15, minute: 15, days: new Set([1, 2, 3, 4, 5]) },
  { hour: 21, minute: 15, days: new Set([1, 2, 3, 4, 5]) },
  { hour: 5, minute: 15, days: new Set([2, 3, 4, 5, 6]) },
];

function nextUpdateAt(now = new Date()) {
  const shanghai = new Date(now.getTime() + SHANGHAI_OFFSET_HOURS * 3_600_000);
  const year = shanghai.getUTCFullYear();
  const month = shanghai.getUTCMonth();
  const day = shanghai.getUTCDate();
  let next = null;

  for (let offset = 0; offset < 8; offset += 1) {
    const localDay = new Date(Date.UTC(year, month, day + offset));
    for (const update of UPDATE_TIMES) {
      if (!update.days.has(localDay.getUTCDay())) continue;
      const candidate = new Date(
        Date.UTC(
          localDay.getUTCFullYear(),
          localDay.getUTCMonth(),
          localDay.getUTCDate(),
          update.hour - SHANGHAI_OFFSET_HOURS,
          update.minute,
        ),
      );
      if (candidate > now && (!next || candidate < next)) next = candidate;
    }
  }

  return next;
}

function scheduleUpdates(runUpdate, timers = {}) {
  const clock = {
    now: timers.now ?? (() => new Date()),
    setTimeout: timers.setTimeout ?? globalThis.setTimeout,
    clearTimeout: timers.clearTimeout ?? globalThis.clearTimeout,
  };
  let disposed = false;
  let timer;

  function arm() {
    const now = clock.now();
    timer = clock.setTimeout(async () => {
      try {
        await runUpdate();
      } finally {
        if (!disposed) arm();
      }
    }, nextUpdateAt(now).getTime() - now.getTime());
  }

  arm();
  return {
    dispose() {
      disposed = true;
      clock.clearTimeout(timer);
    },
  };
}

module.exports = { nextUpdateAt, scheduleUpdates };
