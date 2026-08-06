import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { registerWatchlistCommands } = require("../vscode/watchlist-editor.cjs");

function config() {
  return {
    groups: [{
      id: "broad",
      label: "大盘",
      items: [{
        market: "us",
        symbol: "QQQ",
        name: "纳指",
        bollinger: {
          enabled: true,
          timeframes: ["daily", "weekly", "monthly"],
          window: 20,
          standardDeviations: 2,
        },
      }],
    }],
  };
}

test("registers every native watchlist edit command and toggles an item", async () => {
  const commands = new Map();
  const errors = [];
  let current = config();
  const vscode = {
    commands: {
      registerCommand(name, callback) {
        commands.set(name, callback);
        return { dispose() {} };
      },
    },
    window: {
      showErrorMessage: (message) => errors.push(message),
    },
  };

  const disposables = registerWatchlistCommands(vscode, {
    getWatchlist: () => current,
    persist: async (next) => { current = next; },
    randomId: () => "fixed-id",
  });

  assert.deepEqual([...commands.keys()].sort(), [
    "relifePortfolio.watchlist.addGroup",
    "relifePortfolio.watchlist.addItem",
    "relifePortfolio.watchlist.configureItem",
    "relifePortfolio.watchlist.deleteGroup",
    "relifePortfolio.watchlist.moveGroupDown",
    "relifePortfolio.watchlist.moveGroupUp",
    "relifePortfolio.watchlist.moveItem",
    "relifePortfolio.watchlist.removeItem",
    "relifePortfolio.watchlist.renameGroup",
    "relifePortfolio.watchlist.toggleItem",
  ]);
  assert.equal(disposables.length, 10);

  await commands.get("relifePortfolio.watchlist.toggleItem")({
    watchItem: current.groups[0].items[0],
  });

  assert.equal(current.groups[0].items[0].bollinger.enabled, false);
  assert.deepEqual(errors, []);
});
