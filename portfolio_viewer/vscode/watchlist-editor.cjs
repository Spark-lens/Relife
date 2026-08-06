/* eslint-disable @typescript-eslint/no-require-imports */
const { randomUUID } = require("node:crypto");

const {
  addGroup,
  addItem,
  deleteGroup,
  moveGroupUp,
  moveGroupDown,
  moveItem,
  removeItem,
  renameGroup,
  updateItem,
} = require("./watchlist.cjs");

const DEFAULT_RULE = {
  enabled: true,
  timeframes: ["daily", "weekly", "monthly"],
  window: 20,
  standardDeviations: 2,
};
const TIMEFRAMES = [
  { label: "日线", value: "daily" },
  { label: "周线", value: "weekly" },
  { label: "月线", value: "monthly" },
];

function registerWatchlistCommands(vscode, options) {
  const getWatchlist = options.getWatchlist;
  const persist = options.persist;
  const makeId = options.randomId ?? randomUUID;

  function register(name, operation) {
    return vscode.commands.registerCommand(name, (...args) =>
      Promise.resolve(operation(...args)).catch((error) => {
        vscode.window.showErrorMessage(`Relife 观察列表更新失败：${error.message}`);
      }),
    );
  }

  return [
    register("relifePortfolio.watchlist.addGroup", async () => {
      const label = await vscode.window.showInputBox({
        prompt: "分组名称",
        validateInput: required,
      });
      if (label === undefined) return;
      await persist(addGroup(getWatchlist(), { id: `group-${makeId()}`, label: label.trim() }));
    }),

    register("relifePortfolio.watchlist.renameGroup", async (node) => {
      const label = await vscode.window.showInputBox({
        prompt: "新的分组名称",
        value: node.group.label,
        validateInput: required,
      });
      if (label === undefined) return;
      await persist(renameGroup(getWatchlist(), node.group.id, label.trim()));
    }),

    register("relifePortfolio.watchlist.deleteGroup", async (node) => {
      if (node.group.items.length) {
        const answer = await vscode.window.showWarningMessage(
          `删除分组“${node.group.label}”及其中 ${node.group.items.length} 个标的？`,
          { modal: true },
          "删除",
        );
        if (answer !== "删除") return;
      }
      await persist(deleteGroup(getWatchlist(), node.group.id));
    }),

    register("relifePortfolio.watchlist.moveGroupUp", async (node) => {
      await persist(moveGroupUp(getWatchlist(), node.group.id));
    }),

    register("relifePortfolio.watchlist.moveGroupDown", async (node) => {
      await persist(moveGroupDown(getWatchlist(), node.group.id));
    }),

    register("relifePortfolio.watchlist.addItem", async (node) => {
      const marketChoice = await vscode.window.showQuickPick(
        [{ label: "美股", value: "us" }, { label: "A股", value: "cn" }],
        { placeHolder: "选择市场" },
      );
      if (!marketChoice) return;
      const symbol = await vscode.window.showInputBox({
        prompt: "标的代码",
        validateInput: required,
      });
      if (symbol === undefined) return;
      const name = await vscode.window.showInputBox({
        prompt: "标的名称",
        value: symbol.trim().toUpperCase(),
        validateInput: required,
      });
      if (name === undefined) return;
      await persist(addItem(getWatchlist(), node.group.id, {
        market: marketChoice.value,
        symbol: symbol.trim().toUpperCase(),
        name: name.trim(),
        bollinger: { ...DEFAULT_RULE },
      }));
    }),

    register("relifePortfolio.watchlist.moveItem", async (node) => {
      const groups = getWatchlist().groups
        .filter((group) => group.id !== node.groupId)
        .map((group) => ({ label: group.label, groupId: group.id }));
      if (!groups.length) {
        vscode.window.showErrorMessage("没有可移动到的其他分组");
        return;
      }
      const target = await vscode.window.showQuickPick(groups, { placeHolder: "移动到分组" });
      if (!target) return;
      await persist(moveItem(
        getWatchlist(),
        node.watchItem.market,
        node.watchItem.symbol,
        target.groupId,
      ));
    }),

    register("relifePortfolio.watchlist.removeItem", async (node) => {
      const answer = await vscode.window.showWarningMessage(
        `从观察列表删除 ${node.watchItem.symbol}？`,
        { modal: true },
        "删除",
      );
      if (answer !== "删除") return;
      await persist(removeItem(getWatchlist(), node.watchItem.market, node.watchItem.symbol));
    }),

    register("relifePortfolio.watchlist.toggleItem", async (node) => {
      await persist(updateItem(getWatchlist(), node.watchItem.market, node.watchItem.symbol, {
        bollinger: {
          ...node.watchItem.bollinger,
          enabled: !node.watchItem.bollinger.enabled,
        },
      }));
    }),

    register("relifePortfolio.watchlist.configureItem", async (node) => {
      const selected = await vscode.window.showQuickPick(
        TIMEFRAMES.map((item) => ({
          ...item,
          picked: node.watchItem.bollinger.timeframes.includes(item.value),
        })),
        { canPickMany: true, placeHolder: "选择观察周期" },
      );
      if (!selected) return;
      if (!selected.length) throw new Error("至少选择一个观察周期");
      const windowValue = await vscode.window.showInputBox({
        prompt: "布林窗口",
        value: String(node.watchItem.bollinger.window),
        validateInput: positiveIntegerAtLeastTwo,
      });
      if (windowValue === undefined) return;
      const deviations = await vscode.window.showInputBox({
        prompt: "标准差倍数",
        value: String(node.watchItem.bollinger.standardDeviations),
        validateInput: positiveNumber,
      });
      if (deviations === undefined) return;
      await persist(updateItem(getWatchlist(), node.watchItem.market, node.watchItem.symbol, {
        bollinger: {
          ...node.watchItem.bollinger,
          timeframes: selected.map((item) => item.value),
          window: Number(windowValue),
          standardDeviations: Number(deviations),
        },
      }));
    }),
  ];
}

function required(value) {
  return value?.trim() ? undefined : "不能为空";
}

function positiveIntegerAtLeastTwo(value) {
  return Number.isInteger(Number(value)) && Number(value) >= 2
    ? undefined
    : "请输入不小于 2 的整数";
}

function positiveNumber(value) {
  return Number.isFinite(Number(value)) && Number(value) > 0
    ? undefined
    : "请输入正数";
}

module.exports = { registerWatchlistCommands };
