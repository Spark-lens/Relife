/* eslint-disable @typescript-eslint/no-require-imports */
const { promises: fs } = require("node:fs");
const path = require("node:path");

const WATCHLIST = path.join("portfolio_viewer", "data", "watchlist.json");
const TIMEFRAMES = new Set(["daily", "weekly", "monthly"]);

function watchlistPath(repositoryRoot) {
  return path.join(repositoryRoot, WATCHLIST);
}

async function loadWatchlist(repositoryRoot, readFile = fs.readFile) {
  const data = JSON.parse(await readFile(watchlistPath(repositoryRoot), "utf8"));
  return validateWatchlist(data);
}

async function saveWatchlist(file, data, fileSystem = fs, pid = process.pid) {
  validateWatchlist(data);
  const temporary = `${file}.${pid}.tmp`;
  try {
    await fileSystem.writeFile(temporary, `${JSON.stringify(data, null, 2)}\n`, "utf8");
    await fileSystem.rename(temporary, file);
  } catch (error) {
    await fileSystem.unlink(temporary).catch(() => {});
    throw error;
  }
}

function validateWatchlist(data) {
  if (!data || !Array.isArray(data.groups)) {
    throw new Error("观察配置缺少 groups 数组");
  }
  const groupIds = new Set();
  const symbols = new Set();
  for (const group of data.groups) {
    const id = requireText(group?.id, "分组 id");
    requireText(group?.label, "分组名称");
    if (groupIds.has(id)) throw new Error(`重复分组 id ${id}`);
    groupIds.add(id);
    if (!Array.isArray(group.items)) throw new Error(`分组 ${id} 缺少 items 数组`);
    for (const item of group.items) validateItem(item, symbols);
  }
  return data;
}

function validateItem(item, symbols = new Set()) {
  const market = requireText(item?.market, "标的 market").toLowerCase();
  const symbol = requireText(item?.symbol, "标的 symbol").toUpperCase();
  requireText(item?.name, "标的 name");
  if (!new Set(["us", "cn"]).has(market)) throw new Error(`非法市场 ${market}`);
  const key = `${market}:${symbol}`;
  if (symbols.has(key)) throw new Error(`重复标的 ${key}`);
  symbols.add(key);
  const rule = item?.bollinger;
  if (!rule || typeof rule !== "object") throw new Error(`${key} 缺少 bollinger 配置`);
  if (typeof rule.enabled !== "boolean") throw new Error(`${key} 的 enabled 必须是布尔值`);
  if (!Array.isArray(rule.timeframes) || rule.timeframes.length === 0) {
    throw new Error(`${key} 至少需要一个观察周期`);
  }
  if (rule.timeframes.some((value) => !TIMEFRAMES.has(value))) {
    throw new Error(`${key} 包含非法周期`);
  }
  if (!Number.isInteger(rule.window) || rule.window < 2) {
    throw new Error(`${key} 的 window 必须是不小于 2 的整数`);
  }
  if (!Number.isFinite(rule.standardDeviations) || rule.standardDeviations <= 0) {
    throw new Error(`${key} 的 standardDeviations 必须是正数`);
  }
  return item;
}

function requireText(value, label) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label}不能为空`);
  return value.trim();
}

function edit(data, operation) {
  const next = structuredClone(data);
  operation(next);
  return validateWatchlist(next);
}

function groupById(data, groupId) {
  const group = data.groups.find((candidate) => candidate.id === groupId);
  if (!group) throw new Error(`找不到分组 ${groupId}`);
  return group;
}

function findItem(data, market, symbol) {
  const key = `${market.toLowerCase()}:${symbol.toUpperCase()}`;
  for (const group of data.groups) {
    const index = group.items.findIndex(
      (item) => `${item.market.toLowerCase()}:${item.symbol.toUpperCase()}` === key,
    );
    if (index >= 0) return { group, index };
  }
  throw new Error(`找不到标的 ${key}`);
}

function addGroup(data, group) {
  return edit(data, (next) => next.groups.push({ ...group, items: [] }));
}

function renameGroup(data, groupId, label) {
  return edit(data, (next) => {
    groupById(next, groupId).label = requireText(label, "分组名称");
  });
}

function deleteGroup(data, groupId) {
  return edit(data, (next) => {
    const index = next.groups.findIndex((group) => group.id === groupId);
    if (index < 0) throw new Error(`找不到分组 ${groupId}`);
    next.groups.splice(index, 1);
  });
}

function moveGroup(data, groupId, direction) {
  return edit(data, (next) => {
    const index = next.groups.findIndex((group) => group.id === groupId);
    if (index < 0) throw new Error(`找不到分组 ${groupId}`);
    const target = index + direction;
    if (target < 0 || target >= next.groups.length) return; // 已在边界
    const [group] = next.groups.splice(index, 1);
    next.groups.splice(target, 0, group);
  });
}

function moveGroupUp(data, groupId) {
  return moveGroup(data, groupId, -1);
}

function moveGroupDown(data, groupId) {
  return moveGroup(data, groupId, 1);
}

function addItem(data, groupId, item) {
  return edit(data, (next) => groupById(next, groupId).items.push(item));
}

function moveItem(data, market, symbol, targetGroupId) {
  return edit(data, (next) => {
    const source = findItem(next, market, symbol);
    const [item] = source.group.items.splice(source.index, 1);
    groupById(next, targetGroupId).items.push(item);
  });
}

function removeItem(data, market, symbol) {
  return edit(data, (next) => {
    const found = findItem(next, market, symbol);
    found.group.items.splice(found.index, 1);
  });
}

function updateItem(data, market, symbol, changes) {
  return edit(data, (next) => {
    const found = findItem(next, market, symbol);
    found.group.items[found.index] = { ...found.group.items[found.index], ...changes };
  });
}

module.exports = {
  WATCHLIST,
  addGroup,
  addItem,
  deleteGroup,
  moveGroupUp,
  moveGroupDown,
  loadWatchlist,
  moveItem,
  removeItem,
  renameGroup,
  saveWatchlist,
  updateItem,
  validateItem,
  validateWatchlist,
  watchlistPath,
};
