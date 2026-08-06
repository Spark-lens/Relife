import { randomUUID } from "node:crypto";

const DEFAULT_SYMBOLS = [
  // 示例行情仅用于目录首屏预览，刷新后会优先使用快照行情。
  ["现金流", [["us", "BOXX", "短债", 117.70, 0.42, 0.0036]]],
  ["大盘", [["us", "QQQ", "纳斯达克100 ETF", 53.84, 0.80, 0.0151], ["cn", "512100", "中证1000 ETF", 1.58, -0.03, -0.0156], ["cn", "159915", "创业板 ETF", 2.41, 0.03, 0.0126]]],
  ["股息", [["us", "XQQI", "期权股息", 48.46, 1.04, 0.0219], ["us", "QQQI", "期权股息", 53.84, 0.80, 0.0151], ["us", "SCHD", "红利股息", 33.56, 0.09, 0.0027], ["cn", "513530", "港股红利", 1.58, -0.03, -0.0156]]],
  ["个股", [["us", "BRK.B", "伯克希尔", 513.14, 1.60, 0.0031], ["us", "KO", "可口可乐", 68.22, 0.31, 0.0046], ["us", "GOOGL", "Alphabet", 192.37, 2.10, 0.0110], ["us", "JD", "京东", 32.14, -0.22, -0.0068]]],
  ["杠杆", [["us", "SOXL", "半导体做多", 45.12, 0.68, 0.0153], ["us", "SOXS", "半导体做空", 17.60, -0.21, -0.0118], ["us", "YINN", "中国做多", 38.40, 0.35, 0.0092], ["us", "YANG", "中国做空", 39.13, -0.44, -0.0111], ["us", "TQQQ", "纳指做多", 84.22, 1.30, 0.0157], ["us", "SQQQ", "纳指做空", 13.11, -0.18, -0.0135]]],
  ["比特币", [["us", "IBIT", "比特币 ETF", 61.20, 0.72, 0.0119]]],
];

export function displayName(item) {
  return String(item?.note || "").trim() || item?.name || item?.symbol || "未命名标的";
}

export function defaultWatchlist() {
  return {
    categories: DEFAULT_SYMBOLS.map(([name, entries], index) => ({
      id: `category-${index + 1}`,
      name,
      symbols: entries.map(([market, symbol, itemName, latest, change, changePercent]) => ({ market, symbol, name: itemName, note: "", key: `${market}:${symbol}`, latest, change, changePercent })),
    })),
  };
}

export function addSymbol(state, categoryId, symbol) {
  const normalized = { ...symbol, market: symbol.market.toLowerCase(), symbol: symbol.symbol.trim().toUpperCase() };
  normalized.key = `${normalized.market}:${normalized.symbol}`;
  for (const category of state.categories) {
    if (category.symbols.some((item) => item.key === normalized.key)) {
      const error = new Error(`${normalized.key} 已存在`);
      error.code = "DUPLICATE_SYMBOL";
      error.existingCategoryId = category.id;
      throw error;
    }
  }
  return { categories: state.categories.map((category) => category.id === categoryId ? { ...category, symbols: [...category.symbols, normalized] } : category) };
}

export function addCategory(state, name) {
  const normalized = name.trim();
  if (!normalized) throw new Error("分类名称不能为空");
  if (state.categories.some((item) => item.name === normalized)) throw new Error(`分类 ${normalized} 已存在`);
  return { categories: [...state.categories, { id: `category-${randomUUID()}`, name: normalized, symbols: [] }] };
}

export function renameCategory(state, categoryId, name) {
  const normalized = name.trim();
  if (!normalized) throw new Error("分类名称不能为空");
  if (state.categories.some((item) => item.id !== categoryId && item.name === normalized)) throw new Error(`分类 ${normalized} 已存在`);
  return { categories: state.categories.map((item) => item.id === categoryId ? { ...item, name: normalized } : item) };
}

export function removeCategory(state, categoryId) {
  return { categories: state.categories.filter((item) => item.id !== categoryId) };
}

export function editSymbol(state, key, changes) {
  const existing = state.categories.flatMap((item) => item.symbols).find((item) => item.key === key);
  if (!existing) return state;
  const next = {
    ...existing, ...changes,
    market: (changes.market ?? existing.market).toLowerCase(),
    symbol: (changes.symbol ?? existing.symbol).trim().toUpperCase(),
  };
  next.key = `${next.market}:${next.symbol}`;
  if (next.key !== key && state.categories.some((category) => category.symbols.some((item) => item.key === next.key))) {
    const error = new Error(`${next.key} 已存在`);
    error.code = "DUPLICATE_SYMBOL";
    throw error;
  }
  return { categories: state.categories.map((category) => ({ ...category, symbols: category.symbols.map((item) => item.key === key ? next : item) })) };
}

export function removeSymbol(state, key) {
  return { categories: state.categories.map((category) => ({ ...category, symbols: category.symbols.filter((item) => item.key !== key) })) };
}

export function moveCategory(state, categoryId, offset) {
  const categories = [...state.categories];
  const from = categories.findIndex((item) => item.id === categoryId);
  const to = Math.max(0, Math.min(categories.length - 1, from + offset));
  if (from < 0 || from === to) return state;
  categories.splice(to, 0, categories.splice(from, 1)[0]);
  return { categories };
}

export function moveSymbol(state, key, targetCategoryId, targetIndex) {
  let moving;
  const categories = state.categories.map((category) => ({
    ...category,
    symbols: category.symbols.filter((symbol) => {
      if (symbol.key === key) moving = symbol;
      return symbol.key !== key;
    }),
  }));
  if (!moving) return state;
  return { categories: categories.map((category) => {
    if (category.id !== targetCategoryId) return category;
    const symbols = [...category.symbols];
    symbols.splice(Math.max(0, Math.min(symbols.length, targetIndex)), 0, moving);
    return { ...category, symbols };
  }) };
}
