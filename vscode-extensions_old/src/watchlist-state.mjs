import { randomUUID } from "node:crypto";

const DEFAULT_NAMES = ["现金流", "大盘", "股息", "个股", "杠杆", "比特币"];

export function defaultWatchlist() {
  return { categories: DEFAULT_NAMES.map((name, index) => ({ id: `category-${index + 1}`, name, symbols: [] })) };
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
