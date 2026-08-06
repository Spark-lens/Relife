import path from "node:path";

const REQUESTS = new Set([
  "ready", "refresh", "select-source", "reset-source", "open-portfolio", "open-strategy",
  "watchlist-add-category", "watchlist-rename-category", "watchlist-delete-category",
  "watchlist-move-category", "watchlist-add-symbol", "watchlist-edit-symbol",
  "watchlist-delete-symbol", "watchlist-move-symbol", "open-strategy-file",
]);

const PATTERNS = {
  us: { label: "tradingview_full_latest_YYYY-MM-DD.csv", expression: /^tradingview_full_latest_\d{4}-\d{2}-\d{2}\.csv$/ },
  cn: { label: "交割单_YYYY-MM-DD.csv", expression: /^交割单_\d{4}-\d{2}-\d{2}\.csv$/ },
};

export function defaultSources() {
  return Object.fromEntries(["us", "cn"].map((market) => [market, {
    market, mode: "sample", directory: null, pattern: PATTERNS[market].label, lastValidFile: null,
  }]));
}

export function sourceFromFile(market, file) {
  if (!PATTERNS[market]?.expression.test(path.basename(file))) throw new Error(`文件名不符合 ${PATTERNS[market]?.label ?? market}`);
  return { market, mode: "directory", directory: path.dirname(file), pattern: PATTERNS[market].label, lastValidFile: file };
}

export function allowedRequest(message) {
  return Boolean(message && typeof message.type === "string" && REQUESTS.has(message.type));
}
