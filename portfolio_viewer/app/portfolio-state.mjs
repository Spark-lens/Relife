export const DEFAULT_COLUMNS = [
  { id: "symbol", label: "标的", visible: true, locked: true },
  { id: "quantity", label: "数量", visible: true },
  { id: "weight", label: "占总组合比例", visible: true },
  { id: "totalCost", label: "总成本", visible: true },
  { id: "lastClose", label: "最后收盘价", visible: true },
  { id: "marketValue", label: "总现价", visible: true },
  { id: "dailyPnl", label: "每日收益", visible: true },
  { id: "dailyPnlPct", label: "每日收益%", visible: true },
  { id: "totalPnl", label: "总收益", visible: true },
  { id: "totalPnlPct", label: "总收益%", visible: true },
  { id: "unrealizedPnl", label: "未实现收益", visible: true },
  { id: "unrealizedPnlPct", label: "未实现收益%", visible: true },
  {
    id: "portfolioContributionPct",
    label: "组合贡献%",
    visible: true,
  },
];


export function selectRange(points, range, now = new Date()) {
  if (range === "all") return points;
  const cutoff = new Date(now);
  if (range === "1m") cutoff.setMonth(cutoff.getMonth() - 1);
  if (range === "3m") cutoff.setMonth(cutoff.getMonth() - 3);
  if (range === "ytd") cutoff.setMonth(0, 1);
  const cutoffDate = cutoff.toISOString().slice(0, 10);
  return points.filter((point) => point.date >= cutoffDate);
}


export function filterTransactions(transactions, filter) {
  const cashKinds = new Set(["deposit", "withdrawal", "interest", "cash"]);
  return [...transactions]
    .filter((row) => {
      if (filter === "all") return true;
      if (filter === "cash") return cashKinds.has(row.kind);
      return row.kind === filter;
    })
    .sort((left, right) => right.timestamp.localeCompare(left.timestamp));
}


export function normalizeColumnPreferences(value) {
  const defaults = new Map(
    DEFAULT_COLUMNS.map((column) => [column.id, { ...column }]),
  );
  const result = [];
  const seen = new Set();

  if (Array.isArray(value)) {
    for (const preference of value) {
      const source = defaults.get(preference?.id);
      if (!source || seen.has(source.id) || source.locked) continue;
      result.push({
        ...source,
        visible: preference.visible !== false,
      });
      seen.add(source.id);
    }
  }

  const symbol = { ...defaults.get("symbol"), visible: true };
  const remaining = DEFAULT_COLUMNS
    .filter((column) => column.id !== "symbol" && !seen.has(column.id))
    .map((column) => ({ ...column }));
  return [symbol, ...result, ...remaining];
}


export function formatMoney(value, currency) {
  if (value == null) return "—";
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(value);
}


export function formatNumber(value, maximumFractionDigits = 2) {
  if (value == null) return "—";
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits,
  }).format(value);
}


export function formatPercent(value) {
  if (value == null) return "—";
  return new Intl.NumberFormat("zh-CN", {
    style: "percent",
    signDisplay: "exceptZero",
    maximumFractionDigits: 2,
  }).format(value);
}
