import { portfolioSeed } from "./seed-data.js";

const state = {
  market: "ALL",
  account: "ALL",
  alert: "ALL",
  search: "",
};

const marketOrder = new Map([
  ["US", 0],
  ["CN", 1],
  ["CASH", 2],
]);

const accountOrder = new Map([
  ["schwab", 0],
  ["ibkr", 1],
  ["yinhe", 2],
]);

const marketFilters = [
  { value: "ALL", label: "全部" },
  { value: "US", label: "美股" },
  { value: "CN", label: "A股" },
  { value: "CASH", label: "现金" },
];

const accountFilters = [
  { value: "ALL", label: "全部" },
  { value: "schwab", label: "嘉信" },
  { value: "ibkr", label: "盈透" },
  { value: "yinhe", label: "银河证券" },
];

const alertFilters = [
  { value: "ALL", label: "全部" },
  { value: "enabled", label: "已启用" },
  { value: "disabled", label: "未启用" },
];

const el = {
  snapshotLabel: document.querySelector("#portfolio-snapshot-label"),
  heroMetrics: document.querySelector("#hero-metrics"),
  portfolioChart: document.querySelector("#portfolio-chart"),
  chartBadge: document.querySelector("#chart-badge"),
  chartStart: document.querySelector("#chart-start"),
  chartEnd: document.querySelector("#chart-end"),
  marketFilters: document.querySelector("#market-filters"),
  accountFilters: document.querySelector("#account-filters"),
  alertFilters: document.querySelector("#alert-filters"),
  searchInput: document.querySelector("#search-input"),
  summaryGrid: document.querySelector("#summary-grid"),
  holdingsBody: document.querySelector("#holdings-table-body"),
  cashBody: document.querySelector("#cash-table-body"),
  marketDistribution: document.querySelector("#market-distribution"),
  accountDistribution: document.querySelector("#account-distribution"),
  marketTotalLabel: document.querySelector("#market-total-label"),
  accountTotalLabel: document.querySelector("#account-total-label"),
  statusQuotes: document.querySelector("#status-quotes"),
  statusAlerts: document.querySelector("#status-alerts"),
  statusMail: document.querySelector("#status-mail"),
};

function addThousands(value, decimals = 2) {
  const safeValue = Number.isFinite(value) ? value : 0;
  const sign = safeValue < 0 ? "-" : "";
  const fixed = Math.abs(safeValue).toFixed(decimals);
  const [integerPart, fractionalPart] = fixed.split(".");
  const groupedInteger = integerPart.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return `${sign}${groupedInteger}${decimals > 0 ? `.${fractionalPart}` : ""}`;
}

function trimTrailingZeros(text) {
  return text.replace(/\.?0+$/, "");
}

function formatCurrency(value, currency) {
  const prefix = currency === "CNY" ? "¥" : "$";
  return `${prefix}${addThousands(value, 2)}`;
}

function formatNumber(value) {
  return trimTrailingZeros(addThousands(value, 4));
}

function formatCompact(value) {
  const safeValue = Number.isFinite(value) ? Math.abs(value) : 0;
  const sign = value < 0 ? "-" : "";
  if (safeValue >= 1e8) return `${sign}${(safeValue / 1e8).toFixed(2)}亿`;
  if (safeValue >= 1e4) return `${sign}${(safeValue / 1e4).toFixed(2)}万`;
  return `${sign}${trimTrailingZeros(addThousands(safeValue, 2))}`;
}

function formatPercent(value) {
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
}

function hashString(input) {
  let hash = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function calcMetrics(item) {
  const costValue = item.quantity * item.avgCost;
  const marketValue = item.quantity * item.currentPrice;
  const pnl = marketValue - costValue;
  const pnlPct = costValue === 0 ? 0 : pnl / costValue;

  return {
    ...item,
    costValue,
    marketValue,
    pnl,
    pnlPct,
  };
}

function getAllHoldings() {
  return portfolioSeed.holdings.map(calcMetrics);
}

function getAllCash() {
  return portfolioSeed.cashBalances.map((item) => ({ ...item }));
}

function filteredHoldings() {
  const query = state.search.trim().toLowerCase();
  return getAllHoldings()
    .filter((item) => (state.market === "ALL" ? true : item.market === state.market))
    .filter((item) => (state.account === "ALL" ? true : item.account === state.account))
    .filter((item) => {
      if (state.alert === "ALL") {
        return true;
      }
      return state.alert === "enabled" ? item.alertEnabled : !item.alertEnabled;
    })
    .filter((item) => {
      if (!query) {
        return true;
      }
      const haystack = [item.symbol, item.name, item.account, item.broker, item.marketLabel, item.assetClass]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    })
    .sort((a, b) => {
      const alertRank = Number(b.alertEnabled) - Number(a.alertEnabled);
      if (alertRank !== 0) return alertRank;
      if (a.priority !== b.priority) return a.priority - b.priority;
      const marketRank = (marketOrder.get(a.market) ?? 99) - (marketOrder.get(b.market) ?? 99);
      if (marketRank !== 0) return marketRank;
      const accountRank = (accountOrder.get(a.account) ?? 99) - (accountOrder.get(b.account) ?? 99);
      if (accountRank !== 0) return accountRank;
      return a.symbol.localeCompare(b.symbol);
    });
}

function groupedDistribution(items, field) {
  const totals = new Map();
  const grandTotal = items.reduce((sum, item) => sum + item.marketValue, 0);
  for (const item of items) {
    const key = item[field];
    const existing = totals.get(key) ?? { value: 0, currency: item.currency, items: [] };
    existing.value += item.marketValue;
    existing.items.push(item);
    if (existing.currency !== item.currency) {
      existing.currency = "MIXED";
    }
    totals.set(key, existing);
  }
  return [...totals.entries()]
    .map(([key, payload]) => ({
      key,
      value: payload.value,
      currency: payload.currency,
      items: payload.items,
      ratio: grandTotal === 0 ? 0 : payload.value / grandTotal,
    }))
    .sort((a, b) => b.value - a.value);
}

function renderFilterGroup(container, filters, key) {
  container.innerHTML = filters
    .map(
      (filter) => `
        <button
          type="button"
          class="segment-btn ${state[key] === filter.value ? "active" : ""}"
          data-key="${key}"
          data-value="${filter.value}"
        >
          ${filter.label}
        </button>
      `,
    )
    .join("");
}

function renderSummary(items, cashItems) {
  const totalByCurrency = new Map();
  const pnlByCurrency = new Map();

  for (const item of items) {
    totalByCurrency.set(item.currency, (totalByCurrency.get(item.currency) ?? 0) + item.marketValue);
    pnlByCurrency.set(item.currency, (pnlByCurrency.get(item.currency) ?? 0) + item.pnl);
  }

  const cashByCurrency = new Map();
  for (const cash of cashItems) {
    cashByCurrency.set(cash.currency, (cashByCurrency.get(cash.currency) ?? 0) + cash.cash);
  }

  const enabledCount = items.filter((item) => item.alertEnabled).length;
  const usdMarket = totalByCurrency.get("USD") ?? 0;
  const cnyMarket = totalByCurrency.get("CNY") ?? 0;
  const cnyCash = cashByCurrency.get("CNY") ?? 0;
  const usdPnl = pnlByCurrency.get("USD") ?? 0;
  const cnyPnl = pnlByCurrency.get("CNY") ?? 0;

  const cards = [
    {
      label: "USD 持仓市值",
      value: formatCurrency(usdMarket, "USD"),
      subvalue: `浮盈亏 ${formatCurrency(usdPnl, "USD")}`,
      trend: usdPnl >= 0 ? "up" : "down",
      trendLabel: usdPnl >= 0 ? "USD 浮盈" : "USD 浮亏",
    },
    {
      label: "CNY 持仓市值",
      value: formatCurrency(cnyMarket, "CNY"),
      subvalue: `浮盈亏 ${formatCurrency(cnyPnl, "CNY")}`,
      trend: cnyPnl >= 0 ? "up" : "down",
      trendLabel: cnyPnl >= 0 ? "CNY 浮盈" : "CNY 浮亏",
    },
    {
      label: "现金余额",
      value: formatCurrency(cnyCash, "CNY"),
      subvalue: "IBKR 现金单独展示",
      trend: "flat",
      trendLabel: "现金独立",
    },
    {
      label: "启用预警",
      value: `${enabledCount} 个标的`,
      subvalue: `当前筛选下共有 ${items.length} 条持仓`,
      trend: "up",
      trendLabel: "优先排序在前",
    },
  ];

  el.summaryGrid.innerHTML = cards
    .map(
      (card) => `
        <article class="summary-card">
          <div class="topline">
            <span class="label">${card.label}</span>
            <span class="trend ${card.trend}">${card.trendLabel}</span>
          </div>
          <div class="value">${card.value}</div>
          <div class="subvalue">${card.subvalue}</div>
        </article>
      `,
    )
    .join("");

  const heroMetrics = [
    {
      label: "组合条目",
      value: `${items.length}`,
      subvalue: "按市场、账户、预警多条件筛选",
    },
    {
      label: "USD 持仓",
      value: formatCurrency(usdMarket, "USD"),
      subvalue: "美股持仓按原币种展示",
    },
    {
      label: "CNY 持仓",
      value: formatCurrency(cnyMarket, "CNY"),
      subvalue: "A 股持仓按原币种展示",
    },
    {
      label: "现金",
      value: formatCurrency(cnyCash, "CNY"),
      subvalue: "IBKR 现金单独展示",
    },
  ];

  el.heroMetrics.innerHTML = heroMetrics
    .map(
      (item) => `
        <div class="hero-stat">
          <div class="label">${item.label}</div>
          <div class="value">${item.value}</div>
          <div class="subvalue">${item.subvalue}</div>
        </div>
      `,
    )
    .join("");

  el.marketTotalLabel.textContent = `${formatCurrency(usdMarket, "USD")} + ${formatCurrency(cnyMarket, "CNY")}`;
  el.accountTotalLabel.textContent = `${items.length} positions`;
}

function renderChart(items) {
  const svg = el.portfolioChart;
  const width = 900;
  const height = 300;
  const paddingX = 28;
  const paddingY = 28;
  if (!items.length) {
    svg.innerHTML = `
      <rect x="0" y="0" width="900" height="300" fill="transparent"></rect>
      <text x="450" y="150" fill="rgba(146, 163, 186, 0.9)" text-anchor="middle" font-size="22">
        无可展示持仓
      </text>
    `;
    el.chartBadge.textContent = "0 只标的";
    el.chartStart.textContent = "-";
    el.chartEnd.textContent = "-";
    return;
  }

  const series = buildSeries(items, 34);
  const min = Math.min(...series);
  const max = Math.max(...series);
  const range = max - min || 1;
  const step = (width - paddingX * 2) / (series.length - 1);

  const points = series.map((value, index) => {
    const x = paddingX + index * step;
    const y = height - paddingY - ((value - min) / range) * (height - paddingY * 2);
    return { x, y, value };
  });

  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");

  const areaPath = `${linePath} L ${points.at(-1).x.toFixed(2)} ${height - paddingY} L ${points[0].x.toFixed(2)} ${height - paddingY} Z`;
  const last = points.at(-1)?.value ?? 0;
  const first = points[0]?.value ?? 0;
  const deltaPct = first === 0 ? 0 : (last - first) / first;

  svg.innerHTML = `
    <defs>
      <linearGradient id="chartFill" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#60a5fa" stop-opacity="0.42" />
        <stop offset="100%" stop-color="#60a5fa" stop-opacity="0.02" />
      </linearGradient>
      <linearGradient id="chartLine" x1="0" x2="1">
        <stop offset="0%" stop-color="#60a5fa" />
        <stop offset="55%" stop-color="#34d399" />
        <stop offset="100%" stop-color="#fbbf24" />
      </linearGradient>
    </defs>

    <g opacity="0.25">
      ${Array.from({ length: 6 }, (_, index) => {
        const y = paddingY + index * ((height - paddingY * 2) / 5);
        return `<line x1="${paddingX}" x2="${width - paddingX}" y1="${y}" y2="${y}" stroke="rgba(160, 180, 210, 0.18)" />`;
      }).join("")}
    </g>

    <path d="${areaPath}" fill="url(#chartFill)"></path>
    <path d="${linePath}" fill="none" stroke="url(#chartLine)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></path>
    ${points
      .map(
        (point, index) => `
          <circle cx="${point.x}" cy="${point.y}" r="${index === points.length - 1 ? 5 : 3.2}" fill="${index === points.length - 1 ? "#fbbf24" : "#60a5fa"}" opacity="${index === points.length - 1 ? 1 : 0.7}"></circle>
        `,
      )
      .join("")}
  `;

  el.chartBadge.textContent = `${formatPercent(deltaPct)} · ${items.length} 只标的`;
  el.chartStart.textContent = `start ${formatNumber(first)}`;
  el.chartEnd.textContent = `now ${formatNumber(last)}`;
}

function buildSeries(items, length) {
  const base = items.reduce((sum, item) => sum + item.marketValue, 0) || 1;
  const seed = hashString(items.map((item) => item.symbol).join("|") || "Relife");
  const random = mulberry32(seed);
  let level = 0.94 + random() * 0.08;
  const series = [];

  for (let index = 0; index < length; index += 1) {
    const drift = (random() - 0.46) * 0.045;
    level = Math.max(0.84, Math.min(1.16, level + drift));
    series.push(base * level);
  }

  if (series.length > 0) {
    series[0] = base * 0.92;
    series[series.length - 1] = base;
  }

  return series;
}

function renderDistribution(items) {
  const marketGroups = groupedDistribution(items, "market");
  const accountGroups = groupedDistribution(items, "account");

  el.marketDistribution.innerHTML = renderStackBars(marketGroups, (group) => group.key);
  el.accountDistribution.innerHTML = renderStackBars(accountGroups, (group) => accountLabel(group.key));
}

function renderStackBars(groups, labelResolver) {
  if (!groups.length) {
    return `<div class="stack-row"><span class="muted">无数据</span></div>`;
  }

  const palette = ["#60a5fa", "#34d399", "#fbbf24", "#fb7185"];
  return groups
    .map(
      (group, index) => `
        <div class="stack-row">
          <div class="stack-row-head">
            <span>${labelResolver(group)}</span>
            <span class="muted">${formatPercent(group.ratio)} · ${
              group.currency === "MIXED" ? "mixed" : formatCurrency(group.value, group.currency)
            }</span>
          </div>
          <div class="bar">
            <span style="width:${Math.max(6, group.ratio * 100)}%; background:${palette[index % palette.length]};"></span>
          </div>
        </div>
      `,
    )
    .join("");
}

function accountLabel(account) {
  if (account === "schwab") return "嘉信";
  if (account === "ibkr") return "盈透";
  if (account === "yinhe") return "银河证券";
  return account;
}

function renderHoldingsTable(items) {
  if (!items.length) {
    el.holdingsBody.innerHTML = `
      <tr>
        <td colspan="12">
          <div class="empty-state">
            没有符合当前筛选条件的持仓
          </div>
        </td>
      </tr>
    `;
    return;
  }

  el.holdingsBody.innerHTML = items
    .map((item) => {
      const pnlClass = item.pnl > 0 ? "positive" : item.pnl < 0 ? "negative" : "neutral";
      const alertTone = item.alertEnabled ? item.alertTone : "blue";
      const alertText = item.alertEnabled ? item.alertState : "未启用";
      const alertLabel = item.alertEnabled ? "已启用" : "未启用";

      return `
        <tr>
          <td>
            <span class="badge ${item.market === "US" ? "green" : "blue"}">${item.marketLabel}</span>
          </td>
          <td>${accountLabel(item.account)}</td>
          <td>
            <div class="symbol-cell">
              <span class="symbol">${item.symbol}</span>
              <span class="symbol-sub">${item.assetClass}</span>
            </div>
          </td>
          <td>
            <div class="symbol-cell">
              <span class="symbol">${item.name}</span>
              <span class="symbol-sub">${item.broker}</span>
            </div>
          </td>
          <td class="num">${formatNumber(item.quantity)}</td>
          <td class="num">${formatCurrency(item.avgCost, item.currency)}</td>
          <td class="num">${formatCurrency(item.currentPrice, item.currency)}</td>
          <td class="num">${formatCurrency(item.marketValue, item.currency)}</td>
          <td class="num pnl ${pnlClass}">${formatCurrency(item.pnl, item.currency)}</td>
          <td class="num pnl ${pnlClass}">${formatPercent(item.pnlPct)}</td>
          <td class="num">${item.priority}</td>
          <td>
            <span class="badge ${alertTone}">
              ${alertLabel} · ${alertText}
            </span>
          </td>
        </tr>
      `;
    })
    .join("");
}

function renderCashTable(items) {
  el.cashBody.innerHTML = items
    .map(
      (item) => `
        <tr>
          <td>${accountLabel(item.account)}</td>
          <td>${item.broker}</td>
          <td>${item.currency}</td>
          <td class="num">${formatCurrency(item.cash, item.currency)}</td>
        </tr>
      `,
    )
    .join("");
}

function syncStatus() {
  el.snapshotLabel.textContent = portfolioSeed.snapshotLabel;
  el.statusQuotes.textContent = portfolioSeed.quotesUpdatedAt;
  el.statusAlerts.textContent = portfolioSeed.alertsCheckedAt;
  el.statusMail.textContent = portfolioSeed.mailStatus;
}

function bindFilters() {
  renderFilterGroup(el.marketFilters, marketFilters, "market");
  renderFilterGroup(el.accountFilters, accountFilters, "account");
  renderFilterGroup(el.alertFilters, alertFilters, "alert");

  el.marketFilters.addEventListener("click", handleFilterClick);
  el.accountFilters.addEventListener("click", handleFilterClick);
  el.alertFilters.addEventListener("click", handleFilterClick);
  el.searchInput.addEventListener("input", (event) => {
    state.search = event.target.value;
    render();
  });
}

function handleFilterClick(event) {
  const button = event.target.closest("button[data-key][data-value]");
  if (!button) return;
  const { key, value } = button.dataset;
  state[key] = value;
  render();
}

function render() {
  const items = filteredHoldings();
  const cashItems = getAllCash();

  renderFilterGroup(el.marketFilters, marketFilters, "market");
  renderFilterGroup(el.accountFilters, accountFilters, "account");
  renderFilterGroup(el.alertFilters, alertFilters, "alert");
  renderSummary(items, cashItems);
  renderChart(items);
  renderDistribution(items);
  renderHoldingsTable(items);
  renderCashTable(cashItems);
  syncStatus();
}

bindFilters();
render();
