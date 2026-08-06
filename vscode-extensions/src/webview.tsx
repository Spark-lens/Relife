import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./webview.css";

declare function acquireVsCodeApi(): { postMessage(message: unknown): void };
const vscode = acquireVsCodeApi();
const post = (type: string, data: Record<string, unknown> = {}) => vscode.postMessage({ type, ...data });

type AppState = {
  snapshot: any;
  watchlist: any;
  sources: any;
  refresh: any;
  error: any;
  content: { view: "portfolio" | "watchlist"; title?: string; symbol?: string };
};

function useHostState(): AppState {
  const [state, setState] = useState<AppState>({
    snapshot: null, watchlist: null, sources: null, refresh: null, error: null,
    content: { view: "portfolio" },
  });
  useEffect(() => {
    const listener = (event: MessageEvent) => {
      const message = event.data;
      if (message.type === "snapshot") setState((old) => ({ ...old, snapshot: message.data, error: null }));
      if (message.type === "watchlist") setState((old) => ({ ...old, watchlist: message.data }));
      if (message.type === "source-status") setState((old) => ({ ...old, sources: message.data }));
      if (message.type === "refresh-status") setState((old) => ({ ...old, refresh: message }));
      if (message.type === "error") setState((old) => ({ ...old, error: message }));
      if (message.type === "content-view") setState((old) => ({ ...old, content: message.data || { view: "portfolio" } }));
    };
    window.addEventListener("message", listener);
    post("ready");
    return () => window.removeEventListener("message", listener);
  }, []);
  return state;
}

const money = (value: number | null | undefined, currency = "USD") => {
  if (value == null) return "—";
  const formatted = new Intl.NumberFormat("en-US", { style: "currency", currency, minimumFractionDigits: 2 }).format(value);
  // 去除 CNY/HKD 等货币的本地化前缀（CN¥ → ¥, HK$ → $, NT$ → $, ¥/$/£/₩ 保持不变）
  return formatted.replace(/^(CN|HK|NT|MOP|SGP|MYA|THB)[\$¥]/, (m) => m.replace(/^CN|HK|NT|MOP|SGP|MYA|THB/, ""));
};
const number = (value: number | null | undefined, digits = 2) => value == null ? "—" : new Intl.NumberFormat("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
const signed = (value: number | null | undefined, suffix = "") => value == null ? "—" : `${value >= 0 ? "+" : ""}${number(value)}${suffix}`;
const percent = (value: number | null | undefined) => value == null ? "—" : `${value >= 0 ? "+" : ""}${number(value * 100)}%`;
const tone = (value: number | null | undefined) => value == null || value === 0 ? "flat" : value > 0 ? "up" : "down";

function SymbolIcon({ item }: { item: any }) {
  if (typeof item.icon === "string" && item.icon) return <img className="symbol-icon own" src={item.icon} alt="" />;
  const market = item.market === "cn" ? "cn" : item.market === "us" ? "us" : "fallback";
  return <span className={`symbol-icon ${market}`} aria-label={market === "cn" ? "中国市场" : market === "us" ? "美国市场" : "默认市场图标"} />;
}

function Sidebar({ state }: { state: AppState }) {
  const watchlist = state.watchlist || { categories: [] };
  const prices = useMemo(() => {
    const map = new Map<string, any>();
    for (const market of ["us", "cn"]) for (const row of state.snapshot?.markets?.[market]?.prices || []) map.set(`${market}:${row.symbol}`.toUpperCase(), row);
    return map;
  }, [state.snapshot]);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({ home: true, symbols: true, strategy: true });
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({});
  const [pending, setPending] = useState<{ key: string; categoryId: string } | null>(null);
  useEffect(() => {
    if (state.error?.code === "DUPLICATE_SYMBOL" && pending && window.confirm("该标的已存在。是否移动到目标分类？")) {
      post("watchlist-move-symbol", { key: pending.key, targetCategoryId: pending.categoryId, targetIndex: 9999 });
    }
    if (state.error) setPending(null);
  }, [state.error]);
  const addCategory = () => { const name = window.prompt("新分类名称"); if (name) post("watchlist-add-category", { name }); };
  const addSymbol = (categoryId: string) => {
    const market = (window.prompt("市场：us 或 cn", "us") || "").toLowerCase();
    if (!market) return;
    const symbol = window.prompt("标的代码"); if (!symbol) return;
    const name = window.prompt("中文名称", symbol) || symbol;
    const note = window.prompt("备注（可留空）", "") || "";
    setPending({ key: `${market}:${symbol.toUpperCase()}`, categoryId });
    post("watchlist-add-symbol", { categoryId, symbol: { market, symbol, name, note } });
  };
  const moveSymbol = (key: string, categoryId: string, index: number) => post("watchlist-move-symbol", { key, targetCategoryId: categoryId, targetIndex: index });
  const toggle = (key: string) => setExpanded((old) => ({ ...old, [key]: !old[key] }));
  const toggleCategory = (key: string) => setExpandedCategories((old) => ({ ...old, [key]: old[key] === false }));
  const editItem = (item: any) => {
    const market = window.prompt("市场：us 或 cn", item.market); if (!market) return;
    const symbol = window.prompt("标的代码", item.symbol); if (!symbol) return;
    const name = window.prompt("中文名称", item.name); if (name == null) return;
    const note = window.prompt("备注（留空则显示中文名称）", item.note || "");
    post("watchlist-edit-symbol", { key: item.key, symbol: { market, symbol, name, note: note ?? item.note } });
  };
  const rootRow = (key: string, label: string, action?: React.ReactNode) => <div className="tree-head">
    <button className="tree-row root-row" onClick={() => toggle(key)}><span className={`tree-chevron ${expanded[key] ? "expanded" : ""}`} aria-hidden="true" />{label}</button>
    {action}
  </div>;
  return <aside className="sidebar">
    <section className="tree-section">
      {rootRow("home", "HOME")}
      {expanded.home && <div className="tree-children"><button className="tree-row leaf-row" onClick={() => post("open-portfolio")}><span className="tree-indent" />投资组合</button></div>}
    </section>
    <section className="tree-section">
      {rootRow("symbols", "标的", <button className="tree-action" title="新建分类" aria-label="新建分类" onClick={addCategory}>＋</button>)}
      {expanded.symbols && <div className="tree-children">
        {watchlist.categories.map((category: any, categoryIndex: number) => {
          const isOpen = expandedCategories[category.id] !== false;
          return <div className="category-group" key={category.id} onDragOver={(event) => event.preventDefault()} onDrop={(event) => moveSymbol(event.dataTransfer.getData("text/plain"), category.id, category.symbols.length)}>
            <div className="tree-row category-row" onClick={() => post("open-watchlist", { title: category.name })}>
              <button className={`tree-chevron ${isOpen ? "expanded" : ""}`} aria-label={`${isOpen ? "收起" : "展开"}${category.name}`} onClick={(event) => { event.stopPropagation(); toggleCategory(category.id); }} />
              <span className="tree-label" title={category.name}>{category.name}</span>
              <span className="row-actions">
                <button title="新增标的" onClick={(event) => { event.stopPropagation(); addSymbol(category.id); }}>＋</button>
                <button title="重命名" onClick={(event) => { event.stopPropagation(); const name = window.prompt("分类名称", category.name); if (name) post("watchlist-rename-category", { categoryId: category.id, name }); }}>✎</button>
                <button title="上移" disabled={!categoryIndex} onClick={(event) => { event.stopPropagation(); post("watchlist-move-category", { categoryId: category.id, offset: -1 }); }}>↑</button>
                <button title="下移" disabled={categoryIndex === watchlist.categories.length - 1} onClick={(event) => { event.stopPropagation(); post("watchlist-move-category", { categoryId: category.id, offset: 1 }); }}>↓</button>
                <button title="删除分类" onClick={(event) => { event.stopPropagation(); if (window.confirm(`删除“${category.name}”及其标的？`)) post("watchlist-delete-category", { categoryId: category.id }); }}>×</button>
              </span>
            </div>
            {isOpen && category.symbols.map((item: any, index: number) => {
              const price = prices.get(item.key.toUpperCase()) || item;
              const latest = Number.isFinite(price?.latest) ? price.latest : 100;
              const changePercent = Number.isFinite(price?.changePercent) ? price.changePercent : 0.01;
              const label = String(item.note || "").trim() || item.name || item.symbol;
              return <div className="tree-row symbol-row" key={item.key} draggable onDragStart={(event) => event.dataTransfer.setData("text/plain", item.key)} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.stopPropagation(); moveSymbol(event.dataTransfer.getData("text/plain"), category.id, index); }} onClick={() => post("open-watchlist", { title: label, symbol: item.symbol })}>
                <SymbolIcon item={item} />
                <span className="symbol-label"><strong>{item.symbol}</strong><small title={label}>{label}</small></span>
                <span className="symbol-metrics"><b>{number(latest)}</b><em className={tone(changePercent)}>{percent(changePercent)}</em></span>
                <span className="symbol-actions">
                  <button title="编辑备注" onClick={(event) => { event.stopPropagation(); editItem(item); }}>✎</button>
                  <button title="删除" onClick={(event) => { event.stopPropagation(); if (window.confirm(`删除 ${item.symbol}？`)) post("watchlist-delete-symbol", { key: item.key }); }}>×</button>
                </span>
              </div>;
            })}
          </div>;
        })}
      </div>}
    </section>
    <section className="tree-section">
      {rootRow("strategy", "策略")}
      {expanded.strategy && <div className="tree-children"><button className="tree-row leaf-row" onClick={() => post("open-strategy")}><span className="tree-indent" />布林带策略<span className="readonly-label">只读</span></button></div>}
    </section>
    {state.error && <div className="side-error">{state.error.message}</div>}
  </aside>;
}

// 时段筛选：限定到对应区间；returns 起始索引
function rangeStart(curve: any[], range: string): number {
  if (!curve.length) return 0;
  if (range === "all") return 0;
  const last = new Date(curve[curve.length - 1].date);
  let earliest: Date;
  if (range === "1m") earliest = new Date(last.getFullYear(), last.getMonth() - 1, last.getDate());
  else if (range === "3m") earliest = new Date(last.getFullYear(), last.getMonth() - 3, last.getDate());
  else if (range === "6m") earliest = new Date(last.getFullYear(), last.getMonth() - 6, last.getDate());
  else if (range === "ytd") earliest = new Date(last.getFullYear(), 0, 1);
  else if (range === "1y") earliest = new Date(last.getFullYear() - 1, last.getMonth(), last.getDate());
  else return 0;
  for (let i = 0; i < curve.length; i++) if (new Date(curve[i].date) >= earliest) return i;
  return 0;
}

// 走势对比：
// - mode="value"：显示投资组合 + 现金流（自由现金）的绝对值
// - mode="percent"：显示投资组合 vs 基准（benchmark）的百分比变化（以起点为 0%）
// 时段筛选：6 列等宽，每列两行（标签 + 该区间百分比变化），选中整列高亮
const RANGE_LABELS: [string, string][] = [["1m", "1个月"], ["3m", "3个月"], ["6m", "6个月"], ["ytd", "年初至今"], ["1y", "1年"], ["all", "全部"]];

function pickRange(curve: any[], range: string) {
  if (!curve.length) return { series: curve, startIdx: 0 };
  if (range === "all") return { series: curve.slice(), startIdx: 0 };
  const last = new Date(curve[curve.length - 1].date);
  const months = range === "1m" ? 1 : range === "3m" ? 3 : range === "6m" ? 6 : 0;
  let earliest: Date;
  if (months) earliest = new Date(last.getFullYear(), last.getMonth() - months, last.getDate());
  else if (range === "ytd") earliest = new Date(last.getFullYear(), 0, 1);
  else if (range === "1y") earliest = new Date(last.getFullYear() - 1, last.getMonth(), last.getDate());
  else earliest = new Date(0);
  let startIdx = 0;
  for (let i = 0; i < curve.length; i++) if (new Date(curve[i].date) >= earliest) { startIdx = i; break; }
  return { series: curve.slice(startIdx), startIdx };
}

function LineChart({ curve, mode, secondKey, secondLabel, range, onRangeChange }: {
  curve: any[]; mode: "value" | "percent"; secondKey: "flow" | "benchmark"; secondLabel: string;
  range: string; onRangeChange: (v: string) => void;
}) {
  if (!curve?.length) return <div className="empty">暂无曲线数据</div>;
  const { series } = pickRange(curve, range);
  // 计算时段回报（基于投资组合 value）
  const portfolioReturn = (slice: any[]) => {
    const first = slice.find((r) => r.portfolio != null)?.portfolio;
    const last = [...slice].reverse().find((r) => r.portfolio != null)?.portfolio;
    if (first == null || last == null || !first) return NaN;
    return (last / first - 1) * 100;
  };
  const seriesReturns: Record<string, number> = {};
  for (const [key] of RANGE_LABELS) {
    const { series: s } = pickRange(curve, key);
    seriesReturns[key] = portfolioReturn(s);
  }
  // 归一化数据
  const normalize = (row: any, key: string) => {
    if (row[key] == null) return null;
    if (mode === "value") return row[key];
    const first = series.find((r) => r[key] != null)?.[key];
    if (first == null || !first) return null;
    return ((row[key] / first) - 1) * 100;
  };
  const points = (key: string) => series.map((row, index, arr) => {
    const v = normalize(row, key);
    if (v == null) return null;
    const x = (index / Math.max(1, arr.length - 1)) * 100;
    const values = arr.map((r) => normalize(r, key)).filter((x) => x != null) as number[];
    const min = Math.min(...values), max = Math.max(...values), range = (max - min) || 1;
    const y = 46 - ((v - min) / range) * 42;
    return `${x},${y}`;
  }).filter(Boolean).join(" ");
  // X 轴
  const axisDates = series.length <= 6 ? series.map((r) => r.date.slice(5)) : [series[0].date.slice(5), series[Math.floor(series.length / 3)].date.slice(5), series[Math.floor(2 * series.length / 3)].date.slice(5), series[series.length - 1].date.slice(5)];
  // Y 轴：取两 series union
  const allValues: number[] = [];
  for (const r of series) {
    const p = normalize(r, "portfolio");
    const s = normalize(r, secondKey);
    if (p != null) allValues.push(p);
    if (s != null) allValues.push(s);
  }
  if (mode === "percent") allValues.push(0);
  const minY = allValues.length ? Math.min(...allValues) : 0;
  const maxY = allValues.length ? Math.max(...allValues) : 1;
  const yLabels = [maxY, (maxY + minY) / 2, minY].map((v) => mode === "value"
    ? `${v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(0)}`
    : `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`);
  // 末端浮动标签
  const lastPortfolio = series[series.length - 1]?.portfolio;
  const lastSecond = series[series.length - 1]?.[secondKey];
  return <div className="chart">
    <div className="chart-svg-wrap">
      <svg viewBox="0 0 100 50" preserveAspectRatio="none" role="img" aria-label="组合与基准走势">
        <line x1="0" y1="46" x2="100" y2="46" className="grid"/>
        <polyline points={points(secondKey)} className="benchmark-line"/>
        <polyline points={points("portfolio")} className="portfolio-line"/>
      </svg>
      <div className="chart-endpoint">
        <span className="portfolio-endpoint">{mode === "value" ? number(lastPortfolio) : `${(seriesReturns[range] >= 0 ? "+" : "") + (seriesReturns[range] ?? 0).toFixed(1)}%`}</span>
        <span className="benchmark-endpoint">{mode === "value" ? number(lastSecond) : ""}</span>
      </div>
      <div className="chart-axis-y">{yLabels.map((label, i) => <span key={i}>{label}</span>)}</div>
    </div>
    <div className="chart-axis-x">{axisDates.map((d, i) => <span key={i}>{d}</span>)}</div>
    <div className="legend"><span className="portfolio-dot"/>投资组合 <span className="benchmark-dot"/>{secondLabel}</div>
    <RangeTabs value={range} onChange={onRangeChange} seriesReturns={seriesReturns}/>
  </div>;
}

function RangeTabs({ value, onChange, seriesReturns }: { value: string; onChange: (v: string) => void; seriesReturns: Record<string, number> }) {
  return <nav className="range-tabs">{RANGE_LABELS.map(([k, label]) => {
    const r = seriesReturns[k];
    const ret = isNaN(r) ? "—" : `${r >= 0 ? "+" : ""}${r.toFixed(2)}%`;
    const flat = isNaN(r) ? "flat" : r > 0 ? "up" : r < 0 ? "down" : "flat";
    return <button key={k} className={value === k ? "active" : ""} onClick={() => onChange(k)}>
      <span className="range-label">{label}</span>
      <span className={`range-return ${flat}`}>{ret}</span>
    </button>;
  })}</nav>;
}

function Donut({ rows }: { rows: any[] }) {
  const total = rows.reduce((sum, row) => sum + row.value, 0) || 1;
  let offset = 0;
  const colors = ["#2962ff", "#7c4dff", "#00bcd4", "#ff9800", "#e91e63"];
  return <div className="donut-wrap"><svg viewBox="0 0 42 42" className="donut" role="img" aria-label="资产分布空心饼图"><circle cx="21" cy="21" r="15.9" className="donut-base"/>{rows.map((row, index) => { const share = row.value / total * 100; const circle = <circle key={row.symbol} cx="21" cy="21" r="15.9" pathLength="100" stroke={colors[index % colors.length]} strokeDasharray={`${share} ${100 - share}`} strokeDashoffset={-offset}/>; offset += share; return circle; })}</svg><div className="donut-legend">{rows.map((row, index) => <div key={row.symbol}><i style={{ background: colors[index % colors.length] }}/><span>{row.symbol}</span><b>{percent(row.value / total)}</b></div>)}</div></div>;
}

const HOLDINGS_COLS = ["标的", "数量", "平均成本", "最新价", "市值", "当日", "未实现收益", "总收益"];
const TRANSACTIONS_COLS = ["标的", "买/卖", "日期", "数量", "价格", "手续费", "总计", "笔记"];
const PRICES_COLS = ["标的", "最新价", "涨跌", "涨跌%"];

function SortIndicator({ col, sortKey, sortDir }: { col: string; sortKey: string; sortDir: string }) {
  if (col !== sortKey) return null;
  return <span className="sort-indicator">{sortDir === "asc" ? " ↑" : " ↓"}</span>;
}

function DataTable({ rows, kind, currency, sortKey, sortDir, onSort }: {
  rows: any[]; kind: string; currency: string;
  sortKey?: string; sortDir?: string; onSort?: (key: string) => void;
}) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const th = (col: string, index: number) => {
    if (index === 0) {
      return <th key={col}>
        <span className="th-search-wrap">
          <button className="search-icon-btn" onClick={(e) => { e.stopPropagation(); setSearchOpen(!searchOpen); setSearchQuery(""); }} title="搜索">🔍</button>
          {searchOpen && <input className="search-inline" autoFocus placeholder="搜索…" value={searchQuery} onChange={(e) => setSearchQuery((e.target as HTMLInputElement).value)} onBlur={() => { if (!searchQuery) setSearchOpen(false); }} onKeyDown={(e) => { if (e.key === "Escape") { setSearchQuery(""); setSearchOpen(false); } }} />}
          {!searchOpen && <span className="th-clickable" onClick={() => onSort?.(col)}>标的<SortIndicator col={col} sortKey={sortKey || ""} sortDir={sortDir || ""}/></span>}
        </span>
      </th>;
    }
    if (onSort) {
      return <th key={col} className="sortable" onClick={() => onSort(col)}>{col}<SortIndicator col={col} sortKey={sortKey || ""} sortDir={sortDir || ""}/></th>;
    }
    return <th key={col}>{col}</th>;
  };

  const visibleRows = useMemo(() => {
    let result = [...rows];
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      result = result.filter((r) => (r.symbol || "").toLowerCase().includes(q) || (r.name || "").toLowerCase().includes(q));
    }
    if (sortKey && onSort) {
      result.sort((a, b) => {
        let va: any, vb: any;
        if (kind === "holdings") {
          if (sortKey === "标的") { va = (a.symbol || "").toLowerCase(); vb = (b.symbol || "").toLowerCase(); }
          else if (sortKey === "数量") { va = a.quantity || 0; vb = b.quantity || 0; }
          else if (sortKey === "平均成本") { va = a.averageCost || 0; vb = b.averageCost || 0; }
          else if (sortKey === "最新价") { va = a.lastPrice || 0; vb = b.lastPrice || 0; }
          else if (sortKey === "市值") { va = a.marketValue || 0; vb = b.marketValue || 0; }
          else if (sortKey === "当日") { va = a.lastDayAmount || 0; vb = b.lastDayAmount || 0; }
          else if (sortKey === "未实现收益") { va = a.unrealized || 0; vb = b.unrealized || 0; }
          else if (sortKey === "总收益") { va = (a.unrealized || 0) + (a.realized || 0) + (a.netDividends || 0); vb = (b.unrealized || 0) + (b.realized || 0) + (b.netDividends || 0); }
          else return 0;
        } else if (kind === "transactions") {
          if (sortKey === "标的") { va = (a.symbol || "").toLowerCase(); vb = (b.symbol || "").toLowerCase(); }
          else if (sortKey === "买/卖") { va = a.action || a.kind || ""; vb = b.action || b.kind || ""; }
          else if (sortKey === "日期") { va = a.date || ""; vb = b.date || ""; }
          else if (sortKey === "数量") { va = a.quantity || 0; vb = b.quantity || 0; }
          else if (sortKey === "价格") { va = a.price || 0; vb = b.price || 0; }
          else if (sortKey === "手续费") { va = a.fee || 0; vb = b.fee || 0; }
          else if (sortKey === "总计") { va = a.amount || 0; vb = b.amount || 0; }
          else if (sortKey === "笔记") { va = a.note || ""; vb = b.note || ""; }
          else return 0;
        } else if (kind === "prices") {
          if (sortKey === "标的") { va = (a.symbol || "").toLowerCase(); vb = (b.symbol || "").toLowerCase(); }
          else if (sortKey === "最新价") { va = a.latest || 0; vb = b.latest || 0; }
          else if (sortKey === "涨跌") { va = a.change || 0; vb = b.change || 0; }
          else if (sortKey === "涨跌%") { va = a.changePercent || 0; vb = b.changePercent || 0; }
          else return 0;
        }
        if (typeof va === "string" && typeof vb === "string") return sortDir === "asc" ? va.localeCompare(vb) : vb.localeCompare(va);
        return sortDir === "asc" ? (va - vb) : (vb - va);
      });
    }
    return result;
  }, [rows, kind, searchQuery, sortKey, sortDir]);

  const cols = kind === "prices" ? PRICES_COLS : kind === "transactions" ? TRANSACTIONS_COLS : HOLDINGS_COLS;

  if (!rows?.length) return <div className="empty">暂无数据</div>;
  if (kind === "prices") return <div className="table-scroll"><table><thead><tr>{cols.map(th)}</tr></thead><tbody>{visibleRows.map((row) => <tr key={row.symbol}><td><strong>{row.symbol}</strong><small title={row.name}>{row.name}</small></td><td>{number(row.latest)}</td><td className={tone(row.change)}>{signed(row.change)}</td><td className={tone(row.changePercent)}>{percent(row.changePercent)}</td></tr>)}</tbody></table></div>;
  if (kind === "transactions") return <div className="table-scroll"><table><thead><tr>{cols.map(th)}</tr></thead><tbody>{visibleRows.map((row, index) => <tr key={`${row.date}-${row.symbol}-${index}`}><td><strong>{row.symbol || "—"}</strong><small>{row.name}</small></td><td>{row.action || row.kind}</td><td>{row.date}</td><td>{number(row.quantity)}</td><td>{number(row.price)}</td><td>{money(row.fee, currency)}</td><td className={tone(row.amount)}>{money(row.amount, currency)}</td><td>—</td></tr>)}</tbody></table></div>;
  return <div className="table-scroll"><table><thead><tr>{cols.map(th)}</tr></thead><tbody>{visibleRows.map((row) => { const total = (row.unrealized || 0) + (row.realized || 0) + (row.netDividends || 0); return <tr key={row.symbol}><td><strong>{row.symbol}</strong><small title={row.name}>{row.name}</small></td><td>{number(row.quantity)}</td><td>{number(row.averageCost)}</td><td>{number(row.lastPrice)}</td><td>{money(row.marketValue, currency)}</td><td className={tone(row.lastDayAmount)}>{money(row.lastDayAmount, currency)}<small>{percent(row.lastDayPercent)}</small></td><td className={tone(row.unrealized)}>{money(row.unrealized, currency)}<small>{percent(row.unrealizedPercent)}</small></td><td className={tone(total)}>{money(total, currency)}</td></tr>; })}</tbody></table></div>;
}

const TAB_MAP: Record<string, string[]> = {
  "概览": ["值", "表现"], "控股": ["持仓", "价格"], "交易": ["交易", "现金", "股息"],
  "分析": ["投资组合盈利", "获利", "股利", "风险", "持股表现"],
};

function SummaryCards({ market }: { market: any }) {
  const summary = market.summary;
  return <div className="summary-grid">
    <article><span>投资组合价值</span><strong>{money(summary.portfolioValue, market.currency)}</strong><div className="summary-subline"><span>现金</span><b>{money(summary.cash, market.currency)}</b></div></article>
    <article><span>未实现收益</span><strong className={tone(summary.unrealized)}>{money(summary.unrealized, market.currency)}</strong><div className="summary-subline"><span>最后一天</span><b className={tone(summary.lastDayAmount)}>{money(summary.lastDayAmount, market.currency)} · {percent(summary.lastDayPercent)}</b></div></article>
    <article><span>已实现收益</span><strong className={tone(summary.realized)}>{money(summary.realized, market.currency)}</strong><div className="summary-subline"><span>净股息</span><b className={tone(summary.netDividends)}>{money(summary.netDividends, market.currency)}</b></div></article>
    <article><span>总收益</span><strong className={tone(summary.totalReturn)}>{money(summary.totalReturn, market.currency)}</strong><div className="summary-subline"><span>年化收益率</span><b className={tone(summary.annualizedReturn)}>{percent(summary.annualizedReturn)}</b></div></article>
  </div>;
}

function Analysis({ view, market, chartRange, setChartRange }: { view: string; market: any; chartRange: string; setChartRange: (v: string) => void }) {
  if (view === "风险") return <div className="risk-grid"><Metric label="Beta" value={number(market.risk.beta)}/><Metric label="Sharpe" value={number(market.risk.sharpe)}/><Metric label="Sortino" value={number(market.risk.sortino)}/><Metric label="共同交易日" value={String(market.risk.sampleDays || "—")}/></div>;
  if (view === "股利") return <div className="split"><section className="panel"><h3>已收股息</h3><DataTable rows={market.dividends} kind="transactions" currency={market.currency}/></section><section className="panel"><h3>派息日历</h3><Calendar rows={market.dividendCalendar} currency={market.currency}/></section></div>;
  if (view === "投资组合盈利") return <div className="panel"><LineChart curve={market.curve} mode="percent" secondKey="benchmark" secondLabel={market.benchmark.name} range={chartRange} onRangeChange={setChartRange}/></div>;
  return <DataTable rows={market.holdings} kind="holdings" currency={market.currency}/>;
}

function Metric({ label, value }: { label: string; value: string }) { return <article><span>{label}</span><strong>{value}</strong></article>; }
function Calendar({ rows, currency }: { rows: any[]; currency: string }) { return rows?.length ? <div className="calendar">{rows.map((row, index) => <div key={`${row.date}-${row.symbol}-${index}`}><time>{row.date}</time><strong>{row.symbol}</strong><span>{row.status}</span><b>{money(row.amount, currency)}</b></div>)}</div> : <div className="empty">暂无派息日历</div>; }

function Content({ tab, subtab, market, showSold, setShowSold, chartRange, setChartRange, sortKey, sortDir, onSort }: any) {
  if (tab === "概览" && subtab === "值") return <><section className="chart-panel"><div className="chart-caption"><span>投资组合</span><span>自由现金</span></div><LineChart curve={market.curve} mode="value" secondKey="flow" secondLabel="自由现金" range={chartRange} onRangeChange={setChartRange}/></section><div className="overview-grid"><section className="panel"><h3>资产分布</h3><Donut rows={market.distribution}/></section><section className="panel"><h3>派息监控</h3><Calendar rows={market.dividendCalendar} currency={market.currency}/></section></div></>;
  if (tab === "概览") return <section className="chart-panel"><div className="chart-caption"><span>投资组合</span><span>{market.benchmark.name}</span></div><LineChart curve={market.curve} mode="percent" secondKey="benchmark" secondLabel={market.benchmark.name} range={chartRange} onRangeChange={setChartRange}/></section>;
  if (tab === "控股" && subtab === "价格") return <DataTable rows={market.prices} kind="prices" currency={market.currency} sortKey={sortKey} sortDir={sortDir} onSort={onSort}/>;
  if (tab === "控股") return <><label className="sold-toggle"><input type="checkbox" checked={showSold} onChange={(event) => setShowSold(event.target.checked)}/> 显示已卖出标的</label><DataTable rows={[...market.holdings, ...(showSold ? market.soldHoldings || [] : [])]} kind="holdings" currency={market.currency} sortKey={sortKey} sortDir={sortDir} onSort={onSort}/></>;
  if (tab === "交易") return <DataTable rows={subtab === "现金" ? market.cashTransactions : subtab === "股息" ? market.dividends : market.transactions} kind="transactions" currency={market.currency} sortKey={sortKey} sortDir={sortDir} onSort={onSort}/>;
  return <Analysis view={subtab} market={market} chartRange={chartRange} setChartRange={setChartRange}/>;
}

function Portfolio({ state }: { state: AppState }) {
  const [marketKey, setMarketKey] = useState("us");
  const [tab, setTab] = useState("概览");
  const [subtab, setSubtab] = useState("值");
  const [showSold, setShowSold] = useState(false);
  const [chartRange, setChartRange] = useState("all");
  const [sortKey, setSortKey] = useState("");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const market = state.snapshot?.markets?.[marketKey];
  if (state.content.view === "watchlist") {
    return <main className="content-placeholder"><span className="eyebrow">标的 · 只读展示</span><h1>{state.content.title || "标的"}</h1><p>当前先保留目录、中文名称和行情入口；标的详情、走势图与指标将在后续版本接入。</p><span className="readonly-pill">后续开发</span></main>;
  }
  if (!market) return <div className="loading">正在读取投资组合…</div>;
  const chooseTab = (next: string) => { setTab(next); setSubtab(TAB_MAP[next][0]); setSortKey(""); setSortDir("desc"); };
  const handleSort = (key: string) => {
    if (sortKey === key) setSortDir((prev) => prev === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("desc"); }
  };
  return <main className="portfolio-v2">
    <header className="workbench-head">
      <div className="market-tabs" role="tablist" aria-label="市场切换">
        <button className={marketKey === "us" ? "active" : ""} role="tab" aria-selected={marketKey === "us"} onClick={() => { setMarketKey("us"); setTab("概览"); setSubtab("值"); setShowSold(false); setChartRange("all"); }}>美股持仓</button>
        <button className={marketKey === "cn" ? "active" : ""} role="tab" aria-selected={marketKey === "cn"} onClick={() => { setMarketKey("cn"); setTab("概览"); setSubtab("值"); setShowSold(false); setChartRange("all"); }}>A 股持仓</button>
      </div>
      <div className="workbench-actions"><span className="asof">截至 {market.asOf || "—"} · {market.source?.label || "示例数据"}{market.incomplete && <span className="warning-badge">数据不完整</span>}</span><div className="toolbar"><button onClick={() => post("select-source", { market: marketKey })}>选择文件</button><button className="primary" onClick={() => post("refresh")} disabled={state.refresh?.status === "loading"}>{state.refresh?.status === "loading" ? "刷新中…" : "刷新"}</button></div></div>
    </header>
    {state.refresh?.status === "error" && <div className="error-banner">刷新失败：{state.refresh.message}；继续显示 {state.refresh.staleAt || market.asOf} 的数据。</div>}
    <SummaryCards market={market}/>
    <nav className="main-tabs">{Object.keys(TAB_MAP).map((name) => <button className={tab === name ? "active" : ""} onClick={() => chooseTab(name)} key={name}>{name}</button>)}</nav>
    <div className="subbar"><nav className="sub-tabs">{TAB_MAP[tab].map((name) => <button className={subtab === name ? "active" : ""} onClick={() => setSubtab(name)} key={name}>{name}</button>)}</nav><span className="subbar-asof">投资组合 · {market.benchmark.name}</span></div>
    <Content tab={tab} subtab={subtab} market={market} showSold={showSold} setShowSold={setShowSold} chartRange={chartRange} setChartRange={setChartRange} sortKey={sortKey} sortDir={sortDir} onSort={handleSort}/>
  </main>;
}

function Strategy() {
  return <main className="strategy-page"><span className="eyebrow">策略 · 只读展示</span><h1>布林带均值回归策略</h1><p>用于记录并验证 TradingView 策略思路，以多标的纸面交易观察信号质量和资金占用。</p><div className="strategy-grid"><Metric label="状态" value="后续版本开发"/><Metric label="当前能力" value="仅展示，不运行"/><Metric label="模式" value="Paper / Dry-run"/></div><section className="panel"><h3>源码位置</h3><code>strategies/bollinger_band_reversion/</code><button className="primary" onClick={() => post("open-strategy-file")}>打开策略入口</button></section><section className="panel"><h3>后续版本</h3><p>策略参数、回测结果、持续运行、Webhook 与飞书交易预警将在后续版本单独设计。本页不会启动、配置或修改策略。</p></section></main>;
}

function App() {
  const state = useHostState();
  const kind = document.body.dataset.view;
  if (kind === "sidebar") return <Sidebar state={state}/>;
  if (kind === "strategy") return <Strategy/>;
  return <Portfolio state={state}/>;
}

createRoot(document.getElementById("root")!).render(<App/>);
