import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./webview.css";

declare function acquireVsCodeApi(): { postMessage(message: unknown): void };
const vscode = acquireVsCodeApi();
const post = (type: string, data: Record<string, unknown> = {}) => vscode.postMessage({ type, ...data });

type AppState = { snapshot: any; watchlist: any; sources: any; refresh: any; error: any };

function useHostState(): AppState {
  const [state, setState] = useState<AppState>({ snapshot: null, watchlist: null, sources: null, refresh: null, error: null });
  useEffect(() => {
    const listener = (event: MessageEvent) => {
      const message = event.data;
      if (message.type === "snapshot") setState((old) => ({ ...old, snapshot: message.data, error: null }));
      if (message.type === "watchlist") setState((old) => ({ ...old, watchlist: message.data }));
      if (message.type === "source-status") setState((old) => ({ ...old, sources: message.data }));
      if (message.type === "refresh-status") setState((old) => ({ ...old, refresh: message }));
      if (message.type === "error") setState((old) => ({ ...old, error: message }));
    };
    window.addEventListener("message", listener);
    post("ready");
    return () => window.removeEventListener("message", listener);
  }, []);
  return state;
}

const money = (value: number | null | undefined, currency = "USD") => value == null ? "—" : new Intl.NumberFormat("zh-CN", { style: "currency", currency, minimumFractionDigits: 2 }).format(value);
const number = (value: number | null | undefined, digits = 2) => value == null ? "—" : new Intl.NumberFormat("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
const signed = (value: number | null | undefined, suffix = "") => value == null ? "—" : `${value >= 0 ? "+" : ""}${number(value)}${suffix}`;
const percent = (value: number | null | undefined) => value == null ? "—" : `${value >= 0 ? "+" : ""}${number(value * 100)}%`;
const tone = (value: number | null | undefined) => value == null || value === 0 ? "flat" : value > 0 ? "up" : "down";

function DashboardIcon() {
  return <svg className="dashboard-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 11a6 6 0 0 1 12 0M8 8l3-2M3.5 12.5h9" /></svg>;
}

function Sidebar({ state }: { state: AppState }) {
  const watchlist = state.watchlist || { categories: [] };
  const prices = useMemo(() => {
    const map = new Map<string, any>();
    for (const market of ["us", "cn"]) for (const row of state.snapshot?.markets?.[market]?.prices || []) map.set(`${market}:${row.symbol}`.toUpperCase(), row);
    return map;
  }, [state.snapshot]);
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
  return <aside className="sidebar">
    <section><h2>HOME</h2><button className="nav-item portfolio-link" onClick={() => post("open-portfolio")}><DashboardIcon /><span>投资组合</span></button></section>
    <section>
      <div className="section-title"><h2>标的</h2><button title="新建分类" aria-label="新建分类" onClick={addCategory}>＋</button></div>
      <div className="watch-head"><span>商品</span><span>最新价</span><span className="change-amount">涨跌</span><span>涨跌%</span></div>
      {watchlist.categories.map((category: any, categoryIndex: number) => <details key={category.id} open onDragOver={(event) => event.preventDefault()} onDrop={(event) => moveSymbol(event.dataTransfer.getData("text/plain"), category.id, category.symbols.length)}>
        <summary>
          <span title={category.name}>{category.name}</span>
          <span className="row-actions">
            <button title="新增标的" onClick={(event) => { event.preventDefault(); addSymbol(category.id); }}>＋</button>
            <button title="重命名" onClick={(event) => { event.preventDefault(); const name = window.prompt("分类名称", category.name); if (name) post("watchlist-rename-category", { categoryId: category.id, name }); }}>✎</button>
            <button title="上移" disabled={!categoryIndex} onClick={(event) => { event.preventDefault(); post("watchlist-move-category", { categoryId: category.id, offset: -1 }); }}>↑</button>
            <button title="下移" disabled={categoryIndex === watchlist.categories.length - 1} onClick={(event) => { event.preventDefault(); post("watchlist-move-category", { categoryId: category.id, offset: 1 }); }}>↓</button>
            <button title="删除分类" onClick={(event) => { event.preventDefault(); if (window.confirm(`删除“${category.name}”及其标的？`)) post("watchlist-delete-category", { categoryId: category.id }); }}>×</button>
          </span>
        </summary>
        {category.symbols.map((item: any, index: number) => {
          const price = prices.get(item.key.toUpperCase());
          return <div className="watch-row" key={item.key} draggable onDragStart={(event) => event.dataTransfer.setData("text/plain", item.key)} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.stopPropagation(); moveSymbol(event.dataTransfer.getData("text/plain"), category.id, index); }}>
            <div className="commodity"><strong>{item.symbol}</strong><div><span className="watch-name" title={item.name}>{item.name}</span><span className="watch-note" title={item.note}>{item.note}</span></div></div>
            <span>{number(price?.latest)}</span><span className={`change-amount ${tone(price?.change)}`}>{signed(price?.change)}</span><span className={tone(price?.changePercent)}>{percent(price?.changePercent)}</span>
            <div className="symbol-actions">
              <button title="编辑" onClick={() => { const market = window.prompt("市场：us 或 cn", item.market); if (!market) return; const symbol = window.prompt("标的代码", item.symbol); if (!symbol) return; const name = window.prompt("中文名称", item.name); if (name == null) return; const note = window.prompt("备注", item.note) ?? item.note; post("watchlist-edit-symbol", { key: item.key, symbol: { market, symbol, name, note } }); }}>✎</button>
              <button title="上移" disabled={!index} onClick={() => moveSymbol(item.key, category.id, index - 1)}>↑</button>
              <button title="下移" disabled={index === category.symbols.length - 1} onClick={() => moveSymbol(item.key, category.id, index + 1)}>↓</button>
              <select aria-label={`移动 ${item.symbol} 到分类`} value="" onChange={(event) => { if (event.target.value) moveSymbol(item.key, event.target.value, 9999); }}><option value="">移动到…</option>{watchlist.categories.filter((entry: any) => entry.id !== category.id).map((entry: any) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}</select>
              <button title="删除" onClick={() => { if (window.confirm(`删除 ${item.symbol}？`)) post("watchlist-delete-symbol", { key: item.key }); }}>×</button>
            </div>
          </div>;
        })}
      </details>)}
    </section>
    <section><h2>策略</h2><button className="nav-item" onClick={() => post("open-strategy")}>布林带策略</button></section>
    {state.error && <div className="side-error">{state.error.message}</div>}
  </aside>;
}

function LineChart({ curve, benchmark }: { curve: any[]; benchmark: string }) {
  if (!curve?.length) return <div className="empty">暂无曲线数据</div>;
  const values = curve.flatMap((row) => [row.portfolio, row.benchmark]).filter((value) => value != null);
  const min = Math.min(...values), max = Math.max(...values), range = max - min || 1;
  const points = (key: string) => curve.map((row, index) => row[key] == null ? null : `${(index / Math.max(1, curve.length - 1)) * 100},${46 - ((row[key] - min) / range) * 42}`).filter(Boolean).join(" ");
  return <div className="chart"><svg viewBox="0 0 100 50" preserveAspectRatio="none" role="img" aria-label="组合与基准表现折线图"><line x1="0" y1="46" x2="100" y2="46" className="grid"/><polyline points={points("benchmark")} className="benchmark-line"/><polyline points={points("portfolio")} className="portfolio-line"/></svg><div className="legend"><span className="portfolio-dot"/>投资组合 <span className="benchmark-dot"/>{benchmark}</div></div>;
}

function Donut({ rows }: { rows: any[] }) {
  const total = rows.reduce((sum, row) => sum + row.value, 0) || 1;
  let offset = 0;
  const colors = ["#2962ff", "#7c4dff", "#00bcd4", "#ff9800", "#e91e63"];
  return <div className="donut-wrap"><svg viewBox="0 0 42 42" className="donut" role="img" aria-label="资产分布空心饼图"><circle cx="21" cy="21" r="15.9" className="donut-base"/>{rows.map((row, index) => { const share = row.value / total * 100; const circle = <circle key={row.symbol} cx="21" cy="21" r="15.9" pathLength="100" stroke={colors[index % colors.length]} strokeDasharray={`${share} ${100 - share}`} strokeDashoffset={-offset}/>; offset += share; return circle; })}</svg><div className="donut-legend">{rows.map((row, index) => <div key={row.symbol}><i style={{ background: colors[index % colors.length] }}/><span>{row.symbol}</span><b>{percent(row.value / total)}</b></div>)}</div></div>;
}

function DataTable({ rows, kind, currency }: { rows: any[]; kind: string; currency: string }) {
  if (!rows?.length) return <div className="empty">暂无数据</div>;
  if (kind === "prices") return <div className="table-scroll"><table><thead><tr><th>商品</th><th>最新价</th><th>涨跌</th><th>涨跌%</th></tr></thead><tbody>{rows.map((row) => <tr key={row.symbol}><td><strong>{row.symbol}</strong><small title={row.name}>{row.name}</small></td><td>{number(row.latest)}</td><td className={tone(row.change)}>{signed(row.change)}</td><td className={tone(row.changePercent)}>{percent(row.changePercent)}</td></tr>)}</tbody></table></div>;
  if (kind === "transactions") return <div className="table-scroll"><table><thead><tr><th>日期</th><th>商品</th><th>操作</th><th>数量</th><th>价格</th><th>费用</th><th>金额</th></tr></thead><tbody>{rows.map((row, index) => <tr key={`${row.date}-${row.symbol}-${index}`}><td>{row.date}</td><td><strong>{row.symbol || "—"}</strong><small>{row.name}</small></td><td>{row.action || row.kind}</td><td>{number(row.quantity)}</td><td>{number(row.price)}</td><td>{money(row.fee, currency)}</td><td className={tone(row.amount)}>{money(row.amount, currency)}</td></tr>)}</tbody></table></div>;
  return <div className="table-scroll"><table><thead><tr><th>商品</th><th>数量</th><th>平均成本</th><th>最新价</th><th>市值</th><th>当日</th><th>未实现收益</th><th>总收益</th></tr></thead><tbody>{rows.map((row) => { const total = (row.unrealized || 0) + (row.realized || 0) + (row.netDividends || 0); return <tr key={row.symbol}><td><strong>{row.symbol}</strong><small title={row.name}>{row.name}</small></td><td>{number(row.quantity)}</td><td>{number(row.averageCost)}</td><td>{number(row.lastPrice)}</td><td>{money(row.marketValue, currency)}</td><td className={tone(row.lastDayAmount)}>{money(row.lastDayAmount, currency)}<small>{percent(row.lastDayPercent)}</small></td><td className={tone(row.unrealized)}>{money(row.unrealized, currency)}<small>{percent(row.unrealizedPercent)}</small></td><td className={tone(total)}>{money(total, currency)}</td></tr>; })}</tbody></table></div>;
}

const TAB_MAP: Record<string, string[]> = {
  "概览": ["值", "表现"], "控股": ["持仓", "价格"], "交易": ["交易", "现金", "股息"],
  "分析": ["投资组合盈利", "获利", "股利", "风险", "持股表现"],
};

function SummaryCards({ market }: { market: any }) {
  const summary = market.summary;
  return <div className="summary-grid">
    <article><span>投资组合价值</span><strong>{money(summary.portfolioValue, market.currency)}</strong><small>现金 {money(summary.cash, market.currency)}</small></article>
    <article><span>未实现收益</span><strong className={tone(summary.unrealized)}>{money(summary.unrealized, market.currency)}</strong><small className={tone(summary.lastDayAmount)}>最后一天 {money(summary.lastDayAmount, market.currency)} · {percent(summary.lastDayPercent)}</small></article>
    <article className="realized-card"><div><span>已实现收益</span><strong className={tone(summary.realized)}>{money(summary.realized, market.currency)}</strong></div><div className="realized-detail"><small>交易收益：<b className={tone(summary.tradingRealized)}>{money(summary.tradingRealized, market.currency)}</b></small><small>净股息：<b className={tone(summary.netDividends)}>{money(summary.netDividends, market.currency)}</b></small></div></article>
    <article><span>总收益</span><strong className={tone(summary.totalReturn)}>{money(summary.totalReturn, market.currency)} <em>{percent(summary.totalReturnRate)}</em></strong><small>年化收益率 {percent(summary.annualizedReturn)}</small></article>
  </div>;
}

function Analysis({ view, market }: { view: string; market: any }) {
  if (view === "风险") return <div className="risk-grid"><Metric label="Beta" value={number(market.risk.beta)}/><Metric label="Sharpe" value={number(market.risk.sharpe)}/><Metric label="Sortino" value={number(market.risk.sortino)}/><Metric label="共同交易日" value={String(market.risk.sampleDays || "—")}/></div>;
  if (view === "股利") return <div className="split"><section className="panel"><h3>已收股息</h3><DataTable rows={market.dividends} kind="transactions" currency={market.currency}/></section><section className="panel"><h3>派息日历</h3><Calendar rows={market.dividendCalendar} currency={market.currency}/></section></div>;
  if (view === "投资组合盈利") return <div className="panel"><h3>投资组合与 {market.benchmark.name}</h3><LineChart curve={market.curve} benchmark={market.benchmark.name}/></div>;
  return <DataTable rows={market.holdings} kind="holdings" currency={market.currency}/>;
}

function Metric({ label, value }: { label: string; value: string }) { return <article><span>{label}</span><strong>{value}</strong></article>; }
function Calendar({ rows, currency }: { rows: any[]; currency: string }) { return rows?.length ? <div className="calendar">{rows.map((row, index) => <div key={`${row.date}-${row.symbol}-${index}`}><time>{row.date}</time><strong>{row.symbol}</strong><span>{row.status}</span><b>{money(row.amount, currency)}</b></div>)}</div> : <div className="empty">暂无派息日历</div>; }

function Content({ tab, subtab, market, showSold, setShowSold }: any) {
  if (tab === "概览" && subtab === "值") return <><div className="panel"><h3>组合价值走势</h3><LineChart curve={market.curve} benchmark={market.benchmark.name}/></div><div className="overview-grid"><section className="panel"><h3>资产分布</h3><Donut rows={market.distribution}/></section><section className="panel"><h3>派息监控</h3><Calendar rows={market.dividendCalendar} currency={market.currency}/></section></div></>;
  if (tab === "概览") return <div className="panel"><h3>表现</h3><LineChart curve={market.curve} benchmark={market.benchmark.name}/></div>;
  if (tab === "控股" && subtab === "价格") return <DataTable rows={market.prices} kind="prices" currency={market.currency}/>;
  if (tab === "控股") return <><label className="sold-toggle"><input type="checkbox" checked={showSold} onChange={(event) => setShowSold(event.target.checked)}/> 显示已卖出标的</label><DataTable rows={[...market.holdings, ...(showSold ? market.soldHoldings || [] : [])]} kind="holdings" currency={market.currency}/></>;
  if (tab === "交易") return <DataTable rows={subtab === "现金" ? market.cashTransactions : subtab === "股息" ? market.dividends : market.transactions} kind="transactions" currency={market.currency}/>;
  return <Analysis view={subtab} market={market}/>;
}

function Portfolio({ state }: { state: AppState }) {
  const [marketKey, setMarketKey] = useState("us");
  const [tab, setTab] = useState("控股");
  const [subtab, setSubtab] = useState("持仓");
  const [showSold, setShowSold] = useState(false);
  const market = state.snapshot?.markets?.[marketKey];
  if (!market) return <div className="loading">正在读取投资组合…</div>;
  const chooseTab = (next: string) => { setTab(next); setSubtab(TAB_MAP[next][0]); };
  return <main className="portfolio">
    <header className="topbar">
      <div><select className="market-select" value={marketKey} onChange={(event) => { setMarketKey(event.target.value); setTab("控股"); setSubtab("持仓"); setShowSold(false); }}><option value="us">美股持仓</option><option value="cn">A股持仓</option></select>{market.source.mode === "sample" && <span className="sample-badge">示例数据</span>}{market.incomplete && <span className="warning-badge">数据不完整</span>}</div>
      <div className="toolbar"><button onClick={() => post("select-source", { market: marketKey })}>选择文件</button><button onClick={() => post("reset-source", { market: marketKey })}>恢复示例</button><button onClick={() => post("refresh")} disabled={state.refresh?.status === "loading"}>{state.refresh?.status === "loading" ? "刷新中…" : "刷新"}</button></div>
    </header>
    {state.refresh?.status === "error" && <div className="error-banner">刷新失败：{state.refresh.message}；继续显示 {state.refresh.staleAt || market.asOf} 的数据。</div>}
    <div className="asof">截至 {market.asOf || "—"} · {market.source.label}</div>
    <SummaryCards market={market}/>
    <nav className="main-tabs">{Object.keys(TAB_MAP).map((name) => <button className={tab === name ? "active" : ""} onClick={() => chooseTab(name)} key={name}>{name}</button>)}</nav>
    <nav className="sub-tabs">{TAB_MAP[tab].map((name) => <button className={subtab === name ? "active" : ""} onClick={() => setSubtab(name)} key={name}>{name}</button>)}</nav>
    <Content tab={tab} subtab={subtab} market={market} showSold={showSold} setShowSold={setShowSold}/>
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
