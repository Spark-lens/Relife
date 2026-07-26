"use client";

import { useEffect, useMemo, useState } from "react";

import {
  DEFAULT_COLUMNS,
  formatMoney,
  formatPercent,
  normalizeColumnPreferences,
} from "../portfolio-state.mjs";
import type {
  ColumnPreference,
  DashboardPayload,
  MarketKey,
  PageKey,
  RangeKey,
} from "../portfolio-types";
import { DividendsView } from "./DividendsView";
import { HoldingsTable } from "./HoldingsTable";
import { PerformanceChart } from "./PerformanceChart";
import { TransactionsView } from "./TransactionsView";

const PAGES: { id: PageKey; label: string }[] = [
  { id: "overview", label: "总览" },
  { id: "holdings", label: "持仓" },
  { id: "transactions", label: "交易记录" },
  { id: "dividends", label: "股息" },
];
const RANGES: { id: RangeKey; label: string }[] = [
  { id: "1m", label: "1 月" },
  { id: "3m", label: "3 月" },
  { id: "ytd", label: "今年" },
  { id: "all", label: "全部" },
];

function valueClass(value: number) {
  return value > 0 ? "positive" : value < 0 ? "negative" : "";
}

export function PortfolioDashboard({ data }: { data: DashboardPayload }) {
  const [marketKey, setMarketKey] = useState<MarketKey>("us");
  const [page, setPage] = useState<PageKey>("overview");
  const [range, setRange] = useState<RangeKey>("all");
  const [columns, setColumns] = useState<ColumnPreference[]>(
    normalizeColumnPreferences(DEFAULT_COLUMNS),
  );
  const market = data.markets[marketKey];

  useEffect(() => {
    let active = true;
    queueMicrotask(() => {
      if (!active) return;
      const savedMarket = window.localStorage.getItem("relife.market");
      if (savedMarket === "us" || savedMarket === "cn") {
        setMarketKey(savedMarket);
      }
      try {
        const savedColumns = JSON.parse(
          window.localStorage.getItem("relife.holdings.columns.v1") ?? "null",
        );
        setColumns(normalizeColumnPreferences(savedColumns));
      } catch {
        setColumns(normalizeColumnPreferences(DEFAULT_COLUMNS));
      }
    });
    return () => {
      active = false;
    };
  }, []);

  function updateMarket(next: MarketKey) {
    setMarketKey(next);
    window.localStorage.setItem("relife.market", next);
  }

  function updateColumns(next: ColumnPreference[]) {
    const normalized = normalizeColumnPreferences(next);
    setColumns(normalized);
    window.localStorage.setItem(
      "relife.holdings.columns.v1",
      JSON.stringify(normalized.map(({ id, visible }) => ({ id, visible }))),
    );
  }

  const positionCount = useMemo(
    () => market.groups.reduce((sum, group) => sum + group.positions.length, 0),
    [market],
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <button
          type="button"
          className="brand"
          onClick={() => setPage("overview")}
          aria-label="返回投资组合总览"
        >
          <span className="brand-mark">R</span>
          <span>
            <strong>Relife</strong>
            <small>PORTFOLIO</small>
          </span>
        </button>
        <div className="market-switch" aria-label="市场切换">
          <button
            type="button"
            className={marketKey === "us" ? "active" : ""}
            onClick={() => updateMarket("us")}
          >
            <span>US</span> 美股
          </button>
          <button
            type="button"
            className={marketKey === "cn" ? "active" : ""}
            onClick={() => updateMarket("cn")}
          >
            <span>CN</span> A 股
          </button>
        </div>
        <div className="topbar-meta">
          <span className="status-dot" />
          <span>收盘数据 · {market.asOf}</span>
          <span className="avatar">R</span>
        </div>
      </header>

      <nav className="page-nav" aria-label="投资组合页面">
        {PAGES.map((item) => (
          <button
            type="button"
            key={item.id}
            className={page === item.id ? "active" : ""}
            onClick={() => setPage(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <main>
        <div className="market-heading">
          <div>
            <span className="eyebrow">
              {marketKey === "us" ? "US PORTFOLIO" : "A-SHARE PORTFOLIO"}
            </span>
            <h1>
              {marketKey === "us" ? "美股投资组合" : "A 股投资组合"}
            </h1>
          </div>
          <div className="market-facts">
            <span>{positionCount} 个持仓</span>
            <span>{market.currency}</span>
            <span>
              更新于{" "}
              {new Date(data.generatedAt).toLocaleString("zh-CN", {
                month: "2-digit",
                day: "2-digit",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          </div>
        </div>

        {page === "overview" && (
          <div className="overview-grid">
            {positionCount === 0 && (
              <div className="cash-banner">
                <span>当前空仓</span>
                <strong>现金 100%</strong>
                <small>组合历史与基准对比继续保留</small>
              </div>
            )}
            <section className="summary-grid" aria-label="组合摘要">
              <article className="metric-card primary-metric">
                <span>总资产</span>
                <strong>
                  {formatMoney(
                    market.summary.totalAssets,
                    market.currency,
                  )}
                </strong>
                <small>按最后收盘价计算</small>
              </article>
              <article className="metric-card">
                <span>累计收益率</span>
                <strong
                  className={valueClass(market.summary.cumulativeReturn)}
                >
                  {formatPercent(market.summary.cumulativeReturn)}
                </strong>
                <small>现金流调整后</small>
              </article>
              <article className="metric-card">
                <span>未实现收益</span>
                <strong className={valueClass(market.summary.unrealizedPnl)}>
                  {formatMoney(
                    market.summary.unrealizedPnl,
                    market.currency,
                  )}
                </strong>
                <small>当前持仓浮动盈亏</small>
              </article>
              <article className="metric-card">
                <span>可用现金</span>
                <strong>
                  {formatMoney(market.summary.cash, market.currency)}
                </strong>
                <small>
                  {formatPercent(
                    market.summary.cash / market.summary.totalAssets,
                  )}{" "}
                  占组合
                </small>
              </article>
            </section>

            <section className="panel chart-panel">
              <div className="panel-heading responsive-heading">
                <div>
                  <span className="eyebrow">PERFORMANCE</span>
                  <h2>持仓走势对比</h2>
                  <p>现金流调整后的归一化收益，起点 = 100</p>
                </div>
                <div className="range-switch" aria-label="收益区间">
                  {RANGES.map((item) => (
                    <button
                      type="button"
                      key={item.id}
                      className={range === item.id ? "active" : ""}
                      onClick={() => setRange(item.id)}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>
              <PerformanceChart market={market} range={range} />
            </section>

            <section className="panel allocation-panel">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">ALLOCATION</span>
                  <h2>策略分组</h2>
                </div>
              </div>
              {market.groups.length === 0 ? (
                <div className="empty-state">当前没有持仓分组</div>
              ) : (
                <div className="allocation-list">
                  {market.groups.map((group) => (
                    <div className="allocation-row" key={group.id}>
                      <div>
                        <strong>{group.label}</strong>
                        <span>{group.badge || group.subgroup}</span>
                      </div>
                      <div className="allocation-track">
                        <i
                          style={{
                            width: `${Math.max((group.weight ?? 0) * 100, 1)}%`,
                          }}
                        />
                      </div>
                      <b>{formatPercent(group.weight)}</b>
                    </div>
                  ))}
                  <div className="allocation-row">
                    <div>
                      <strong>现金</strong>
                      <span>可用余额</span>
                    </div>
                    <div className="allocation-track cash-track">
                      <i
                        style={{
                          width: `${Math.max(
                            (market.summary.cash /
                              market.summary.totalAssets) *
                              100,
                            1,
                          )}%`,
                        }}
                      />
                    </div>
                    <b>
                      {formatPercent(
                        market.summary.cash / market.summary.totalAssets,
                      )}
                    </b>
                  </div>
                </div>
              )}
            </section>

            <HoldingsTable
              market={market}
              columns={columns}
              onColumnsChange={updateColumns}
              onResetColumns={() => updateColumns(DEFAULT_COLUMNS)}
              compact
            />
          </div>
        )}

        {page === "holdings" && (
          <HoldingsTable
            market={market}
            columns={columns}
            onColumnsChange={updateColumns}
            onResetColumns={() => updateColumns(DEFAULT_COLUMNS)}
          />
        )}
        {page === "transactions" && <TransactionsView market={market} />}
        {page === "dividends" && <DividendsView market={market} />}
      </main>

      <footer>
        <span>Relife Private Portfolio</span>
        <span>A 股与美股独立核算 · 仅使用收盘数据</span>
      </footer>
    </div>
  );
}
