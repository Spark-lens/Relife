"use client";

import { useState } from "react";

import {
  filterTransactions,
  formatMoney,
  formatNumber,
} from "../portfolio-state.mjs";
import type { MarketPayload } from "../portfolio-types";

const FILTERS = [
  ["all", "全部"],
  ["buy", "买入"],
  ["sell", "卖出"],
  ["dividend", "股息"],
  ["cash", "资金"],
] as const;

export function TransactionsView({ market }: { market: MarketPayload }) {
  const [filter, setFilter] = useState("all");
  const rows = filterTransactions(market.transactions, filter);

  return (
    <section className="panel">
      <div className="panel-heading responsive-heading">
        <div>
          <span className="eyebrow">ACTIVITY</span>
          <h2>交易记录</h2>
          <p>默认按时间倒序，共 {market.transactions.length} 条</p>
        </div>
        <div className="filter-group" aria-label="交易类型筛选">
          {FILTERS.map(([id, label]) => (
            <button
              type="button"
              key={id}
              className={filter === id ? "active" : ""}
              onClick={() => setFilter(id)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      {rows.length === 0 ? (
        <div className="empty-state">没有符合条件的交易记录</div>
      ) : (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>时间</th>
                <th>标的</th>
                <th>操作</th>
                <th>数量</th>
                <th>成交价</th>
                <th>发生金额</th>
                <th>费用</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row: MarketPayload["transactions"][number]) => (
                <tr key={row.id}>
                  <td className="muted-cell">
                    {row.timestamp.replace("T", " ").slice(0, 16)}
                  </td>
                  <td>
                    <div className="symbol-cell">
                      <strong>{row.symbol || "现金"}</strong>
                      <span>{row.name !== row.symbol ? row.name : ""}</span>
                    </div>
                  </td>
                  <td>
                    <span className={`action-badge ${row.kind}`}>
                      {row.action}
                    </span>
                  </td>
                  <td>{row.quantity ? formatNumber(row.quantity, 4) : "—"}</td>
                  <td>
                    {row.price
                      ? formatMoney(row.price, market.currency)
                      : "—"}
                  </td>
                  <td
                    className={
                      row.amount > 0
                        ? "positive"
                        : row.amount < 0
                          ? "negative"
                          : ""
                    }
                  >
                    {formatMoney(row.amount, market.currency)}
                  </td>
                  <td>{formatMoney(row.fee, market.currency)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
