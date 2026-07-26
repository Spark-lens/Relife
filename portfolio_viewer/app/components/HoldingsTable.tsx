"use client";

import {
  formatMoney,
  formatNumber,
  formatPercent,
} from "../portfolio-state.mjs";
import type {
  ColumnPreference,
  MarketPayload,
  Position,
} from "../portfolio-types";
import { ColumnSettings } from "./ColumnSettings";

const PERCENT_FIELDS = new Set([
  "weight",
  "dailyPnlPct",
  "totalPnlPct",
  "unrealizedPnlPct",
  "portfolioContributionPct",
]);
const MONEY_FIELDS = new Set([
  "totalCost",
  "lastClose",
  "marketValue",
  "dailyPnl",
  "totalPnl",
  "unrealizedPnl",
]);
const SIGNED_FIELDS = new Set([
  "dailyPnl",
  "dailyPnlPct",
  "totalPnl",
  "totalPnlPct",
  "unrealizedPnl",
  "unrealizedPnlPct",
  "portfolioContributionPct",
]);

function cell(
  position: Position,
  field: string,
  currency: string,
) {
  if (field === "symbol") {
    return (
      <div className="symbol-cell">
        <strong>{position.symbol}</strong>
        <span>{position.name !== position.symbol ? position.name : ""}</span>
      </div>
    );
  }
  const value = position[field as keyof Position] as number | null;
  if (PERCENT_FIELDS.has(field)) return formatPercent(value);
  if (MONEY_FIELDS.has(field)) return formatMoney(value, currency);
  return formatNumber(value, 4);
}

export function HoldingsTable({
  market,
  columns,
  onColumnsChange,
  onResetColumns,
  compact = false,
}: {
  market: MarketPayload;
  columns: ColumnPreference[];
  onColumnsChange: (columns: ColumnPreference[]) => void;
  onResetColumns: () => void;
  compact?: boolean;
}) {
  const visible = columns.filter((column) => column.visible);
  const groups = compact ? market.groups.slice(0, 3) : market.groups;

  return (
    <section className="panel holdings-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">CURRENT POSITIONS</span>
          <h2>{compact ? "持仓概览" : "全部持仓"}</h2>
        </div>
        <ColumnSettings
          columns={columns}
          onChange={onColumnsChange}
          onReset={onResetColumns}
        />
      </div>
      {groups.length === 0 ? (
        <div className="empty-state">
          <strong>当前空仓</strong>
          <span>现金占组合 100%，历史收益曲线仍保留。</span>
        </div>
      ) : (
        <div className="table-scroll">
          <table className="data-table holdings-table">
            <thead>
              <tr>
                {visible.map((column) => (
                  <th
                    key={column.id}
                    className={column.id === "symbol" ? "sticky-column" : ""}
                  >
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {groups.flatMap((group) => [
                <tr className="group-row" key={`${group.id}:heading`}>
                  <td colSpan={visible.length}>
                    <span>{group.label}</span>
                    {group.subgroup && <em>{group.subgroup}</em>}
                    {group.badge && <b>{group.badge}</b>}
                    <small>
                      {formatMoney(group.marketValue, market.currency)} ·{" "}
                      {formatPercent(group.weight)}
                    </small>
                  </td>
                </tr>,
                ...group.positions.map((position) => (
                  <tr key={position.symbol}>
                    {visible.map((column) => {
                      const value = position[
                        column.id as keyof Position
                      ] as number | null;
                      const signed = SIGNED_FIELDS.has(column.id);
                      return (
                        <td
                          key={column.id}
                          className={[
                            column.id === "symbol" ? "sticky-column" : "",
                            signed && (value ?? 0) > 0 ? "positive" : "",
                            signed && (value ?? 0) < 0 ? "negative" : "",
                          ]
                            .filter(Boolean)
                            .join(" ")}
                        >
                          {cell(position, column.id, market.currency)}
                        </td>
                      );
                    })}
                  </tr>
                )),
              ])}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
