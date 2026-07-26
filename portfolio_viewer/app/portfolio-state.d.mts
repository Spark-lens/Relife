import type {
  ColumnPreference,
  PerformancePoint,
  RangeKey,
  Transaction,
} from "./portfolio-types";

export const DEFAULT_COLUMNS: ColumnPreference[];

export function selectRange<T extends Pick<PerformancePoint, "date">>(
  points: T[],
  range: RangeKey,
  now?: Date,
): T[];

export function filterTransactions<T extends Pick<Transaction, "kind" | "timestamp">>(
  transactions: T[],
  filter: string,
): T[];

export function normalizeColumnPreferences(
  value: unknown,
): ColumnPreference[];

export function formatMoney(
  value: number | null | undefined,
  currency: string,
): string;

export function formatNumber(
  value: number | null | undefined,
  maximumFractionDigits?: number,
): string;

export function formatPercent(
  value: number | null | undefined,
): string;
