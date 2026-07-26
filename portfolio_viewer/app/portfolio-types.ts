export type MarketKey = "us" | "cn";
export type PageKey = "overview" | "holdings" | "transactions" | "dividends";
export type RangeKey = "1m" | "3m" | "ytd" | "all";

export type PerformancePoint = {
  date: string;
  portfolio: number;
  [key: string]: string | number;
};

export type Position = {
  symbol: string;
  name: string;
  quantity: number;
  weight: number | null;
  totalCost: number;
  averageCost: number;
  lastClose: number;
  marketValue: number;
  dailyPnl: number;
  dailyPnlPct: number | null;
  unrealizedPnl: number;
  unrealizedPnlPct: number | null;
  portfolioContributionPct: number | null;
  totalPnl: number;
  totalPnlPct: number | null;
};

export type PositionGroup = {
  id: string;
  label: string;
  subgroup: string;
  badge: string;
  marketValue: number;
  weight: number | null;
  positions: Position[];
};

export type Transaction = {
  id: string;
  timestamp: string;
  symbol: string;
  name: string;
  kind: string;
  action: string;
  quantity: number;
  price: number;
  amount: number;
  fee: number;
};

export type Dividend = {
  date: string;
  symbol: string;
  name: string;
  gross: number | null;
  taxAdjustment: number | null;
  net: number;
};

export type MarketPayload = {
  currency: "USD" | "CNY";
  asOf: string;
  summary: {
    totalAssets: number;
    cumulativeReturn: number;
    unrealizedPnl: number;
    cash: number;
  };
  performance: PerformancePoint[];
  benchmarks: { id: string; label: string }[];
  groups: PositionGroup[];
  transactions: Transaction[];
  dividends: Dividend[];
  dividendMonths: { month: string; net: number }[];
};

export type DashboardPayload = {
  generatedAt: string;
  markets: Record<MarketKey, MarketPayload>;
};

export type ColumnPreference = {
  id: string;
  label: string;
  visible: boolean;
  locked?: boolean;
};
