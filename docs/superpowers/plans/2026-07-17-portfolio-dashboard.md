# Relife Portfolio Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and privately deploy a TradingView-style A-share and US portfolio dashboard from the repository's latest transaction CSV files.

**Architecture:** A Python package parses broker records, reconstructs ledgers with moving-average cost, downloads daily close data, and writes one static `public/data/portfolio.json` contract. A vinext/React single-page dashboard reads that contract and provides market/page switching, benchmark charts, grouped holdings, transactions, dividends, and device-local column preferences. The validated build is deployed through Sites with owner-only access.

**Tech Stack:** Python 3.12 standard library, pandas, yfinance, akshare, unittest; vinext, React 19, TypeScript, CSS, Node test runner; Codex Sites.

## Global Constraints

- A 股与美股完全隔离核算，不做汇率换算或跨市场合计。
- 行情只使用每日收盘价；美股基准为 `QQQ`、`SPY`，A 股基准为上证指数、沪深 300。
- 主题使用中性炭黑背景，红涨绿跌；背景不得带蓝色倾向。
- Sites 必须以仅所有者可访问模式部署。
- 不覆盖原始交易文件、历史快照、`.vscode/launch.json` 或用户未提交的 `docs/plans/trading-strategy-v0.2.md`。
- 当前持仓缺少行情时生成失败；失败不得覆盖上一次成功的 JSON。
- Python 新增第三方依赖时同步更新根目录 `requirements.txt`；本计划不新增现有清单之外的 Python 包。

---

### Task 1: Scaffold the Sites application and lock the JSON contract

**Files:**
- Create through Sites initializer: `package.json`, `package-lock.json`, `app/layout.tsx`, `app/page.tsx`, `app/globals.css`, `vite.config.ts`, `next.config.ts`, `tsconfig.json`
- Create: `portfolio_dashboard/__init__.py`
- Create: `portfolio_dashboard/schema.py`
- Create: `tests/test_portfolio_schema.py`
- Create: `public/data/portfolio.json`

**Interfaces:**
- Produces: `validate_dashboard_payload(payload: dict[str, object]) -> None`
- Produces JSON top-level shape: `{"generatedAt": str, "markets": {"us": MarketPayload, "cn": MarketPayload}}`
- Each `MarketPayload` contains `currency`, `asOf`, `summary`, `performance`, `benchmarks`, `groups`, `transactions`, `dividends`, and `dividendMonths`.

- [ ] **Step 1: Initialize the vinext starter in the repository root**

Run:

```bash
/mnt/c/Users/yuemi/.codex/plugins/cache/openai-bundled/sites/0.1.27/scripts/init-site.sh "$PWD"
```

Expected: dependencies install successfully and the starter creates `app/`, `package.json`, and `.openai/hosting.json` without changing existing Python/data paths.

- [ ] **Step 2: Write the failing schema test**

```python
# tests/test_portfolio_schema.py
import unittest

from portfolio_dashboard.schema import validate_dashboard_payload


class DashboardSchemaTest(unittest.TestCase):
    def test_requires_both_isolated_markets(self) -> None:
        with self.assertRaisesRegex(ValueError, "markets.us"):
            validate_dashboard_payload({"generatedAt": "2026-07-17T00:00:00+08:00", "markets": {}})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the schema test and confirm the red state**

Run: `python3 -m unittest tests.test_portfolio_schema -v`

Expected: `ModuleNotFoundError: No module named 'portfolio_dashboard.schema'`.

- [ ] **Step 4: Implement the contract validator**

```python
# portfolio_dashboard/schema.py
REQUIRED_MARKET_KEYS = {
    "currency", "asOf", "summary", "performance", "benchmarks",
    "groups", "transactions", "dividends", "dividendMonths",
}


def validate_dashboard_payload(payload: dict[str, object]) -> None:
    markets = payload.get("markets")
    if not isinstance(markets, dict):
        raise ValueError("markets 必须是对象")
    for market in ("us", "cn"):
        value = markets.get(market)
        if not isinstance(value, dict):
            raise ValueError(f"缺少 markets.{market}")
        missing = REQUIRED_MARKET_KEYS - value.keys()
        if missing:
            raise ValueError(f"markets.{market} 缺少字段：{sorted(missing)}")
```

Add a minimal valid two-market fixture to `public/data/portfolio.json` so the starter can be replaced incrementally.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m unittest tests.test_portfolio_schema -v`

Expected: one passing test.

Commit:

```bash
git add package.json package-lock.json app vite.config.ts next.config.ts tsconfig.json portfolio_dashboard tests/test_portfolio_schema.py public/data/portfolio.json .openai/hosting.json
git commit -m "chore: scaffold portfolio dashboard"
```

### Task 2: Parse US and A-share transaction sources

**Files:**
- Create: `portfolio_dashboard/models.py`
- Create: `portfolio_dashboard/parsers.py`
- Create: `tests/test_portfolio_parsers.py`
- Create: `tests/fixtures/portfolio/tradingview.csv`
- Create: `tests/fixtures/portfolio/yinhe.csv`

**Interfaces:**
- Produces `Transaction` dataclass with `market`, `timestamp`, `source_index`, `symbol`, `name`, `kind`, `quantity`, `price`, `fee`, `cash_delta`, `external_cash_flow`, `source_id`.
- Produces `parse_tradingview(path: Path) -> list[Transaction]`.
- Produces `parse_yinhe(path: Path) -> list[Transaction]`.
- Produces `latest_source(pattern: str, root: Path) -> Path`.

- [ ] **Step 1: Write parser tests for supported rows and stable sorting**

```python
class TradingViewParserTest(unittest.TestCase):
    def test_parses_buy_dividend_and_deposit(self) -> None:
        rows = parse_tradingview(self.fixture("tradingview.csv"))
        self.assertEqual(["deposit", "buy", "dividend"], [row.kind for row in rows])
        self.assertEqual(Decimal("0.08"), rows[-1].cash_delta)


class YinheParserTest(unittest.TestCase):
    def test_preserves_same_day_source_order(self) -> None:
        rows = parse_yinhe(self.fixture("yinhe.csv"))
        same_day = [row for row in rows if row.timestamp.date().isoformat() == "2026-07-14"]
        self.assertEqual([0, 1, 2], [row.source_index for row in same_day])
```

The fixtures include `证券买入`, `证券卖出`, `银行转证券`, `证券转银行`, `利息归本`, `红利入账`, and reverse-repo actions.

- [ ] **Step 2: Run parser tests and confirm failure**

Run: `python3 -m unittest tests.test_portfolio_parsers -v`

Expected: import failure for `portfolio_dashboard.parsers`.

- [ ] **Step 3: Implement normalized models and parsers**

```python
@dataclass(frozen=True)
class Transaction:
    market: Literal["us", "cn"]
    timestamp: datetime
    source_index: int
    symbol: str
    name: str
    kind: Literal["buy", "sell", "dividend", "deposit", "withdrawal", "interest", "cash"]
    quantity: Decimal
    price: Decimal
    fee: Decimal
    cash_delta: Decimal
    external_cash_flow: Decimal
    source_id: str
```

Open each source with `path.open(encoding="utf-8-sig", newline="")`, pass the handle to `csv.DictReader(handle)`, validate exact source headers, normalize US exchange prefixes away for lookup, and retain the original source row index. For A shares, deduplicate by `(成交编号, 合同编号)` when present and by full-row fingerprint otherwise.

- [ ] **Step 4: Verify parsing against fixtures and current files**

Run:

```bash
python3 -m unittest tests.test_portfolio_parsers -v
python3 -c "from pathlib import Path; from portfolio_dashboard.parsers import parse_tradingview,parse_yinhe; print(len(parse_tradingview(Path('data/tradingview/tradingview_full_latest_2026-07-16.csv'))), len(parse_yinhe(Path('data/transactions/yinhe/交割单_2026-07-15.csv'))))"
```

Expected: all tests pass and the sample counts print `111 175`.

- [ ] **Step 5: Commit parser work**

```bash
git add portfolio_dashboard/models.py portfolio_dashboard/parsers.py tests/test_portfolio_parsers.py
git commit -m "feat: parse portfolio transaction sources"
```

### Task 3: Reconstruct ledgers, returns, dividends, and strategy groups

**Files:**
- Create: `portfolio_dashboard/ledger.py`
- Create: `portfolio_dashboard/classification.py`
- Create: `data/templates/portfolio/strategy_groups.json`
- Create: `tests/test_portfolio_ledger.py`

**Interfaces:**
- Consumes: `Transaction`.
- Produces `replay_ledger(transactions: Sequence[Transaction], closes: PriceMatrix) -> LedgerResult`.
- Produces `classify_symbol(market: str, symbol: str, name: str, config: dict) -> Classification`.
- `LedgerResult` exposes `positions`, `cash`, `transactions`, `dividends`, `daily_values`, and `external_flows`.

- [ ] **Step 1: Write failing moving-average and return formula tests**

```python
def test_partial_sell_uses_weighted_average_and_net_proceeds(self) -> None:
    ledger = replay_ledger(
        [
            tx("buy", qty="2", price="10", fee="1"),
            tx("buy", qty="2", price="14", fee="1"),
            tx("sell", qty="1", price="15", fee="0.5"),
        ],
        closes={"ABC": [("2026-07-16", "16"), ("2026-07-17", "17")]},
    )
    position = ledger.positions["ABC"]
    self.assertEqual(Decimal("3"), position.quantity)
    self.assertEqual(Decimal("37.5"), position.total_cost)
    self.assertEqual(Decimal("2"), position.realized_pnl)
    self.assertEqual(Decimal("13.5"), position.unrealized_pnl)


def test_contribution_uses_market_total_assets(self) -> None:
    metrics = position_metrics(position("10", cost="80"), close="10", previous="9", total_assets="200")
    self.assertEqual(Decimal("0.25"), metrics.unrealized_pnl_pct)
    self.assertEqual(Decimal("0.10"), metrics.portfolio_contribution_pct)
```

- [ ] **Step 2: Run ledger tests and confirm failure**

Run: `python3 -m unittest tests.test_portfolio_ledger -v`

Expected: import failure for `portfolio_dashboard.ledger`.

- [ ] **Step 3: Implement ledger replay and metrics**

Use `Decimal` for transaction and cost calculations. On buy, allocate price × quantity + fee into moving-average cost. On sell, allocate average cost, use price × quantity − fee as net proceeds, and reject negative holdings. Aggregate dividends by `(date, symbol)`, splitting positive amounts into gross and negative amounts into tax adjustment.

Implement:

```python
def total_return(position: PositionState) -> Decimal:
    return position.realized_pnl + position.unrealized_pnl + position.net_dividends


def safe_ratio(numerator: Decimal, denominator: Decimal) -> Decimal | None:
    return None if denominator == 0 else numerator / denominator
```

- [ ] **Step 4: Add strategy classification configuration**

```json
{
  "groups": [
    {"id": "cashflow", "label": "现金流", "members": ["$CASH", "BOXX"]},
    {"id": "broad-market", "label": "大盘", "members": ["QQQ", "512100"]},
    {"id": "dividend", "label": "股息", "members": ["XQQI", "QQQI", "SCHD", "513530"]},
    {"id": "stock", "label": "个股", "members": ["BRK.B", "KO", "GOOGL", "JD"]},
    {"id": "leverage", "label": "杠杆", "members": ["SOXL", "SOXS", "YINN", "YANG", "TQQQ", "SQQQ"]},
    {"id": "bitcoin", "label": "比特币", "members": ["IBIT"]}
  ],
  "fallback": {"id": "other", "label": "其他", "badge": "策略外"}
}
```

Assert that `XBI` classifies as `other`, `BOXX` as `cashflow`, `QQQI` as `dividend`, and `SOXS` as `leverage`.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m unittest tests.test_portfolio_ledger -v`

Expected: all ledger and classification tests pass.

Commit:

```bash
git add portfolio_dashboard/ledger.py portfolio_dashboard/classification.py data/templates/portfolio/strategy_groups.json tests/test_portfolio_ledger.py
git commit -m "feat: calculate portfolio ledger metrics"
```

### Task 4: Download close data and generate the atomic dashboard JSON

**Files:**
- Create: `portfolio_dashboard/market_data.py`
- Create: `portfolio_dashboard/generator.py`
- Create: `scripts/generate_portfolio_dashboard.py`
- Create: `tests/test_portfolio_generator.py`
- Modify: `package.json`

**Interfaces:**
- Produces `YahooPriceProvider.history(symbols, start, end) -> PriceMatrix`.
- Produces `AksharePriceProvider.history(symbols, start, end) -> PriceMatrix`.
- Produces `build_dashboard(us_path, cn_path, output_path, providers, generated_at) -> dict`.
- CLI options: `--us-transactions`, `--cn-transactions`, `--output`, `--offline-fixtures`.

- [ ] **Step 1: Write failing cash-flow-adjusted performance and atomic-write tests**

```python
def test_twr_removes_external_deposit(self) -> None:
    points = build_performance(
        values=[("2026-01-01", "100"), ("2026-01-02", "210")],
        external_flows={"2026-01-02": Decimal("100")},
    )
    self.assertEqual(Decimal("1.10"), points[-1].index / Decimal("100"))


def test_generation_does_not_replace_previous_json_on_missing_current_price(self) -> None:
    output.write_text('{"sentinel": true}', encoding="utf-8")
    with self.assertRaisesRegex(MissingPriceError, "QQQI"):
        build_dashboard(
            us_path=self.fixture("tradingview.csv"),
            cn_path=self.fixture("yinhe.csv"),
            output_path=output,
            providers=MissingCurrentPriceProviders(),
            generated_at=datetime.fromisoformat("2026-07-17T12:00:00+08:00"),
        )
    self.assertEqual('{"sentinel": true}', output.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run generator tests and confirm failure**

Run: `python3 -m unittest tests.test_portfolio_generator -v`

Expected: import failure for `portfolio_dashboard.generator`.

- [ ] **Step 3: Implement providers and performance alignment**

Use yfinance adjusted daily closes for US symbols, `QQQ`, and `SPY`. Use AkShare daily close series for A-share securities, `sh000001`, and `sh000300`. Normalize every date to ISO `YYYY-MM-DD`, sort ascending, and forward-fill only between first and last valid observations.

Implement daily cash-flow adjustment:

```python
factor = (today_value - external_flow_today) / previous_value
index_value = previous_index * factor
```

Skip the first value and any period where `previous_value == 0`. Normalize benchmarks and the portfolio to 100 on their first common valid date.

- [ ] **Step 4: Implement generator and atomic output**

Write to `public/data/.portfolio.json.tmp`, call `validate_dashboard_payload`, then replace `public/data/portfolio.json` with `Path.replace`. On any exception, delete only the temporary file and preserve the previous JSON.

Add:

```json
"portfolio:data": "python3 scripts/generate_portfolio_dashboard.py",
"portfolio:update": "npm run portfolio:data && npm run build"
```

to `package.json`.

- [ ] **Step 5: Verify offline fixtures and commit**

Run:

```bash
python3 -m unittest tests.test_portfolio_generator -v
python3 scripts/generate_portfolio_dashboard.py --offline-fixtures tests/fixtures/market_data
python3 -c "import json; d=json.load(open('public/data/portfolio.json')); print([p['symbol'] for g in d['markets']['us']['groups'] for p in g['positions']])"
```

Expected: tests pass and current US symbols are `BOXX`, `QQQI`, `SCHD`, `SOXS`, and `XBI`; the A-share position list is empty.

Commit:

```bash
git add portfolio_dashboard/market_data.py portfolio_dashboard/generator.py scripts/generate_portfolio_dashboard.py tests/test_portfolio_generator.py tests/fixtures/market_data package.json public/data/portfolio.json
git commit -m "feat: generate portfolio dashboard data"
```

### Task 5: Build the neutral-dark dashboard shell and overview charts

**Files:**
- Modify: `app/layout.tsx`
- Modify: `app/page.tsx`
- Modify: `app/globals.css`
- Create: `app/portfolio-types.ts`
- Create: `app/portfolio-state.mjs`
- Create: `tests/portfolio-state.test.mjs`
- Delete: `app/_sites-preview/SkeletonPreview.tsx`
- Delete: `app/_sites-preview/preview.css`

**Interfaces:**
- Consumes `public/data/portfolio.json`.
- Produces pure helpers `selectRange(points, range, now)`, `formatMoney`, `formatPercent`, `filterTransactions`.
- UI state: market `us | cn`, page `overview | holdings | transactions | dividends`, range `1m | 3m | ytd | all`.

- [ ] **Step 1: Write failing Node tests for range and transaction behavior**

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { filterTransactions, selectRange } from "../app/portfolio-state.mjs";

test("all keeps the full history and transaction filters preserve reverse order", () => {
  assert.deepEqual(selectRange([{ date: "2026-01-01" }, { date: "2026-07-17" }], "all", new Date("2026-07-17")), [
    { date: "2026-01-01" }, { date: "2026-07-17" },
  ]);
  assert.deepEqual(filterTransactions([{ kind: "sell", timestamp: "2026-02-01" }, { kind: "buy", timestamp: "2026-03-01" }], "buy"), [
    { kind: "buy", timestamp: "2026-03-01" },
  ]);
});
```

- [ ] **Step 2: Run Node tests and confirm failure**

Run: `node --test tests/portfolio-state.test.mjs`

Expected: module-not-found failure.

- [ ] **Step 3: Implement helpers and dashboard shell**

Replace the starter with a client component that loads static JSON, restores `relife.market`, and renders:

```tsx
<MarketSwitch value={market} onChange={setMarket} />
<PageTabs value={page} onChange={setPage} items={["overview", "holdings", "transactions", "dividends"]} />
```

The overview contains four summary metrics and one responsive SVG chart with directly labeled series for the portfolio and two market benchmarks. Add 1 月、3 月、今年、全部 buttons; default to `all`.

- [ ] **Step 4: Apply final visual system and metadata**

Use neutral values such as `#121212`, `#1c1c1c`, `#303030`, and `#e5e5e5`; use red for positive and green for negative. Keep blue only for the portfolio line, not backgrounds. Update metadata title/description, remove the temporary preview marker/imports, and remove `react-loading-skeleton` if unused.

- [ ] **Step 5: Run tests/build and commit**

Run:

```bash
node --test tests/portfolio-state.test.mjs
npm run build
```

Expected: Node tests pass and vinext build completes.

Commit:

```bash
git add app package.json package-lock.json tests/portfolio-state.test.mjs
git commit -m "feat: add portfolio overview dashboard"
```

### Task 6: Add grouped holdings and icon-triggered column preferences

**Files:**
- Create: `app/components/HoldingsTable.tsx`
- Create: `app/components/ColumnSettings.tsx`
- Modify: `app/page.tsx`
- Modify: `app/globals.css`
- Modify: `app/portfolio-state.mjs`
- Modify: `tests/portfolio-state.test.mjs`

**Interfaces:**
- Produces `DEFAULT_COLUMNS` in the approved 13-column order.
- Produces `normalizeColumnPreferences(value) -> ColumnPreference[]`.
- Stores preferences under `relife.holdings.columns.v1`.
- `ColumnSettings` receives `columns`, `onChange`, and `onReset`.

- [ ] **Step 1: Add failing preference normalization tests**

```javascript
test("column preferences keep symbol locked and reject unknown ids", () => {
  const normalized = normalizeColumnPreferences([
    { id: "dailyPnl", visible: false },
    { id: "unknown", visible: true },
    { id: "symbol", visible: false },
  ]);
  assert.equal(normalized[0].id, "symbol");
  assert.equal(normalized[0].visible, true);
  assert.equal(normalized.some((column) => column.id === "unknown"), false);
});
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `node --test tests/portfolio-state.test.mjs`

Expected: missing export `normalizeColumnPreferences`.

- [ ] **Step 3: Implement holdings table and settings popover**

Render strategy group headers and current positions. The settings surface is closed by default and opens only from an icon-only button with `aria-label="设置持仓列"`. It supports checkboxes, pointer drag/drop, keyboard up/down buttons, and “恢复默认”. Keep the symbol column locked and sticky; use horizontal scrolling on small screens.

- [ ] **Step 4: Persist shared market preferences**

Read/write `relife.holdings.columns.v1` in `localStorage`; A 股 and美股 use the same preference list on the current device. Invalid saved values fall back through `normalizeColumnPreferences`.

- [ ] **Step 5: Test/build and commit**

Run:

```bash
node --test tests/portfolio-state.test.mjs
npm run build
```

Expected: all state tests and build pass.

Commit:

```bash
git add app/components app/page.tsx app/globals.css app/portfolio-state.mjs tests/portfolio-state.test.mjs
git commit -m "feat: add configurable grouped holdings"
```

### Task 7: Add reverse-chronological transactions and dividend analytics

**Files:**
- Create: `app/components/TransactionsView.tsx`
- Create: `app/components/DividendsView.tsx`
- Modify: `app/page.tsx`
- Modify: `app/globals.css`
- Modify: `tests/portfolio-state.test.mjs`

**Interfaces:**
- `TransactionsView` consumes normalized transactions already sorted descending.
- `DividendsView` consumes `dividends` and `dividendMonths`.
- Clicking the transaction “股息” filter remains within the transaction page; the primary “股息” page shows totals, monthly bars, and details.

- [ ] **Step 1: Add failing transaction and month aggregation rendering-state tests**

```javascript
test("transaction all filter sorts timestamp descending", () => {
  const result = filterTransactions([
    { id: "old", kind: "buy", timestamp: "2026-01-01T00:00:00" },
    { id: "new", kind: "dividend", timestamp: "2026-07-01T00:00:00" },
  ], "all");
  assert.deepEqual(result.map((row) => row.id), ["new", "old"]);
});
```

- [ ] **Step 2: Run test and confirm failure**

Run: `node --test tests/portfolio-state.test.mjs`

Expected: order assertion fails before helper sorting is added.

- [ ] **Step 3: Implement transaction and dividend pages**

Transactions render date, symbol/name, action, quantity, price, amount, and fee with filters `全部 / 买入 / 卖出 / 股息 / 资金`. Dividends render cumulative net dividend, an accessible monthly net-dividend bar chart, and descending details with gross, tax adjustment, and net values; unavailable A-share gross/tax values render `—`.

- [ ] **Step 4: Add empty states and responsive behavior**

Render explicit empty text for no positions, no matching transactions, and no dividends. On A-share overview with no positions, show “当前空仓 · 现金 100%” while retaining the historical chart.

- [ ] **Step 5: Test/build and commit**

Run:

```bash
node --test tests/portfolio-state.test.mjs
npm run build
```

Expected: all tests pass and production build completes.

Commit:

```bash
git add app/components app/page.tsx app/globals.css tests/portfolio-state.test.mjs
git commit -m "feat: add transactions and dividends views"
```

### Task 8: Integrate live sample data, document refresh, verify, and deploy privately

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt` only if imports differ from its existing `pandas`, `yfinance`, `akshare` entries
- Regenerate: `public/data/portfolio.json`
- Modify: `.openai/hosting.json` only to persist the exact Sites `project_id`

**Interfaces:**
- Operator command: `npm run portfolio:update`.
- Deployment source is the exact validated commit and build.

- [ ] **Step 1: Run the complete Python suite**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_portfolio_*.py' -v
```

Expected: all portfolio Python tests pass.

- [ ] **Step 2: Generate current data from the two repository CSV files**

Run: `npm run portfolio:data`

Expected:

- A 股 current position count is `0` and cash is approximately `3749.16 CNY`.
- 美股 positions are `BOXX`, `QQQI`, `SCHD`, `SOXS`, and `XBI`.
- `XBI` is in `其他 / 策略外`.
- Transactions and dividend details are reverse chronological.

- [ ] **Step 3: Run frontend tests and production build**

Run:

```bash
node --test tests/portfolio-state.test.mjs
npm run build
```

Expected: all tests pass and `dist/server/index.js` exists.

- [ ] **Step 4: Update README**

Document:

```markdown
## 私有投资组合仪表盘

刷新交易、行情并构建：

```bash
npm run portfolio:update
```

数据源自动选择最新美股 TradingView 全量文件和最新银河证券交割单。页面使用最近收盘价；A 股、美股独立核算。
```

- [ ] **Step 5: Create the private Sites project and persist its ID**

Call `create_site` once with title `Relife Portfolio`, description `A 股与美股独立核算的私人投资组合仪表盘`, and an available slug beginning with `relife-portfolio`. Copy the returned opaque ID exactly into `.openai/hosting.json` as `project_id`.

- [ ] **Step 6: Commit the exact validated source**

```bash
git add README.md requirements.txt public/data/portfolio.json .openai/hosting.json
git commit -m "docs: add portfolio dashboard refresh workflow"
```

Do not stage `.vscode/launch.json` or `docs/plans/trading-strategy-v0.2.md`.

- [ ] **Step 7: Save and deploy the exact version through Sites**

Push the exact validated commit with the credential returned by `create_site`, package with the Sites `package-site.sh` helper, save one version, and call the private deployment tool. Poll until the deployment reports `succeeded`.

Expected: an owner-only production URL that opens the dashboard successfully.

- [ ] **Step 8: Final verification**

Run:

```bash
git status --short
git log -8 --oneline
```

Expected: only the user's pre-existing `.vscode/launch.json` modification and `docs/plans/trading-strategy-v0.2.md` untracked file remain unrelated; all dashboard commits are present.
