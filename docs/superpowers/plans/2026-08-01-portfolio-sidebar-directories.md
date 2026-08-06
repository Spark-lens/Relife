# Relife Portfolio 0.3.0 目录化侧栏实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有观察列表重构为 HOME、标的、因子三级根目录，并用富行情 Webview 显示“代码/中文名称、价格、涨跌幅”，同时把布林参数上移到因子目录。

**Architecture:** `watchlist.json` 分离标的分类 `symbolGroups` 与因子 `factors`；Python 对两者的标的并集获取行情，只对因子成员计算布林信号。VS Code 侧栏改为 `WebviewViewProvider`，扩展宿主继续负责配置编辑、原子保存、调度、弹窗和飞书投递，Webview 仅负责目录与行情渲染及发送受限操作消息。

**Tech Stack:** Python 3.12、Node.js 22、VS Code Extension API、原生 HTML/CSS/JavaScript、`unittest`、`node:test`；不新增第三方依赖。

## Global Constraints

- 在当前 `0.1-xql` 分支工作，不建立隔离 worktree。
- 版本保持 `0.3.0`；最终重新生成 `portfolio_viewer/relife-portfolio-0.3.0.vsix`。
- 不覆盖交易 CSV、`portfolio.json`、环境变量或用户未提交修改。
- 0.3.0 只支持预置 `bollinger_lower` 因子，不实现通用公式、自动交易或后台常驻服务。
- Webhook URL 与签名密钥仍只存入 VS Code SecretStorage。
- 标的行固定为：左侧代码主标题与中文显示名，右侧价格、涨跌幅；不显示价格变化值。
- 显示名使用 `note.trim() || name`；悬浮详情同时展示市场、代码、正式中文名称、备注和行情日期。
- 中国市场涨为红色、跌为绿色；颜色之外必须保留正负号，确保可访问性。

---

## 目标配置

```json
{
  "symbolGroups": [
    {
      "id": "broad-market",
      "label": "大盘",
      "items": [
        {
          "market": "cn",
          "symbol": "512100",
          "name": "中证1000ETF",
          "note": ""
        }
      ]
    }
  ],
  "factors": [
    {
      "id": "bollinger-lower",
      "type": "bollinger_lower",
      "label": "布林下轨",
      "enabled": true,
      "timeframes": ["daily", "weekly", "monthly"],
      "window": 20,
      "standardDeviations": 2,
      "items": [
        {
          "market": "us",
          "symbol": "GOOGL",
          "name": "谷歌-A",
          "note": "等待日线收回下轨"
        }
      ]
    }
  ]
}
```

规则：

- `symbolGroups` 内的 `market + symbol` 全局唯一。
- 同一因子内的 `market + symbol` 唯一；不同因子之间允许重复。
- 因子标的不要求存在于 `symbolGroups`，可直接输入任意 A 股或美股代码。
- `bollinger_lower` 的开关、周期、窗口、标准差倍数只存放在因子对象上，标的对象不再包含 `bollinger`。
- 旧 `groups` 配置迁移时去除每个标的的 `bollinger` 并生成一个布林下轨因子；若旧标的参数不一致，迁移必须报错且保留旧文件，禁止静默丢失规则。

---

### Task 1: 重构配置模型并提供安全迁移

**Files:**
- Modify: `portfolio_viewer/vscode/watchlist.cjs`
- Modify: `portfolio_viewer/data/watchlist.json`
- Modify: `portfolio_viewer/tests/vscode-watchlist.test.mjs`

**Interfaces:**
- Produces: `validateWatchlist(data) -> data`
- Produces: `migrateWatchlist(data) -> { data, migrated }`
- Produces: `addSymbolGroup`、`renameSymbolGroup`、`deleteSymbolGroup`
- Produces: `addCatalogItem`、`moveCatalogItem`、`removeCatalogItem`、`updateCatalogItem`
- Produces: `updateFactor`、`addFactorItem`、`removeFactorItem`、`updateFactorItem`
- Duplicate scope: 标的目录全局去重；因子仅在单个因子内去重。

- [ ] **Step 1: 写新结构、重复范围、备注与迁移失败测试**

```js
function item(market, symbol, name, note = "") {
  return { market, symbol, name, note };
}

function fixture() {
  return {
    symbolGroups: [{
      id: "broad-market",
      label: "大盘",
      items: [item("cn", "512100", "中证1000ETF")],
    }],
    factors: [
      {
        id: "bollinger-lower",
        type: "bollinger_lower",
        label: "布林下轨",
        enabled: true,
        timeframes: ["daily", "weekly", "monthly"],
        window: 20,
        standardDeviations: 2,
        items: [item("us", "GOOGL", "谷歌-A")],
      },
      {
        id: "bollinger-lower-2",
        type: "bollinger_lower",
        label: "布林下轨 2",
        enabled: false,
        timeframes: ["daily"],
        window: 30,
        standardDeviations: 2,
        items: [],
      },
    ],
  };
}

test("validates catalog and factor duplicate scopes", () => {
  const data = fixture();
  assert.equal(validateWatchlist(data), data);
  assert.throws(
    () => addCatalogItem(data, "broad-market", data.symbolGroups[0].items[0]),
    /重复标的 cn:512100/,
  );
  assert.doesNotThrow(() => addFactorItem(data, "bollinger-lower-2", {
    market: "us", symbol: "GOOGL", name: "谷歌-A", note: "",
  }));
});

test("rejects inconsistent legacy factor parameters without overwriting", () => {
  const legacy = {
    groups: [
      { id: "broad", label: "大盘", items: [{
        ...item("us", "QQQ", "纳斯达克100ETF"),
        bollinger: { enabled: true, timeframes: ["daily"], window: 20, standardDeviations: 2 },
      }] },
      { id: "stock", label: "个股", items: [{
        ...item("us", "GOOGL", "谷歌-A"),
        bollinger: { enabled: true, timeframes: ["daily"], window: 30, standardDeviations: 2 },
      }] },
    ],
  };
  assert.throws(() => migrateWatchlist(legacy), /旧配置包含不同的布林参数/);
});
```

- [ ] **Step 2: 运行测试并确认旧实现失败**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/node --test tests/vscode-watchlist.test.mjs`

Expected: FAIL，提示缺少 `symbolGroups`、`factors` 或迁移函数。

- [ ] **Step 3: 用最小实现替换旧的 `groups + item.bollinger` 模型**

```js
function validateWatchlist(data) {
  if (!data || !Array.isArray(data.symbolGroups) || !Array.isArray(data.factors)) {
    throw new Error("观察配置缺少 symbolGroups 或 factors 数组");
  }
  validateSymbolGroups(data.symbolGroups);
  validateFactors(data.factors);
  return data;
}

function displayName(item) {
  return item.note?.trim() || item.name;
}

function stripLegacyRule({ bollinger, note = "", ...item }) {
  void bollinger;
  return { ...item, note };
}

function legacyRuleKey(rule) {
  return JSON.stringify([
    rule.enabled,
    rule.timeframes,
    rule.window,
    rule.standardDeviations,
  ]);
}

function assertEquivalentLegacyRules(rules) {
  if (new Set(rules.map(legacyRuleKey)).size > 1) {
    throw new Error("旧配置包含不同的布林参数，请先统一后再迁移");
  }
}
```

验证必须覆盖：空 id/label/name/symbol、非法市场、重复目录 id、非法因子类型、空周期、`window < 2`、非正标准差倍数以及非字符串备注。`saveWatchlist` 保留现有同目录临时文件加原子 `rename` 行为。

- [ ] **Step 4: 实现确定性的旧配置迁移**

```js
function migrateWatchlist(data) {
  if (!Array.isArray(data?.groups)) return { data: validateWatchlist(data), migrated: false };
  const rules = data.groups.flatMap((group) => group.items.map((item) => item.bollinger));
  assertEquivalentLegacyRules(rules);
  const first = rules[0] ?? {
    enabled: true,
    timeframes: ["daily", "weekly", "monthly"],
    window: 20,
    standardDeviations: 2,
  };
  const symbolGroups = data.groups.map((group) => ({
    id: group.id,
    label: group.label,
    items: group.items.map(stripLegacyRule),
  }));
  return {
    migrated: true,
    data: validateWatchlist({
      symbolGroups,
      factors: [{
        id: "bollinger-lower",
        type: "bollinger_lower",
        label: "布林下轨",
        ...first,
        items: symbolGroups.flatMap((group) => group.items.map((item) => ({ ...item }))),
      }],
    }),
  };
}
```

- [ ] **Step 5: 更新仓库初始配置**

将现金流、大盘、股息、个股、杠杆、比特币六组写入 `symbolGroups`；将其标的复制到唯一预置因子 `bollinger-lower.items`。保留 `BOXX`、`159915` 等现有代码，并把正式名称统一为中文，例如 `GOOGL` 使用 `谷歌-A`、`512100` 使用 `中证1000ETF`。

- [ ] **Step 6: 运行 Node 配置测试**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/node --test tests/vscode-watchlist.test.mjs`

Expected: PASS。

- [ ] **Step 7: 提交本任务**

```bash
git add portfolio_viewer/vscode/watchlist.cjs portfolio_viewer/data/watchlist.json portfolio_viewer/tests/vscode-watchlist.test.mjs
git commit -m "refactor: split symbols and factors"
```

---

### Task 2: 行情快照与因子级布林检查

**Files:**
- Modify: `portfolio_viewer/portfolio_dashboard/watchlist.py`
- Modify: `portfolio_viewer/scripts/check_watchlist.py`
- Modify: `portfolio_viewer/tests/test_watchlist.py`

**Interfaces:**
- Produces: `collect_symbols(config) -> list[tuple[str, str]]`
- Produces: `latest_quote(closes) -> { price, changePercent, dataDate }`
- Produces: `check_watchlist(...) -> { checkedAt, quotes, results, alerts, errors }`
- `results` 与 `alerts` 使用 `factorId`、`factorLabel`，不再使用 `groupId`、`groupLabel`。

- [ ] **Step 1: 写标的并集、行情百分比和因子参数测试**

```python
def test_collects_catalog_and_factor_only_symbols_once(self) -> None:
    config = {
        "symbolGroups": [{
            "id": "broad-market", "label": "大盘",
            "items": [{"market": "cn", "symbol": "512100", "name": "中证1000ETF", "note": ""}],
        }],
        "factors": [{
            "id": "bollinger-lower", "type": "bollinger_lower", "label": "布林下轨",
            "enabled": True, "timeframes": ["daily"], "window": 20,
            "standardDeviations": 2,
            "items": [{"market": "us", "symbol": "GOOGL", "name": "谷歌-A", "note": ""}],
        }],
    }
    config["factors"][0]["items"].append(
        {"market": "us", "symbol": "MSFT", "name": "微软", "note": ""}
    )
    self.assertEqual(
        [("cn", "512100"), ("us", "GOOGL"), ("us", "MSFT")],
        collect_symbols(config),
    )

def test_latest_quote_returns_price_percent_and_date(self) -> None:
    quote = latest_quote({
        date(2026, 7, 30): Decimal("100"),
        date(2026, 7, 31): Decimal("102"),
    })
    self.assertEqual(Decimal("102"), quote["price"])
    self.assertEqual(Decimal("2"), quote["changePercent"])
    self.assertEqual(date(2026, 7, 31), quote["dataDate"])
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `/home/clannad/miniforge3/envs/istorm_rag_gpu/bin/python -m unittest portfolio_viewer.tests.test_watchlist`

Expected: FAIL，提示缺少 `collect_symbols`、`latest_quote` 或仍读取 `groups`。

- [ ] **Step 3: 先按市场和代码构造唯一行情集合**

`check_watchlist` 对标的目录和因子目录的并集获取约三年日线。同一 `market + symbol` 只调用一次行情源，获取到的日线同时用于最新价、涨跌幅和因子计算；单个标的失败只增加一条错误，其他标的继续。

```python
def latest_quote(closes: Mapping[date, Decimal]) -> dict[str, Any]:
    ordered = sorted(closes.items())
    if len(ordered) < 2:
        raise ValueError("少于两个收盘价，无法计算涨跌幅")
    (previous_day, previous), (data_day, price) = ordered[-2:]
    del previous_day
    if previous == 0:
        raise ValueError("前收盘价为 0，无法计算涨跌幅")
    return {
        "price": price,
        "changePercent": (price - previous) / previous * Decimal("100"),
        "dataDate": data_day,
    }
```

- [ ] **Step 4: 把规则读取移动到因子层**

```python
for factor in config["factors"]:
    if not factor["enabled"]:
        continue
    for item in factor["items"]:
        for timeframe in factor["timeframes"]:
            evaluated = evaluate_bollinger(
                aggregate_closes(histories[(item["market"], item["symbol"])], timeframe),
                window=factor["window"],
                standard_deviations=Decimal(str(factor["standardDeviations"])),
            )
```

每条结果必须包含 `factorId`、`factorLabel`、`market`、`symbol`、`name`、`note`、周期与规则参数。`quotes` 每项只包含 `market`、`symbol`、`price`、`changePercent`、`dataDate`，不输出绝对价格变化。

- [ ] **Step 5: 保持脚本顶层错误 JSON 结构一致**

`portfolio_viewer/scripts/check_watchlist.py` 的异常回退增加空 `quotes`：

```python
payload = {
    "checkedAt": now.isoformat(),
    "quotes": [],
    "results": [],
    "alerts": [],
    "errors": [{"symbol": None, "timeframe": None, "message": str(exc)}],
}
```

- [ ] **Step 6: 运行 Python 全量测试**

Run: `/home/clannad/miniforge3/envs/istorm_rag_gpu/bin/python -m unittest discover -s portfolio_viewer/tests`

Expected: PASS，且旧日/周/月聚合和单标的失败隔离测试继续通过。

- [ ] **Step 7: 提交本任务**

```bash
git add portfolio_viewer/portfolio_dashboard/watchlist.py portfolio_viewer/scripts/check_watchlist.py portfolio_viewer/tests/test_watchlist.py
git commit -m "feat: return quotes for factor symbols"
```

---

### Task 3: 将编辑能力改为目录与因子操作

**Files:**
- Modify: `portfolio_viewer/vscode/watchlist-editor.cjs`
- Modify: `portfolio_viewer/tests/vscode-watchlist-editor.test.mjs`

**Interfaces:**
- Consumes: Task 1 的目录和标的操作函数。
- Produces: `registerWatchlistCommands(vscode, { getWatchlist, persist, refreshQuotes })`
- Webview 命令参数统一为普通对象，例如 `{ kind: "factor", id: "bollinger-lower" }` 或 `{ kind: "factorItem", factorId, market, symbol }`。

- [ ] **Step 1: 写目录、因子标的、备注与因子参数命令测试**

覆盖以下命令及取消输入不保存：

```text
relifePortfolio.symbolGroup.add
relifePortfolio.symbolGroup.rename
relifePortfolio.symbolGroup.delete
relifePortfolio.symbol.add
relifePortfolio.symbol.move
relifePortfolio.symbol.remove
relifePortfolio.symbol.editNote
relifePortfolio.factor.configure
relifePortfolio.factor.toggle
relifePortfolio.factor.addItem
relifePortfolio.factor.removeItem
relifePortfolio.factor.editNote
```

因子添加标的测试必须直接输入不在 `symbolGroups` 中的 `{ market: "us", symbol: "MSFT", name: "微软" }` 并成功保存。

- [ ] **Step 2: 运行编辑器测试并确认失败**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/node --test tests/vscode-watchlist-editor.test.mjs`

Expected: FAIL，旧命令仍绑定标的级 `bollinger`。

- [ ] **Step 3: 复用 VS Code 原生输入控件实现新命令**

标的添加流程固定为：选择市场 → 输入代码 → 输入中文名称 → 输入可选备注。代码标准化为大写；备注取消时中止，空字符串时保存为 `""`。

因子配置流程固定为：多选日/周/月 → 输入窗口 → 输入标准差倍数；更新因子对象本身。因子标的仅编辑 `market`、`symbol`、`name`、`note`。

- [ ] **Step 4: 保存成功后触发行情刷新**

`persist(next)` 完成原子保存与侧栏刷新；增加或删除标的后调用 `refreshQuotes()`，仅编辑备注或目录名称时不请求行情。

- [ ] **Step 5: 运行编辑器测试**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/node --test tests/vscode-watchlist-editor.test.mjs`

Expected: PASS。

- [ ] **Step 6: 提交本任务**

```bash
git add portfolio_viewer/vscode/watchlist-editor.cjs portfolio_viewer/tests/vscode-watchlist-editor.test.mjs
git commit -m "feat: edit symbol and factor directories"
```

---

### Task 4: 用富行情 Webview 实现 HOME、标的、因子侧栏

**Files:**
- Rewrite: `portfolio_viewer/vscode/sidebar.cjs`
- Create: `portfolio_viewer/vscode/sidebar-webview.js`
- Create: `portfolio_viewer/vscode/sidebar-webview.css`
- Rewrite: `portfolio_viewer/tests/vscode-sidebar.test.mjs`

**Interfaces:**
- Produces: `createPortfolioSidebarProvider(vscode, context, options)`，实现 `resolveWebviewView(view)`、`refresh(config, quotes)`。
- Produces: `buildSidebarModel(config, quotes)`，输出 HOME、标的、因子三棵可序列化目录树。
- Webview -> host: `{ type: "command", command, target }`、`{ type: "refresh" }`、`{ type: "openPortfolio" }`。
- Host -> Webview: `{ type: "state", data: model }`、`{ type: "refreshing", value: boolean }`。

- [ ] **Step 1: 写根目录、显示名和行情模型测试**

```js
test("builds HOME symbols and factors with code-first quotes", () => {
  const model = buildSidebarModel(config, [{
    market: "us", symbol: "GOOGL", price: 353.58,
    changePercent: 1.26, dataDate: "2026-07-31",
  }]);
  assert.deepEqual(model.roots.map((root) => root.label), ["HOME", "标的", "因子"]);
  assert.equal(model.roots[0].children[0].label, "打开投资组合");
  const row = model.roots[2].children[0].children[0];
  assert.equal(row.code, "GOOGL");
  assert.equal(row.displayName, "等待日线收回下轨");
  assert.equal(row.price, 353.58);
  assert.equal(row.changePercent, 1.26);
  assert.equal("priceChange" in row, false);
});
```

- [ ] **Step 2: 运行侧栏测试并确认失败**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/node --test tests/vscode-sidebar.test.mjs`

Expected: FAIL，旧实现仍返回 `TreeItem`。

- [ ] **Step 3: 实现 WebviewViewProvider 与严格 CSP**

`sidebar.cjs` 生成仅允许扩展本地 CSS 与带 nonce 脚本的 HTML：

```html
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src WEBVIEW_CSP; script-src 'nonce-NONCE';">
<link rel="stylesheet" href="SIDEBAR_CSS">
<div id="app"></div>
<script nonce="NONCE" src="SIDEBAR_JS"></script>
```

消息处理必须使用命令白名单，拒绝 Webview 提交的任意命令字符串。允许列表仅包含 Task 3 的命令以及打开组合、立即刷新。

- [ ] **Step 4: 实现最终选定的行情布局**

每个标的行使用三列：

```text
名称                         价格       涨跌幅
GOOGL                       353.58      +1.26%
谷歌-A
```

左侧首行为 `symbol`，第二行为 `note || name`。价格用 `Intl.NumberFormat("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 3 })`；涨跌幅固定两位小数并保留正负号。没有行情时显示 `--`，悬浮提示显示市场、正式名称、备注和行情日期。

使用 `--vscode-charts-red` 与 `--vscode-charts-green` 显示涨跌；零涨跌使用普通前景色。窄侧栏不得横向滚动，代码与显示名可省略，价格和涨跌幅保持可读。

- [ ] **Step 5: 实现目录交互和编辑入口**

HOME 下只有“打开投资组合”。标的下展示六个自定义分组，因子下每个因子一个子目录。目录使用原生 `<button>` 控制展开；标的与目录提供可聚焦的 `⋯` 操作按钮，并在右键时打开同一操作菜单，避免只支持鼠标右键。

- [ ] **Step 6: 运行侧栏测试**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/node --test tests/vscode-sidebar.test.mjs`

Expected: PASS，并验证 CSP、三根目录、显示顺序、无 `priceChange` 字段和命令白名单。

- [ ] **Step 7: 提交本任务**

```bash
git add portfolio_viewer/vscode/sidebar.cjs portfolio_viewer/vscode/sidebar-webview.js portfolio_viewer/vscode/sidebar-webview.css portfolio_viewer/tests/vscode-sidebar.test.mjs
git commit -m "feat: add quote-rich portfolio sidebar"
```

---

### Task 5: 接入扩展生命周期、行情刷新与通知

**Files:**
- Modify: `portfolio_viewer/vscode/extension.cjs`
- Modify: `portfolio_viewer/vscode/alerts.cjs`
- Modify: `portfolio_viewer/tests/vscode-extension.test.mjs`
- Modify: `portfolio_viewer/tests/vscode-alerts.test.mjs`

**Interfaces:**
- Consumes: Task 2 检查器返回的 `quotes`。
- Consumes: Task 4 的 `createPortfolioSidebarProvider`。
- Preserves: 既有 `runUpdates() -> Promise.allSettled([refresh(), checkAlerts()])` 双任务隔离。

- [ ] **Step 1: 写 Webview 注册、迁移顺序和行情推送测试**

测试必须证明：

1. 使用 `registerWebviewViewProvider("relifePortfolio.actions", provider)`，不再注册 TreeDataProvider。
2. 激活时先加载并在需要时原子保存迁移后的配置，再运行首次检查。
3. `checkAlerts()` 将 `payload.quotes` 推送到侧栏，即使没有信号也更新行情。
4. 检查失败保留上一份行情，投资组合刷新失败不阻断策略检查。

- [ ] **Step 2: 运行扩展测试并确认失败**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/node --test tests/vscode-extension.test.mjs tests/vscode-alerts.test.mjs`

Expected: FAIL，旧实现仍注册原生树且通知使用 `groupLabel`。

- [ ] **Step 3: 串行完成配置启动，再恢复并行任务**

```js
async function initializeRepository() {
  const loaded = await loadWatchlist(repositoryRoot);
  watchlist = loaded.data;
  if (loaded.migrated) await saveWatchlist(watchlistPath(repositoryRoot), watchlist);
  sidebar.refresh(watchlist, quotes);
  const existingPortfolio = loadPortfolio(repositoryRoot)
    .then((data) => { portfolio = data; sendPortfolio(); })
    .catch((error) => output.appendLine(`读取现有组合数据失败：${error.message}`));
  await Promise.allSettled([existingPortfolio, runUpdates()]);
}
```

`loadWatchlist` 的返回类型在 Task 1 中同步调整为 `{ data, migrated }`。初始化失败写入输出并显示一次错误，不覆盖旧配置。

- [ ] **Step 4: 更新行情与通知文案**

`checkAlerts` 成功后保存内存 `quotes = payload.quotes ?? []` 并调用 `sidebar.refresh(watchlist, quotes)`。`formatAlert` 第二行改为 `factorLabel · 周期`；首行仍包含 `[策略触发]`、代码和正式名称，候选池免责声明不变。

- [ ] **Step 5: 保持调度与重试语义**

激活、手动“立即更新”和既有四个定时点继续运行组合刷新与因子检查。飞书失败在下一次检查重试；不重复 VS Code 弹窗；关闭 VS Code 期间不监控、不补发。

- [ ] **Step 6: 运行扩展与通知测试**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/node --test tests/vscode-extension.test.mjs tests/vscode-alerts.test.mjs`

Expected: PASS。

- [ ] **Step 7: 提交本任务**

```bash
git add portfolio_viewer/vscode/extension.cjs portfolio_viewer/vscode/alerts.cjs portfolio_viewer/tests/vscode-extension.test.mjs portfolio_viewer/tests/vscode-alerts.test.mjs
git commit -m "feat: connect quotes and factors to sidebar"
```

---

### Task 6: 更新扩展清单与使用文档

**Files:**
- Modify: `portfolio_viewer/package.json`
- Modify: `portfolio_viewer/package-lock.json`
- Modify: `portfolio_viewer/tests/vscode-package.test.mjs`
- Modify: `portfolio_viewer/README.md`

**Interfaces:**
- `relifePortfolio.actions` 的 view 声明增加 `"type": "webview"`。
- VSIX 包含 `vscode/sidebar-webview.js` 与 `vscode/sidebar-webview.css`。

- [ ] **Step 1: 先更新清单测试**

```js
assert.deepEqual(view, {
  id: "relifePortfolio.actions",
  name: "Relife Portfolio",
  type: "webview",
});
assert(packageJson.files.includes("vscode/sidebar-webview.js"));
assert(packageJson.files.includes("vscode/sidebar-webview.css"));
```

命令断言改为 Task 3 的新命令集合，并删除标的级 `configureItem`、`toggleItem` 断言。原生 `view/item/context` 菜单不再适用于 Webview，应从清单移除；编辑入口由 Webview 的 `⋯` 与右键菜单提供。

- [ ] **Step 2: 运行清单测试并确认失败**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/node --test tests/vscode-package.test.mjs`

Expected: FAIL，view 尚未声明为 Webview 或缺少新资源。

- [ ] **Step 3: 更新 `package.json`、锁文件和 README**

README 必须说明：

- HOME／标的／因子目录职责。
- 标的目录与因子目录的成员互相独立。
- 代码为主标题，显示名为 `备注或中文名称`，右侧只显示价格和涨跌幅。
- 行情日期可在悬浮详情查看，`--` 表示尚无可用行情。
- 布林参数位于因子目录，所有因子成员共享。
- Webview 操作入口、四个更新时间、Webhook 安全与失败排查。

- [ ] **Step 4: 运行清单测试**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/node --test tests/vscode-package.test.mjs`

Expected: PASS。

- [ ] **Step 5: 提交本任务**

```bash
git add portfolio_viewer/package.json portfolio_viewer/package-lock.json portfolio_viewer/tests/vscode-package.test.mjs portfolio_viewer/README.md
git commit -m "docs: describe portfolio directory sidebar"
```

---

### Task 7: 完整回归、构建、打包与隐私检查

**Files:**
- Regenerate: `portfolio_viewer/vscode/dist/webview.js`
- Regenerate: `portfolio_viewer/vscode/dist/webview.css`
- Regenerate: `portfolio_viewer/relife-portfolio-0.3.0.vsix`

**Interfaces:**
- Produces: 可安装的 `relife-portfolio-0.3.0.vsix`。

- [ ] **Step 1: 运行 Python 全量测试**

Run: `/home/clannad/miniforge3/envs/istorm_rag_gpu/bin/python -m unittest discover -s portfolio_viewer/tests`

Expected: PASS。

- [ ] **Step 2: 运行 VS Code 扩展全量测试**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/npm run vscode:test`

Expected: PASS。

- [ ] **Step 3: 构建投资组合主面板资源**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/npm run vscode:build`

Expected: `vscode/dist/webview.js` 与 `vscode/dist/webview.css` 生成成功；侧栏 Webview 使用独立原生资源，无需第二套构建工具。

- [ ] **Step 4: 运行打包清单测试**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/npm run vscode:package:test`

Expected: PASS。

- [ ] **Step 5: 重新生成 VSIX**

Run: `cd portfolio_viewer && /home/clannad/.nvm/versions/node/v22.19.0/bin/npm run vscode:package`

Expected: 生成 `relife-portfolio-0.3.0.vsix`。

- [ ] **Step 6: 检查 VSIX 文件列表与敏感内容**

Run: `cd portfolio_viewer && unzip -l relife-portfolio-0.3.0.vsix`

Expected: 包含侧栏 JS/CSS、扩展代码和 README；不包含 `portfolio_viewer/data/watchlist.json`、Webhook、签名密钥、交易 CSV、`public/data/portfolio.json` 或环境变量文件。

- [ ] **Step 7: 手工冒烟验证**

在 Extension Development Host 中确认：

1. 根目录顺序为 HOME、标的、因子。
2. “打开投资组合”只位于 HOME 下。
3. 标的行显示代码、备注或中文名称、价格、涨跌幅，不显示绝对价格变化。
4. 因子可添加不在标的目录中的 A 股/美股代码。
5. 修改因子参数后所有因子成员共同使用新参数。
6. 手动刷新、定时检查、VS Code 弹窗和飞书重试正常。

- [ ] **Step 8: 提交生成物与最终变更**

```bash
git add portfolio_viewer/vscode/dist portfolio_viewer/relife-portfolio-0.3.0.vsix
git commit -m "build: package relife portfolio 0.3.0"
```

---

## 完成标准

- 配置中不存在标的级 `bollinger` 字段。
- 侧栏采用已确认的第二种富行情 Webview 布局。
- 行情列顺序为价格、涨跌幅，不显示价格变化。
- 所有自动化测试、构建、清单测试、VSIX 打包和隐私检查通过。
- 不修改或提交任务范围外的交易数据与用户已有变更。
