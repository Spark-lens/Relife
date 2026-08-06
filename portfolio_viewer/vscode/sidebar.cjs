/* eslint-disable @typescript-eslint/no-require-imports */

// Tree item kinds
const KIND = {
  HOME: "home",
  HOME_PORTFOLIO: "home-portfolio",
  WATCHLIST: "watchlist",
  WATCHLIST_GROUP: "watchlist-group",
  WATCHLIST_SYMBOL: "watchlist-symbol",
};

function createPortfolioTreeDataProvider(vscode, initialWatchlist = { groups: [] }) {
  let watchlist = initialWatchlist;
  const emitter = new vscode.EventEmitter();
  let selectedSymbol = null;

  function homeItem() {
    const item = new vscode.TreeItem("HOME", vscode.TreeItemCollapsibleState.Collapsed);
    item._kind = KIND.HOME;
    item.contextValue = "relifeHome";
    return item;
  }

  function homePortfolioItem() {
    const item = new vscode.TreeItem("投资组合", vscode.TreeItemCollapsibleState.None);
    item.command = { command: "relifePortfolio.open", title: "投资组合" };
    item._kind = KIND.HOME_PORTFOLIO;
    item.contextValue = "relifeHomePortfolio";
    return item;
  }

  function watchlistItem() {
    const item = new vscode.TreeItem("观察列表", vscode.TreeItemCollapsibleState.Collapsed);
    item._kind = KIND.WATCHLIST;
    item.contextValue = "relifeWatchlistRoot";
    return item;
  }

  function groupItem(group) {
    const label = group.subtitle ? `${group.label} - ${group.subtitle}` : group.label;
    const item = new vscode.TreeItem(label, vscode.TreeItemCollapsibleState.Collapsed);
    item.contextValue = "relifeWatchlistGroup";
    item._kind = KIND.WATCHLIST_GROUP;
    item._group = group;
    return item;
  }

  function symbolItem(group, value) {
    const label = value.name ? `${value.symbol} ${value.name}` : value.symbol;
    const item = new vscode.TreeItem(label, vscode.TreeItemCollapsibleState.None);
    // No strategy info in description per user request
    item.tooltip = value.name || value.symbol;
    item.contextValue = value.bollinger.enabled
      ? "relifeWatchlistItemEnabled"
      : "relifeWatchlistItemDisabled";
    item._kind = KIND.WATCHLIST_SYMBOL;
    item._groupId = group.id;
    item._symbol = value;
    if (selectedSymbol &&
        selectedSymbol.market === value.market &&
        selectedSymbol.symbol === value.symbol) {
      item.iconPath = new vscode.ThemeIcon("circle-filled", new vscode.ThemeColor("charts.blue"));
    }
    return item;
  }

  return {
    onDidChangeTreeData: emitter.event,
    getTreeItem: (treeItem) => treeItem,
    getChildren(element) {
      // Top level: HOME + Watchlist
      if (!element) {
        return [homeItem(), watchlistItem()];
      }
      switch (element._kind) {
        case KIND.HOME:
          return [homePortfolioItem()];
        case KIND.WATCHLIST:
          return watchlist.groups.map(groupItem);
        case KIND.WATCHLIST_GROUP:
          return element._group.items.map((it) => symbolItem(element._group, it));
        default:
          return [];
      }
    },
    setSelected(market, symbol) {
      selectedSymbol = symbol ? { market, symbol } : null;
      emitter.fire();
    },
    refresh(nextWatchlist = watchlist) {
      watchlist = nextWatchlist;
      emitter.fire();
    },
  };
}

module.exports = { createPortfolioTreeDataProvider };
