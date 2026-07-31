function createPortfolioTreeDataProvider(vscode) {
  const item = new vscode.TreeItem(
    "打开投资组合",
    vscode.TreeItemCollapsibleState.None,
  );
  item.command = {
    command: "relifePortfolio.open",
    title: "打开投资组合",
  };
  return {
    getTreeItem: (treeItem) => treeItem,
    getChildren: () => [item],
  };
}

module.exports = { createPortfolioTreeDataProvider };
