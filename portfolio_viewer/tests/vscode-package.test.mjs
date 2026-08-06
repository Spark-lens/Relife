import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

test("declares the 0.4.0 watchlist and notification commands", async () => {
  const [packageJson, packageLock] = await Promise.all([
    readFile("package.json", "utf8").then(JSON.parse),
    readFile("package-lock.json", "utf8").then(JSON.parse),
  ]);
  const container = packageJson.contributes.viewsContainers.activitybar[0];
  const view = packageJson.contributes.views[container.id][0];

  assert.equal(packageJson.version, "0.4.0");
  assert.equal(packageLock.version, "0.4.0");
  assert.equal(packageLock.packages[""].version, "0.4.0");
  assert.deepEqual(container, {
    id: "relifePortfolio",
    title: "Relife Portfolio",
    icon: "media/activitybar.svg",
  });
  assert.deepEqual(view, {
    id: "relifePortfolio.actions",
    name: "Relife Portfolio",
  });
  assert(packageJson.files.includes("media/activitybar.svg"));
  assert(packageJson.files.includes("vscode/sidebar.cjs"));
  assert(packageJson.files.includes("vscode/watchlist.cjs"));
  assert(packageJson.files.includes("vscode/watchlist-editor.cjs"));
  assert(packageJson.files.includes("vscode/alerts.cjs"));
  const commands = new Set(packageJson.contributes.commands.map((item) => item.command));
  for (const command of [
    "relifePortfolio.configureFeishu",
    "relifePortfolio.testFeishu",
    "relifePortfolio.watchlist.addGroup",
    "relifePortfolio.watchlist.addItem",
    "relifePortfolio.watchlist.configureItem",
    "relifePortfolio.watchlist.deleteGroup",
    "relifePortfolio.watchlist.moveGroupDown",
    "relifePortfolio.watchlist.moveGroupUp",
    "relifePortfolio.watchlist.moveItem",
    "relifePortfolio.watchlist.removeItem",
    "relifePortfolio.watchlist.renameGroup",
    "relifePortfolio.watchlist.toggleItem",
  ]) assert(commands.has(command), `missing command ${command}`);
  assert(packageJson.contributes.menus["view/title"].length >= 2);
  assert(packageJson.contributes.menus["view/item/context"].length >= 7);
  await access(container.icon);
});

test("webview build contains only bundled code and styles", async () => {
  assert.deepEqual((await readdir("vscode/dist")).sort(), [
    "webview.css",
    "webview.js",
  ]);
  const webviewBundle = await readFile("vscode/dist/webview.js", "utf8");
  assert.match(webviewBundle, /立即更新/);
  assert.doesNotMatch(webviewBundle, /\bprocess\.env\.NODE_ENV\b/);
});
