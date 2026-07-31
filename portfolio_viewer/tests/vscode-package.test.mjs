import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

test("declares the 0.2.0 Activity Bar portfolio view", async () => {
  const [packageJson, packageLock] = await Promise.all([
    readFile("package.json", "utf8").then(JSON.parse),
    readFile("package-lock.json", "utf8").then(JSON.parse),
  ]);
  const container = packageJson.contributes.viewsContainers.activitybar[0];
  const view = packageJson.contributes.views[container.id][0];

  assert.equal(packageJson.version, "0.2.0");
  assert.equal(packageLock.version, "0.2.0");
  assert.equal(packageLock.packages[""].version, "0.2.0");
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
