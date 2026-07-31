import assert from "node:assert/strict";
import { readdir } from "node:fs/promises";
import test from "node:test";

test("webview build contains only bundled code and styles", async () => {
  assert.deepEqual((await readdir("vscode/dist")).sort(), [
    "webview.css",
    "webview.js",
  ]);
});
