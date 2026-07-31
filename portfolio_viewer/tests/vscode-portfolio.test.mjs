import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const {
  PYTHON,
  findRelifeRoot,
  loadPortfolio,
  refreshPortfolio,
  singleFlight,
} = require("../vscode/portfolio.cjs");

async function makeRepository() {
  const root = await mkdtemp(path.join("/tmp", "relife-vscode-"));
  await mkdir(path.join(root, "data", "transactions"), { recursive: true });
  await mkdir(path.join(root, "portfolio_viewer", "scripts"), { recursive: true });
  await mkdir(path.join(root, "portfolio_viewer", "public", "data"), {
    recursive: true,
  });
  await writeFile(
    path.join(
      root,
      "portfolio_viewer",
      "scripts",
      "generate_portfolio_dashboard.py",
    ),
    "",
  );
  await writeFile(
    path.join(root, "portfolio_viewer", "public", "data", "portfolio.json"),
    '{"generatedAt":"old"}',
  );
  return root;
}

test("finds only a workspace containing the Relife portfolio inputs", async () => {
  const invalid = await mkdtemp(path.join("/tmp", "not-relife-"));
  const root = await makeRepository();
  try {
    assert.equal(findRelifeRoot([invalid, root]), root);
    assert.equal(findRelifeRoot([invalid]), null);
  } finally {
    await rm(invalid, { recursive: true });
    await rm(root, { recursive: true });
  }
});

test("refreshes with the configured Python and then returns generated JSON", async () => {
  const root = await makeRepository();
  const calls = [];
  try {
    const data = await refreshPortfolio(root, async (file, args, options) => {
      calls.push({ file, args, options });
      await writeFile(
        path.join(
          root,
          "portfolio_viewer",
          "public",
          "data",
          "portfolio.json",
        ),
        '{"generatedAt":"new"}',
      );
    });

    assert.deepEqual(data, { generatedAt: "new" });
    assert.equal(calls[0].file, PYTHON);
    assert.deepEqual(calls[0].args, [
      path.join(
        root,
        "portfolio_viewer",
        "scripts",
        "generate_portfolio_dashboard.py",
      ),
    ]);
    assert.equal(calls[0].options.cwd, root);
  } finally {
    await rm(root, { recursive: true });
  }
});

test("keeps the last generated JSON when refresh fails", async () => {
  const root = await makeRepository();
  const output = path.join(
    root,
    "portfolio_viewer",
    "public",
    "data",
    "portfolio.json",
  );
  try {
    await assert.rejects(
      refreshPortfolio(root, async () => {
        throw new Error("provider unavailable");
      }),
      /provider unavailable/,
    );
    assert.deepEqual(await loadPortfolio(root), { generatedAt: "old" });
    assert.equal(JSON.parse(await readFile(output, "utf8")).generatedAt, "old");
  } finally {
    await rm(root, { recursive: true });
  }
});

test("combines overlapping refresh requests and permits the next run", async () => {
  let release;
  let runs = 0;
  const refresh = singleFlight(() => {
    runs += 1;
    if (runs > 1) return Promise.resolve();
    return new Promise((resolve) => {
      release = resolve;
    });
  });

  const first = refresh();
  const overlapping = refresh();
  assert.equal(overlapping, first);
  assert.equal(runs, 1);

  release();
  await first;
  await refresh();
  assert.equal(runs, 2);
});
