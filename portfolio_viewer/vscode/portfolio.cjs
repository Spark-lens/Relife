/* eslint-disable @typescript-eslint/no-require-imports */
const { execFile } = require("node:child_process");
const { existsSync } = require("node:fs");
const { readFile } = require("node:fs/promises");
const path = require("node:path");

const PYTHON = "/home/clannad/miniforge3/envs/istorm_rag_gpu/bin/python";
const GENERATOR = path.join(
  "portfolio_viewer",
  "scripts",
  "generate_portfolio_dashboard.py",
);
const OUTPUT = path.join(
  "portfolio_viewer",
  "public",
  "data",
  "portfolio.json",
);

function findRelifeRoot(folderPaths) {
  return (
    folderPaths.find(
      (root) =>
        existsSync(path.join(root, "data", "transactions")) &&
        existsSync(path.join(root, GENERATOR)) &&
        existsSync(path.join(root, OUTPUT)),
    ) ?? null
  );
}

async function loadPortfolio(repositoryRoot) {
  return JSON.parse(
    await readFile(path.join(repositoryRoot, OUTPUT), "utf8"),
  );
}

function runPython(file, args, options) {
  return new Promise((resolve, reject) => {
    execFile(file, args, options, (error, stdout, stderr) => {
      if (error) {
        error.stdout = stdout;
        error.stderr = stderr;
        reject(error);
      } else {
        resolve({ stdout, stderr });
      }
    });
  });
}

async function refreshPortfolio(repositoryRoot, run = runPython) {
  await run(PYTHON, [path.join(repositoryRoot, GENERATOR)], {
    cwd: repositoryRoot,
    maxBuffer: 10 * 1024 * 1024,
  });
  return loadPortfolio(repositoryRoot);
}

function singleFlight(operation) {
  let running = null;
  return (...args) => {
    if (!running) {
      try {
        running = Promise.resolve(operation(...args)).finally(() => {
          running = null;
        });
      } catch (error) {
        running = Promise.reject(error).finally(() => {
          running = null;
        });
      }
    }
    return running;
  };
}

module.exports = {
  PYTHON,
  findRelifeRoot,
  loadPortfolio,
  refreshPortfolio,
  singleFlight,
};
