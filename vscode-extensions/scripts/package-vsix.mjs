import { existsSync, statSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const packageJson = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8"));
const outputName = `${packageJson.publisher}.${packageJson.name}-${packageJson.version}.vsix`;
const outputPath = path.join(root, outputName);
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const vsce = path.join(root, "node_modules", ".bin", process.platform === "win32" ? "vsce.cmd" : "vsce");
const nodeBin = path.dirname(process.execPath);
const env = { ...process.env, PATH: `${nodeBin}${path.delimiter}${process.env.PATH || ""}` };

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, env, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function requireFile(relativePath) {
  const file = path.join(root, relativePath);
  if (!existsSync(file) || statSync(file).size === 0) throw new Error(`缺少构建产物：${relativePath}`);
}

const nodeMajor = Number(process.versions.node.split(".")[0]);
if (nodeMajor < 22) throw new Error(`需要 Node.js 22+，当前为 ${process.versions.node}`);

console.log(`打包 ${outputName}`);
run(npm, ["test"]);
run(npm, ["run", "build"]);
requireFile("dist/webview.js");
requireFile("dist/webview.css");

const bundle = readFileSync(path.join(root, "dist/webview.js"), "utf8");
for (const marker of ["HOME", "投资组合", "布林带策略", "选择文件"]) {
  if (!bundle.includes(marker)) throw new Error(`构建产物未包含新 UI 标记：${marker}`);
}

run(vsce, ["package", "--out", outputName]);
requireFile(outputName);
console.log(`VSIX 已生成：${outputPath}`);
