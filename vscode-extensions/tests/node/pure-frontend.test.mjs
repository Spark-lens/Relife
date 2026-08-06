import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

test("宿主只读取示例快照，不启动 Python 数据引擎", () => {
  const extension = fs.readFileSync(new URL("../../extension.cjs", import.meta.url), "utf8");
  assert.match(extension, /loadInitialSnapshot/);
  assert.match(extension, /resources.*portfolio-snapshot\.json/);
  assert.doesNotMatch(extension, /child_process|spawn\(|detectAndApplyPythonEnv|runEngine|pythonPath/);
});

test("选择文件入口明确保留为后续数据接入入口", () => {
  const extension = fs.readFileSync(new URL("../../extension.cjs", import.meta.url), "utf8");
  assert.match(extension, /showOpenDialog/);
  assert.match(extension, /当前 UI 仍使用示例快照/);
});
