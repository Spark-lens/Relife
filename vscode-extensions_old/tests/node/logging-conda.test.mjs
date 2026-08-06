import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

test("conda 环境在激活时检测并自动回填，未命中则弹窗引导", () => {
  const manifest = JSON.parse(fs.readFileSync(new URL("../../package.json", import.meta.url), "utf8"));
  assert.match(
    manifest.contributes.configuration.properties["relife.pythonPath"].description,
    /自动检测 conda 环境/,
  );

  const extension = fs.readFileSync(new URL("../../extension.cjs", import.meta.url), "utf8");
  assert.match(extension, /async function detectAndApplyPythonEnv/);
  // 简单中文日志
  assert.match(extension, /检测到 conda 环境/);
  assert.match(extension, /已自动填写 relife\.pythonPath/);
  assert.match(extension, /未检测到可用的 conda 环境/);
  assert.match(extension, /已手动指定 Python 解释器/);
  assert.match(extension, /其他可用环境/);
  assert.match(extension, /showWarningMessage/);
  // 配置监听
  assert.match(extension, /onDidChangeConfiguration/);
  assert.match(extension, /affectsConfiguration\("relife\.pythonPath"\)/);
  assert.match(extension, /detectAndApplyPythonEnv\(\)\.catch/);
  // 配置变更日志用中文
  assert.match(extension, /配置变更: pythonPath/);
});

test("刷新日志使用中文文本行 + 复杂数据 JSON，环境信息不重复", () => {
  const extension = fs.readFileSync(new URL("../../extension.cjs", import.meta.url), "utf8");
  // 简单中文文本行
  assert.match(extension, /开始刷新/);
  assert.match(extension, /刷新完成/);
  assert.match(extension, /刷新失败/);
  assert.match(extension, /数据获取警告/);
  assert.match(extension, /数据获取错误明细/);
  // 引擎命令也用中文
  assert.match(extension, /校验源文件/);
  assert.match(extension, /构建快照/);
  assert.match(extension, /启动引擎/);
  assert.match(extension, /完成.*ms/);
  assert.match(extension, /失败.*ms/);
  // 时区使用 Asia/Shanghai
  assert.match(extension, /Asia\/Shanghai/);
  // ts() 不通过 toISOString 输出（避免转回 UTC）
  assert.doesNotMatch(extension, /slice\(0, 19\)/);
  // runEngine 不再打印 conda 环境（收敛）
  assert.doesNotMatch(extension, /发现 conda 环境/);
  // 错误透传
  assert.match(extension, /err\.type = response\.error\?\.type/);
  assert.match(extension, /err\.errors = response\.result\?\.errors/);
  // yfinance 噪音过滤
  assert.match(extension, /isYFinanceNoise/);
  assert.match(extension, /quoteSummary/);
  assert.match(extension, /possibly delisted/);
  // 缓存读取失败用中文
  assert.match(extension, /读取缓存失败，改用示例数据/);
});
