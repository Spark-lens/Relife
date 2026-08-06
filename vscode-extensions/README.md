# Relife 0.0.1

Relife 是 TradingView 风格的固定黑色主题双市场投资组合 VS Code 插件。当前版本专注于 UI 初始界面，首次启动直接使用内置示例快照，运行时不依赖 Python 或网络。

## 环境要求

- VS Code >= 1.95
- Node.js >= 22.13（仅用于开发和打包）

## 目录

侧栏按 VS Code 原生风格保留三个一级目录：`HOME`、`标的`、`策略`。标的下固定展示 `现金流`、`大盘`、`股息`、`个股`、`杠杆`、`比特币` 六类；标的备注默认显示中文名称，修改后显示修改值。

## 打包

```bash
npm run package       # 测试 + 构建 + 打包
npm run package:quick # 仅构建 + 打包（跳过测试）
npm run package:local # 使用当前 Node 22：测试 + 构建 + 产物校验 + 打包
```

生成文件：`clannad0710.relife-0.0.1.vsix`。

## 预览、构建与验收约定

以后所有预览必须由真实源码构建；未完成源码构建、VSIX 内容校验和安装后界面验证，不允许打包。

预览以 `src/` 为唯一真源，使用 `resources/sample/portfolio-snapshot.json` 作为纯前端示例数据。独立 HTML 只用于早期讨论稿，不能作为最终交付依据。正式交付前依次完成：真实源码构建、Node 测试、VSIX 文件清单与压缩完整性校验，以及安装 VSIX 后的界面确认。

## 数据

`选择文件` 和 `刷新` 入口已经保留，用于确认最终交互位置；正式 CSV 解析、行情请求和数据快照接入放到后续版本。当前插件只读取 `resources/sample/portfolio-snapshot.json`。

## 隐私

VSIX 仅包含内置示例快照。真实交易文件、缓存、密钥和本机路径不会打包；插件运行时不启动 Python 数据引擎。
