# VS Code 左侧导航入口设计

## 目标

为 `Relife Portfolio` 0.2.0 增加符合 VS Code 使用习惯的左侧入口，同时保留现有适合宽屏展示的编辑区投资组合页面。

## 交互

- Activity Bar 显示一个由 `Re` 两个字符组成的自制单色 SVG 图标。
- 点击图标后打开 Relife 侧栏，侧栏只提供“打开投资组合”入口。
- 点击入口后，在编辑区打开或聚焦现有投资组合 Webview。
- 投资组合界面顶部提供“立即更新”按钮。
- 更新期间按钮禁用并显示更新状态；成功后刷新页面数据，失败时保留旧数据并显示错误提示。
- 命令面板中的“打开投资组合”和“立即更新数据”命令继续保留。

## 实现

- 在扩展清单中使用 VS Code 原生 `viewsContainers.activitybar` 和 `views` 声明左侧容器与入口。
- 扩展宿主注册轻量 Tree View 数据提供器；只有一个可点击入口，不引入额外 UI 依赖。
- 复用现有 `openPortfolio` 和 `singleFlight` 更新逻辑。
- Webview 通过消息请求更新；扩展宿主执行更新并把结果或错误回传给 Webview。
- 图标使用仓库内 SVG 资源，Marketplace 图标继续使用现有 PNG。

## 错误处理

- 非 Relife 工作区点击入口时显示现有错误提示。
- 数据更新失败时输出详细日志、显示 VS Code 错误通知，并通知 Webview 恢复按钮状态。
- 重复点击更新按钮仍合并为同一次更新，避免并发写入组合数据。

## 验证

- 清单测试验证 Activity Bar 容器、Tree View、图标和版本号。
- 扩展宿主测试验证侧栏入口调用打开命令。
- Webview 测试验证更新按钮与消息协议。
- 运行现有扩展测试、构建、VSIX 隐私检查和完整打包。

## 发布

- 版本升级为 `0.2.0`。
- 生成 `relife-portfolio-0.2.0.vsix`，代码提交后由 Publisher `clannad0710` 更新 Marketplace。
