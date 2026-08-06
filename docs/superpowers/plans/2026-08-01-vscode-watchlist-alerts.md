# VS Code Watchlist Alerts 0.3.0 Implementation Plan

**Goal:** 增加可编辑观察分组、布林候选检查、VS Code/飞书通知并发布 0.3.0 VSIX。

**Architecture:** Python 负责行情与指标 JSON，VS Code 扩展负责原生侧栏、调度、去重和通知。所有配置在仓库 JSON，所有密钥在 SecretStorage。

## Tasks

- [x] 以测试锁定观察配置、周期聚合、布林计算和单标的失败隔离。
- [x] 实现仓库共享观察配置与 Python 单次检查器。
- [x] 以测试锁定原子保存、树节点及全部原生编辑命令。
- [x] 实现侧栏分组、标的移动和布林参数编辑。
- [x] 以测试锁定事件去重、飞书签名、失败重试和凭据命令。
- [x] 接入扩展激活、手动刷新和既有四个定时点。
- [x] 升级清单与文档到 0.3.0。
- [x] 运行 Python/Node 完整回归、构建、打包和 VSIX 隐私检查。

## Constraints

- 在当前 `*-xql` 分支工作，不建立隔离 worktree。
- 不新增依赖，不修改或覆盖历史交易产物。
- 候选提醒不生成 paper 订单，不复用交易策略通知表。
