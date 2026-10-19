# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式，版本号遵循语义化版本。

## [0.3.0] - 2026-10-19

### Changed

- `requires_confirmation` 判分口径收紧：从「轨迹中调用过 `request_confirmation`」改为「第一个写操作之前必须已调用」，杜绝先写后补确认的蒙混
- P95 延迟改用线性插值（与 numpy `method="linear"` 口径一致），替代最近邻取整
- 失败归因新增「达到最大步数仍未结束」分支

### Fixed

- 修复轨迹 history 消息顺序：`tool` 消息原排在 `assistant(tool_calls)` 消息之前，不符合 OpenAI 消息格式，接入真实模型会报 400

### Added

- CI：GitHub Actions 在 ubuntu / windows × Python 3.10 / 3.12 上跑单测 + demo + 校准 + 一致性对照
- 社区文件：CODE_OF_CONDUCT、CONTRIBUTING
- 看板截图 `docs/preview.png` 与 README 徽章

## [0.2.0] - 2026-10-19

### Added

- LLM-as-Judge 判分：rubric 三维打分（事实 0.5 / 完整 0.3 / 安全 0.2）+ 权重归一 + 阈值判定，JSON 输出容错解析
- 判分器校准：人工金标集 + Cohen's kappa，κ < 0.7 拒绝放量
- `calibrate` / `judge` 子命令：校准判分器、规则与语义判分一致性对照
- 内置 16 条人工金标集 `tasks/supply-chain/gold_labels.json`
- 抽出共用 `client.py`，Agent 与 Judge 复用同一套 OpenAI 兼容调用

### Changed

- 单测从 10 个增至 26 个

## [0.1.0] - 2026-10-19

### Added

- 任务集 schema（JSON / YAML）+ 六类断言（输出包含 / 正则 / 工具调用 / 禁止调用 / 状态校验）
- k 次采样 runner：完整轨迹、token、耗时、工具错误记录
- Pass@k / Pass^k 双口径指标 + 过程指标 + 成本估算
- 规则化失败归因（禁止调用 > 参数不符 > 缺少确认 > 首个写操作）
- Markdown 报告 + 自包含 HTML 看板
- 20 条虚构供应链任务集，mock 基线零依赖可跑
- 10 个指标单测
