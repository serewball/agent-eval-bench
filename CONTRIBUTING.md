# 贡献指南

感谢你的兴趣。这个项目刻意保持小，欢迎的贡献也按价值排序：

## 优先级最高的贡献

1. **新任务集**：带上你自己的领域数据（脱敏后）与断言，是让这个工具覆盖更多场景最直接的方式
2. **失败归因规则**：`evalbench/judge.py` 的 `attribute_failure` 目前是启发式优先级规则，能补一条普适性强的规则很有价值
3. **Agent 适配**：`evalbench/agents.py` 新增适配器（如 LangGraph / AutoGen 的桥接示例）
4. **文档修正**：README 与 `docs/interview-qa.md` 里的口径错误

## 不做的事

- Agent 框架、训练、RL——定位是**中立的评测工具**，不是选手
- 云服务 / 多租户 / 账号体系
- 真实企业数据（数据集必须虚构或脱敏）

## 开发约定

```bash
python -m unittest discover -s tests   # 全部单测，必须通过
python -m evalbench demo               # 端到端演示，必须可跑
```

- 零第三方依赖是硬约束：核心路径只用标准库；`pyyaml` 是可选扩展，缺失时必须降级给出明确提示（见 `schema.py` 的 `load_tasks`）
- 判分与归因逻辑必须可解释：每条断言失败要带 `detail`，归因要能定位到具体步骤
- 改判分口径（`judge.py` / `judge_llm.py`）后必须重跑 `python -m evalbench calibrate`，并更新 README 中的校准数字
- 指标口径改动必须同步更新 `docs/interview-qa.md` 对应条目

## 提交任务集

在 `tasks/<你的领域>/` 下放两个文件：

- `tasks.json`：任务定义，每条任务要有 `id`、`instruction`、`assertions`，写操作类任务请设置 `requires_confirmation` 或 `forbidden_tools`
- `state.json`（可选）：环境初始状态，与任务集中的 `state_equals` 路径对应

任务集里不要出现任何真实订单号、人名、公司名。

## 分支与提交

- 提交信息用中文或英文皆可，但必须说清楚「改了什么 + 为什么」
- 破坏性改动请在 PR 描述里写迁移说明
