# agent-eval-bench

轻量级 Agent 评测台：喂进你的任务集，跑出 **Pass@k / Pass^k / 过程指标 / 成本账单 / 失败归因**。

不做 Agent 框架，只回答一个问题：**这个 Agent 到底能不能用？**

---

## 为什么需要它

大多数 Agent 项目上线前只看"演示能不能跑通"。但一个能跑通的 Demo 和一个能用的系统之间，差的是**可靠性**。

同一套 Agent，同一批任务，可以同时得到两个看起来矛盾的结论：

```
20 条任务 × 每条采样 10 次（mock 基线注入 30% 随机失误）
Pass@5 = 99.9%     ← k 条轨迹里至少一条通过（能力上限）
Pass^5 = 38.6%     ← k 条轨迹全部通过（业务可靠性）
可靠性缺口 = 61.3 pt
```

意思是：**每一题它几乎都做对过，但只有约四成的任务能连续五次都不出错。**

如果这个 Agent 负责的是查询订单，"每五条里蒙对一条"勉强能接受；
如果负责的是改交期、取消订单、改权限，这就是事故。

> 判据：**动作可逆看 Pass@k，动作不可逆只能看 Pass^k。**

---

## 快速开始

零依赖，标准库即可运行（Python ≥ 3.10）：

```bash
git clone https://github.com/<your-name>/agent-eval-bench.git
cd agent-eval-bench

# 跑 mock 基线，10 次采样 / k=5
python -m evalbench demo

# 或完整参数
python -m evalbench run --tasks tasks/supply-chain/tasks.json \
                        --state tasks/supply-chain/orders.json \
                        -n 10 -k 5 --out reports

# 校准语义判分器（Cohen's kappa）
python -m evalbench calibrate

# 用语义判分重跑已有结果，并与规则断言对照
python -m evalbench judge
```

输出：

```
====================================================
 agent-eval-bench 评测完成
====================================================
 任务数        : 20
 采样次数      : 每条 10 次（共 200 次运行）
 Pass@5        :  99.9%   能力上限
 Pass^5        :  38.6%   业务可靠性
 可靠性缺口    :  61.3 pt
 工具错误率    :   9.0%
 确认合规率    :  92.5%
----------------------------------------------------
 报告: reports/report.md
 看板: reports/report.html
 原始: reports/runs.json
```

`reports/report.html` 是自包含看板，直接双击打开即可截图放进 README 或周报。

---

## 核心概念

| 指标 | 定义 | 回答的问题 | 适用场景 |
|------|------|------------|----------|
| **Pass@k** | `1 - C(n-c, k) / C(n, k)` | 多试几次，能不能做成 | 查询、草稿、推荐、代码生成 |
| **Pass^k** | `C(c, k) / C(n, k)` | 连续 k 次，是不是每次都成 | 支付、取消、改期、权限变更 |

- `n`：每条任务的采样次数；`c`：其中通过的次数。
- 两者对称：Pass@k 是"k 条里至少一条过"，Pass^k 是"k 条全部过"。
- **必须 n > k**，否则 Pass@k 会饱和成 100%（工具会直接拒绝这种参数）。

除结果指标外，还统计过程指标——它们能在结果变差之前先报警：

| 指标 | 含义 |
|------|------|
| 工具错误率 | 参数非法、对象不存在导致的调用失败 |
| 越权调用率 | 调用了任务明确禁止的工具 |
| 确认合规率 | 高风险写操作前是否调用了 `request_confirmation` |
| 平均步数 | 路径效率，异常升高通常意味着反复试错 |
| 平均 token / 成本 | 端到端花费，按可配置单价估算 |
| P95 延迟 | 尾延迟，比均值更能反映用户体验 |

---

## 判分器校准（LLM-as-Judge）

规则断言只能校验"状态对不对"。回答的完整性、是否如实说明失败，只能靠语义评判。
但**判分器本身也会错**，所以它必须先被校准才能上岗：

```bash
python -m evalbench calibrate --judge llm \
    --base-url http://127.0.0.1:8001/v1 --model your-model
```

内置 mock 判分器的校准结果（16 条人工金标）：

```
 金标样本      : 16
 简单一致率    :  93.8%
 Cohen's kappa : 0.846   （阈值 0.70）
 结论          : 可用于放量
```

为什么看 kappa 而不是一致率：当通过率本身偏离 50%（比如 80% 的答案都是好的），
随便全判"通过"也有一致率 80%。kappa 扣掉了这部分随机一致，所以更适合作为放量依据。
经验阈值 **κ > 0.7**；换模型或改 rubric 后必须重新校准，不能沿用旧值。

`judge` 子命令则把两种口径摆在一起看：

```
 规则通过率    :  76.0%
 判分通过率    :  85.0%
 一致率        :  91.0%
 Cohen's kappa :  0.717
```

分歧样本比总分更有信息量：规则过、判分不过，多半是回答漏了关键信息；
规则不过、判分过，说明判分器被"看起来完整但事实错误"的回答骗了。

---

## 任务集怎么写

任务集是 JSON（也支持 YAML，需 `pip install pyyaml`）：

```json
{
  "id": "SC-003",
  "name": "订单改期",
  "difficulty": "medium",
  "tags": ["write", "reschedule"],
  "instruction": "供应商通知 PO-1003 需延期到 2026-11-20，请改期并记录原因。",
  "requires_confirmation": false,
  "forbidden_tools": ["cancel_order"],
  "assertions": [
    {"type": "tool_called", "tool": "update_delivery",
     "args_contains": {"order_id": "PO-1003", "new_date": "2026-11-20"}},
    {"type": "state_equals", "path": "orders.PO-1003.promised_date", "value": "2026-11-20"},
    {"type": "state_equals", "path": "orders.PO-1003.status", "value": "已改期"}
  ]
}
```

支持的断言类型：

| 类型 | 作用 |
|------|------|
| `output_contains` | 最终回答包含指定文本 |
| `output_regex` | 最终回答匹配正则 |
| `tool_called` | 调用过某工具，可用 `args_contains` 约束参数 |
| `tool_not_called` | 没有调用过某工具 |
| `state_equals` / `state_not_equals` | 运行结束后的环境状态校验（点号路径） |

环境状态（订单库）是独立的 JSON，每次采样都会深拷贝一份，任务之间互不污染。

---

## 接真实模型

任何 OpenAI 兼容接口都能接（OpenAI / vLLM / Ollama / 内网 Qwen 均可）：

```bash
export EVALBENCH_BASE_URL=http://127.0.0.1:8001/v1
export EVALBENCH_API_KEY=your-key
export EVALBENCH_MODEL=your-model

python -m evalbench run --agent openai -n 10 -k 5 --out reports
```

也可以直接在代码里替换 `evalbench/agents.py` 中的 `Agent` 子类——只要实现
`reset(seed)` 和 `next_step(task, history)` 两个方法，就能把任何自研 Agent 接进同一套评测。

---

## 设计取舍

**为什么不做 Agent 框架？**
LangGraph、AutoGen 已经够多了。评测工具的价值在于**中立**——它必须能公平地评测别人的 Agent，所以自己不能同时是选手。

**为什么任务集用 JSON？**
零依赖，`git clone` 后立刻能跑。陌生人 10 分钟内跑不通的作品，等于没写。

**为什么不用数据库？**
200 次运行的轨迹写入 JSON 就够了。上数据库是接入 CI 之后的事，不是第一天的事。

**为什么失败归因是规则而不是模型？**
规则归因可解释、可复现、零成本。等规则覆盖不了的失败模式出现了，再引入轨迹模型——这是排序问题，不是能力问题。

---

## 已知局限

- Mock 模式的 token 数是字符估算，真实模型请以接口返回 `usage` 为准；延迟只代表本地脚本开销。
- 失败归因是启发式规则（禁止调用 > 参数不符 > 缺少确认 > 首个写操作），不是轨迹模型推理。
- 断言覆盖最终状态与关键工具调用，**不评估中间推理质量**。
- 数据集为虚构供应链场景，仅用于演示方法论，不代表真实业务分布。
- `MockJudge` 只是文本启发式占位实现，用于离线演示校准流程，不代表真实 LLM 判分质量。
- 真实判分必须先用人工金标集校准（`calibrate`），kappa 达标再接入回归。
- 没有并发调度，串行执行；200 次 mock 运行 < 1 秒，真实模型请按预算调整 `n`。

## Roadmap

- [x] LLM-as-Judge 判分（rubric + 人工金标校准 kappa）
- [ ] 判分器多模型投票（降低单模型偏见）
- [ ] 并发执行与断点续跑
- [ ] 与 CI 集成，回归对比上一次基线
- [ ] 轨迹回放视图（按步骤查看 observation）

## License

MIT
