"""报告输出：Markdown 与自包含 HTML。

设计原则：先结论、再证据、最后局限。
报告本身要能直接贴进周报或 README，而不是一堆原始数字。
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .metrics import Summary


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_markdown(
    path: Path,
    summary: Summary,
    meta: Dict[str, Any],
    cases: List[Dict[str, Any]],
) -> Path:
    k = summary.k
    gap = summary.pass_at_k - summary.pass_pow_k

    lines: List[str] = []
    lines.append("# Agent 评测报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 任务集：`{meta.get('tasks_path', '-')}`（{summary.tasks} 条任务）")
    lines.append(f"- Agent：`{meta.get('agent', '-')}`　模型：`{meta.get('model', 'mock')}`")
    lines.append(
        f"- 采样：每条任务 {summary.samples_per_task} 次，共 {summary.total_runs} 次运行；"
        f"按 k={k} 评估"
    )
    lines.append("")

    lines.append("## 一、核心结论")
    lines.append("")
    lines.append("| 指标 | 数值 | 含义 |")
    lines.append("|------|------|------|")
    lines.append(f"| **Pass@{k}** | **{_pct(summary.pass_at_k)}** | k 条轨迹里至少一条通过（能力上限） |")
    lines.append(f"| **Pass^{k}** | **{_pct(summary.pass_pow_k)}** | k 条轨迹全部通过（业务可靠性） |")
    lines.append(f"| 可靠性缺口 | {_pct(gap)} | 两者差值越大，越不适合无人值守上线 |")
    lines.append(f"| 任务触达率 | {_pct(summary.task_pass_rate)} | 至少成功过一次的任务占比 |")
    lines.append("")

    lines.append("## 二、可靠性缺口")
    lines.append("")
    lines.append(
        f"同一套 Agent，Pass@{k} 为 {_pct(summary.pass_at_k)}，"
        f"但 Pass^{k} 只有 {_pct(summary.pass_pow_k)}。"
        "这不是矛盾，而是两种口径在回答不同的问题：前者允许人工从多条结果里挑选，"
        "后者要求系统自己在每一次都做对。"
    )
    lines.append("")
    lines.append(
        f"> 注意：k 越接近采样次数，「至少成功一次」越容易成立，Pass@{k} 会自然抬高。"
        "所以拿 Pass@k 做上线决策是危险的——它衡量的是上限，不是稳定性。"
    )
    lines.append("")
    lines.append("> 判据：若动作可逆（查询、草稿、推荐），看 Pass@k；")
    lines.append("> 若动作不可逆（改期、取消、权限变更、对外发送），只能看 Pass^k。")
    lines.append("")

    lines.append("## 三、过程指标")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 平均步数 | {summary.avg_steps:.2f} |")
    lines.append(f"| 工具错误率 | {_pct(summary.tool_error_rate)} |")
    lines.append(f"| 越权调用率 | {_pct(summary.forbidden_rate)} |")
    lines.append(f"| 确认合规率 | {_pct(summary.confirmation_rate)} |")
    lines.append(f"| 平均 token | {summary.avg_tokens:,.0f} |")
    lines.append(f"| 估算成本 | ¥{summary.cost:.4f} |")
    lines.append(f"| P95 延迟 | {summary.p95_latency_ms:,.1f} ms |")
    lines.append("")

    lines.append("## 四、逐任务结果")
    lines.append("")
    lines.append(f"| 任务 | 难度 | 通过/采样 | Pass@{k} | Pass^{k} | 平均步数 |")
    lines.append("|------|------|-----------|--------|--------|----------|")
    for row in summary.rows:
        mark = "✅" if row.pass_pow_k >= 0.99 else ("⚠️" if row.passed else "❌")
        lines.append(
            f"| {mark} {row.task_id} {row.name} | {row.difficulty} | "
            f"{row.passed}/{row.runs} | {_pct(row.pass_at_k)} | {_pct(row.pass_pow_k)} | {row.avg_steps:.1f} |"
        )
    lines.append("")

    lines.append("## 五、失败归因（启发式）")
    lines.append("")
    if cases:
        lines.append("| 任务 | 采样序号 | 首个可疑步骤 | 结论 | 未通过的断言 |")
        lines.append("|------|----------|--------------|------|--------------|")
        for case in cases:
            failed = "；".join(case["failed_assertions"][:2]) or "-"
            lines.append(
                f"| {case['task_id']} | #{case['run_idx']} | 第 {case['step']} 步 | "
                f"{case['reason']} | {failed} |"
            )
    else:
        lines.append("本次没有失败样本。")
    lines.append("")

    lines.append("## 六、已知局限")
    lines.append("")
    lines.append("- Mock 模式的 token 数为字符估算，真实模型请以接口返回 usage 为准。")
    lines.append("- Mock 模式延迟只代表本地脚本开销，不代表真实模型响应时间。")
    lines.append("- 失败归因是规则启发式（禁止调用 > 首个写操作 > 首个动作），非轨迹模型推理。")
    lines.append("- 断言覆盖最终状态与关键工具调用，不评估中间推理质量。")
    lines.append("- 数据集为虚构供应链场景，仅用于演示方法论，不代表真实业务分布。")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_html(path: Path, summary: Summary, meta: Dict[str, Any]) -> Path:
    k = summary.k
    pass_k_width = max(0.0, min(100.0, summary.pass_at_k * 100))
    pass_pow_width = max(0.0, min(100.0, summary.pass_pow_k * 100))

    rows = []
    for row in summary.rows:
        badge = "ok" if row.pass_pow_k >= 0.99 else ("warn" if row.passed else "bad")
        rows.append(
            f"<tr><td><span class='dot {badge}'></span>{html.escape(row.task_id)} "
            f"{html.escape(row.name)}</td><td>{html.escape(row.difficulty)}</td>"
            f"<td>{row.passed}/{row.runs}</td><td>{row.pass_at_k * 100:.0f}%</td>"
            f"<td>{row.pass_pow_k * 100:.0f}%</td><td>{row.avg_steps:.1f}</td></tr>"
        )

    document = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent 评测报告 · agent-eval-bench</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; padding: 40px 24px; background: #FAFAF7; color: #2C2C2A; }}
  main {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 6px; }}
  .sub {{ color: #6B6A64; font-size: 13px; margin-bottom: 28px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }}
  .card {{ background: #fff; border: 1px solid #E4E2DA; border-radius: 10px; padding: 14px 16px; }}
  .card .label {{ font-size: 12px; color: #6B6A64; }}
  .card .value {{ font-size: 24px; font-weight: 600; margin-top: 6px; }}
  .bar-row {{ margin-bottom: 14px; }}
  .bar-label {{ font-size: 12px; color: #6B6A64; margin-bottom: 6px; }}
  .bar {{ background: #EFEDE6; border-radius: 6px; height: 26px; position: relative; }}
  .bar span {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px; }}
  .bar b {{ position: absolute; right: 10px; top: 4px; font-size: 12px; font-weight: 500; color: #2C2C2A; }}
  .blue {{ background: #7FA8DC; }} .amber {{ background: #E0A83F; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #E4E2DA;
           border-radius: 10px; overflow: hidden; font-size: 13px; }}
  th, td {{ text-align: left; padding: 9px 12px; border-bottom: 1px solid #F0EEE7; }}
  th {{ background: #F7F6F1; font-weight: 500; color: #55544F; }}
  .dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; }}
  .ok {{ background: #639922; }} .warn {{ background: #BA7517; }} .bad {{ background: #E24B4A; }}
  .note {{ margin-top: 26px; font-size: 12px; color: #6B6A64; line-height: 1.7; }}
</style></head>
<body><main>
  <h1>Agent 评测报告</h1>
  <div class="sub">{html.escape(meta.get('agent', '-'))} · 模型 {html.escape(str(meta.get('model', 'mock')))}
    · {summary.tasks} 条任务 × {summary.samples_per_task} 次采样 · 按 k={k} 评估</div>
  <div class="cards">
    <div class="card"><div class="label">Pass@{k}</div><div class="value">{summary.pass_at_k * 100:.1f}%</div></div>
    <div class="card"><div class="label">Pass^{k}</div><div class="value">{summary.pass_pow_k * 100:.1f}%</div></div>
    <div class="card"><div class="label">工具错误率</div><div class="value">{summary.tool_error_rate * 100:.1f}%</div></div>
    <div class="card"><div class="label">P95 延迟</div><div class="value">{summary.p95_latency_ms:,.1f}ms</div></div>
  </div>
  <div class="bar-row">
    <div class="bar-label">Pass@{k} · 能力上限（k 条里至少一条通过）</div>
    <div class="bar"><span class="blue" style="width:{pass_k_width:.1f}%"></span><b>{summary.pass_at_k * 100:.1f}%</b></div>
  </div>
  <div class="bar-row">
    <div class="bar-label">Pass^{k} · 业务可靠性（k 条全部通过）</div>
    <div class="bar"><span class="amber" style="width:{pass_pow_width:.1f}%"></span><b>{summary.pass_pow_k * 100:.1f}%</b></div>
  </div>
  <table>
    <thead><tr><th>任务</th><th>难度</th><th>通过/采样</th><th>Pass@{k}</th><th>Pass^{k}</th><th>平均步数</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <div class="note">成本 ¥{summary.cost:.4f} · 总 token {summary.total_tokens:,} · 确认合规率 {summary.confirmation_rate * 100:.1f}%<br>
  每个指标的口径与局限见同目录 Markdown 报告。</div>
</main></body></html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
