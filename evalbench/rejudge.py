"""对已有评测结果重跑语义判分，并与规则断言做一致性对照。

用途：规则断言和 LLM Judge 是两种独立口径。二者在哪些样本上分歧，
往往就是「断言的覆盖盲区」或「判分器的偏见」——比单个总分更有信息量。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .calibration import cohen_kappa


@dataclass
class RejudgeRow:
    task_id: str
    run_idx: int
    rule_passed: bool
    judge_score: float
    judge_passed: bool
    agree: bool
    reason: str


@dataclass
class RejudgeSummary:
    items: int
    rule_pass_rate: float
    judge_pass_rate: float
    agreement: float
    kappa: float
    rows: List[RejudgeRow] = field(default_factory=list)


def rejudge_runs(
    tasks: Dict[str, Any], runs_map: Dict[str, List[Dict[str, Any]]], judge: Any
) -> RejudgeSummary:
    rows: List[RejudgeRow] = []
    pairs = []

    for task_id, runs in runs_map.items():
        task = tasks.get(task_id)
        if task is None:
            continue
        for run in runs:
            result = judge.score(task.instruction, run.get("final_output", ""), run.get("steps", []))
            judge_passed = bool(result.passed and not result.error)
            rule_passed = bool(run.get("passed"))
            pairs.append((1 if judge_passed else 0, 1 if rule_passed else 0))
            rows.append(
                RejudgeRow(
                    task_id=task_id,
                    run_idx=int(run.get("run_idx", 0)),
                    rule_passed=rule_passed,
                    judge_score=result.score,
                    judge_passed=judge_passed,
                    agree=judge_passed == rule_passed,
                    reason=result.reason or result.error,
                )
            )

    n = len(pairs) or 1
    return RejudgeSummary(
        items=len(pairs),
        rule_pass_rate=sum(1 for _, r in pairs if r) / n,
        judge_pass_rate=sum(1 for j, _ in pairs if j) / n,
        agreement=sum(1 for j, r in pairs if j == r) / n,
        kappa=cohen_kappa(pairs),
        rows=rows,
    )


def load_runs(path: str) -> Dict[str, List[Dict[str, Any]]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_rejudge_report(path: Path, summary: RejudgeSummary, meta: Dict[str, Any]) -> Path:
    lines: List[str] = []
    lines.append("# 判分一致性对照")
    lines.append("")
    lines.append(f"- 判分器：`{meta.get('judge', '-')}`　模型：`{meta.get('model', '-')}`")
    lines.append(f"- 结果文件：`{meta.get('runs', '-')}`（{summary.items} 条运行记录）")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 规则断言通过率 | {summary.rule_pass_rate * 100:.1f}% |")
    lines.append(f"| 语义判分通过率 | {summary.judge_pass_rate * 100:.1f}% |")
    lines.append(f"| 一致率 | {summary.agreement * 100:.1f}% |")
    lines.append(f"| Cohen's kappa | {summary.kappa:.3f} |")
    lines.append("")

    disagreements = [r for r in summary.rows if not r.agree]
    lines.append("## 分歧样本")
    lines.append("")
    if disagreements:
        lines.append("| 任务 | 采样 | 规则 | 判分 | 分数 | 判分理由 |")
        lines.append("|------|------|------|------|------|----------|")
        for row in disagreements[:15]:
            lines.append(
                f"| {row.task_id} | #{row.run_idx} | {'通过' if row.rule_passed else '不通过'} | "
                f"{'通过' if row.judge_passed else '不通过'} | {row.judge_score:.0f} | {row.reason or '-'} |"
            )
    else:
        lines.append("两种口径完全一致。")
    lines.append("")
    lines.append("## 怎么看分歧")
    lines.append("")
    lines.append("- 规则通过、判分不通过：回答可能漏了关键信息或语气不当，属于断言覆盖不到的部分。")
    lines.append("- 规则不通过、判分通过：判分器被表面完整但事实错误的回答骗过，需要补 rubric 或收紧阈值。")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
