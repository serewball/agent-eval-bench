"""判分器校准：用人工金标集衡量 LLM Judge 是否可信。

核心指标是 Cohen's kappa——它剔除了"瞎猜也能对上"的随机一致，
所以比简单一致率严格。工程上的经验阈值：kappa > 0.7 才可用于放量判分。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


def cohen_kappa(pairs: List[Tuple[int, int]]) -> float:
    """pairs: [(judge_label, human_label)]，标签取 0/1。"""
    if not pairs:
        return 0.0
    n = len(pairs)
    observed = sum(1 for j, h in pairs if j == h) / n
    judge_pos = sum(j for j, _ in pairs) / n
    human_pos = sum(h for _, h in pairs) / n
    expected = judge_pos * human_pos + (1 - judge_pos) * (1 - human_pos)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


@dataclass
class CalibrationRow:
    item_id: str
    task_id: str
    judge_score: float
    judge_label: int
    human_label: int
    agree: bool
    reason: str


@dataclass
class CalibrationResult:
    items: int
    agreement: float
    kappa: float
    judge_pass_rate: float
    human_pass_rate: float
    usable: bool
    rows: List[CalibrationRow] = field(default_factory=list)


def load_gold(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "items" not in data:
        raise ValueError("金标集缺少 items 字段")
    return data


def run_calibration(
    gold: Dict[str, Any],
    tasks: Dict[str, Any],
    judge: Any,
    min_kappa: float = 0.7,
) -> CalibrationResult:
    pairs: List[Tuple[int, int]] = []
    rows: List[CalibrationRow] = []

    for item in gold["items"]:
        task = tasks.get(item["task_id"])
        if task is None:
            continue
        result = judge.score(task.instruction, item.get("output", ""), item.get("steps", []))
        judge_label = 1 if (result.passed and not result.error) else 0
        human_label = int(item["human"])
        pairs.append((judge_label, human_label))
        rows.append(
            CalibrationRow(
                item_id=item["id"],
                task_id=item["task_id"],
                judge_score=result.score,
                judge_label=judge_label,
                human_label=human_label,
                agree=judge_label == human_label,
                reason=result.reason or result.error,
            )
        )

    n = len(pairs) or 1
    agreement = sum(1 for j, h in pairs if j == h) / n
    kappa = cohen_kappa(pairs)

    return CalibrationResult(
        items=len(pairs),
        agreement=agreement,
        kappa=kappa,
        judge_pass_rate=sum(j for j, _ in pairs) / n,
        human_pass_rate=sum(h for _, h in pairs) / n,
        usable=kappa >= min_kappa,
        rows=rows,
    )


def write_calibration_report(
    path: Path,
    result: CalibrationResult,
    meta: Dict[str, Any],
    min_kappa: float = 0.7,
) -> Path:
    lines: List[str] = []
    lines.append("# 判分器校准报告")
    lines.append("")
    lines.append(f"- 判分器：`{meta.get('judge', '-')}`　模型：`{meta.get('model', '-')}`")
    lines.append(f"- 金标集：`{meta.get('gold', '-')}`（{result.items} 条）")
    lines.append("")
    lines.append("## 一、结论")
    lines.append("")
    lines.append("| 指标 | 数值 | 说明 |")
    lines.append("|------|------|------|")
    lines.append(f"| 简单一致率 | {result.agreement * 100:.1f}% | 与人工标注完全一致的比例 |")
    lines.append(f"| **Cohen's kappa** | **{result.kappa:.3f}** | 剔除随机一致后的真实一致性 |")
    lines.append(f"| 判分通过率 | {result.judge_pass_rate * 100:.1f}% | 判分器判为通过的比例 |")
    lines.append(f"| 人工通过率 | {result.human_pass_rate * 100:.1f}% | 金标集中人工判为通过的比例 |")
    verdict = "可用于放量" if result.usable else "不可用于放量，需调整 rubric 或换模型"
    lines.append(f"| 可用性（κ ≥ {min_kappa}） | **{verdict}** | 阈值可按业务风险调整 |")
    lines.append("")
    lines.append(
        "> kappa 而非一致率：当通过率本身偏离 50% 时，简单一致率会被随机猜中抬高。"
        "kappa 扣掉了这部分，因此更适合作为判分器能否上岗的依据。"
    )
    lines.append("")
    lines.append("## 二、逐条对照")
    lines.append("")
    lines.append("| 样本 | 任务 | 判分 | 判分结论 | 人工结论 | 是否一致 | 理由 |")
    lines.append("|------|------|------|----------|----------|----------|------|")
    for row in result.rows:
        lines.append(
            f"| {row.item_id} | {row.task_id} | {row.judge_score:.0f} | "
            f"{'通过' if row.judge_label else '不通过'} | "
            f"{'通过' if row.human_label else '不通过'} | "
            f"{'✅' if row.agree else '❌'} | {row.reason or '-'} |"
        )
    lines.append("")
    lines.append("## 三、怎么用这个结果")
    lines.append("")
    lines.append("- kappa 达标：可把判分器接进日常回归，替代部分人工抽检，但保留 5%–10% 人工复核。")
    lines.append("- kappa 不达标：先看分歧样本集中在哪个维度（事实 / 完整 / 安全），再改 rubric 描述。")
    lines.append("- 换模型或改 rubric 后必须重新校准，不能沿用旧的 kappa。")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
