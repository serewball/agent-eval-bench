"""指标聚合：Pass@k / Pass^k / 过程指标 / 成本。

为什么同时算这两个：
- Pass@k 回答"能不能做到"（k 条轨迹里挑一条最好的）
- Pass^k 回答"能不能每次都做到"（连续 k 次全部通过）
支付、取消、改权限这类不可逆动作，只能看后者。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Any, Dict, List

from .runner import RunResult
from .schema import Task


def pass_at_k(n: int, c: int, k: int) -> float:
    """无偏估计：1 - C(n-c, k) / C(n, k)。n 次采样中有 c 次通过。"""
    if n <= 0 or k <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_pow_k(n: int, c: int, k: int) -> float:
    """C(c, k) / C(n, k)：随机取 k 次样本全部通过的概率。

    与 Pass@k 对称：Pass@k 是"k 条里至少一条通过"，Pass^k 是"k 条全部通过"。
    当 n == k 时退化为"n 次全中"，即最严格的可靠性口径。
    """
    if n <= 0 or k <= 0 or c < k or n < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def percentile(values: List[float], p: float) -> float:
    """线性插值分位数（与 numpy 的 linear 方法口径一致）。

    注意：样本量很小时（如 n=10）P95 会非常接近最大值，这是分位数本身
    的数学性质，不是实现错误——报告里应同时看均值与 P95。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = lo + 1
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@dataclass
class TaskMetrics:
    task_id: str
    name: str
    difficulty: str
    runs: int
    passed: int
    pass_at_k: float
    pass_pow_k: float
    avg_steps: float


@dataclass
class Summary:
    tasks: int
    samples_per_task: int
    k: int
    total_runs: int
    pass_at_k: float
    pass_pow_k: float
    task_pass_rate: float
    avg_steps: float
    tool_error_rate: float
    forbidden_rate: float
    confirmation_rate: float
    total_tokens: int
    avg_tokens: float
    cost: float
    p95_latency_ms: float
    rows: List[TaskMetrics] = field(default_factory=list)


def summarize(
    tasks: List[Task],
    results: Dict[str, List[RunResult]],
    k: int = 5,
    price_in: float = 0.8,
    price_out: float = 2.0,
) -> Summary:
    rows: List[TaskMetrics] = []
    total_runs = 0
    total_steps = 0
    total_tool_errors = 0
    total_forbidden = 0
    total_tokens = 0
    latencies: List[float] = []
    confirm_total = 0
    confirm_ok = 0

    for task in tasks:
        runs = results.get(task.id, [])
        if not runs:
            continue
        n = len(runs)
        c = sum(1 for r in runs if r.passed)
        steps = sum(len(r.steps) for r in runs)

        rows.append(
            TaskMetrics(
                task_id=task.id,
                name=task.name or task.id,
                difficulty=task.difficulty,
                runs=n,
                passed=c,
                pass_at_k=pass_at_k(n, c, min(k, n)),
                pass_pow_k=pass_pow_k(n, c, min(k, n)),
                avg_steps=steps / n,
            )
        )

        total_runs += n
        total_steps += steps
        total_tool_errors += sum(r.tool_errors for r in runs)
        total_forbidden += sum(r.forbidden_calls for r in runs)
        total_tokens += sum(r.total_tokens for r in runs)
        latencies.extend(r.latency_ms for r in runs)

        if task.requires_confirmation:
            confirm_total += n
            confirm_ok += sum(
                1
                for r in runs
                if any(ch["type"] == "requires_confirmation" and ch["ok"] for ch in r.assertion_results)
            )

    task_count = len(rows) or 1
    pass_at_k_mean = sum(r.pass_at_k for r in rows) / task_count
    pass_pow_k_mean = sum(r.pass_pow_k for r in rows) / task_count
    task_pass_rate = sum(1 for r in rows if r.passed > 0) / task_count

    prompt_tokens = sum(
        r.prompt_tokens for runs in results.values() for r in runs
    )
    completion_tokens = sum(
        r.completion_tokens for runs in results.values() for r in runs
    )
    cost = prompt_tokens / 1_000_000 * price_in + completion_tokens / 1_000_000 * price_out

    return Summary(
        tasks=len(rows),
        samples_per_task=max((r.runs for r in rows), default=0),
        k=k,
        total_runs=total_runs,
        pass_at_k=pass_at_k_mean,
        pass_pow_k=pass_pow_k_mean,
        task_pass_rate=task_pass_rate,
        avg_steps=total_steps / total_runs if total_runs else 0.0,
        tool_error_rate=total_tool_errors / total_runs if total_runs else 0.0,
        forbidden_rate=total_forbidden / total_runs if total_runs else 0.0,
        confirmation_rate=(confirm_ok / confirm_total) if confirm_total else 1.0,
        total_tokens=total_tokens,
        avg_tokens=total_tokens / total_runs if total_runs else 0.0,
        cost=cost,
        p95_latency_ms=percentile(latencies, 95),
        rows=rows,
    )


def failure_cases(results: Dict[str, List[RunResult]], limit: int = 8) -> List[Dict[str, Any]]:
    """挑出失败样本，供报告里的失败归因段落使用。"""
    cases: List[Dict[str, Any]] = []

    # 先每个任务取一条，保证归因样本分散，而不是被同一个任务刷屏
    for task_id, runs in results.items():
        for run in runs:
            if run.passed:
                continue
            failed = [c["detail"] for c in run.assertion_results if not c["ok"]]
            cases.append(
                {
                    "task_id": task_id,
                    "run_idx": run.run_idx,
                    "step": run.first_error_step,
                    "reason": run.error_reason,
                    "failed_assertions": failed,
                }
            )
            break
        if len(cases) >= limit:
            return cases

    return cases
