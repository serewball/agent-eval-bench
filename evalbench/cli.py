"""命令行入口：python -m evalbench run / demo / calibrate / judge"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .agents import build_agent
from .calibration import load_gold, run_calibration, write_calibration_report
from .judge import evaluate
from .judge_llm import build_judge
from .metrics import failure_cases, summarize
from .rejudge import load_runs, rejudge_runs, write_rejudge_report
from .report import write_html, write_markdown
from .runner import run_task
from .schema import load_state, load_tasks
from .tools import build_tools


def _run(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    base_state = load_state(args.state)
    tools = build_tools()
    tool_specs = [tool.spec() for tool in tools.values()]

    agent = build_agent(
        args.agent,
        tool_specs,
        error_rate=args.error_rate,
        seed=args.seed,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )

    results: Dict[str, List[Any]] = {}
    for task in tasks:
        runs = []
        for run_idx in range(args.samples):
            result = run_task(task, agent, tools, base_state, run_idx, max_steps=args.max_steps)
            runs.append(evaluate(task, result))
        results[task.id] = runs

    summary = summarize(tasks, results, k=args.k, price_in=args.price_in, price_out=args.price_out)
    meta = {
        "tasks_path": args.tasks,
        "agent": args.agent,
        "model": args.model or ("mock" if args.agent == "mock" else "env"),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = write_markdown(out_dir / "report.md", summary, meta, failure_cases(results))
    html_path = write_html(out_dir / "report.html", summary, meta)
    raw_path = out_dir / "runs.json"
    raw_path.write_text(
        json.dumps(
            {task_id: [r.to_dict() for r in runs] for task_id, runs in results.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    k = summary.k
    print("=" * 52)
    print(" agent-eval-bench 评测完成")
    print("=" * 52)
    print(f" 任务数        : {summary.tasks}")
    print(f" 采样次数      : 每条 {summary.samples_per_task} 次（共 {summary.total_runs} 次运行）")
    print(f" Pass@{k}        : {summary.pass_at_k * 100:5.1f}%   能力上限")
    print(f" Pass^{k}        : {summary.pass_pow_k * 100:5.1f}%   业务可靠性")
    print(f" 可靠性缺口    : {summary.pass_at_k * 100 - summary.pass_pow_k * 100:5.1f} pt")
    print(f" 工具错误率    : {summary.tool_error_rate * 100:5.1f}%")
    print(f" 确认合规率    : {summary.confirmation_rate * 100:5.1f}%")
    print(f" P95 延迟      : {summary.p95_latency_ms:,.1f} ms")
    print(f" 估算成本      : ¥{summary.cost:.4f}")
    print("-" * 52)
    print(f" 报告: {md_path}")
    print(f" 看板: {html_path}")
    print(f" 原始: {raw_path}")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    tasks = {task.id: task for task in load_tasks(args.tasks)}
    gold = load_gold(args.gold)
    judge = build_judge(
        args.judge,
        threshold=args.threshold,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )

    result = run_calibration(gold, tasks, judge, min_kappa=args.min_kappa)
    out_dir = Path(args.out)
    path = write_calibration_report(
        out_dir / "calibration.md",
        result,
        {"judge": args.judge, "model": args.model or ("mock" if args.judge == "mock" else "env"), "gold": args.gold},
        min_kappa=args.min_kappa,
    )

    print("=" * 52)
    print(" 判分器校准")
    print("=" * 52)
    print(f" 金标样本      : {result.items}")
    print(f" 简单一致率    : {result.agreement * 100:5.1f}%")
    print(f" Cohen's kappa : {result.kappa:5.3f}   （阈值 {args.min_kappa:.2f}）")
    print(f" 结论          : {'可用于放量' if result.usable else '不可用于放量'}")
    print("-" * 52)
    print(f" 报告: {path}")
    return 0


def _judge(args: argparse.Namespace) -> int:
    tasks = {task.id: task for task in load_tasks(args.tasks)}
    runs_map = load_runs(args.runs)
    judge = build_judge(
        args.judge,
        threshold=args.threshold,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )

    summary = rejudge_runs(tasks, runs_map, judge)
    out_dir = Path(args.out)
    path = write_rejudge_report(
        out_dir / "judge.md",
        summary,
        {"judge": args.judge, "model": args.model or ("mock" if args.judge == "mock" else "env"), "runs": args.runs},
    )

    print("=" * 52)
    print(" 判分一致性对照")
    print("=" * 52)
    print(f" 运行记录      : {summary.items}")
    print(f" 规则通过率    : {summary.rule_pass_rate * 100:5.1f}%")
    print(f" 判分通过率    : {summary.judge_pass_rate * 100:5.1f}%")
    print(f" 一致率        : {summary.agreement * 100:5.1f}%")
    print(f" Cohen's kappa : {summary.kappa:5.3f}")
    print("-" * 52)
    print(f" 报告: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalbench", description="Agent 评测台：Pass@k / Pass^k / 过程指标 / 失败归因"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "demo"):
        cmd = sub.add_parser(name, help="运行评测" + ("（mock 基线）" if name == "demo" else ""))
        cmd.add_argument("--tasks", default="tasks/supply-chain/tasks.json")
        cmd.add_argument("--state", default="tasks/supply-chain/orders.json")
        cmd.add_argument("--agent", default="mock", choices=["mock", "openai"])
        cmd.add_argument(
            "-n", "--samples", type=int, default=10, help="每条任务采样次数（必须 >= k）"
        )
        cmd.add_argument("-k", type=int, default=5, help="评估窗口 k")
        cmd.add_argument("--max-steps", type=int, default=8)
        cmd.add_argument("--error-rate", type=float, default=0.30, help="mock 注入失误的概率")
        cmd.add_argument("--seed", type=int, default=7)
        cmd.add_argument("--model", default=None)
        cmd.add_argument("--base-url", dest="base_url", default=None)
        cmd.add_argument("--api-key", dest="api_key", default=None)
        cmd.add_argument("--price-in", type=float, default=0.8, help="输入 token 单价（元/百万）")
        cmd.add_argument("--price-out", type=float, default=2.0, help="输出 token 单价（元/百万）")
        cmd.add_argument("--out", default="reports")

    calibrate = sub.add_parser("calibrate", help="用人工金标集校准判分器（Cohen's kappa）")
    calibrate.add_argument("--gold", default="tasks/supply-chain/gold_labels.json")
    calibrate.add_argument("--tasks", default="tasks/supply-chain/tasks.json")
    calibrate.add_argument("--judge", default="mock", choices=["mock", "llm"])
    calibrate.add_argument("--threshold", type=float, default=70.0, help="判为通过的分数阈值")
    calibrate.add_argument("--min-kappa", type=float, default=0.7, help="放量所需的最低 kappa")
    calibrate.add_argument("--model", default=None)
    calibrate.add_argument("--base-url", dest="base_url", default=None)
    calibrate.add_argument("--api-key", dest="api_key", default=None)
    calibrate.add_argument("--out", default="reports")

    judge = sub.add_parser("judge", help="对已有结果重跑语义判分，并与规则断言对照")
    judge.add_argument("--runs", default="reports/runs.json")
    judge.add_argument("--tasks", default="tasks/supply-chain/tasks.json")
    judge.add_argument("--judge", default="mock", choices=["mock", "llm"])
    judge.add_argument("--threshold", type=float, default=70.0)
    judge.add_argument("--model", default=None)
    judge.add_argument("--base-url", dest="base_url", default=None)
    judge.add_argument("--api-key", dest="api_key", default=None)
    judge.add_argument("--out", default="reports")

    return parser


def main(argv: List[str] | None = None) -> int:
    # 报告里包含中文：英文 locale 的 Windows（如 GitHub Actions runner）默认
    # stdout 编码是 cp1252，print 中文会 UnicodeEncodeError。这里强制 UTF-8，
    # 与报告文件的编码保持一致。
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # 已被重定向到非文本流时忽略
                pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in ("run", "demo"):
        if args.samples < args.k:
            parser.error("采样次数 n 必须大于等于评估窗口 k，否则 Pass@k 会饱和")
        return _run(args)
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "judge":
        return _judge(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
