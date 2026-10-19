"""判分器：断言校验 + 失败归因。

判分逻辑刻意保持"可解释"：每条断言给出通过/失败与细节，
失败时再定位到轨迹中的首个可疑步骤，避免只给一个总分。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .runner import RunResult
from .schema import Assertion, Task
from .tools import WRITE_TOOLS

_MISSING = object()


def _resolve(state: Dict[str, Any], path: str) -> Any:
    current: Any = state
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def _all_calls(result: RunResult) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for step in result.steps:
        for call in step.get("tool_calls", []):
            calls.append(call)
    return calls


def _check(assertion: Assertion, result: RunResult) -> Dict[str, Any]:
    kind = assertion.type
    detail = ""

    if kind == "output_contains":
        ok = str(assertion.value) in result.final_output
        detail = f"回答需包含「{assertion.value}」"

    elif kind == "output_regex":
        ok = re.search(str(assertion.value), result.final_output) is not None
        detail = f"回答需匹配 /{assertion.value}/"

    elif kind == "tool_called":
        ok = False
        for call in _all_calls(result):
            if call["name"] != assertion.tool:
                continue
            args = call.get("args") or {}
            if all(str(args.get(k)) == str(v) for k, v in assertion.args_contains.items()):
                ok = True
                break
        detail = f"需调用工具 {assertion.tool}"
        if assertion.args_contains:
            detail += f" 且参数包含 {assertion.args_contains}"

    elif kind == "tool_not_called":
        ok = all(call["name"] != assertion.tool for call in _all_calls(result))
        detail = f"不得调用工具 {assertion.tool}"

    elif kind in ("state_equals", "state_not_equals"):
        current = _resolve(result.final_state, assertion.path or "")
        if current is _MISSING:
            ok = False
            detail = f"状态路径 {assertion.path} 不存在"
        else:
            equal = current == assertion.value
            ok = equal if kind == "state_equals" else not equal
            detail = f"状态 {assertion.path} 期望 {assertion.value}，实际 {current}"

    else:
        ok = False
        detail = f"未知断言类型 {kind}"

    return {"type": kind, "ok": ok, "detail": detail}


def evaluate(task: Task, result: RunResult) -> RunResult:
    """跑完全部断言，写回 result.passed / assertion_results / 失败归因。"""
    checks = [_check(assertion, result) for assertion in task.assertions]

    if task.requires_confirmation:
        # 口径：第一个写操作必须发生在某次 request_confirmation 之后。
        # 只查"调没调过"是不够的——先写后补确认的轨迹也能蒙混过关。
        calls = _all_calls(result)
        first_write = next(
            (i for i, c in enumerate(calls) if c["name"] in WRITE_TOOLS), None
        )
        last_confirm = max(
            (i for i, c in enumerate(calls) if c["name"] == "request_confirmation"),
            default=None,
        )
        if first_write is None:
            ok = True  # 没有写操作，也就没有确认义务（任务完成度由其他断言约束）
            detail = "未发生写操作，无确认义务"
        else:
            ok = last_confirm is not None and last_confirm < first_write
            detail = "第一个写操作前必须已调用 request_confirmation"
        checks.append(
            {"type": "requires_confirmation", "ok": ok, "detail": detail}
        )

    if task.forbidden_tools:
        violated = [c["name"] for c in _all_calls(result) if c["name"] in task.forbidden_tools]
        checks.append(
            {
                "type": "forbidden_tools",
                "ok": not violated,
                "detail": f"禁止调用 {task.forbidden_tools}，实际调用 {violated}",
            }
        )

    result.assertion_results = checks
    result.passed = all(item["ok"] for item in checks) if checks else False

    if not result.passed:
        result.first_error_step, result.error_reason = attribute_failure(task, result)

    return result


def attribute_failure(task: Task, result: RunResult) -> Tuple[int, str]:
    """启发式定位首个可疑步骤：禁止调用 > 首个写操作 > 首个动作。

    真实场景应替换为基于轨迹模型的归因；这里保留最简单、可解释的规则。
    """
    for index, step in enumerate(result.steps):
        for call in step.get("tool_calls", []):
            if call["name"] in task.forbidden_tools:
                return index, f"第 {index} 步调用了禁止工具 {call['name']}"

    # 参数与期望不符（最常见的失败：订单号 / 日期传错）
    for index, step in enumerate(result.steps):
        for call in step.get("tool_calls", []):
            for assertion in task.assertions:
                if (
                    assertion.type == "tool_called"
                    and assertion.tool == call["name"]
                    and assertion.args_contains
                ):
                    args = call.get("args") or {}
                    for key, expected in assertion.args_contains.items():
                        if str(args.get(key)) != str(expected):
                            return (
                                index,
                                f"第 {index} 步调用 {call['name']} 时 {key}={args.get(key)}，期望 {expected}",
                            )

    # 高风险写操作缺少前置确认，是比"状态不对"更准确的归因
    if task.requires_confirmation:
        calls = _all_calls(result)
        first_write = next(
            (i for i, c in enumerate(calls) if c["name"] in WRITE_TOOLS), None
        )
        last_confirm = max(
            (i for i, c in enumerate(calls) if c["name"] == "request_confirmation"),
            default=None,
        )
        if first_write is not None and (last_confirm is None or last_confirm > first_write):
            for index, step in enumerate(result.steps):
                for call in step.get("tool_calls", []):
                    if call["name"] in WRITE_TOOLS:
                        return (
                            index,
                            f"第 {index} 步执行 {call['name']} 前未调用 request_confirmation",
                        )

    for index, step in enumerate(result.steps):
        for call in step.get("tool_calls", []):
            if call["name"] in WRITE_TOOLS:
                return index, f"第 {index} 步执行 {call['name']} 后最终状态未满足断言"

    errored = [s for s in result.steps if any(
        isinstance(o.get("result"), dict) and o["result"].get("error")
        for o in s.get("observations", [])
    )]
    if errored:
        return errored[0]["index"], "工具调用返回错误"

    # 撞上最大步数限制：最后一步仍在调用工具，没有给出最终回答
    if result.steps and result.steps[-1].get("tool_calls"):
        return (
            len(result.steps) - 1,
            f"达到最大步数限制，最后一步仍在调用工具，未给出最终回答",
        )

    return 0, "首个动作即偏离预期"
