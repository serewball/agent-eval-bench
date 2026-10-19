"""单任务多次采样的执行器：记录完整轨迹、token、耗时与工具错误。"""

from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agents import Agent
from .schema import Task
from .tools import ToolError


@dataclass
class RunResult:
    task_id: str
    run_idx: int
    passed: bool = False
    assertion_results: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_output: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    tool_errors: int = 0
    forbidden_calls: int = 0
    first_error_step: Optional[int] = None
    error_reason: str = ""
    final_state: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_idx": self.run_idx,
            "passed": self.passed,
            "assertion_results": self.assertion_results,
            "steps": self.steps,
            "final_output": self.final_output,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "tool_errors": self.tool_errors,
            "forbidden_calls": self.forbidden_calls,
            "first_error_step": self.first_error_step,
            "error_reason": self.error_reason,
        }


def run_task(
    task: Task,
    agent: Agent,
    tools: Dict[str, Any],
    base_state: Dict[str, Any],
    run_idx: int,
    max_steps: int = 8,
) -> RunResult:
    state = copy.deepcopy(base_state)
    agent.reset(seed=run_idx)
    history: List[Dict[str, Any]] = []
    result = RunResult(task_id=task.id, run_idx=run_idx)

    started = time.perf_counter()
    for step_idx in range(max_steps):
        try:
            step = agent.next_step(task, history)
        except Exception as exc:  # 模型侧异常也算一次失败
            result.error_reason = f"Agent 异常：{exc}"
            break

        result.prompt_tokens += step.prompt_tokens
        result.completion_tokens += step.completion_tokens

        record: Dict[str, Any] = {
            "index": step_idx,
            "content": step.content,
            "tool_calls": [{"name": c.name, "args": c.args} for c in step.tool_calls],
            "observations": [],
            "prompt_tokens": step.prompt_tokens,
            "completion_tokens": step.completion_tokens,
        }

        if not step.tool_calls:
            result.final_output = step.content
            result.steps.append(record)
            break

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": step.content,
            "tool_calls": [],
        }

        for call in step.tool_calls:
            call_id = call.id or f"call_{uuid.uuid4().hex[:8]}"
            assistant_message["tool_calls"].append(
                {"id": call_id, "name": call.name, "args": call.args}
            )

            tool = tools.get(call.name)
            if tool is None:
                observation = {"error": f"工具 {call.name} 不存在"}
                result.tool_errors += 1
            else:
                try:
                    observation = tool.func(state, **call.args)
                except ToolError as exc:
                    observation = {"error": str(exc)}
                    result.tool_errors += 1
                except TypeError as exc:
                    observation = {"error": f"参数不合法：{exc}"}
                    result.tool_errors += 1

            if call.name in task.forbidden_tools:
                result.forbidden_calls += 1

            record["observations"].append({"tool": call.name, "result": observation})
            history.append(
                {
                    "role": "tool",
                    "name": call.name,
                    "tool_call_id": call_id,
                    "content": json.dumps(observation, ensure_ascii=False),
                }
            )

        history.append(assistant_message)
        result.steps.append(record)

    result.latency_ms = (time.perf_counter() - started) * 1000
    result.final_state = state
    return result
