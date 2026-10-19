"""Runner 轨迹协议的单测：重点锁住 OpenAI 消息顺序回归。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalbench.agents import Agent, Step, ToolCall  # noqa: E402
from evalbench.judge import evaluate  # noqa: E402
from evalbench.runner import run_task  # noqa: E402
from evalbench.schema import Assertion, Task  # noqa: E402
from evalbench.tools import build_tools  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class _SpyAgent(Agent):
    """第一步调工具，第二步记录 history 快照后给出最终回答。"""

    name = "spy"

    def __init__(self) -> None:
        self.turn = 0
        self.snapshot: List[Dict[str, Any]] = []

    def reset(self, seed: int) -> None:
        self.turn = 0
        self.snapshot = []

    def next_step(self, task: Task, history: List[Dict[str, Any]]) -> Step:
        self.turn += 1
        if self.turn == 1:
            return Step(
                content="查询订单",
                tool_calls=[ToolCall(name="get_order", args={"order_id": "PO-1001"})],
            )
        self.snapshot = list(history)
        return Step(content="订单 PO-1001 当前状态：运输中。")


class TestHistoryOrder(unittest.TestCase):
    def setUp(self) -> None:
        self.task = Task(
            id="T-1",
            instruction="查询订单 PO-1001 的状态。",
            assertions=[
                Assertion(type="output_contains", value="PO-1001"),
                Assertion(type="tool_called", tool="get_order", args_contains={"order_id": "PO-1001"}),
            ],
        )
        state_path = ROOT / "tasks" / "supply-chain" / "orders.json"
        self.state = json.loads(state_path.read_text(encoding="utf-8"))

    def test_assistant_message_precedes_tool_message(self) -> None:
        """OpenAI 协议：tool 消息必须紧跟对应的 assistant(tool_calls) 消息。"""
        agent = _SpyAgent()
        result = run_task(self.task, agent, build_tools(), self.state, run_idx=0)
        evaluate(self.task, result)
        self.assertTrue(result.passed, result.assertion_results)

        roles = [m["role"] for m in agent.snapshot]
        # 顺序必须是 assistant(tool_calls) -> tool，反之 OpenAI 接口报 400
        self.assertEqual(roles, ["assistant", "tool"])
        self.assertEqual(agent.snapshot[0]["tool_calls"][0]["name"], "get_order")
        self.assertEqual(agent.snapshot[1]["role"], "tool")


class TestMaxSteps(unittest.TestCase):
    def test_step_cap_aborts_with_failure(self) -> None:
        """Agent 永远调工具不结束：max_steps 兜底，判失败并给出归因。"""

        class _LoopAgent(Agent):
            name = "loop"

            def next_step(self, task: Task, history: List[Dict[str, Any]]) -> Step:
                return Step(
                    content="继续查询",
                    tool_calls=[ToolCall(name="get_order", args={"order_id": "PO-1001"})],
                )

        task = Task(
            id="T-2",
            instruction="查询订单 PO-1001 的状态。",
            assertions=[Assertion(type="output_contains", value="PO-1001")],
        )
        state_path = ROOT / "tasks" / "supply-chain" / "orders.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))

        result = run_task(task, _LoopAgent(), build_tools(), state, run_idx=0, max_steps=3)
        evaluate(task, result)
        self.assertFalse(result.passed)
        self.assertIn("最大步数", result.error_reason)


if __name__ == "__main__":
    unittest.main()
