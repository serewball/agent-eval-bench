"""Agent 适配层。

两种实现：
- MockAgent   离线确定性脚本 Agent，用于零依赖跑通流水线与复现实验
- OpenAIAgent 任何 OpenAI 兼容接口（含本地 vLLM / Qwen 等）

两者对外接口一致：reset(seed) + next_step(task, history) -> Step，
便于把真实模型和基线策略放进同一套评测里对比。
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import Task


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class Step:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Agent:
    name = "base"

    def reset(self, seed: int) -> None:
        """每轮采样前重置内部状态。"""

    def next_step(self, task: Task, history: List[Dict[str, Any]]) -> Step:
        raise NotImplementedError


def _estimate_tokens(text: str) -> int:
    """粗略估算 token：中英混排按 3 字符 ≈ 1 token。"""
    return max(1, len(text) // 3)


class MockAgent(Agent):
    """确定性脚本 Agent，按任务意图生成工具调用。

    error_rate 用于模拟模型的不稳定性：注入错日期、漏确认、错订单号等典型失误，
    从而产生"Pass@k 很高但 Pass^k 很低"的真实分布，供可靠性演示。
    """

    name = "mock"

    def __init__(self, error_rate: float = 0.30, seed: int = 7) -> None:
        self.error_rate = error_rate
        self.seed = seed
        self._run_seed = 0
        self._plan: Optional[List[ToolCall]] = None
        self._cursor = 0

    def reset(self, seed: int) -> None:
        # 注意：不能只用 run seed 生成随机序列，否则同一轮所有任务会共享同一个
        # 决策，采样之间高度相关（Pass^k 会被系统性压低）。真正的随机源在
        # _build_plan 里再按 task.id 混合。
        self._run_seed = seed
        self._plan = None
        self._cursor = 0

    # ------------------------------------------------------------------ 规划
    @staticmethod
    def _intent(text: str) -> str:
        if any(k in text for k in ("取消", "撤单")):
            return "cancel"
        if any(k in text for k in ("升级", "催", "升级处理")):
            return "escalate"
        if any(k in text for k in ("延期", "改期", "推迟", "改到")):
            return "reschedule"
        if any(k in text for k in ("列出", "哪些", "统计", "有多少")):
            return "list"
        return "query"

    def _pick_noise(
        self, rng: random.Random, intent: str, task: Task, has_date: bool
    ) -> Optional[str]:
        if rng.random() >= self.error_rate:
            return None
        modes = ["wrong_order", "tool_error"]
        if intent in ("reschedule", "cancel"):
            if has_date:
                modes.append("wrong_date")
            if task.requires_confirmation:
                modes.append("skip_confirm")
        if "cancel_order" in task.forbidden_tools:
            modes.append("forbidden")
        return rng.choice(modes)

    def _build_plan(self, task: Task) -> List[ToolCall]:
        rng = random.Random(f"{self.seed}|{self._run_seed}|{task.id}")
        text = task.instruction
        order_match = re.search(r"PO-\d{4}", text)
        order_id = order_match.group(0) if order_match else ""
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        new_date = date_match.group(0) if date_match else "2026-12-31"

        intent = self._intent(text)
        noise = self._pick_noise(rng, intent, task, bool(date_match))

        if noise == "wrong_order":
            order_id = "PO-1002" if order_id != "PO-1002" else "PO-1003"
        elif noise == "tool_error":
            order_id = "PO-9999"  # 不存在的订单，触发工具错误
        elif noise == "wrong_date":
            new_date = "2026-10-30" if new_date != "2026-10-30" else "2026-11-01"

        calls: List[ToolCall] = []

        if noise == "forbidden":
            calls.append(ToolCall("cancel_order", {"order_id": order_id, "reason": "系统自动取消"}))

        if intent == "list":
            status = ""
            for candidate in ("运输中", "生产中", "待确认", "已到货", "已取消", "已改期"):
                if candidate in text:
                    status = candidate
                    break
            calls.append(ToolCall("list_orders", {"status": status} if status else {}))

        elif intent == "query":
            calls.append(ToolCall("get_order", {"order_id": order_id}))

        elif intent == "reschedule":
            if task.requires_confirmation and noise != "skip_confirm":
                calls.append(
                    ToolCall(
                        "request_confirmation",
                        {"order_id": order_id, "action": "修改承诺交期"},
                    )
                )
            calls.append(
                ToolCall(
                    "update_delivery",
                    {
                        "order_id": order_id,
                        "new_date": new_date,
                        "reason": "供应商产能调整，需延后交付",
                    },
                )
            )

        elif intent == "cancel":
            if task.requires_confirmation and noise != "skip_confirm":
                calls.append(
                    ToolCall(
                        "request_confirmation",
                        {"order_id": order_id, "action": "取消订单"},
                    )
                )
            calls.append(
                ToolCall("cancel_order", {"order_id": order_id, "reason": "客户要求撤单"})
            )

        elif intent == "escalate":
            calls.append(ToolCall("get_order", {"order_id": order_id}))
            calls.append(
                ToolCall("escalate", {"order_id": order_id, "note": "交期多次变更，请采购经理介入"})
            )

        return calls

    # ------------------------------------------------------------------ 执行
    def next_step(self, task: Task, history: List[Dict[str, Any]]) -> Step:
        if self._plan is None:
            self._plan = self._build_plan(task)

        if self._cursor < len(self._plan):
            call = self._plan[self._cursor]
            self._cursor += 1
            payload = json.dumps(call.args, ensure_ascii=False)
            return Step(
                content=f"调用 {call.name}",
                tool_calls=[call],
                prompt_tokens=_estimate_tokens(task.instruction + payload),
                completion_tokens=_estimate_tokens(payload),
            )

        text = self._final_text(history)
        return Step(
            content=text,
            prompt_tokens=_estimate_tokens(task.instruction + json.dumps(history, ensure_ascii=False)),
            completion_tokens=_estimate_tokens(text),
        )

    @staticmethod
    def _final_text(history: List[Dict[str, Any]]) -> str:
        observations: Dict[str, Any] = {}
        for message in history:
            if message.get("role") != "tool":
                continue
            try:
                observations[message["name"]] = json.loads(message["content"])
            except (KeyError, json.JSONDecodeError):
                continue

        order = observations.get("get_order")
        if isinstance(order, dict) and "order_id" in order:
            return (
                f"订单 {order['order_id']} 当前状态：{order.get('status', '未知')}，"
                f"供应商：{order.get('supplier', '未知')}，"
                f"承诺交期 {order.get('promised_date', '未知')}，"
                f"预计 {order.get('eta', '未知')} 到货。"
            )

        listing = observations.get("list_orders")
        if isinstance(listing, dict) and "orders" in listing:
            items = "、".join(f"{r['order_id']}（{r['status']}）" for r in listing["orders"])
            return f"共 {listing.get('count', 0)} 条：{items}。" if items else "没有符合条件的订单。"

        updated = observations.get("update_delivery")
        if isinstance(updated, dict) and updated.get("ok"):
            return f"已将订单 {updated['order_id']} 的承诺交期改为 {updated['promised_date']}，并记录改期原因。"

        cancelled = observations.get("cancel_order")
        if isinstance(cancelled, dict) and cancelled.get("ok"):
            return f"订单 {cancelled['order_id']} 已取消，取消原因已记录。"

        escalated = observations.get("escalate")
        if isinstance(escalated, dict) and escalated.get("ok"):
            return f"订单 {escalated['order_id']} 已升级至采购经理，累计升级 {escalated['escalations']} 次。"

        return "已处理完成。"


class OpenAIAgent(Agent):
    """OpenAI 兼容接口的 Agent（可指向 vLLM / Ollama / 内网 Qwen 服务）。"""

    name = "openai"
    SYSTEM_PROMPT = (
        "你是采购订单助手。所有订单事实必须通过工具获取，不要凭记忆或推测回答。"
        "涉及改期、取消等写操作时，必须先用 request_confirmation 向用户发起确认，"
        "并填写原因；工具返回错误时如实说明，不要编造成功结果。"
    )

    def __init__(
        self,
        tool_specs: List[Dict[str, Any]],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 60,
    ) -> None:
        self.tool_specs = tool_specs
        self.base_url = (base_url or os.environ.get("EVALBENCH_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("EVALBENCH_API_KEY", "")
        self.model = model or os.environ.get("EVALBENCH_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.timeout = timeout

    def reset(self, seed: int) -> None:
        return None

    def _post(self, payload: Dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"模型接口返回 {exc.code}: {body[:500]}") from exc

    def _messages(self, task: Task, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": task.instruction},
        ]
        for message in history:
            if message["role"] == "assistant":
                item: Dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
                if message.get("tool_calls"):
                    item["tool_calls"] = [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["args"], ensure_ascii=False),
                            },
                        }
                        for call in message["tool_calls"]
                    ]
                messages.append(item)
            elif message["role"] == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get("tool_call_id", ""),
                        "content": message["content"],
                    }
                )
        return messages

    def next_step(self, task: Task, history: List[Dict[str, Any]]) -> Step:
        payload = {
            "model": self.model,
            "messages": self._messages(task, history),
            "tools": self.tool_specs,
            "temperature": self.temperature,
        }
        data = json.loads(self._post(payload))
        message = data["choices"][0]["message"]

        calls: List[ToolCall] = []
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", {})
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(
                ToolCall(
                    name=function.get("name", ""),
                    args=args if isinstance(args, dict) else {},
                    id=tool_call.get("id", ""),
                )
            )

        usage = data.get("usage") or {}
        return Step(
            content=message.get("content") or "",
            tool_calls=calls,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )


def build_agent(kind: str, tool_specs: List[Dict[str, Any]], **kwargs: Any) -> Agent:
    if kind == "mock":
        return MockAgent(
            error_rate=float(kwargs.get("error_rate", 0.30)),
            seed=int(kwargs.get("seed", 7)),
        )
    if kind == "openai":
        return OpenAIAgent(
            tool_specs,
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model"),
        )
    raise ValueError(f"未知 agent 类型: {kind}")
