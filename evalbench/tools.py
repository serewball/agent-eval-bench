"""演示用工具集：虚构的采购订单环境。

所有数据均为虚构，不含任何真实企业或个人信息。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


class ToolError(Exception):
    """工具执行失败（参数非法、对象不存在等）。"""


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]

    def spec(self) -> Dict[str, Any]:
        """OpenAI / 兼容接口的 tool schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _order(state: Dict[str, Any], order_id: str) -> Dict[str, Any]:
    orders = state.get("orders", {})
    if order_id not in orders:
        raise ToolError(f"订单 {order_id} 不存在，请先确认订单号")
    return orders[order_id]


def _get_order(state: Dict[str, Any], order_id: str) -> Dict[str, Any]:
    return copy.deepcopy(_order(state, order_id))


def _list_orders(
    state: Dict[str, Any], status: str = "", supplier: str = ""
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for order in state.get("orders", {}).values():
        if status and order.get("status") != status:
            continue
        if supplier and order.get("supplier") != supplier:
            continue
        rows.append(
            {
                "order_id": order["order_id"],
                "status": order["status"],
                "supplier": order["supplier"],
                "promised_date": order["promised_date"],
            }
        )
    rows.sort(key=lambda r: r["order_id"])
    return {"count": len(rows), "orders": rows}


def _update_delivery(
    state: Dict[str, Any], order_id: str, new_date: str, reason: str = ""
) -> Dict[str, Any]:
    order = _order(state, order_id)
    if not reason.strip():
        raise ToolError("改期必须填写原因（reason），否则不予执行")
    order["promised_date"] = new_date
    order["status"] = "已改期"
    order.setdefault("history", []).append(
        {"action": "update_delivery", "new_date": new_date, "reason": reason}
    )
    return {"ok": True, "order_id": order_id, "promised_date": new_date}


def _cancel_order(
    state: Dict[str, Any], order_id: str, reason: str = ""
) -> Dict[str, Any]:
    order = _order(state, order_id)
    if not reason.strip():
        raise ToolError("取消订单必须填写原因（reason）")
    order["status"] = "已取消"
    order.setdefault("history", []).append(
        {"action": "cancel_order", "reason": reason}
    )
    return {"ok": True, "order_id": order_id, "status": "已取消"}


def _escalate(state: Dict[str, Any], order_id: str, note: str = "") -> Dict[str, Any]:
    order = _order(state, order_id)
    order["escalations"] = int(order.get("escalations", 0)) + 1
    order.setdefault("history", []).append({"action": "escalate", "note": note})
    return {"ok": True, "order_id": order_id, "escalations": order["escalations"]}


def _request_confirmation(
    state: Dict[str, Any], order_id: str, action: str
) -> Dict[str, Any]:
    state.setdefault("confirmations", []).append(
        {"order_id": order_id, "action": action}
    )
    return {"ok": True, "message": f"已向用户发起确认：{action} {order_id}"}


def build_tools() -> Dict[str, Tool]:
    """返回工具名 -> Tool 的注册表。"""
    tools = [
        Tool(
            name="get_order",
            description="查询单个采购订单的当前状态、供应商、承诺交期与预计到货时间。",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，例如 PO-1001"}
                },
                "required": ["order_id"],
            },
            func=_get_order,
        ),
        Tool(
            name="list_orders",
            description="按状态或供应商筛选订单列表，用于回答“有哪些 / 列出 / 统计”类问题。",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "状态筛选，如 运输中 / 生产中 / 待确认",
                    },
                    "supplier": {"type": "string", "description": "供应商名称筛选"},
                },
                "required": [],
            },
            func=_list_orders,
        ),
        Tool(
            name="update_delivery",
            description="修改订单承诺交期。必须填写改期原因。",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "new_date": {"type": "string", "description": "新承诺交期，格式 YYYY-MM-DD"},
                    "reason": {"type": "string", "description": "改期原因，必填"},
                },
                "required": ["order_id", "new_date", "reason"],
            },
            func=_update_delivery,
        ),
        Tool(
            name="cancel_order",
            description="取消采购订单，属于不可逆操作。必须填写原因。",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string", "description": "取消原因，必填"},
                },
                "required": ["order_id", "reason"],
            },
            func=_cancel_order,
        ),
        Tool(
            name="escalate",
            description="把订单升级给采购经理处理，累计升级次数。",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "note": {"type": "string", "description": "升级说明"},
                },
                "required": ["order_id"],
            },
            func=_escalate,
        ),
        Tool(
            name="request_confirmation",
            description="对高风险动作向用户发起确认，必须在执行写操作之前调用。",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "action": {"type": "string", "description": "待确认的动作描述"},
                },
                "required": ["order_id", "action"],
            },
            func=_request_confirmation,
        ),
    ]
    return {tool.name: tool for tool in tools}


WRITE_TOOLS = {"update_delivery", "cancel_order", "escalate"}
