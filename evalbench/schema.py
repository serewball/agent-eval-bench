"""Task / assertion 数据模型。仅依赖标准库。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Assertion:
    """一条成功判据。

    支持的类型：
    - output_contains  最终回答包含指定文本
    - output_regex     最终回答匹配正则
    - tool_called      调用过某工具（可用 args_contains 约束参数）
    - tool_not_called  没有调用过某工具
    - state_equals     运行结束后的环境状态等于期望值（path 用点号分隔）
    - state_not_equals 运行结束后的环境状态不等于期望值
    """

    type: str
    value: Any = None
    path: Optional[str] = None
    tool: Optional[str] = None
    args_contains: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Assertion":
        if "type" not in data:
            raise ValueError(f"assertion 缺少 type 字段: {data}")
        return cls(
            type=data["type"],
            value=data.get("value"),
            path=data.get("path"),
            tool=data.get("tool"),
            args_contains=data.get("args_contains") or {},
        )


@dataclass
class Task:
    id: str
    instruction: str
    assertions: List[Assertion]
    name: str = ""
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    forbidden_tools: List[str] = field(default_factory=list)
    requires_confirmation: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        if "id" not in data or "instruction" not in data:
            raise ValueError(f"task 缺少 id 或 instruction: {data}")
        return cls(
            id=data["id"],
            instruction=data["instruction"],
            assertions=[Assertion.from_dict(a) for a in data.get("assertions", [])],
            name=data.get("name", data["id"]),
            tags=data.get("tags", []),
            difficulty=data.get("difficulty", "medium"),
            forbidden_tools=data.get("forbidden_tools", []),
            requires_confirmation=data.get("requires_confirmation", False),
        )


def load_tasks(path: str) -> List[Task]:
    """读取任务集。.json 零依赖；.yaml/.yml 需要 PyYAML。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"任务集不存在: {p}")

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "读取 YAML 任务集需要 PyYAML：pip install pyyaml\n"
                "或直接使用自带的 JSON 任务集。"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    raw_tasks = data["tasks"] if isinstance(data, dict) else data
    return [Task.from_dict(item) for item in raw_tasks]


def load_state(path: str) -> Dict[str, Any]:
    """读取环境初始状态（订单库等）。每次运行会深拷贝一份。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
