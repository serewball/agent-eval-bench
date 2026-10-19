"""OpenAI 兼容接口的最小客户端（标准库 urllib 实现）。

Agent 与 Judge 共用同一份调用逻辑，避免两处各写一遍 HTTP。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    pass


def chat(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    timeout: int = 60,
) -> Dict[str, Any]:
    """调用 /chat/completions，返回原始响应 dict。"""
    url = (base_url or os.environ.get("EVALBENCH_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    key = api_key or os.environ.get("EVALBENCH_API_KEY", "")
    model_name = model or os.environ.get("EVALBENCH_MODEL", "gpt-4o-mini")

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    request = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"模型接口返回 {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"无法连接模型接口 {url}: {exc.reason}") from exc


def message_text(response: Dict[str, Any]) -> str:
    """从响应里取出 assistant 文本，兼容缺失字段的粗糙实现。"""
    try:
        return response["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"响应结构异常: {response}") from exc
