from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from qwen_agent.config import Config
from qwen_agent.tools import TOOL_SCHEMAS


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolRequest:
    id: str
    name: str
    arguments: dict[str, Any]
    arguments_raw: str


@dataclass
class StreamDone:
    finish_reason: str | None
    content: str
    tool_calls: list[ToolRequest]


@dataclass
class StreamError:
    message: str


def host_of(url: str) -> str:
    p = urlparse(url)
    return p.netloc or url


class QwenClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.api.timeout_s, connect=15.0),
            headers={
                "Authorization": f"Bearer {cfg.api.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, messages: list[dict[str, Any]], stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.cfg.api.model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "stream": stream,
            "temperature": 0.2,
        }
        if stream:
            body["stream_options"] = {"include_usage": False}
        return body

    async def stream(
        self, messages: list[dict[str, Any]]
    ) -> AsyncIterator[TextDelta | StreamDone | StreamError]:
        url = f"{self.cfg.api.base_url}/chat/completions"
        try:
            async with self._client.stream("POST", url, json=self._payload(messages, True)) as resp:
                if resp.status_code >= 400:
                    err_body = (await resp.aread()).decode("utf-8", "replace")
                    yield StreamError(self._explain_http(resp.status_code, err_body))
                    return
                async for item in self._consume_sse(resp):
                    yield item
        except httpx.ConnectError as exc:
            yield StreamError(self._explain_connect(exc))
        except httpx.TimeoutException:
            yield StreamError(
                f"timeout talking to {host_of(self.cfg.api.base_url)}. "
                "Is Tailscale up? Try: qwen-agent doctor"
            )
        except httpx.HTTPError as exc:
            yield StreamError(f"HTTP error: {exc}")

    async def _consume_sse(
        self, resp: httpx.Response
    ) -> AsyncIterator[TextDelta | StreamDone | StreamError]:
        content_parts: list[str] = []
        tools: dict[int, dict[str, str]] = {}
        finish: str | None = None
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith(":") :
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                yield StreamError(str(chunk["error"]))
                return
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                content_parts.append(piece)
                yield TextDelta(piece)
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index") or 0)
                slot = tools.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
        yield StreamDone(
            finish_reason=finish,
            content="".join(content_parts),
            tool_calls=_finalize_tools(tools),
        )

    def _explain_http(self, status: int, body: str) -> str:
        host = host_of(self.cfg.api.base_url)
        snippet = body[:400].replace("\n", " ")
        if status in (401, 403):
            return (
                f"HTTP {status} from {host}. Tailscale can see the node but the ACL "
                "or API auth is denying you. Ask the GPU host admin to allow your "
                f"user to this port. Body: {snippet}"
            )
        if status == 404:
            return (
                f"HTTP 404 from {host}{self.cfg.api.base_url}. "
                "Wrong port or missing /v1. Ollama is usually :11434/v1; llama.cpp often :8080/v1."
            )
        return f"HTTP {status} from {host}: {snippet}"

    def _explain_connect(self, exc: BaseException) -> str:
        host = host_of(self.cfg.api.base_url)
        return (
            f"cannot connect to {host} ({exc}). "
            "Usual causes: Tailscale logged out, Cloudflare WARP still up, "
            "the GPU host bound only to 10.40.0.10/127.0.0.1, or ACL drop. "
            "Run: qwen-agent doctor"
        )


def _finalize_tools(tools: dict[int, dict[str, str]]) -> list[ToolRequest]:
    out: list[ToolRequest] = []
    for idx in sorted(tools):
        slot = tools[idx]
        raw = slot.get("arguments") or "{}"
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        out.append(
            ToolRequest(
                id=slot.get("id") or f"call_{idx}",
                name=slot.get("name") or "",
                arguments=parsed,
                arguments_raw=raw,
            )
        )
    return out


async def list_models(cfg: Config) -> tuple[int | None, str]:
    url = f"{cfg.api.base_url}/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {cfg.api.api_key}"},
            )
            return resp.status_code, resp.text[:2000]
    except httpx.ConnectError as exc:
        return None, f"connect error: {exc}"
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.HTTPError as exc:
        return None, str(exc)
