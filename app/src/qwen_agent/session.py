from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_agent.config import HOME

KEEP_MESSAGES = 30


def cwd_hash(cwd: Path) -> str:
    return hashlib.sha256(str(cwd.resolve()).encode("utf-8")).hexdigest()[:16]


def session_path(cwd: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = HOME / "sessions" / cwd_hash(cwd)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{stamp}.jsonl"


class SessionLog:
    def __init__(self, cwd: Path) -> None:
        self.path = session_path(cwd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def rotate(self, cwd: Path) -> None:
        self.path = session_path(cwd)
        self.path.touch(exist_ok=True)


def _starts_broken(msg: dict[str, Any]) -> bool:
    role = msg.get("role")
    if role == "tool":
        return True
    if role == "assistant" and msg.get("tool_calls"):
        return True
    return False


def trim_messages(messages: list[dict[str, Any]], keep: int = KEEP_MESSAGES) -> list[dict[str, Any]]:
    if not messages:
        return messages
    system, rest = [], messages
    if messages[0].get("role") == "system":
        system = [messages[0]]
        rest = messages[1:]
    if len(rest) <= keep:
        return system + rest
    trimmed = rest[-keep:]
    while trimmed and _starts_broken(trimmed[0]):
        trimmed = trimmed[1:]
    return system + trimmed
