from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Awaitable

READ_CAP = 80_000
BASH_TIMEOUT = 60
GLOB_CAP = 400
GREP_CAP = 80

AskPerm = Callable[[str, str], Awaitable[str]]


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file under the project cwd. Returns numbered lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "1-based start line"},
                    "limit": {"type": "integer", "description": "max lines to return"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Overwrite or create a file under cwd. Creates parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exactly one unique occurrence of old_string with new_string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents under cwd (ripgrep if installed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "subdir or file, default cwd"},
                    "glob": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Recursive glob under cwd. Returns matching paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "e.g. **/*.py"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a Linux/WSL bash command in cwd. Not PowerShell. 60s timeout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]


class ToolError(Exception):
    pass


def resolve_in_cwd(path: str, cwd: Path) -> Path:
    cwd = cwd.resolve()
    raw = Path(path).expanduser()
    candidate = raw if raw.is_absolute() else (cwd / raw)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError as exc:
        raise ToolError(f"path escapes cwd: {path}") from exc
    return resolved


_BLOCKED = [
    re.compile(r"\brm\s+-[a-zA-Z]*f[a-zA-Z]*\s+(--no-preserve-root\s+)?/(\s|$)"),
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r"\bdd\s+.*\bof=/dev/"),
    re.compile(r"\b(wipefs|shred)\b"),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*;\s*\}\s*;"),
    re.compile(r"\bchmod\s+-R\s+777\s+/"),
    re.compile(r"\b(forkbomb|format\s+[cC]:)"),
]


def command_blocked(command: str) -> str | None:
    for rx in _BLOCKED:
        if rx.search(command):
            return f"blocked dangerous command: {command!r}"
    return None


def _number_lines(text: str, start: int = 1) -> str:
    lines = text.splitlines()
    width = len(str(start + max(len(lines) - 1, 0)))
    return "\n".join(f"{i:>{width}}|{line}" for i, line in enumerate(lines, start))


def _clip(text: str, cap: int = READ_CAP) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n...[truncated {len(text) - cap} chars]"


async def _read_file(cwd: Path, args: dict[str, Any]) -> str:
    path = resolve_in_cwd(str(args.get("path") or ""), cwd)
    if not path.is_file():
        raise ToolError(f"not a file: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    offset = int(args.get("offset") or 1)
    if offset < 1:
        offset = 1
    limit = args.get("limit")
    if limit is None:
        chunk = lines[offset - 1 :]
    else:
        chunk = lines[offset - 1 : offset - 1 + int(limit)]
    body = _number_lines("\n".join(chunk), start=offset)
    return _clip(body)


async def _write_file(cwd: Path, args: dict[str, Any]) -> str:
    path = resolve_in_cwd(str(args.get("path") or ""), cwd)
    content = args.get("content")
    if content is None:
        raise ToolError("content is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content), encoding="utf-8")
    return f"wrote {path} ({len(str(content))} bytes)"


async def _edit_file(cwd: Path, args: dict[str, Any]) -> str:
    path = resolve_in_cwd(str(args.get("path") or ""), cwd)
    old = args.get("old_string")
    new = args.get("new_string")
    if old is None or new is None:
        raise ToolError("old_string and new_string are required")
    if not path.is_file():
        raise ToolError(f"not a file: {path}")
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        raise ToolError("old_string not found")
    if count > 1:
        raise ToolError(f"old_string is not unique ({count} matches)")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"edited {path}"


def _iter_files(root: Path) -> list[Path]:
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__"}
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for name in filenames:
            out.append(Path(dirpath) / name)
    return out


async def _grep(cwd: Path, args: dict[str, Any]) -> str:
    pattern = str(args.get("pattern") or "")
    if not pattern:
        raise ToolError("pattern is required")
    rel = str(args.get("path") or ".")
    target = resolve_in_cwd(rel, cwd)
    glob = args.get("glob")
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--line-number", "--color", "never", "-e", pattern]
        if glob:
            cmd.extend(["--glob", str(glob)])
        cmd.append(str(target))
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        text = stdout.decode("utf-8", "replace")
        if proc.returncode not in (0, 1):
            err = stderr.decode("utf-8", "replace").strip()
            raise ToolError(err or f"rg exit {proc.returncode}")
        lines = text.splitlines()[:GREP_CAP]
        if not lines:
            return "no matches"
        extra = ""
        if len(text.splitlines()) > GREP_CAP:
            extra = f"\n...[{len(text.splitlines()) - GREP_CAP} more]"
        return "\n".join(lines) + extra

    rx = re.compile(pattern)
    hits: list[str] = []
    files = [target] if target.is_file() else _iter_files(target)
    for fp in files:
        if glob:
            from fnmatch import fnmatch

            if not fnmatch(fp.name, str(glob)) and not fnmatch(str(fp.relative_to(cwd)), str(glob)):
                continue
        try:
            data = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(data.splitlines(), 1):
            if rx.search(line):
                rel_p = fp.relative_to(cwd)
                hits.append(f"{rel_p}:{i}:{line}")
                if len(hits) >= GREP_CAP:
                    return "\n".join(hits) + "\n...[truncated]"
    return "\n".join(hits) if hits else "no matches"


async def _glob(cwd: Path, args: dict[str, Any]) -> str:
    pattern = str(args.get("pattern") or "")
    if not pattern:
        raise ToolError("pattern is required")
    cwd = cwd.resolve()
    matches = sorted(p for p in cwd.glob(pattern) if p.is_file())
    rels = []
    for p in matches[:GLOB_CAP]:
        try:
            rels.append(str(p.resolve().relative_to(cwd)))
        except ValueError:
            continue
    extra = f"\n...[{len(matches) - GLOB_CAP} more]" if len(matches) > GLOB_CAP else ""
    return "\n".join(rels) + extra if rels else "no matches"


async def _bash(cwd: Path, args: dict[str, Any]) -> str:
    command = str(args.get("command") or "")
    if not command.strip():
        raise ToolError("command is required")
    blocked = command_blocked(command)
    if blocked:
        raise ToolError(blocked)
    env = os.environ.copy()
    env["PAGER"] = "cat"
    env["GIT_PAGER"] = "cat"
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=BASH_TIMEOUT)
    except asyncio.TimeoutError as exc:
        raise ToolError(f"timed out after {BASH_TIMEOUT}s") from exc
    out = stdout.decode("utf-8", "replace")
    err = stderr.decode("utf-8", "replace")
    parts = [f"exit {proc.returncode}"]
    if out:
        parts.append("stdout:\n" + out)
    if err:
        parts.append("stderr:\n" + err)
    return _clip("\n".join(parts), cap=READ_CAP)


_HANDLERS = {
    "read_file": _read_file,
    "write_file": _write_file,
    "edit_file": _edit_file,
    "grep": _grep,
    "glob": _glob,
    "bash": _bash,
}


def summary(name: str, args: dict[str, Any]) -> str:
    if name == "bash":
        return f"bash {args.get('command', '')}"
    if name in {"read_file", "write_file", "edit_file"}:
        return f"{name} {args.get('path', '')}"
    if name == "grep":
        return f"grep {args.get('pattern', '')}"
    if name == "glob":
        return f"glob {args.get('pattern', '')}"
    return name


async def run_tool(name: str, arguments: dict[str, Any] | str, cwd: Path) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return f"invalid JSON arguments: {arguments[:200]}"
    if not isinstance(arguments, dict):
        arguments = {}
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"unknown tool: {name}"
    try:
        return await handler(cwd, arguments)
    except ToolError as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: {exc}"
