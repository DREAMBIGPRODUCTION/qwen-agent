from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

from qwen_agent.api import list_models
from qwen_agent.config import Config, HOME, load_config


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def _git_rev(cwd: Path) -> str | None:
    import subprocess

    try:
        p = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip() or None


async def collect_report(cfg: Config | None = None, cwd: Path | None = None) -> str:
    cfg = cfg or load_config()
    cwd = (cwd or Path.cwd()).resolve()
    lines: list[str] = []

    def ok(flag: bool, name: str, detail: str = "") -> None:
        mark = "ok" if flag else "FAIL"
        extra = f" — {detail}" if detail else ""
        lines.append(f"[{mark}] {name}{extra}")

    linux = sys.platform.startswith("linux")
    wsl = _is_wsl()
    ok(linux, "Linux kernel", "WSL2" if wsl else sys.platform)
    if not linux:
        lines.append("    Run inside WSL: wsl.exe")
    ok(sys.version_info >= (3, 11), f"Python {sys.version.split()[0]}", "need 3.11+")
    venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    ok(True, "venv" if venv else "interpreter", sys.prefix)
    ok(shutil.which("git") is not None, "git", shutil.which("git") or "missing")
    rg = shutil.which("rg")
    ok(True, "ripgrep", rg or "optional — Python grep fallback will be used")
    ok(cfg.path.is_file(), "config", str(cfg.path))
    ok(HOME.is_dir(), "data dir", str(HOME))

    host = urlparse(cfg.api.base_url).netloc
    ok(True, "API", f"{cfg.api.model} @ {cfg.api.base_url}")

    status, body = await list_models(cfg)
    if status == 200:
        names = []
        try:
            data = json.loads(body)
            names = [m.get("id") or m.get("name") for m in (data.get("data") or []) if m]
        except json.JSONDecodeError:
            names = []
        ok(True, f"GET {host}/models", ", ".join(n for n in names if n)[:200] or "reachable")
        if cfg.api.model and names and cfg.api.model not in names:
            lines.append(
                f"    warning: configured model {cfg.api.model!r} not in /models list"
            )
    elif status in (401, 403):
        ok(False, f"GET {host}/models HTTP {status}", "ACL or auth denied")
        lines.append(
            "    Tailscale may see the node, but this user is not allowed to that port. "
            "Ask the GPU host admin to add an ACL src for your Tailscale login to portal8k:11434 "
            "(or :8080 for llama.cpp)."
        )
    elif status == 404:
        ok(False, f"GET {host}/models HTTP 404", "wrong path/port")
        lines.append("    Ollama: http://<tailscale-ip>:11434/v1")
        lines.append("    llama.cpp: http://<tailscale-ip>:8080/v1")
    elif status is None and body == "timeout":
        ok(False, f"GET {host}/models", "timeout")
        lines.append(
            "    Packet is dropping. Disconnect Cloudflare WARP, confirm tailscale status, "
            "and do not use the LAN IP 10.40.0.10 from another network."
        )
    else:
        ok(False, f"GET {host}/models", body[:180])
        lines.append(
            "    Cannot connect. Check Tailscale login, WARP off, and that Ollama/llama.cpp "
            "listens on 0.0.0.0 (not 127.0.0.1 / not only 10.40.0.10)."
        )

    branch = _git_rev(cwd)
    if branch:
        ok(True, "git repo", f"{cwd} ({branch})")
    else:
        ok(True, "cwd", f"{cwd} (not a git repo — that's fine)")

    posix = cwd.as_posix()
    if posix.startswith("/mnt/c/") or posix.startswith("/mnt/d/"):
        lines.append(
            f"[WARN] cwd is {cwd} — /mnt/c is a slow Windows mount. "
            "Clone under $HOME/src/... inside WSL."
        )

    from qwen_agent.context import build_pack

    pack = build_pack(cwd, cfg)
    ok(True, "local context pack", f"{pack.chars} chars will be sent in each request")
    for label, path, body in pack.instructions + pack.references:
        lines.append(f"    {label}: {path} ({len(body)} chars)")
    if not pack.instructions and not pack.references:
        lines.append("    none yet — add ~/.qwen-agent/instructions.md, AGENTS.md, or .qwen-agent/ref/")
    for s in pack.skipped:
        lines.append(f"    skipped: {s}")

    return "\n".join(lines)


def main() -> int:
    import asyncio

    from qwen_agent.config import ensure_home

    ensure_home()
    print(asyncio.run(collect_report()))
    return 0
