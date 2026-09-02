from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


HOME = Path.home() / ".qwen-agent"
DEFAULT_CONFIG = HOME / "config.toml"
DEFAULT_BASE = "http://100.64.11.62:11434/v1"
DEFAULT_MODEL = "qwen3.8:27b"


@dataclass
class ApiConfig:
    base_url: str = DEFAULT_BASE
    api_key: str = "dummy"
    model: str = DEFAULT_MODEL
    timeout_s: float = 300.0


@dataclass
class UiConfig:
    vim_mode: bool = False


@dataclass
class SafetyConfig:
    auto_approve_read: bool = True


@dataclass
class ContextConfig:
    """Local files injected into each API request. Not stored on the GPU host."""

    files: list[str]


@dataclass
class Config:
    api: ApiConfig
    ui: UiConfig
    safety: SafetyConfig
    context: ContextConfig
    path: Path


def normalize_base_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return DEFAULT_BASE
    if u.endswith("/v1"):
        return u
    if u.endswith("/v1/chat/completions"):
        return u[: -len("/chat/completions")]
    return u + "/v1"


def _load_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f) or {}


def load_config(path: Path | None = None) -> Config:
    cfg_path = path or DEFAULT_CONFIG
    data = _load_toml(cfg_path) if cfg_path.is_file() else {}
    api_d = data.get("api") or {}
    ui_d = data.get("ui") or {}
    safety_d = data.get("safety") or {}

    api = ApiConfig(
        base_url=normalize_base_url(
            os.environ.get("QWEN_BASE_URL") or api_d.get("base_url") or DEFAULT_BASE
        ),
        api_key=os.environ.get("QWEN_API_KEY") or str(api_d.get("api_key") or "dummy"),
        model=os.environ.get("QWEN_MODEL") or str(api_d.get("model") or DEFAULT_MODEL),
        timeout_s=float(api_d.get("timeout_s") or 300),
    )
    ui = UiConfig(vim_mode=bool(ui_d.get("vim_mode", False)))
    safety = SafetyConfig(auto_approve_read=bool(safety_d.get("auto_approve_read", True)))
    ctx_d = data.get("context") or {}
    files = ctx_d.get("files") or []
    if isinstance(files, str):
        files = [files]
    context = ContextConfig(files=[str(x) for x in files])
    return Config(api=api, ui=ui, safety=safety, context=context, path=cfg_path)


def default_config_text() -> str:
    return """[api]
base_url = "http://100.64.11.62:11434/v1"
api_key = "dummy"
model = "qwen3.8:27b"
timeout_s = 300

[ui]
vim_mode = false

[safety]
auto_approve_read = true
# bash/write/edit require y/n unless always-allow this session

# Local files attached to every request (not saved on the GPU host).
# Also auto-loads: ~/.qwen-agent/instructions.md, AGENTS.md in the repo,
# and files under .qwen-agent/ref/
[context]
files = []
"""


def ensure_home() -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / "sessions").mkdir(parents=True, exist_ok=True)
    instr = HOME / "instructions.md"
    if not instr.is_file():
        instr.write_text(
            "# Personal instructions\n\n"
            "These stay on this machine and are sent with every request to the remote model.\n"
            "Put preferences here (language, review style, things never to do).\n",
            encoding="utf-8",
        )
    if not DEFAULT_CONFIG.is_file():
        DEFAULT_CONFIG.write_text(default_config_text(), encoding="utf-8")
    else:
        text = DEFAULT_CONFIG.read_text(encoding="utf-8")
        if "[context]" not in text:
            DEFAULT_CONFIG.write_text(
                text.rstrip()
                + "\n\n[context]\nfiles = []\n",
                encoding="utf-8",
            )
