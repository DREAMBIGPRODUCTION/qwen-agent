from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from qwen_agent.config import HOME, Config
from qwen_agent.tools import ToolError, resolve_in_cwd

# Rough char budgets. Qwen3.8-27B has a large window; keep the *always-on*
# pack modest so tools + history still fit. ~4 chars/token.
FILE_CAP = 24_000
TOTAL_CAP = 80_000

PROJECT_INSTRUCTION_NAMES = (
    "AGENTS.md",
    "QWEN.md",
    ".qwen-agent.md",
)


def _read_capped(path: Path, cap: int = FILE_CAP) -> str:
    data = path.read_text(encoding="utf-8", errors="replace")
    if len(data) <= cap:
        return data
    return data[:cap] + f"\n...[truncated {len(data) - cap} chars from {path.name}]"


def _safe_file(path: Path) -> Path | None:
    try:
        if path.is_file():
            return path.resolve()
    except OSError:
        return None
    return None


@dataclass
class ContextPack:
    """Local files assembled into the request body. Nothing is stored on the GPU host."""

    instructions: list[tuple[str, Path, str]] = field(default_factory=list)
    references: list[tuple[str, Path, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    chars: int = 0

    def render(self) -> str:
        parts: list[str] = []
        if self.instructions:
            parts.append("## Standing instructions (local files, injected each request)")
            for label, path, body in self.instructions:
                parts.append(f"### {label} ({path})")
                parts.append(body.rstrip())
        if self.references:
            parts.append("## Pinned reference material (local files, injected each request)")
            parts.append("Treat these as ground truth for this repo. Prefer them over memory.")
            for label, path, body in self.references:
                parts.append(f"### {label} ({path})")
                parts.append(body.rstrip())
        return "\n\n".join(parts)

    def summary(self) -> str:
        lines = [
            "Context is assembled on THIS machine and sent in the API JSON.",
            "The GPU host does not keep a copy after the request finishes.",
            "",
        ]
        if self.instructions:
            lines.append("Instructions:")
            for label, path, body in self.instructions:
                lines.append(f"  - {label}: {path} ({len(body)} chars)")
        else:
            lines.append("Instructions: none (add ~/.qwen-agent/instructions.md or AGENTS.md in the repo)")
        if self.references:
            lines.append("References:")
            for label, path, body in self.references:
                lines.append(f"  - {label}: {path} ({len(body)} chars)")
        else:
            lines.append("References: none (put files in .qwen-agent/ref/ or /ref path)")
        if self.skipped:
            lines.append("Skipped (budget or missing):")
            for s in self.skipped:
                lines.append(f"  - {s}")
        lines.append(f"Total injected: {self.chars} chars (cap {TOTAL_CAP})")
        return "\n".join(lines)


def build_pack(
    cwd: Path,
    cfg: Config,
    extra_pins: list[Path] | None = None,
) -> ContextPack:
    pack = ContextPack()
    budget = TOTAL_CAP

    def absorb(label: str, path: Path, bucket: list[tuple[str, Path, str]]) -> None:
        nonlocal budget
        resolved = _safe_file(path)
        if resolved is None:
            pack.skipped.append(f"missing {path}")
            return
        if any(p == resolved for _, p, _ in pack.instructions + pack.references):
            return
        try:
            body = _read_capped(resolved)
        except OSError as exc:
            pack.skipped.append(f"{path}: {exc}")
            return
        if len(body) > budget:
            pack.skipped.append(f"{path} ({len(body)} chars) exceeds remaining budget {budget}")
            return
        bucket.append((label, resolved, body))
        budget -= len(body)
        pack.chars += len(body)

    global_instr = HOME / "instructions.md"
    absorb("user instructions", global_instr, pack.instructions)

    for name in PROJECT_INSTRUCTION_NAMES:
        candidate = cwd / name
        if candidate.is_file():
            absorb(f"project {name}", candidate, pack.instructions)
            break

    for rel in cfg.context.files:
        rel = str(rel).strip()
        if not rel:
            continue
        try:
            path = resolve_in_cwd(rel, cwd)
        except ToolError:
            raw = Path(rel).expanduser()
            path = raw if raw.is_absolute() else cwd / raw
        absorb(f"config {rel}", path, pack.references)

    ref_dir = cwd / ".qwen-agent" / "ref"
    if ref_dir.is_dir():
        for path in sorted(p for p in ref_dir.rglob("*") if p.is_file()):
            absorb(f"ref/{path.relative_to(ref_dir)}", path, pack.references)

    for pin in extra_pins or []:
        absorb(f"pin {pin}", pin, pack.references)

    return pack


def system_prompt(cwd: Path, pack: ContextPack) -> str:
    import platform
    from datetime import datetime

    posix = cwd.as_posix()
    wsl_note = ""
    if posix.startswith("/mnt/c") or posix.startswith("/mnt/d"):
        wsl_note = "\nWarning: cwd is a Windows mount (/mnt/c). Prefer $HOME/src for speed.\n"

    injected = pack.render()
    extra = f"\n\n{injected}\n" if injected else ""
    return f"""You are a coding agent running inside Linux/WSL on the user's machine.
cwd: {cwd}
os: {platform.system()} {platform.release()}
date: {datetime.now().strftime("%Y-%m-%d")}
{wsl_note}
Rules:
- Use tools. Do not invent file contents.
- Commands run via bash on Linux/WSL. Never PowerShell or cmd.exe.
- Match existing code style. Small diffs. No drive-by refactors.
- After edits, summarize files changed.
- The model is remote; the repo and session stay local. Do not assume you can write to the GPU host.
- Standing instructions and pinned references below are ground truth. Prefer them over guessing.
- Stay inside cwd. If a task is unclear, ask.
{extra}"""
