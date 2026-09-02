from __future__ import annotations

import asyncio
import io
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_toolkit import Application
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, FormattedText, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from rich.console import Console
from rich.markdown import Markdown

from qwen_agent import __version__
from qwen_agent.api import QwenClient, StreamDone, StreamError, TextDelta, ToolRequest, host_of
from qwen_agent.config import Config, ensure_home, load_config
from qwen_agent.context import ContextPack, build_pack, system_prompt
from qwen_agent.doctor import collect_report
from qwen_agent.permissions import PermissionBroker
from qwen_agent.session import SessionLog, trim_messages
from qwen_agent.tools import ToolError, resolve_in_cwd, run_tool, summary as tool_summary

MAX_TOOL_STEPS = 20
HELP = """\
/help        this text
/clear       new local session (history stays on this machine)
/cwd         show working directory
/model       show remote model
/doctor      connectivity + environment checks
/context     what instructions/refs will be sent this turn
/ref PATH    pin a local file into every request (until /unref)
/unref PATH  drop a pin
/quit        exit

Enter send · Ctrl+C cancel in-flight (twice to quit) · Ctrl+D quit

Repos, sessions, instructions, and pins live on YOUR machine.
Each request JSON includes standing instructions + pinned refs + recent chat.
The GPU host infers and discards — it does not store your files.
"""


def _git_branch(cwd: Path) -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if p.returncode != 0:
        return ""
    return p.stdout.strip()


def _render_md(text: str, width: int) -> FormattedText:
    if not text:
        return FormattedText([])
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        color_system="truecolor",
        width=max(40, width),
        highlight=False,
    )
    console.print(Markdown(text))
    return to_formatted_text(ANSI(buf.getvalue().rstrip("\n")))


@dataclass
class Entry:
    kind: str
    text: str
    meta: str = ""
    live: bool = False


@dataclass
class TuiState:
    cfg: Config
    cwd: Path
    entries: list[Entry] = field(default_factory=list)
    busy: bool = False
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    perm_future: asyncio.Future[str] | None = None
    perm_label: str = ""
    status_note: str = ""
    always_quit_c: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    session: SessionLog | None = None
    broker: PermissionBroker = field(default_factory=PermissionBroker)
    client: QwenClient | None = None
    input: TextArea | None = None
    scroll: int = 0
    pins: list[Path] = field(default_factory=list)
    pack: ContextPack | None = None


class AgentTui:
    def __init__(self, cfg: Config, cwd: Path) -> None:
        self.s = TuiState(cfg=cfg, cwd=cwd)
        self.s.broker = PermissionBroker(auto_approve_read=cfg.safety.auto_approve_read)
        self.s.session = SessionLog(cwd)
        self.s.messages = [{"role": "system", "content": ""}]
        self._refresh_system()
        self.s.client = QwenClient(cfg)
        self.s.input = TextArea(
            prompt=FormattedText([("class:prompt", "▸ ")]),
            multiline=True,
            wrap_lines=True,
            scrollbar=False,
        )
        self.scrollback = FormattedTextControl(
            self._scrollback_text, focusable=False, show_cursor=False
        )
        self.status_ctl = FormattedTextControl(self._status_text, focusable=False)
        self.kb = self._bindings()
        input_kb = KeyBindings()

        @input_kb.add("enter")
        def _input_enter(event) -> None:  # noqa: ANN001
            if self.s.perm_future and not self.s.perm_future.done():
                return
            if self.s.busy:
                return
            text = self.s.input.text
            if not text.strip():
                return
            self.s.input.text = ""
            self.s.always_quit_c = 0
            event.app.create_background_task(self.handle_submit(text))

        @input_kb.add("escape", "enter")
        def _input_nl(event) -> None:  # noqa: ANN001
            event.current_buffer.insert_text("\n")

        self.s.input.control.key_bindings = input_kb
        self.app = Application(
            layout=Layout(
                HSplit(
                    [
                        Window(self.status_ctl, height=1, style="class:status"),
                        Window(self.scrollback, wrap_lines=True, allow_scroll_beyond_bottom=True),
                        Window(height=1, char="─", style="class:sep"),
                        self.s.input,
                    ]
                ),
                focused_element=self.s.input,
            ),
            key_bindings=self.kb,
            style=Style.from_dict(
                {
                    "status": "bg:#1c1c1c #d0d0d0",
                    "sep": "#444444",
                    "prompt": "bold #7aa2f7",
                    "user": "bold #9ece6a",
                    "assistant": "#c0caf5",
                    "tool": "#e0af68",
                    "toolok": "#9ece6a",
                    "err": "bold #f7768e",
                    "meta": "#565f89",
                    "perm": "bold #bb9af7",
                }
            ),
            full_screen=True,
            mouse_support=False,
        )
        pack = self.s.pack
        n_ctx = 0 if pack is None else len(pack.instructions) + len(pack.references)
        self._add_system(
            f"qwen-agent {__version__}  ·  local cwd {cwd}  ·  remote {host_of(cfg.api.base_url)}\n"
            f"sessions: {self.s.session.path.parent}  ·  {n_ctx} instruction/ref file(s)  ·  /help /context"
        )

    def _status_text(self) -> FormattedText:
        cfg = self.s.cfg
        branch = _git_branch(self.s.cwd)
        git = f"  {branch}" if branch else ""
        note = f"  {self.s.status_note}" if self.s.status_note else ""
        perm = f"  {self.s.perm_label}" if self.s.perm_label else ""
        cwd = str(self.s.cwd)
        if len(cwd) > 40:
            cwd = "…" + cwd[-39:]
        return FormattedText(
            [
                (
                    "class:status",
                    f" {cfg.api.model}  {host_of(cfg.api.base_url)}  {cwd}{git}{note}{perm} ",
                )
            ]
        )

    def _scrollback_text(self) -> FormattedText:
        width = 80
        try:
            width = max(40, get_app().output.get_size().columns - 2)
        except Exception:
            pass
        parts: list[tuple[str, str]] = []
        for e in self.s.entries:
            if e.kind == "user":
                parts.append(("class:user", "you\n"))
                parts.append(("", e.text.rstrip() + "\n\n"))
            elif e.kind == "assistant":
                parts.append(("class:assistant", "qwen\n"))
                if e.live:
                    parts.append(("class:assistant", e.text + "▊\n\n"))
                else:
                    md = _render_md(e.text, width)
                    parts.extend(list(md))
                    parts.append(("", "\n\n"))
            elif e.kind == "tool":
                parts.append(("class:tool", f"⚙ {e.meta}\n"))
                if e.text:
                    parts.append(("class:meta", e.text.rstrip()[:2000] + "\n\n"))
                else:
                    parts.append(("", "\n"))
            elif e.kind == "err":
                parts.append(("class:err", e.text.rstrip() + "\n\n"))
            elif e.kind == "perm":
                parts.append(("class:perm", e.text.rstrip() + "\n\n"))
            else:
                parts.append(("class:meta", e.text.rstrip() + "\n\n"))
        return FormattedText(parts)

    def _add_system(self, text: str) -> None:
        self.s.entries.append(Entry("sys", text))

    def _refresh_system(self) -> None:
        pack = build_pack(self.s.cwd, self.s.cfg, self.s.pins)
        self.s.pack = pack
        prompt = system_prompt(self.s.cwd, pack)
        if self.s.messages and self.s.messages[0].get("role") == "system":
            self.s.messages[0]["content"] = prompt
        else:
            self.s.messages.insert(0, {"role": "system", "content": prompt})

    def _bindings(self) -> KeyBindings:
        kb = KeyBindings()
        state = self.s

        @kb.add("c-c")
        def _cc(event) -> None:  # noqa: ANN001
            if state.perm_future and not state.perm_future.done():
                state.perm_future.set_result("n")
                return
            if state.busy:
                state.cancel.set()
                state.status_note = "cancelling"
                return
            state.always_quit_c += 1
            if state.always_quit_c >= 2:
                event.app.exit()
            else:
                state.status_note = "Ctrl+C again to quit"
                event.app.invalidate()

        @kb.add("c-d")
        def _cd(event) -> None:  # noqa: ANN001
            event.app.exit()

        @kb.add("pageup")
        def _pu(event) -> None:  # noqa: ANN001
            event.app.layout.focus(self.scrollback)

        awaiting = Condition(lambda: state.perm_future is not None and not state.perm_future.done())

        def _if_perm(letter: str):
            def _inner(event) -> None:  # noqa: ANN001
                if state.perm_future and not state.perm_future.done():
                    state.perm_future.set_result(letter)
                    event.app.layout.focus(state.input)

            return _inner

        kb.add("y", filter=awaiting)(_if_perm("y"))
        kb.add("a", filter=awaiting)(_if_perm("a"))
        kb.add("n", filter=awaiting)(_if_perm("n"))
        kb.add("Y", filter=awaiting)(_if_perm("y"))
        kb.add("A", filter=awaiting)(_if_perm("a"))
        kb.add("N", filter=awaiting)(_if_perm("n"))

        @kb.add("enter", eager=True)
        def _enter(event) -> None:  # noqa: ANN001
            if state.perm_future and not state.perm_future.done():
                return
            if state.busy:
                return
            text = state.input.text
            if not text.strip():
                return
            state.input.text = ""
            state.always_quit_c = 0
            event.app.create_background_task(self.handle_submit(text))

        @kb.add("escape", "enter")
        def _nl(event) -> None:  # noqa: ANN001
            event.current_buffer.insert_text("\n")

        return kb

    async def ask_perm(self, label: str) -> str:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self.s.perm_future = fut
        self.s.perm_label = "[y] once  [a] always  [n] skip"
        self.s.entries.append(Entry("perm", f"Allow {label}?"))
        self.app.invalidate()
        try:
            return await fut
        finally:
            self.s.perm_future = None
            self.s.perm_label = ""
            self.app.invalidate()

    async def handle_submit(self, text: str) -> None:
        raw = text.strip()
        if raw.startswith("/"):
            await self._slash(raw)
            self.app.invalidate()
            return
        self.s.entries.append(Entry("user", raw))
        self.s.messages.append({"role": "user", "content": raw})
        self.s.session.append({"role": "user", "content": raw})
        await self._agent_loop()

    async def _slash(self, raw: str) -> None:
        cmd, _, rest = raw.partition(" ")
        cmd = cmd.lower()
        if cmd in {"/quit", "/exit"}:
            self.app.exit()
        elif cmd == "/help":
            self._add_system(HELP)
        elif cmd == "/clear":
            self.s.messages = [{"role": "system", "content": ""}]
            self._refresh_system()
            self.s.session.rotate(self.s.cwd)
            self.s.entries.clear()
            self._add_system(f"new local session → {self.s.session.path}")
        elif cmd == "/cwd":
            self._add_system(str(self.s.cwd))
        elif cmd == "/model":
            self._add_system(f"{self.s.cfg.api.model}  {self.s.cfg.api.base_url}")
        elif cmd == "/context":
            self._refresh_system()
            assert self.s.pack is not None
            self._add_system(self.s.pack.summary())
        elif cmd == "/ref":
            path = rest.strip()
            if not path:
                self._add_system("usage: /ref path/to/file")
                return
            try:
                resolved = resolve_in_cwd(path, self.s.cwd)
            except ToolError as exc:
                self._add_system(str(exc))
                return
            if not resolved.is_file():
                self._add_system(f"not a file: {resolved}")
                return
            if resolved not in self.s.pins:
                self.s.pins.append(resolved)
            self._refresh_system()
            self._add_system(f"pinned {resolved} — sent with every request until /unref")
        elif cmd == "/unref":
            path = rest.strip()
            if not path:
                self.s.pins.clear()
                self._refresh_system()
                self._add_system("cleared session pins")
                return
            try:
                resolved = resolve_in_cwd(path, self.s.cwd)
            except ToolError:
                resolved = (self.s.cwd / path).resolve()
            self.s.pins = [p for p in self.s.pins if p != resolved]
            self._refresh_system()
            self._add_system(f"unpinned {resolved}")
        elif cmd == "/doctor":
            report = await collect_report(self.s.cfg, self.s.cwd)
            self._add_system(report)
        else:
            self._add_system(f"unknown command {cmd}. /help")

    async def _agent_loop(self) -> None:
        assert self.s.client is not None
        self.s.busy = True
        self.s.cancel = asyncio.Event()
        self.s.status_note = "thinking"
        self._refresh_system()
        self.app.invalidate()
        try:
            for _step in range(MAX_TOOL_STEPS):
                if self.s.cancel.is_set():
                    self.s.entries.append(Entry("err", "cancelled"))
                    break
                assistant = Entry("assistant", "", live=True)
                self.s.entries.append(assistant)
                payload = trim_messages(self.s.messages)
                tool_calls: list[ToolRequest] = []
                finish = None
                err: str | None = None
                async for ev in self.s.client.stream(payload):
                    if self.s.cancel.is_set():
                        err = "cancelled"
                        break
                    if isinstance(ev, TextDelta):
                        assistant.text += ev.text
                        self.app.invalidate()
                    elif isinstance(ev, StreamError):
                        err = ev.message
                        break
                    elif isinstance(ev, StreamDone):
                        finish = ev.finish_reason
                        if ev.content and not assistant.text:
                            assistant.text = ev.content
                        tool_calls = ev.tool_calls
                assistant.live = False
                if err:
                    if not assistant.text:
                        self.s.entries.pop()
                    self.s.entries.append(Entry("err", err))
                    break
                if not tool_calls:
                    if assistant.text:
                        self.s.messages.append({"role": "assistant", "content": assistant.text})
                        self.s.session.append({"role": "assistant", "content": assistant.text})
                    elif not assistant.text:
                        self.s.entries.pop()
                    break
                api_tool_calls = []
                for tc in tool_calls:
                    api_tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments_raw or "{}",
                            },
                        }
                    )
                self.s.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant.text or None,
                        "tool_calls": api_tool_calls,
                    }
                )
                if not assistant.text:
                    # keep the row as a stub; replace with nothing visual
                    self.s.entries.pop()
                for tc in tool_calls:
                    allowed = True
                    label = tool_summary(tc.name, tc.arguments)
                    if self.s.broker.needs_prompt(tc.name):
                        decision = await self.ask_perm(label)
                        if decision == "a":
                            self.s.broker.remember_always(tc.name)
                        elif decision != "y":
                            allowed = False
                    card = Entry("tool", "", meta=label)
                    self.s.entries.append(card)
                    self.s.status_note = label[:40]
                    self.app.invalidate()
                    if not allowed:
                        result = "error: user declined this tool call"
                    else:
                        result = await run_tool(tc.name, tc.arguments, self.s.cwd)
                    card.text = result[:1500]
                    self.s.messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )
                    self.s.session.append(
                        {"role": "tool", "name": tc.name, "content": result[:4000]}
                    )
                    self.app.invalidate()
                if finish == "stop" and not tool_calls:
                    break
            else:
                self.s.entries.append(Entry("err", "stopped after 20 tool steps"))
        finally:
            self.s.busy = False
            self.s.status_note = ""
            self.app.invalidate()

    async def _shutdown(self) -> None:
        if self.s.client:
            await self.s.client.aclose()

    def run(self) -> int:
        try:
            self.app.run()
        finally:
            try:
                asyncio.run(self._shutdown())
            except RuntimeError:
                pass
        return 0


def run_tui() -> int:
    if not sys.platform.startswith("linux"):
        print("qwen-agent runs on Linux or WSL only. Start WSL: wsl.exe", file=sys.stderr)
        return 2
    ensure_home()
    cfg = load_config()
    cwd = Path.cwd().resolve()
    return AgentTui(cfg, cwd).run()
