from __future__ import annotations

from dataclasses import dataclass, field


DESTRUCTIVE_TOOLS = frozenset({"write_file", "edit_file", "bash"})
READ_TOOLS = frozenset({"read_file", "grep", "glob"})


@dataclass
class PermissionBroker:
    auto_approve_read: bool = True
    always_session: set[str] = field(default_factory=set)

    def needs_prompt(self, tool_name: str) -> bool:
        if tool_name in self.always_session:
            return False
        if tool_name in READ_TOOLS and self.auto_approve_read:
            return False
        if tool_name in DESTRUCTIVE_TOOLS:
            return True
        return False

    def remember_always(self, tool_name: str) -> None:
        self.always_session.add(tool_name)
