from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qwen-agent",
        description="Local coding-agent TUI for a remote Qwen OpenAI-compatible API",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="tui",
        choices=["tui", "doctor"],
        help="tui (default) or doctor",
    )
    args = parser.parse_args(argv)

    if not sys.platform.startswith("linux"):
        print("qwen-agent runs on Linux or WSL only. Start WSL: wsl.exe", file=sys.stderr)
        return 2

    if args.command == "doctor":
        from qwen_agent.doctor import main as doctor_main

        return doctor_main()

    from qwen_agent.app import run_tui

    return run_tui()


if __name__ == "__main__":
    raise SystemExit(main())
