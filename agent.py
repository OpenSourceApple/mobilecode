#!/usr/bin/env python3
"""ashellagent — a coding agent for a-Shell on iOS.

    python3 agent.py [WORKSPACE] [options]

Interactive REPL:
    python3 agent.py ~/Documents/myproject

One-shot (Shortcuts-friendly, prints the final answer and exits):
    python3 agent.py ~/Documents/myproject -p "add a LICENSE file"

Resume the saved session for a workspace after a-Shell killed the app:
    python3 agent.py ~/Documents/myproject --resume

Only the Python standard library is used, so there is nothing to pip-install
on the phone.
"""

import sys
from pathlib import Path

# Sibling imports work because sys.path[0] is the script directory, but be
# explicit in case the file is invoked through a symlink or from another cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

import core
import llm
import sessions
import tools
import vcs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description="A coding agent for a-Shell (iOS) and desktop terminals.",
    )
    parser.add_argument("workspace", nargs="?", default=None,
                        help="workspace directory (default: current directory)")
    parser.add_argument("-p", "--prompt", default=None,
                        help="run a single task and exit (one-shot mode)")
    parser.add_argument("--resume", action="store_true",
                        help="load the saved session for this workspace")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="disable automatic lg2 commits")
    parser.add_argument("--quiet", action="store_true",
                        help="hide tool calls and results")
    return parser.parse_args(argv)


def choose_workspace(raw: str) -> Path:
    path = Path(raw or ".").expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return path


def banner(session: core.AgentSession):
    print(f"\nWorkspace: {session.workspace}")
    print(f"Model: {llm.MODEL}  ({llm.BASE_URL})")
    vcs_note = "auto-checkpoint on" if session.auto_checkpoint else "checkpoints off"
    print(f"lg2: {vcs_note}")
    print("quit/exit to end · /clear resets · continue resumes a stalled turn")


def repl(session: core.AgentSession):
    banner(session)
    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not line:
            continue
        lowered = line.lower()
        if lowered in {"quit", "exit"}:
            print("Bye.")
            return
        if lowered == "/clear":
            session.messages = [{"role": "system",
                                 "content": core.SYSTEM_PROMPT}]
            session.save()
            print("Conversation cleared.")
            continue
        if lowered in {"continue", "retry", "c"} and session.turn_is_open():
            run_turn(session, resume=True)
            continue

        run_turn(session, user_prompt=line)


def run_turn(session: core.AgentSession, user_prompt=None, resume=False):
    try:
        if resume:
            session.resume()
        else:
            session.turn(user_prompt)
    except Exception as error:  # noqa: BLE001 — the REPL must survive anything
        print(f"\nRequest failed: {error}")
        if session.turn_is_open():
            print("(your request is still queued — type 'continue' to resume)")
        return
    usage = session.last_usage
    if usage:
        print(f"\n[prompt {usage.get('prompt_tokens', '?')} tokens, "
              f"completion {usage.get('completion_tokens', '?')}]")


def one_shot(session: core.AgentSession, prompt: str, quiet: bool = False) -> int:
    try:
        session.turn(prompt)
    except Exception as error:  # noqa: BLE001
        print(f"Request failed: {error}", file=sys.stderr)
        return 1
    # The loop already prints the final answer unless running quiet.
    if quiet:
        final = next((m.get("content") for m in reversed(session.messages)
                      if m.get("role") == "assistant" and m.get("content")), "")
        if final:
            print(final)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    workspace = choose_workspace(args.workspace)

    saved = sessions.load(workspace) if args.resume else None
    if args.resume and saved is None:
        print(f"No saved session for {workspace}; starting fresh.")

    session = core.AgentSession(
        workspace,
        verbose=not args.quiet,
        auto_checkpoint=not args.no_checkpoint,
        messages=saved,
    )

    if args.prompt is not None:
        return one_shot(session, args.prompt, quiet=args.quiet)
    repl(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
