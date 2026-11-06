"""The agent loop: turn execution, context compaction, checkpointing.

Design notes carried over from testing against the live endpoint:

  - compaction keeps the system prompt, the opening request and a verbatim
    tail, aligned to whole assistant/tool groups so the API never sees an
    orphaned tool result, and never grows the transcript;
  - a model response with empty content and no tool calls is an error, not a
    valid end of turn — treating it as valid once made the agent stop
    silently after a single tool call;
  - after every step the transcript is saved, because a-Shell terminates
    backgrounded apps without warning.
"""

import json

import llm
import sessions
import tools
import vcs
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS, truncate

MAX_STEPS = 40
CONTEXT_BUDGET = 120_000
TAIL_MESSAGES = 12

SYSTEM_PROMPT = """\
You are a coding agent working directly on files in a workspace directory.

Available tools:
- read_file(path, offset?, limit?)  read a text file with line numbers
- write_file(path, content)          create or overwrite a file
- edit_file(path, old_string, new_string, replace_all?)  replace exact text
- list_dir(path?, recursive?)        list files with sizes
- search(query, path?, pattern?)     grep file contents or glob filenames
- run(command, timeout?)             run a shell command in the workspace
- mkdir(path)                        create a directory
- delete_file(path)                  delete a file or empty directory
- checkpoint(message)                commit all changes with lg2

Rules:
- Paths are always relative to the workspace root. Absolute paths fail.
- Prefer search over reading whole files; prefer edit_file over write_file
  when changing existing files.
- run reaches the full command set, including `lg2` — this environment is
  a-Shell on iOS, which has no `git` binary; lg2 is its libgit2-based
  equivalent (status, diff, add, commit, log, branch, ...).
- After editing code, run it or its tests to verify the change works.
- Call checkpoint after completing a unit of work.
- Work until the task is actually complete, then summarize what changed.
- Be concise. Your final message is read on a phone screen.
""".strip()


def approx_tokens(messages) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part.get("text", ""))) // 4
        for call in message.get("tool_calls") or []:
            total += len(str(call.get("function", {}).get("arguments", ""))) // 4
    return total


def compact(messages):
    """Drop the middle of the transcript when it grows past the budget."""
    if approx_tokens(messages) <= CONTEXT_BUDGET:
        return messages

    head = messages[:2]
    rest = messages[2:]
    tail = rest[-TAIL_MESSAGES:]

    # Never start the tail on a tool result whose assistant call was cut off.
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    # Never end the tail on an assistant call whose results were cut off.
    while tail and tail[-1].get("role") == "assistant" and tail[-1].get("tool_calls"):
        tail = tail[:-1]

    dropped = len(rest) - len(tail)
    compacted = list(head)
    if dropped > 0:
        compacted.append({
            "role": "user",
            "content": (
                f"[{dropped} earlier messages elided to stay within the "
                "context budget. Re-read files or re-run search if you need "
                "that detail again.]"
            ),
        })
    compacted.extend(tail)
    return compacted


class AgentSession:
    """One workspace + transcript + settings."""

    def __init__(self, workspace, *, verbose=True, auto_checkpoint=True,
                 messages=None):
        tools.set_workspace(workspace)
        self.workspace = tools.WORKSPACE
        self.verbose = verbose
        self.auto_checkpoint = auto_checkpoint
        self.messages = messages if messages is not None \
            else [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_usage = {}

    # ------------------------------------------------------------------

    def turn(self, user_prompt: str):
        """Run one user turn to completion, dispatching tool calls."""
        self.messages.append({"role": "user", "content": user_prompt})
        try:
            return self._run_loop()
        finally:
            self.save()

    def resume(self):
        """Re-drive the loop for a turn that stopped midway.

        The transcript must already end with a user request or a tool result;
        nothing new is appended.
        """
        if not self.turn_is_open():
            raise RuntimeError("Nothing to resume: the last turn completed.")
        return self._run_loop()

    def turn_is_open(self) -> bool:
        return bool(self.messages) and \
            self.messages[-1].get("role") in ("user", "tool")

    # ------------------------------------------------------------------

    def _run_loop(self):
        for _ in range(MAX_STEPS):
            self.messages = compact(self.messages)
            message, usage = llm.call_model(self.messages, TOOL_SCHEMAS)
            self.last_usage = usage

            entry = {"role": "assistant", "content": message.get("content") or ""}
            if message.get("tool_calls"):
                entry["tool_calls"] = message["tool_calls"]
            self.messages.append(entry)
            self.save()

            if message.get("content"):
                print(f"\nAssistant: {message['content']}", flush=True)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                self._checkpoint()
                return self.messages

            for call in tool_calls:
                self._dispatch(call)

        print(f"\n[stopped after {MAX_STEPS} tool steps]", flush=True)
        return self.messages

    def _dispatch(self, call):
        function = call.get("function", {})
        name = function.get("name", "")
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        if self.verbose:
            print(f"\n[tool] {name} {json.dumps(arguments)[:300]}", flush=True)

        handler = TOOL_FUNCTIONS.get(name)
        if handler is None:
            result = (f"Unknown tool {name!r}. "
                      f"Available: {', '.join(TOOL_FUNCTIONS)}.")
        else:
            try:
                result = handler(**arguments)
            except TypeError as error:
                result = f"Bad arguments for {name}: {error}"
            except Exception as error:
                result = f"{type(error).__name__}: {error}"

        if self.verbose:
            print(f"[result] {result if len(result) <= 800 else result[:800] + ' ...'}",
                  flush=True)

        self.messages.append({
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "name": name,
            "content": truncate(result, tools.MAX_TOOL_OUTPUT),
        })
        self.save()

    def _checkpoint(self):
        """Commit completed work with lg2 so it can be rolled back."""
        if not self.auto_checkpoint:
            return
        if not vcs.is_repo(self.workspace):
            return
        last_user = next((m["content"] for m in reversed(self.messages)
                          if m.get("role") == "user"
                          and not str(m.get("content", "")).startswith("[")),
                         "checkpoint")
        summary = str(last_user).strip().splitlines()[0][:120]
        result = vcs.checkpoint(self.workspace, summary)
        if result.code != 0 and self.verbose:
            print(f"[checkpoint failed] {result.output[:300]}", flush=True)

    # ------------------------------------------------------------------

    def save(self):
        try:
            sessions.save(self.messages, self.workspace)
        except OSError:
            pass  # persistence is best-effort; never break a turn over it
