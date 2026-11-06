"""The agent's tools.

All paths are workspace-relative and jailed: symlinks that resolve outside
the workspace are rejected on read/write/delete and skipped on listing and
search, so a repo can never pull the agent outside its sandbox.

`run` reaches the full a-Shell command set through ios_system (rg, jq, diff,
make, node, python3, ffmpeg, magick, ssh, ...) plus `lg2` for version
control — a-Shell has no `git` binary.
"""

import fnmatch
import json
from pathlib import Path

import shell_exec
import vcs

MAX_FILE_BYTES = 400_000
MAX_TOOL_OUTPUT = 30_000

WORKSPACE: Path = None


def set_workspace(path: Path) -> None:
    global WORKSPACE
    WORKSPACE = Path(path).resolve()


def truncate(text, limit=MAX_TOOL_OUTPUT):
    """Keep head and tail of oversized output, marking the elision."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    elided = len(text) - head - tail
    return text[:head] + f"\n\n... [{elided} characters elided] ...\n\n" + text[-tail:]


def rel(path) -> str:
    try:
        return str(Path(path).relative_to(WORKSPACE))
    except ValueError:
        return str(path)


def workspace_path(user_path, *, must_exist=False) -> Path:
    """Resolve a workspace-relative path and reject anything outside it."""
    if not isinstance(user_path, str) or not user_path.strip():
        raise ValueError("A non-empty relative path is required.")

    candidate = Path(user_path.strip()).expanduser()
    if candidate.is_absolute():
        raise ValueError(
            f"Absolute paths are not allowed; use a path relative to the "
            f"workspace (got {user_path!r})."
        )

    resolved = (WORKSPACE / candidate).resolve()
    try:
        resolved.relative_to(WORKSPACE)
    except ValueError:
        raise ValueError(
            f"Path escapes the workspace: {user_path!r}. "
            f"The workspace is {WORKSPACE}."
        )

    if must_exist and not resolved.exists():
        raise ValueError(f"No such file or directory: {user_path}")

    return resolved


def _inside_workspace(item) -> bool:
    """True if the item's real target still lives inside the workspace.

    Guards against symlinks that point outside the sandbox.
    """
    try:
        item.resolve().relative_to(WORKSPACE)
        return True
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

def tool_read_file(path, offset=None, limit=None):
    target = workspace_path(path, must_exist=True)
    if not target.is_file():
        raise ValueError(f"Not a regular file: {path}")

    size = target.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"{rel(target)} is {size} bytes, over the {MAX_FILE_BYTES}-byte "
            "read limit. Use search or run (head/tail) to inspect it instead."
        )

    raw = target.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return (
            f"{rel(target)} is not UTF-8 text ({size} bytes). "
            f"First 64 bytes (hex): {raw[:64].hex()}"
        )

    numbered = [f"{i + 1}\t{line}" for i, line in enumerate(text.splitlines())]

    if offset is not None or limit is not None:
        start = max(0, int(offset or 1) - 1)
        end = start + int(limit) if limit else len(numbered)
        body = "\n".join(numbered[start:end])
        shown = f"{start + 1}-{min(end, len(numbered))} of {len(numbered)} lines"
        return truncate(f"{rel(target)} (lines {shown})\n{body}")

    return truncate(
        f"{rel(target)} ({len(numbered)} lines, {size} bytes)\n"
        + "\n".join(numbered)
    )


def tool_write_file(path, content):
    if not isinstance(content, str):
        raise ValueError("File content must be a string.")
    target = workspace_path(path)
    if target.exists() and target.is_dir():
        raise ValueError(f"{rel(target)} is a directory.")
    data = content.encode("utf-8")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(
            f"Content is {len(data)} bytes, over the {MAX_FILE_BYTES}-byte "
            "write limit. Write it in pieces with edit_file instead."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {rel(target)} ({len(data)} bytes)."


def tool_edit_file(path, old_string, new_string, replace_all=False):
    if not isinstance(old_string, str) or not old_string:
        raise ValueError("old_string must be a non-empty string.")
    if not isinstance(new_string, str):
        raise ValueError("new_string must be a string.")
    target = workspace_path(path, must_exist=True)
    if not target.is_file():
        raise ValueError(f"Not a regular file: {path}")

    text = target.read_text(encoding="utf-8")
    occurrences = text.count(old_string)
    if occurrences == 0:
        raise ValueError(
            f"old_string not found in {rel(target)}. Read the file first and "
            "copy the text exactly (including indentation and trailing "
            "whitespace)."
        )
    if occurrences > 1 and not replace_all:
        raise ValueError(
            f"old_string appears {occurrences} times in {rel(target)}. "
            "Include more surrounding context to make it unique, or pass "
            "replace_all=true."
        )

    updated = (
        text.replace(old_string, new_string)
        if replace_all
        else text.replace(old_string, new_string, 1)
    )
    target.write_text(updated, encoding="utf-8")
    return (
        f"Edited {rel(target)}: {occurrences if replace_all else 1} "
        "replacement(s)."
    )


def tool_list_dir(path=".", recursive=False):
    target = workspace_path(path, must_exist=True)
    if not target.is_dir():
        raise ValueError(f"Not a directory: {path}")

    rows = []
    if recursive:
        for item in sorted(target.rglob("*")):
            if not _inside_workspace(item):
                continue
            if item.is_dir():
                rows.append(f"dir  {rel(item)}")
            else:
                rows.append(f"file {rel(item)} ({item.stat().st_size} bytes)")
    else:
        for item in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            try:
                resolved = item.resolve()
                resolved.relative_to(WORKSPACE)
            except (OSError, ValueError):
                rows.append(f"link {rel(item)} -> (outside workspace, hidden)")
                continue
            if resolved.is_dir():
                rows.append(f"dir  {rel(item)}")
            else:
                rows.append(f"file {rel(item)} ({resolved.stat().st_size} bytes)")

    if not rows:
        return f"{rel(target)} is empty."
    return truncate("\n".join(rows))


def tool_search(query, path=".", pattern="content", max_results=200):
    target = workspace_path(path, must_exist=True)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    hits = []
    if pattern == "name":
        for item in sorted(target.rglob("*")):
            if not _inside_workspace(item):
                continue
            if fnmatch.fnmatch(item.name, query):
                hits.append(rel(item))
            if len(hits) >= max_results:
                break
    else:
        needle = query.lower()
        for item in sorted(target.rglob("*")):
            if not item.is_file() or not _inside_workspace(item):
                continue
            try:
                if item.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = item.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if needle in line.lower():
                    hits.append(f"{rel(item)}:{number}: {line.strip()[:200]}")
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                break

    if not hits:
        return f"No matches for {query!r} under {rel(target)}."
    return truncate("\n".join(hits))


def tool_run(command, timeout=120):
    """Run a command in the workspace via ios_system / sh."""
    result = shell_exec.run(command, cwd=WORKSPACE, timeout=timeout)
    return truncate(f"{result.output}\n[exit code: {result.code}]")


def tool_mkdir(path):
    target = workspace_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return f"Created directory {rel(target)}."


def tool_delete_file(path):
    target = workspace_path(path, must_exist=True)
    if target == WORKSPACE:
        raise ValueError("Refusing to delete the workspace root.")
    if target.is_dir():
        if any(target.iterdir()):
            raise ValueError(f"{rel(target)} is not empty. Remove its contents first.")
        target.rmdir()
        return f"Removed directory {rel(target)}."
    target.unlink()
    return f"Deleted {rel(target)}."


def tool_checkpoint(message):
    """Commit workspace changes with lg2 (a-Shell's git)."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("A non-empty commit message is required.")
    result = vcs.checkpoint(WORKSPACE, message)
    return result.output if result.code == 0 else f"checkpoint failed: {result.output}"


TOOL_FUNCTIONS = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_dir": tool_list_dir,
    "search": tool_search,
    "run": tool_run,
    "mkdir": tool_mkdir,
    "delete_file": tool_delete_file,
    "checkpoint": tool_checkpoint,
}


def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_SCHEMAS = [
    _schema(
        "read_file", "Read a UTF-8 text file with line numbers.",
        {
            "path": {"type": "string", "description": "Path relative to workspace."},
            "offset": {"type": "integer", "description": "First line to return (1-based)."},
            "limit": {"type": "integer", "description": "Maximum number of lines."},
        },
        ["path"],
    ),
    _schema(
        "write_file", "Create or overwrite a file with complete content.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _schema(
        "edit_file",
        "Replace an exact string in an existing file. Preferred over "
        "write_file for modifying existing code: only the changed text is "
        "sent, not the whole file.",
        {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        ["path", "old_string", "new_string"],
    ),
    _schema(
        "list_dir", "List files and directories with sizes.",
        {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
        [],
    ),
    _schema(
        "search",
        "Search text in file contents, or filenames by glob. Use this instead "
        "of reading every file.",
        {
            "query": {"type": "string"},
            "path": {"type": "string"},
            "pattern": {"type": "string", "enum": ["content", "name"]},
        },
        ["query"],
    ),
    _schema(
        "run",
        "Run a shell command in the workspace. Full a-Shell command set: rg, "
        "jq, diff, make, node, python3, ffmpeg, ssh, and lg2 for git. "
        "Compound commands (&&, ||, ;, |) are run through sh. On a-Shell the "
        "timeout is advisory only.",
        {"command": {"type": "string"}, "timeout": {"type": "integer"}},
        ["command"],
    ),
    _schema(
        "mkdir", "Create a directory, including parents.",
        {"path": {"type": "string"}},
        ["path"],
    ),
    _schema(
        "delete_file", "Delete a file or an empty directory.",
        {"path": {"type": "string"}},
        ["path"],
    ),
    _schema(
        "checkpoint",
        "Commit all workspace changes with lg2 (a-Shell's git, backed by "
        "libgit2). Use after completing a unit of work so it can be rolled "
        "back. Fails cleanly if lg2 is unavailable.",
        {"message": {"type": "string", "description": "Commit message."}},
        ["message"],
    ),
]
