"""lg2 wrapper.

a-Shell has no `git` binary: version control is `lg2`, the libgit2 example
CLI. Its subcommands are add, apply, blame, branch, checkout, clone, commit,
config, describe, diff, fetch, init, log, merge, pull, push, rebase, remote,
reset, rm, stash, status, submodule, tag, worktree.

`lg2 config` has NO --global flag: `lg2 config <KEY> [<VALUE>]` reads/writes
the repository's own config. That is what we want — identity is stored inside
each workspace's .git/config and nothing is written outside the sandbox.
"""

from __future__ import annotations
import shell_exec
from pathlib import Path

_IDENTITY_NAME = "ashellagent"
_IDENTITY_EMAIL = "agent@local"

_backend = None  # resolved once: "lg2", "git", or None


def backend() -> str | None:
    """Which VCS binary to use: lg2 (a-Shell's) or git (desktop fallback).

    lg2 is always preferred — a-Shell has no `git` binary and this harness is
    written for it. `git` is only used when lg2 is absent, which is the case
    when the same workspace is opened on a desktop clone.
    """
    global _backend
    if _backend is None:
        if shell_exec.run("lg2 version", cwd=Path.cwd(), timeout=30).code == 0:
            _backend = "lg2"
        elif shell_exec.run("git --version", cwd=Path.cwd(), timeout=30).code == 0:
            _backend = "git"
        else:
            _backend = ""
    return _backend or None


def is_repo(root: Path) -> bool:
    return (Path(root) / ".git").exists()


def _vcs(args: str, root: Path) -> shell_exec.Result:
    binary = backend()
    if binary is None:
        return shell_exec.Result(
            "no version-control binary found (need lg2 on a-Shell, or git)", 1)
    return shell_exec.run(f"{binary} {args}", cwd=Path(root), timeout=60)


def init(root: Path) -> shell_exec.Result:
    return _vcs("init", root)


def ensure_identity(root: Path) -> str:
    """Set user.name / user.email in the repo config. Idempotent.

    `lg2 config <KEY>` prints the value and exits 0 when set; a missing key
    is an error, which is what we probe for.
    """
    notes = []
    for key, value in (("user.name", _IDENTITY_NAME),
                       ("user.email", _IDENTITY_EMAIL)):
        probe = _vcs(f"config {key}", root)
        if probe.code != 0:
            set_result = _vcs(f'config {key} "{value}"', root)
            if set_result.code != 0:
                notes.append(f"could not set {key}: {set_result.output}")
    return "; ".join(notes)


def checkpoint(root: Path, message: str) -> shell_exec.Result:
    """Stage everything and commit. Creates the repo if there is none.

    Returns a Result whose output is either a confirmation or the raw lg2
    error, so the caller (agent or human) can act on it.
    """
    root = Path(root)
    message = (message or "checkpoint").strip().replace('"', "'")[:200]
    if not message:
        message = "checkpoint"

    if not is_repo(root):
        created = init(root)
        if created.code != 0:
            return shell_exec.Result(
                f"lg2 init failed: {created.output}", created.code)

    identity_problem = ensure_identity(root)
    staged = _vcs("add -A", root)
    if staged.code != 0:
        return shell_exec.Result(
            f"lg2 add failed: {staged.output}", staged.code)

    committed = _vcs(f'commit -m "{message}"', root)
    if committed.code != 0:
        # A clean tree is not a failure — the work is already committed.
        lowered = committed.output.lower()
        if "nothing to commit" in lowered or "working tree clean" in lowered:
            return shell_exec.Result("Already committed (working tree clean).", 0)
        return shell_exec.Result(
            f"lg2 commit failed: {committed.output}"
            + (f" [{identity_problem}]" if identity_problem else ""),
            committed.code,
        )

    note = f" [{identity_problem}]" if identity_problem else ""
    return shell_exec.Result(f"Committed: {message}{note}", 0)


def status(root: Path) -> shell_exec.Result:
    return _vcs("status", root)


def diff(root: Path) -> shell_exec.Result:
    return _vcs("diff", root)


def log(root: Path, count: int = 5) -> shell_exec.Result:
    return _vcs(f"log --oneline -{int(count)}", root) if count > 0 \
        else _vcs("log", root)
