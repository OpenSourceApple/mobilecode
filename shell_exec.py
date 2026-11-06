"""Shell command execution, tuned for a-Shell on iOS.

Two paths:

  iOS (a-Shell)   os.system() -> ios_system. CPython enables os.system on iOS
                  (#define HAVE_SYSTEM 1 under TARGET_OS_IPHONE) and a-Shell
                  routes it to the same command interpreter the terminal uses.
                  subprocess is NOT used there: ios_system's libc replacement
                  fakes fork() as `return ios_nextAvailablePid()` and there is
                  no /bin/sh, so subprocess.run(shell=True) can fail or hang.

  macOS / Linux   subprocess.run(shell=True), which gives real exit codes,
                  separated stderr and an enforceable timeout.

ios_system parses `<`, `>`, `2>`, `2>&1` and `&>` itself but does NOT handle
`&&`, `||` or `;` at the top level, so compound commands are wrapped in
`sh -c '...'` (sh and dash are both shipped with a-Shell).
"""

import os
import subprocess
import tempfile
from collections import namedtuple
from pathlib import Path

Result = namedtuple("Result", ["output", "code"])

_MAX_COMMAND_CHARS = 4000
_METACHARS = ("&&", "||", ";", "|", "$(", "`", ">", "<")


def is_ios() -> bool:
    """True when running inside a-Shell on iOS."""
    if os.environ.get("APPDIR"):
        return True
    return Path("/private/var/mobile/Containers").exists()


def _needs_sh_wrap(command: str) -> bool:
    """True when the command needs a real shell for its compound syntax."""
    return any(token in command for token in _METACHARS)


def _sh_quote(text: str) -> str:
    """Single-quote a string for `sh -c`."""
    return "'" + text.replace("'", "'\\''") + "'"


def _decode_exit_code(code) -> int:
    """Normalize an exit status to a plain exit code.

    POSIX os.system() returns the raw wait status, so a command exiting with
    7 comes back as 7 << 8 == 1792. a-Shell's ios_system returns the exit
    code directly. Decode only when the value could be a wait status, so both
    conventions end up as the same small integer.
    """
    try:
        code = int(code)
    except (TypeError, ValueError):
        return 1
    if code < 0:
        return code
    if code > 255:
        try:
            return os.waitstatus_to_exitcode(code)
        except (ValueError, AttributeError):
            return code >> 8
    return code


def _run_via_os_system(command: str, cwd: Path) -> Result:
    """Run through os.system(), capturing output with `> tmp 2>&1`.

    ios_system writes the output file itself, so no fork is involved.
    """
    handle, tmp = tempfile.mkstemp(prefix="ashellagent-", suffix=".out")
    os.close(handle)

    wrapped = command
    if _needs_sh_wrap(command):
        wrapped = f"sh -c {_sh_quote(command)}"

    previous = os.getcwd()
    try:
        os.chdir(str(cwd))
        code = os.system(f"{wrapped} > \"{tmp}\" 2>&1")
        try:
            output = Path(tmp).read_text(encoding="utf-8", errors="replace")
        except OSError:
            output = ""
    finally:
        try:
            os.chdir(previous)
        except OSError:
            pass
        Path(tmp).unlink(missing_ok=True)

    return Result(output.rstrip(), _decode_exit_code(code))


def _run_via_subprocess(command: str, cwd: Path, timeout: int) -> Result:
    completed = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (completed.stdout or "").rstrip()
    err = (completed.stderr or "").rstrip()
    if err:
        output = (output + "\n[stderr]\n" + err).strip()
    return Result(output, completed.returncode)


def run(command: str, cwd: Path, timeout: int = 120) -> Result:
    """Run `command` in `cwd` and return (combined output, exit code).

    On iOS the timeout parameter is advisory only: os.system() is synchronous
    and cannot be interrupted, so long-running commands must be self-limiting.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string.")
    if len(command) > _MAX_COMMAND_CHARS:
        raise ValueError(
            f"command is too long ({_MAX_COMMAND_CHARS} char limit); "
            "write it to a script file and run that instead."
        )

    if is_ios():
        result = _run_via_os_system(command, cwd)
    else:
        try:
            result = _run_via_subprocess(command, cwd, timeout)
        except subprocess.TimeoutExpired:
            return Result(f"Command timed out after {timeout}s: {command}", 124)
        except Exception:
            # Some desktop sandboxes forbid subprocess; ios_system-style
            # execution still works there.
            result = _run_via_os_system(command, cwd)

    if not result.output:
        result = Result("(no output)", result.code)
    return result
