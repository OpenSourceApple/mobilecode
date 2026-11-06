# mobilecode

A coding agent that runs inside **a-Shell** on iOS. Read, write and edit
files, run shell commands, search the workspace, and checkpoint work with
`lg2` — a-Shell's libgit2-based `git` equivalent.

**Requirements:** Python 3.8+ and the standard library only. Nothing to
`pip install`. (a-Shell ships Python 3.13.)

```
agent.py      entry point: REPL, one-shot mode, CLI flags
core.py       agent loop, context compaction, checkpointing
tools.py      the 9 tools and their JSON schemas
llm.py        model access (NVIDIA NIM, OpenAI-compatible) + rate limiting
shell_exec.py command execution, tuned for ios_system
vcs.py        lg2 wrapper (git fallback on desktop)
sessions.py   transcript persistence across app restarts
tests/        offline unit suite + opt-in live end-to-end test
```

## Why it is built this way

a-Shell is not a small Unix box, and three of its properties decide the
design:

1. **`os.system()` works, `subprocess` does not.** a-Shell's Python enables
   `os.system()` on purpose (CPython's `#define HAVE_SYSTEM 1` under
   `TARGET_OS_IPHONE`) and routes it to `ios_system`, the same command
   interpreter the terminal uses. `subprocess` is unusable there: ios_system
   fakes `fork()` with `return ios_nextAvailablePid()` and there is no
   `/bin/sh`. Every command the agent runs goes through `os.system()`, with
   output captured by `> tmp 2>&1` redirection, which ios_system parses
   itself. Compound commands (`&&`, `|`) are wrapped in `sh -c '...'`
   because ios_system's top level does not handle them.

2. **The app can be killed at any time.** iOS terminates backgrounded apps
   without warning, so the transcript is written to disk after every model
   step and can be resumed with `--resume` (or `continue` in the REPL).

3. **Version control is `lg2`, not `git`.** a-Shell ships no `git` binary.
   `lg2` is the libgit2 CLI: add, apply, blame, branch, checkout, clone,
   commit, config, diff, fetch, init, log, merge, rebase, reset, stash,
   status, tag... The agent uses it through the `checkpoint` tool, and
   automatically commits completed work so anything can be rolled back.
   `lg2 config <KEY> [<VALUE>]` writes repo-local config only — there is no
   `--global` flag — so identity is stored inside each workspace and never
   outside the sandbox.

The model layer assumes an OpenAI-compatible chat-completions endpoint and
uses **native tool calling**, which is far more reliable than asking a model
to emit hand-written JSON (a fenced or prose-wrapped response then breaks the
whole loop).

## Setup on a-Shell

1. Copy the `mobilecode` folder somewhere under `~/Documents/` (the only
   user-writable home directory in a-Shell):

   ```
   cp -r mobilecode ~/Documents/
   ```

2. Put the API key in `~/.profile` — a-Shell runs that file when a new
   window opens:

   ```
   echo 'export NVIDIA_API_KEY=nvapi-...' >> ~/.profile
   ```

   Then open a new a-Shell window. Get a free key at
   <https://build.nvidia.com> (any model's page, "Get API Key"). The default
   model is `minimaxai/minimax-m3.1-flash`; override with `NVIDIA_MODEL`.
   Any OpenAI-compatible endpoint works via `NVIDIA_BASE_URL`.

3. Run it:

   ```
   cd ~/Documents
   python3 mobilecode/agent.py myproject
   ```

### Usage

```
python3 agent.py [WORKSPACE]            interactive session
python3 agent.py WS -p "task"           one-shot: do the task, print, exit
python3 agent.py WS --resume            continue the saved session
python3 agent.py WS --no-checkpoint     disable automatic lg2 commits
python3 agent.py WS --quiet             hide tool calls
```

In the REPL: `quit`/`exit` to leave, `/clear` to reset the conversation,
`continue` to resume a turn that was interrupted (by a rate limit or by the
app being killed).

### One-shot mode and Shortcuts

`-p` makes the agent run a single task and print the answer, which is what
you want from a Shortcuts **Execute Command** action:

```
python3 ~/Documents/mobilecode/agent.py ~/Documents/notes -p "tidy up index.md"
```

Keep extension-mode shortcuts under a few seconds; use "In App" execution for
anything longer.

## Rate limits

The free NVIDIA endpoint allows roughly **40 requests per minute, shared by
every client using that key**. The agent paces itself under the ceiling
(tune with `NVIDIA_RPM`), retries 429s with waits, and if the budget stays
saturated it fails loudly and leaves the turn resumable instead of dropping
your task.

## Tests

```
python3 tests/run_all.py
```

94 offline tests: path jail and symlink escapes, every tool, both execution
paths, compaction invariants, the silent-stall regressions, lg2 wrapper
behaviour (stubbed), session round-trips. The live end-to-end test needs a
key and only runs when asked for:

```
mobilecode_LIVE=1 NVIDIA_API_KEY=... python3 tests/run_all.py
```

## Maintaining this repo on a-Shell

The repo itself is a normal git repository, so it can be cloned from a Mac
and worked on from the phone. Inside a-Shell, `lg2` drives it:

```
lg2 status
lg2 diff
lg2 add -A
lg2 commit -m "describe the change"
lg2 log --oneline
```

`lg2 config` writes repo-local config only (`lg2 config user.name "you"`,
`lg2 config user.email "you@example.com"`). The agent sets identity
automatically on first checkpoint. `push`/`pull` over https have historically
been flaky in lg2; moving work between devices with `lg2 bundle`/`clone` over
ssh, or via the Files app, is the dependable path.

## Known constraints

- `patch` does not exist on a-Shell (only `diff`); the agent's `edit_file`
  tool does exact-string replacement, which covers most of it. `lg2 apply`
  applies a patch inside a repo.
- `clang` needs `pkg install llvm`; pure-Python `pip install` only.
- `getpass` hangs on iOS (no termios echo control) — never prompt for
  secrets; use environment variables.
- Command timeouts are advisory on iOS: `os.system()` is synchronous and
  cannot be interrupted.
- WASM-installed commands (`pkg install rg`) can race when the same command
  appears twice in one pipeline; prefer built-in `grep` there.
