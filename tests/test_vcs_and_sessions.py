"""lg2 wrapper (stubbed) and session persistence."""

import json
import unittest
from pathlib import Path

from helpers import load, fresh_workspace

vcs = load("vcs")
shell_exec = load("shell_exec")
sessions = load("sessions")
tools = load("tools")


class StubLg2:
    """Records lg2 invocations and replays scripted results.

    fail_on    substring match — applied to every command (e.g. "commit").
    fail_probe exact match — simulates a missing config key, i.e. only the
               read form `lg2 config <KEY>` fails, not the write form
               `lg2 config <KEY> <VALUE>`.
    """

    def __init__(self, fail_on=(), fail_probe=()):
        self.calls = []
        self.fail_on = fail_on
        self.fail_probe = fail_probe

    def __call__(self, command, cwd, timeout=120):
        self.calls.append((command, str(cwd)))
        for needle in self.fail_on:
            if needle in command:
                return shell_exec.Result(f"error: {needle}", 1)
        if command in self.fail_probe:
            return shell_exec.Result(f"error: {command} not set", 1)
        return shell_exec.Result("ok", 0)


class VcsTest(unittest.TestCase):
    def setUp(self):
        self.ws = fresh_workspace()
        self.stub = StubLg2()
        self.original_run = vcs.shell_exec.run
        vcs.shell_exec.run = self.stub

    def tearDown(self):
        vcs.shell_exec.run = self.original_run

    def test_is_repo_false_when_no_git(self):
        self.assertFalse(vcs.is_repo(self.ws))

    def test_is_repo_true_with_git_dir(self):
        (self.ws / ".git").mkdir()
        self.assertTrue(vcs.is_repo(self.ws))

    def test_checkpoint_initializes_missing_repo(self):
        result = vcs.checkpoint(self.ws, "first commit")
        self.assertEqual(result.code, 0)
        commands = [c for c, _ in self.stub.calls]
        self.assertTrue(any(c.startswith("lg2 init") for c in commands))
        self.assertTrue(any(c.startswith("lg2 add -A") for c in commands))
        self.assertTrue(any("lg2 commit -m" in c for c in commands))

    def test_checkpoint_sets_repo_local_identity(self):
        # Missing keys: only the read form fails, which is what the code
        # probes with. The write form must then be issued and succeed.
        self.stub.fail_probe = {"lg2 config user.name", "lg2 config user.email"}
        vcs.checkpoint(self.ws, "msg")
        commands = [c for c, _ in self.stub.calls]
        self.assertTrue(any('config user.name "ashellagent"' in c
                            for c in commands))
        self.assertTrue(any('config user.email "agent@local"' in c
                            for c in commands))

    def test_checkpoint_does_not_use_global_flag(self):
        # lg2 config has no --global; a-Shell's guide example is wrong there.
        vcs.checkpoint(self.ws, "msg")
        commands = " ".join(c for c, _ in self.stub.calls)
        self.assertNotIn("--global", commands)

    def test_checkpoint_quotes_message(self):
        # The message's own double quotes are sanitized to single quotes so
        # they cannot break out of the -m "..." wrapper.
        result = vcs.checkpoint(self.ws, 'fix "quoted" name')
        self.assertEqual(result.code, 0)
        commit = next(c for c, _ in self.stub.calls if "commit" in c)
        self.assertIn("fix 'quoted' name", commit)
        self.assertNotIn('"quoted"', commit)

    def test_checkpoint_clean_tree_is_not_a_failure(self):
        # lg2/git exit nonzero with "nothing to commit" — the work is already
        # committed, so this must read as success.
        real_run = vcs.shell_exec.run
        calls = []

        def fake_run(command, cwd, timeout=120):
            calls.append(command)
            if "commit" in command:
                return shell_exec.Result(
                    "On branch main\nnothing to commit, working tree clean", 1)
            return shell_exec.Result("ok", 0)

        vcs.shell_exec.run = fake_run
        try:
            result = vcs.checkpoint(self.ws, "already done")
        finally:
            vcs.shell_exec.run = real_run
        self.assertEqual(result.code, 0)
        self.assertIn("Already committed", result.output)

    def test_checkpoint_reports_commit_failure(self):
        self.stub.fail_on = ("commit",)
        result = vcs.checkpoint(self.ws, "msg")
        self.assertNotEqual(result.code, 0)
        self.assertIn("lg2 commit failed", result.output)

    def test_checkpoint_reports_add_failure(self):
        self.stub.fail_on = ("add",)
        result = vcs.checkpoint(self.ws, "msg")
        self.assertNotEqual(result.code, 0)
        self.assertIn("lg2 add failed", result.output)

    def test_checkpoint_sanitizes_long_message(self):
        vcs.checkpoint(self.ws, "x" * 500)
        commit = next(c for c, _ in self.stub.calls if "commit" in c)
        self.assertLess(len(commit), 260)

    def test_checkpoint_empty_message_defaults(self):
        result = vcs.checkpoint(self.ws, "   ")
        self.assertEqual(result.code, 0)
        commit = next(c for c, _ in self.stub.calls if "commit" in c)
        self.assertIn("checkpoint", commit)

    def test_status_diff_log(self):
        vcs.status(self.ws)
        vcs.diff(self.ws)
        vcs.log(self.ws, 3)
        commands = [c for c, _ in self.stub.calls]
        self.assertIn("lg2 status", commands)
        self.assertIn("lg2 diff", commands)
        self.assertIn("lg2 log --oneline -3", commands)


class SessionsTest(unittest.TestCase):
    def setUp(self):
        self.ws = fresh_workspace()
        sessions.SESSIONS_DIR = fresh_workspace("ashellagent_sessions_")

    def test_save_and_load_roundtrip(self):
        messages = [{"role": "system", "content": "s"},
                    {"role": "user", "content": "café 日本"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function",
                         "function": {"name": "run", "arguments": "{}"}}]}]
        path = sessions.save(messages, self.ws)
        self.assertTrue(path.is_file())
        loaded = sessions.load(self.ws)
        self.assertEqual(loaded, messages)

    def test_load_missing_returns_none(self):
        self.assertIsNone(sessions.load(self.ws))

    def test_load_corrupt_returns_none(self):
        path = sessions.session_path(self.ws)
        path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(sessions.load(self.ws))

    def test_slug_is_filesystem_safe(self):
        # Path separators and spaces collapse to underscores; leading
        # separators are stripped, so the result is a plain filename.
        slug = sessions.slug("/private/var/a b/c")
        self.assertNotIn("/", slug)
        self.assertNotIn(" ", slug)
        self.assertLess(len(slug), 200)
        self.assertEqual(slug, sessions.slug("/private/var/a b/c"))

    def test_save_is_atomic(self):
        path = sessions.save([{"role": "user", "content": "x"}], self.ws)
        self.assertFalse(path.with_suffix(".json.tmp").exists())
        self.assertTrue(json.loads(path.read_text(encoding="utf-8")))


import re  # noqa: E402  (used by test_slug_is_filesystem_safe)

if __name__ == "__main__":
    unittest.main()
