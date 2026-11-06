"""Shell execution: both the iOS path and the desktop path."""

import unittest
from pathlib import Path

from helpers import load, fresh_workspace

shell_exec = load("shell_exec")


class ShellExecTest(unittest.TestCase):
    def setUp(self):
        self.ws = fresh_workspace()
        (self.ws / "data.txt").write_text("alpha\nbeta\n")

    def test_desktop_path(self):
        shell_exec.is_ios = lambda: False
        result = shell_exec.run("echo hello-desktop", cwd=self.ws)
        self.assertEqual(result.code, 0)
        self.assertIn("hello-desktop", result.output)

    def test_desktop_captures_stderr(self):
        shell_exec.is_ios = lambda: False
        result = shell_exec.run("echo out; echo err 1>&2", cwd=self.ws)
        self.assertIn("out", result.output)
        self.assertIn("err", result.output)

    def test_desktop_exit_code(self):
        shell_exec.is_ios = lambda: False
        result = shell_exec.run("exit 3", cwd=self.ws)
        self.assertEqual(result.code, 3)

    def test_desktop_timeout(self):
        shell_exec.is_ios = lambda: False
        result = shell_exec.run("sleep 5", cwd=self.ws, timeout=1)
        self.assertEqual(result.code, 124)
        self.assertIn("timed out", result.output)

    def test_ios_path_basic(self):
        shell_exec.is_ios = lambda: True
        result = shell_exec.run("echo hello-ios", cwd=self.ws)
        self.assertEqual(result.code, 0)
        self.assertIn("hello-ios", result.output)

    def test_ios_path_reads_workspace_file(self):
        shell_exec.is_ios = lambda: True
        result = shell_exec.run("cat data.txt", cwd=self.ws)
        self.assertIn("alpha", result.output)
        self.assertIn("beta", result.output)

    def test_ios_path_compound_command_is_wrapped_in_sh(self):
        shell_exec.is_ios = lambda: True
        result = shell_exec.run("echo a && echo b", cwd=self.ws)
        self.assertIn("a", result.output)
        self.assertIn("b", result.output)

    def test_ios_path_captures_stderr(self):
        shell_exec.is_ios = lambda: True
        result = shell_exec.run("echo err-out 1>&2", cwd=self.ws)
        self.assertIn("err-out", result.output)

    def test_ios_path_nonzero_exit(self):
        shell_exec.is_ios = lambda: True
        result = shell_exec.run("exit 7", cwd=self.ws)
        self.assertEqual(result.code, 7)

    def test_ios_path_missing_command_does_not_crash(self):
        shell_exec.is_ios = lambda: True
        result = shell_exec.run("definitely_not_a_command_xyz", cwd=self.ws)
        self.assertNotEqual(result.code, 0)

    def test_ios_path_cwd_is_workspace_and_restored(self):
        shell_exec.is_ios = lambda: True
        before = Path.cwd()
        result = shell_exec.run("pwd", cwd=self.ws)
        self.assertIn(str(self.ws), result.output)
        self.assertEqual(Path.cwd(), before)

    def test_ios_path_leaves_no_temp_files(self):
        shell_exec.is_ios = lambda: True
        shell_exec.run("echo x", cwd=self.ws)
        import tempfile
        leftovers = list(Path(tempfile.gettempdir()).glob("ashellagent-*.out"))
        self.assertEqual(leftovers, [])

    def test_ios_path_unicode(self):
        shell_exec.is_ios = lambda: True
        (self.ws / "u.txt").write_text("café 日本\n", encoding="utf-8")
        result = shell_exec.run("cat u.txt", cwd=self.ws)
        self.assertIn("café", result.output)
        self.assertIn("日本", result.output)

    def test_empty_command_rejected(self):
        with self.assertRaises(ValueError):
            shell_exec.run("   ", cwd=self.ws)

    def test_oversized_command_rejected(self):
        with self.assertRaises(ValueError):
            shell_exec.run("x" * 5000, cwd=self.ws)

    def test_decode_exit_code_posix_wait_status(self):
        self.assertEqual(shell_exec._decode_exit_code(7 << 8), 7)
        self.assertEqual(shell_exec._decode_exit_code(127 << 8), 127)

    def test_decode_exit_code_direct(self):
        self.assertEqual(shell_exec._decode_exit_code(7), 7)
        self.assertEqual(shell_exec._decode_exit_code(0), 0)

    def test_needs_sh_wrap(self):
        self.assertTrue(shell_exec._needs_sh_wrap("a && b"))
        self.assertTrue(shell_exec._needs_sh_wrap("a | b"))
        self.assertTrue(shell_exec._needs_sh_wrap("a > b"))
        self.assertFalse(shell_exec._needs_sh_wrap("echo plain"))


if __name__ == "__main__":
    unittest.main()
