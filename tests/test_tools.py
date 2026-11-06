"""Tool behaviour and sandbox security."""

import os
import unittest
from pathlib import Path

from helpers import load, fresh_workspace

tools = load("tools")


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.ws = fresh_workspace()
        tools.set_workspace(self.ws)
        self.ws = tools.WORKSPACE

    # ---- path safety -----------------------------------------------------

    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError):
            tools.workspace_path("/etc/passwd")

    def test_dotdot_escape_rejected(self):
        with self.assertRaises(ValueError):
            tools.workspace_path("../../etc/passwd")

    def test_relative_path_ok(self):
        self.assertEqual(str(tools.workspace_path("a.txt")),
                         str(self.ws / "a.txt"))

    def test_symlink_read_blocked(self):
        os.symlink("/etc/passwd", self.ws / "esc.txt")
        with self.assertRaises(ValueError):
            tools.tool_read_file("esc.txt")

    def test_symlink_write_blocked(self):
        os.symlink("/etc/passwd", self.ws / "esc.txt")
        with self.assertRaises(ValueError):
            tools.tool_write_file("esc.txt", "pwned")

    def test_symlink_delete_blocked(self):
        os.symlink("/etc/passwd", self.ws / "esc.txt")
        with self.assertRaises(ValueError):
            tools.tool_delete_file("esc.txt")

    def test_symlinks_not_traversed_in_listing(self):
        os.symlink("/etc", self.ws / "escdir")
        listing = tools.tool_list_dir(".", recursive=True)
        self.assertNotIn("escdir", listing)

    def test_symlinks_not_traversed_in_search(self):
        os.symlink("/etc/passwd", self.ws / "esc.txt")
        hits = tools.tool_search("*", pattern="name")
        self.assertNotIn("esc.txt", hits)

    def test_workspace_root_delete_blocked(self):
        with self.assertRaises(ValueError):
            tools.tool_delete_file(".")

    # ---- file operations -------------------------------------------------

    def test_write_and_read(self):
        tools.tool_write_file("x/y.txt", "hello\nworld\n")
        body = tools.tool_read_file("x/y.txt")
        self.assertIn("hello", body)
        self.assertIn("1\t", body)          # line numbers
        self.assertIn("2 lines", body)

    def test_write_creates_parents(self):
        out = tools.tool_write_file("deep/a/b/c.txt", "z")
        self.assertIn("Wrote", out)
        self.assertTrue((self.ws / "deep/a/b/c.txt").exists())

    def test_write_rejects_directory_target(self):
        (self.ws / "adir").mkdir()
        with self.assertRaises(ValueError):
            tools.tool_write_file("adir", "x")

    def test_write_rejects_oversized_content(self):
        with self.assertRaises(ValueError):
            tools.tool_write_file("big.txt", "x" * (tools.MAX_FILE_BYTES + 1))

    def test_edit_replaces(self):
        tools.tool_write_file("x/y.txt", "hello\nworld\n")
        tools.tool_edit_file("x/y.txt", "world", "there")
        self.assertIn("there", tools.tool_read_file("x/y.txt"))

    def test_edit_missing_old_string_errors(self):
        tools.tool_write_file("x/y.txt", "hello\n")
        with self.assertRaises(ValueError):
            tools.tool_edit_file("x/y.txt", "NOT_THERE", "z")

    def test_edit_ambiguous_errors_without_replace_all(self):
        tools.tool_write_file("dup.txt", "a\na\n")
        with self.assertRaises(ValueError):
            tools.tool_edit_file("dup.txt", "a", "b")

    def test_edit_replace_all(self):
        tools.tool_write_file("dup.txt", "a\na\n")
        tools.tool_edit_file("dup.txt", "a", "b", replace_all=True)
        body = tools.tool_read_file("dup.txt")
        self.assertIn("1\tb", body)
        self.assertIn("2\tb", body)
        self.assertNotIn("1\ta", body)

    def test_read_offset_limit(self):
        tools.tool_write_file("x/y.txt", "one\ntwo\nthree\n")
        body = tools.tool_read_file("x/y.txt", offset=2, limit=1)
        self.assertIn("two", body)
        self.assertNotIn("one", body)

    def test_read_missing_file_errors(self):
        with self.assertRaises(ValueError):
            tools.tool_read_file("nope.txt")

    def test_read_binary_no_crash(self):
        (self.ws / "bin.dat").write_bytes(b"\x00\x01\xff\xfe")
        body = tools.tool_read_file("bin.dat")
        self.assertIn("not UTF-8", body)

    def test_read_oversized_errors(self):
        (self.ws / "big.txt").write_text("x" * (tools.MAX_FILE_BYTES + 1))
        with self.assertRaises(ValueError):
            tools.tool_read_file("big.txt")

    # ---- listing and search ----------------------------------------------

    def test_list_dir(self):
        tools.tool_write_file("a.txt", "x")
        (self.ws / "sub").mkdir()
        listing = tools.tool_list_dir(".")
        self.assertIn("dir  sub", listing)
        self.assertIn("file a.txt", listing)

    def test_list_dir_recursive(self):
        tools.tool_write_file("deep/a/b.txt", "x")
        listing = tools.tool_list_dir(".", recursive=True)
        self.assertIn("deep/a/b.txt", listing)

    def test_list_empty(self):
        self.assertIn("is empty", tools.tool_list_dir("."))

    def test_list_shows_outside_symlinks_as_hidden(self):
        os.symlink("/etc", self.ws / "escdir")
        listing = tools.tool_list_dir(".")
        self.assertIn("link escdir", listing)

    def test_search_content(self):
        tools.tool_write_file("x/y.txt", "hello\nworld\n")
        self.assertIn("x/y.txt:1: hello", tools.tool_search("hello"))

    def test_search_case_insensitive(self):
        tools.tool_write_file("x/y.txt", "hello\n")
        self.assertIn("hello", tools.tool_search("HELLO"))

    def test_search_name_glob(self):
        tools.tool_write_file("a.log", "")
        tools.tool_write_file("b.txt", "")
        hits = tools.tool_search("*.log", pattern="name")
        self.assertIn("a.log", hits)
        self.assertNotIn("b.txt", hits)

    def test_search_no_match(self):
        self.assertIn("No matches", tools.tool_search("zzzznotfound"))

    def test_search_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            tools.tool_search("  ")

    # ---- delete / mkdir --------------------------------------------------

    def test_mkdir_and_delete(self):
        tools.tool_mkdir("newdir")
        self.assertIn("Created", tools.tool_mkdir("newdir"))  # idempotent
        self.assertIn("Removed", tools.tool_delete_file("newdir"))

    def test_delete_nonempty_dir_blocked(self):
        tools.tool_write_file("d/f.txt", "x")
        with self.assertRaises(ValueError):
            tools.tool_delete_file("d")

    def test_delete_file(self):
        tools.tool_write_file("f.txt", "x")
        self.assertIn("Deleted", tools.tool_delete_file("f.txt"))

    # ---- encoding --------------------------------------------------------

    def test_unicode_roundtrip(self):
        tools.tool_write_file("uni.txt", "café 日本 😀\n")
        self.assertIn("café 日本 😀", tools.tool_read_file("uni.txt"))

    def test_unicode_search(self):
        tools.tool_write_file("uni.txt", "café\n")
        self.assertIn("café", tools.tool_search("café"))

    # ---- helpers ---------------------------------------------------------

    def test_truncate_keeps_head_and_tail(self):
        body = tools.truncate("A" * 10000 + "END", 1000)
        self.assertTrue(body.endswith("END"))
        self.assertIn("elided", body)

    def test_truncate_small_noop(self):
        self.assertEqual(tools.truncate("short", 1000), "short")

    def test_every_schema_is_wellformed(self):
        self.assertEqual({t["function"]["name"] for t in tools.TOOL_SCHEMAS},
                         set(tools.TOOL_FUNCTIONS))
        for schema in tools.TOOL_SCHEMAS:
            fn = schema["function"]
            self.assertEqual(schema["type"], "function")
            self.assertEqual(fn["parameters"]["type"], "object")
            self.assertIn("required", fn["parameters"])
            self.assertTrue(set(fn["parameters"]["required"])
                            <= set(fn["parameters"]["properties"]))


if __name__ == "__main__":
    unittest.main()
