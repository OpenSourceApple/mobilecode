"""Optional live end-to-end run against the real endpoint.

Skipped unless ASHELLAGENT_LIVE=1 and NVIDIA_API_KEY are set, so `python3 -m
unittest discover` stays offline and rate-limit free.
"""

import os
import unittest
from pathlib import Path

from helpers import load, fresh_workspace

live = os.environ.get("ASHELLAGENT_LIVE") == "1" and os.environ.get("NVIDIA_API_KEY")


@unittest.skipUnless(live, "set ASHELLAGENT_LIVE=1 + NVIDIA_API_KEY to run")
class LiveEndToEndTest(unittest.TestCase):
    def test_edit_run_verify(self):
        core = load("core")
        tools = load("tools")
        sessions = load("sessions")

        ws = fresh_workspace("ashellagent_live_")
        (ws / "hello.py").write_text(
            "def greet(name):\n"
            "    return 'Hello, ' + name\n"
            "\n\n"
            "if __name__ == '__main__':\n"
            "    print(greet('world'))\n"
        )
        (ws / "data").mkdir()
        (ws / "data" / "values.txt").write_text("alpha\nbeta\n")

        session = core.AgentSession(ws, verbose=True, auto_checkpoint=False)
        session.turn(
            "Look at hello.py and data/values.txt. Add a function shout(text) "
            "to hello.py that returns the text uppercased with an exclamation "
            "mark, print it from __main__, then run the script to verify."
        )

        source = (ws / "hello.py").read_text()
        self.assertIn("def shout(", source)
        self.assertIn("upper()", source)
        self.assertIn("!", source)


if __name__ == "__main__":
    unittest.main()
