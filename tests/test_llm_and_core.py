"""The model layer: empty responses must fail loudly, not end a turn silently.

These regressions came from a real failure: `content: ""` with no tool calls
ended the turn silently after a single tool call, and an empty `choices` list
escaped as IndexError.
"""

import json
import unittest

from helpers import load, fresh_workspace

llm = load("llm")
tools = load("tools")
core = load("core")


def response(content=None, tool_calls=None, finish="stop", reasoning="",
             message_keys=None, choices=None):
    if choices is not None:
        return {"choices": choices}
    message = {}
    if message_keys is None:
        if content is not None:
            message["content"] = content
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
    else:
        for key in message_keys:
            message[key] = None
    return {"choices": [{"message": message, "finish_reason": finish}]}


TC = [{"id": "call_1", "type": "function",
       "function": {"name": "list_dir", "arguments": "{\"path\": \".\"}"}}]


class LlmErrorsTest(unittest.TestCase):
    def setUp(self):
        llm.API_KEY = "stub"
        self.sent = []
        llm.post = self._fake_post
        llm.limiter = llm.RateLimiter(limit=1000)
        self.fake_response = None

    def _fake_post(self, payload, timeout=300):
        self.sent.append(json.loads(json.dumps(payload)))
        return self.fake_response

    def call(self):
        return llm.call_model([{"role": "user", "content": "hi"}],
                              tools.TOOL_SCHEMAS)

    # ---- the silent-stall regressions -----------------------------------

    def test_empty_string_content_raises(self):
        self.fake_response = response("")
        with self.assertRaises(RuntimeError) as caught:
            self.call()
        self.assertIn("finish_reason", str(caught.exception))

    def test_whitespace_content_raises(self):
        self.fake_response = response("   \n ")
        with self.assertRaises(RuntimeError):
            self.call()

    def test_none_content_raises(self):
        self.fake_response = response(None, reasoning="hmm")
        with self.assertRaises(RuntimeError) as caught:
            self.call()
        text = str(caught.exception)
        self.assertIn("finish_reason", text)
        self.assertIn("reasoning", text)

    def test_empty_choices_raises_runtimeerror_not_indexerror(self):
        self.fake_response = response(choices=[])
        with self.assertRaises(RuntimeError):
            self.call()

    def test_error_key_raises(self):
        self.fake_response = {"error": {"message": "bad"}}
        with self.assertRaises(RuntimeError) as caught:
            self.call()
        self.assertIn("API error", str(caught.exception))

    # ---- working shapes --------------------------------------------------

    def test_plain_content_works(self):
        self.fake_response = response("done")
        message, usage = self.call()
        self.assertEqual(message["content"], "done")

    def test_content_with_tool_calls_works(self):
        # observed ~2/5 times on minimax-m3.1-flash
        self.fake_response = response("I'll look.", tool_calls=TC)
        message, _ = self.call()
        self.assertEqual(message["content"], "I'll look.")
        self.assertTrue(message["tool_calls"])

    def test_reasoning_only_with_tool_calls_works(self):
        self.fake_response = response(None, tool_calls=TC, reasoning="plan")
        message, _ = self.call()
        self.assertTrue(message["tool_calls"])

    def test_payload_shape(self):
        self.fake_response = response("ok")
        self.call()
        payload = self.sent[0]
        self.assertEqual(payload["model"], llm.MODEL)
        self.assertIn("tools", payload)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertIn("messages", payload)


class RateLimiterTest(unittest.TestCase):
    def test_first_request_has_no_wait(self):
        limiter = llm.RateLimiter(limit=3, window=60)
        self.assertEqual(limiter.acquire(), 0.0)

    def test_slots_are_counted(self):
        limiter = llm.RateLimiter(limit=2, window=60)
        limiter.acquire()
        limiter.acquire()
        self.assertEqual(len(limiter.times), 2)

    def test_old_entries_expire(self):
        limiter = llm.RateLimiter(limit=1, window=0.05)
        limiter.acquire()
        import time
        time.sleep(0.06)
        self.assertEqual(limiter.acquire(), 0.0)


class CompactionTest(unittest.TestCase):
    def setUp(self):
        self.ws = fresh_workspace()
        tools.set_workspace(self.ws)

    def test_small_transcript_untouched(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "hi"}]
        self.assertIs(core.compact(msgs), msgs)

    def test_compact_does_not_grow(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "f"}]
        msgs += [{"role": "user", "content": "x" * 100000} for _ in range(8)]
        out = core.compact(msgs)
        self.assertLessEqual(len(out), len(msgs))

    def test_big_transcript_shrinks_and_preserves_head(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "first request"}]
        for _ in range(40):
            msgs.append({"role": "user", "content": "x" * 50000})
            msgs.append({"role": "assistant", "content": "y" * 50000})
        out = core.compact(msgs)
        self.assertLess(len(out), len(msgs))
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[1]["content"], "first request")
        self.assertTrue(any("elided" in str(m.get("content", "")) for m in out))

    def test_no_orphaned_tool_results(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "f"}]
        for i in range(30):
            msgs.append({"role": "user", "content": "x" * 50000})
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                         "name": "read_file", "content": "z" * 50000})
        out = core.compact(msgs)
        pending = set()
        for message in out:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                pending.update(c.get("id") for c in message["tool_calls"])
            elif message.get("role") == "tool":
                self.assertIn(message.get("tool_call_id"), pending,
                              "orphaned tool result after compaction")
                pending.discard(message.get("tool_call_id"))

    def test_tail_never_dangles(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "f"}]
        for i in range(30):
            msgs.append({"role": "user", "content": "x" * 50000})
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "run", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                         "name": "run", "content": "z" * 50000})
        out = core.compact(msgs)
        # Ending on a tool result is valid (that is the resumable state), but
        # only if the assistant call it answers is present just before it.
        last = out[-1]
        if last.get("role") == "tool":
            previous = out[-2]
            self.assertEqual(previous.get("role"), "assistant")
            ids = {c.get("id") for c in previous.get("tool_calls") or []}
            self.assertIn(last.get("tool_call_id"), ids)
        # An assistant call with no answer after it is never valid.
        self.assertFalse(out[-1].get("role") == "assistant"
                         and out[-1].get("tool_calls"),
                         "tail must not end on an unanswered assistant call")


class StallResumeTest(unittest.TestCase):
    """A turn that stops midway stays resumable and the task survives."""

    def setUp(self):
        self.ws = fresh_workspace()
        tools.set_workspace(self.ws)
        self.session = core.AgentSession(self.ws, verbose=False)
        core.llm.post = lambda payload, timeout=300: self.next_response
        llm.post = core.llm.post
        core.llm.limiter = llm.RateLimiter(limit=1000)

    def respond(self, response_body):
        self.next_response = response_body

    def test_stall_after_tool_call_is_loud(self):
        self.respond(response(None, tool_calls=TC))
        self.session.messages.append({"role": "user", "content": "make a file"})
        # first step: tool call dispatched fine
        with self.assertRaises(RuntimeError):
            self.respond(response("", finish="stop"))  # the stall
            # drive manually to hit the stall on the second step
            self.session._run_loop()

    def test_resume_completes_task(self):
        write_call = [{"id": "c2", "type": "function",
                       "function": {"name": "write_file",
                                    "arguments": json.dumps(
                                        {"path": "hello.py",
                                         "content": "print('hello')\n"})}}]
        self.session.messages.append({"role": "user",
                                      "content": "make a python file"})
        self.respond(response(None, tool_calls=write_call))
        self.session.turn("make a python file")
        self.respond(response("created it", finish="stop"))
        self.session.resume()
        self.assertTrue((self.ws / "hello.py").exists())
        self.assertFalse(self.session.turn_is_open())

    def test_turn_is_open_states(self):
        self.assertFalse(core.AgentSession(self.ws).turn_is_open())
        session = core.AgentSession(self.ws)
        session.messages.append({"role": "user", "content": "x"})
        self.assertTrue(session.turn_is_open())
        session.messages.append({"role": "tool", "tool_call_id": "c",
                                 "name": "run", "content": "y"})
        self.assertTrue(session.turn_is_open())


if __name__ == "__main__":
    unittest.main()
