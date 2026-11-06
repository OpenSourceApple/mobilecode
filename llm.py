"""Model access over an OpenAI-compatible endpoint (NVIDIA NIM by default).

Facts this module is built around, all verified against
integrate.api.nvidia.com:

  - the free tier is ~40 requests/minute per key, SHARED by every client
    using that key, so local pacing stays under the ceiling and 429s are
    retried with real waits rather than treated as fatal;
  - the response body of a 429 is not in the exception text, so it is read
    and reported explicitly;
  - `content` may be null OR an empty string when reasoning consumed the
    completion budget, which must fail loudly instead of ending a turn
    silently;
  - `choices` can be empty, which would otherwise surface as IndexError.

Transport is stdlib urllib: nothing to pip-install on the phone. The CA
bundle is discovered from a-Shell's app bundle ($APPDIR/cacert.pem) because
a-Shell's SSL_CERT_FILE usually points at ~/Documents/cacert.pem, which
typically does not exist.
"""

from __future__ import annotations
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

API_KEY = os.environ.get("NVIDIA_API_KEY")
BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
).rstrip("/")
MODEL = os.environ.get("NVIDIA_MODEL", "minimaxai/minimax-m3.1-flash")
CHAT_URL = BASE_URL + "/chat/completions"

MAX_TOKENS = 16384
TEMPERATURE = 1.0
TOP_P = 0.95

# Stay under the shared per-key ceiling; tune with NVIDIA_RPM if other
# clients are known to be quiet.
REQUEST_LIMIT = int(os.environ.get("NVIDIA_RPM", "30"))
REQUEST_WINDOW = 60.0
_MAX_ATTEMPTS = 8

_request_times: deque = deque()


class RateLimiter:
    """Rolling-window limiter for a shared requests-per-minute budget."""

    def __init__(self, limit: int = REQUEST_LIMIT, window: float = REQUEST_WINDOW):
        self.limit = limit
        self.window = window
        self.times: deque = deque()

    def acquire(self) -> float:
        """Block until a slot is free. Returns how long it waited (0 if none)."""
        waited = 0.0
        while True:
            now = time.monotonic()
            while self.times and now - self.times[0] >= self.window:
                self.times.popleft()
            if len(self.times) < self.limit:
                self.times.append(now)
                return waited
            sleep_for = self.window - (now - self.times[0]) + 0.5
            print(f"  [local rate limit; waiting {sleep_for:.0f}s]", flush=True)
            time.sleep(sleep_for)
            waited += sleep_for


limiter = RateLimiter()


def ca_bundle():
    """Find a usable CA bundle on this platform."""
    candidates = [
        os.environ.get("SSL_CERT_FILE"),
        os.environ.get("CURL_CA_BUNDLE"),
    ]
    appdir = os.environ.get("APPDIR")
    if appdir:
        candidates.append(os.path.join(appdir, "cacert.pem"))
    candidates.append("/etc/ssl/cert.pem")
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


def ssl_context():
    bundle = ca_bundle()
    if bundle:
        try:
            return ssl.create_default_context(cafile=bundle)
        except ssl.SSLError:
            pass
    return ssl.create_default_context()


def _retry_after(error) -> float:
    header = error.headers.get("Retry-After") if error.headers else None
    try:
        return max(5.0, float(header))
    except (TypeError, ValueError):
        return 30.0


def post(payload: dict, timeout: int = 300) -> dict:
    """POST JSON, retrying 429/5xx with backoff. Raises RuntimeError on
    terminal failures with the response body included."""
    if not API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set.\n"
            "  On a-Shell:  echo 'export NVIDIA_API_KEY=nvapi-...' >> ~/.profile\n"
            "  then open a new a-Shell window (.profile runs on launch)."
        )

    request = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    context = ssl_context()

    delay = 5.0
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            if error.code == 429:
                wait = _retry_after(error)
                print(f"  [rate limited; waiting {wait:.0f}s]", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            if error.code >= 500 and attempt < _MAX_ATTEMPTS - 1:
                print(f"  [server {error.code}; retrying in {delay:.0f}s]", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"HTTP {error.code} {error.reason}\n{detail}") from error
        except urllib.error.URLError as error:
            if attempt < _MAX_ATTEMPTS - 1:
                print(f"  [network error: {error.reason}; retrying]", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"Network error: {error.reason}") from error

    raise RuntimeError(
        f"The provider's per-minute request limit stayed saturated for "
        f"{_MAX_ATTEMPTS} attempts. The API key is shared by every client "
        "using it, so another process may be consuming the budget. Wait a "
        "minute and try again (in the agent, type 'continue'), or lower "
        "NVIDIA_RPM."
    )


def call_model(messages: list, tools: list | None) -> tuple[dict, dict]:
    """One completion request. Returns (message, usage).

    Raises RuntimeError when the model produces nothing actionable, including
    the finish_reason, the reasoning preview and the raw message keys so a
    truncated or reasoning-only response is diagnosable instead of silent.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    limiter.acquire()
    response = post(payload)

    if "error" in response:
        raise RuntimeError(f"API error: {response['error']}")

    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(
            "Response contained no choices "
            f"(keys: {sorted(response.keys())}): "
            + json.dumps(response)[:400]
        )

    choice = choices[0]
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")

    content = message.get("content")
    has_content = isinstance(content, str) and content.strip()
    if not has_content and not message.get("tool_calls"):
        reasoning = message.get("reasoning_content") or ""
        raise RuntimeError(
            "The model returned no content and no tool calls"
            + (f" (reasoning: {reasoning[:200]})" if reasoning else "")
            + f" [finish_reason={finish}]"
            + f" [message keys: {sorted(message.keys())}]"
            + " — this usually means the completion budget was consumed by"
            + " reasoning, or the response was truncated. Type 'continue' to"
            + " retry."
        )
    return message, response.get("usage", {})
