"""Session persistence.

a-Shell terminates backgrounded apps, so the transcript is written to disk
after every model step. Sessions live under
$HOME/.ashellagent/sessions/<workspace-slug>.json — on a-Shell $HOME is
~/Documents, which is the one user-writable location that survives.
"""

import json
import re
from datetime import datetime
from pathlib import Path

SESSIONS_DIR = Path.home() / ".ashellagent" / "sessions"


def slug(workspace_path) -> str:
    text = str(workspace_path)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text or "workspace"


def sessions_dir() -> Path:
    path = Path(SESSIONS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_path(workspace_path) -> Path:
    return sessions_dir() / (slug(workspace_path) + ".json")


def save(messages, workspace_path, extra=None) -> Path:
    path = session_path(workspace_path)
    payload = {
        "workspace": str(workspace_path),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "messages": messages,
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)
    return path


def load(workspace_path):
    """Return the saved messages list, or None if there is no session."""
    path = session_path(workspace_path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    return messages
