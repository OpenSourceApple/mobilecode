"""Shared test helpers: a fresh workspace per test and import of the package."""

import importlib.util
import sys
import tempfile
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
if PACKAGE_DIR.parent not in sys.path:
    # make `import ashellagent.xxx` work AND sibling imports inside the package
    sys.path.insert(0, str(PACKAGE_DIR.parent))
    sys.path.insert(0, str(PACKAGE_DIR))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, str(PACKAGE_DIR / f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_workspace(prefix="ashellagent_test_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix)).resolve()
