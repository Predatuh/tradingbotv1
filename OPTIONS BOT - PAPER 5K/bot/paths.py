"""Path handling that works both as plain Python and as a frozen .exe.

Two kinds of files, two locations:
  * app_dir()      — user-editable / persistent files (config.yaml, .env,
                     state.json, trades_log.jsonl, model.pkl, risk_profile).
                     Plain Python: the project folder.
                     Frozen exe:   the folder the .exe lives in.
  * resource_path()— read-only assets bundled INSIDE the exe (dashboard.html).
                     Frozen exe:   PyInstaller's unpack dir (sys._MEIPASS).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def app_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _PROJECT_ROOT


def resource_path(rel: str) -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", app_dir())) / rel
    return _PROJECT_ROOT / rel
