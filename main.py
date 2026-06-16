from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from voice_clone_dot_tts.main import main
except ModuleNotFoundError as exc:
    missing = exc.name or "a required dependency"
    raise SystemExit(
        "Could not start Voice Clone dots.tts because "
        f"{missing!r} is not installed.\n\n"
        "Install the local development environment first:\n"
        "  python -m venv .venv\n"
        "  .venv/bin/python -m pip install -e .[dev]\n\n"
        "On Windows PowerShell use:\n"
        "  py -3.12 -m venv .venv\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install -e .[dev]"
    ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
