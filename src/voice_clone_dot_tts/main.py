from __future__ import annotations

import sys
from multiprocessing import freeze_support
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from voice_clone_dot_tts.ui import MainWindow, create_app
else:
    from .ui import MainWindow, create_app


def main() -> int:
    freeze_support()
    app = create_app(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
