#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Create it with /opt/homebrew/bin/python3.12 -m venv .venv first." >&2
  exit 1
fi

.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pyinstaller packaging/voice-clone-dot-tts.spec --clean --noconfirm
rm -rf "dist/Voice Clone dots.tts"
