#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -d .venv ]]; then
  echo "Missing .venv. Use the existing Python 3.11 environment that already has Chatterbox installed."
  exit 1
fi
source .venv/bin/activate
exec python -m voice_studio
