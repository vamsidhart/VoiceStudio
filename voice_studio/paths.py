"""Filesystem locations for Voice Studio.

Source lives in this repository. Personal media and projects live under ``data/``
and are not committed.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROJECTS = DATA / "projects"
UPLOADS = DATA / "uploads"
DICTIONARY = DATA / "pronunciations.json"
PREFERENCES = DATA / "preferences.json"
DEFAULT_VOICE = DATA / "voices" / "my_voice.wav"


def ensure_data_dirs() -> None:
    for folder in (PROJECTS, UPLOADS, DEFAULT_VOICE.parent):
        folder.mkdir(parents=True, exist_ok=True)
