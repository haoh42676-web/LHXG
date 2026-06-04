from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
HISTORY_DIR = CACHE_DIR / "history"
NEWS_DIR = CACHE_DIR / "news"
OUTPUT_DIR = DATA_DIR / "outputs"

for directory in (DATA_DIR, CACHE_DIR, HISTORY_DIR, NEWS_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
