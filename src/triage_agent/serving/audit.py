"""Journal d'audit : append-only JSONL (une ligne = un échange)."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def log_exchange(path: str | Path, record: dict) -> None:
    """Ajoute ``record`` (avec horodatage) à la fin du journal, de façon thread-safe."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"_ts": _now_iso(), **record}, ensure_ascii=False)
    with _lock:
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
