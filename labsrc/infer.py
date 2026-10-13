"""Inference scaffolding for E0-E3 predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def write_predictions(rows: Iterable[Dict[str, Any]], out_path: str | Path) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(out)
