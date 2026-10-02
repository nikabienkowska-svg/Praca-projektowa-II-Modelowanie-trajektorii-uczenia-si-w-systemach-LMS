"""Moduł odpowiedzialny za ładowanie i walidację danych Webis-CMV-20 (pary i cechy LIWC)."""

import os
import json
import bz2
from typing import List, Dict, Any, Generator


def load_cmv_sample(jsonl_path: str = "data/processed/cmv_persuasion_pairs_sample.jsonl") -> List[Dict[str, Any]]:
    """Wczytuje przygotowaną próbkę par perswazyjnych z pliku JSONL."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku {jsonl_path}. Uruchom najpierw skrypt:\n"
            "python src/data/prepare_cmv.py"
        )
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def stream_cmv_pairs_raw(
    bz2_path: str = "data/raw/pairs.jsonl.bz2", limit: int = 1000
) -> Generator[Dict[str, Any], None, None]:
    """Strumieniuje rekordy ze skompresowanego pliku bz2."""
    if not os.path.exists(bz2_path):
        raise FileNotFoundError(f"Brak pliku surowego {bz2_path}")
    count = 0
    with bz2.open(bz2_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
                count += 1
                if limit and count >= limit:
                    break
