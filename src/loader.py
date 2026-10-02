"""Prosty moduł do wczytywania danych Webis-CMV-20 w Pythonie."""

import os
import json
import pandas as pd
from typing import Optional


def load_cmv_dataframe(path: str = "data/processed/cmv_persuasion_pairs_sample.jsonl") -> pd.DataFrame:
    """Wczytuje oczyszczoną próbkę par dyskusji do DataFrame Pandas."""
    if not os.path.exists(path):
        # Sprawdzamy czy ścieżka względna z poziomu notebooka (folder wyżej)
        alt_path = os.path.join("..", path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)
