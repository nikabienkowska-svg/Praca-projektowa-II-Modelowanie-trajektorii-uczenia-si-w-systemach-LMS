"""Moduł odpowiedzialny za ładowanie i walidację tabel ze zbioru OULAD."""

import os
from typing import Dict, Optional
import pandas as pd


def load_raw_tables(raw_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """Wczytuje surowe tabele OULAD z podanego katalogu.
    
    Zwraca słownik z ramkami danych dla:
    courses, assessments, vle, studentInfo, studentRegistration,
    studentAssessment, studentVle.
    """
    tables = [
        "courses",
        "assessments",
        "vle",
        "studentInfo",
        "studentRegistration",
        "studentAssessment",
        "studentVle",
    ]
    data = {}
    for table_name in tables:
        path = os.path.join(raw_dir, f"{table_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Nie znaleziono pliku {path}. Upewnij się, że dane OULAD zostały "
                "pobrane do folderu data/raw/ zgodnie z data/README.md."
            )
        print(f"Ładowanie tabeli {table_name}...")
        data[table_name] = pd.read_csv(path)
    return data
