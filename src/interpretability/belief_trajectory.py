"""Moduł analizy trajektorii ewolucji przekonań dyskutantów.
Śledzi zmiany pewności (certainty vs tentative) w wieloetapowej sekwencji odpowiedzi.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from src.features.text_features import extract_linguistic_markers


def compute_belief_trajectory(sequence_of_comments: List[str]) -> pd.DataFrame:
    """Analizuje sekwencję kolejnych wypowiedzi autora posta (OP) w wątku
    i wylicza trajektorię zmiany stopnia pewności w czasie.

    Returns:
        DataFrame ze wskaźnikami: runda, gęstość zwrotów asekuracyjnych,
        gęstość słów kategorycznych, bilans pewności.
    """
    history = []
    for step_idx, text in enumerate(sequence_of_comments):
        stats = extract_linguistic_markers(text)
        certainty_score = stats["certainty_density"] - stats["hedging_density"]

        history.append({
            "step": step_idx,
            "word_count": stats["word_count"],
            "hedging_density": stats["hedging_density"],
            "certainty_density": stats["certainty_density"],
            "net_certainty_score": certainty_score,
            "status": "pewny / kategoryczny" if certainty_score > 0.01 else ("wahający się" if stats["hedging_density"] > 0.01 else "neutralny"),
        })

    return pd.DataFrame(history)


def detect_inflection_point(trajectory_df: pd.DataFrame) -> int:
    """Wykrywa indeks kroku czasowego (rundy), w którym nastąpiło największe
    załamanie pewności siebie (przejście ze stanu pewnego w stan wątpliwości).
    """
    if len(trajectory_df) < 2:
        return 0

    scores = trajectory_df["net_certainty_score"].values
    diffs = np.diff(scores)
    # Największy spadek (najbardziej ujemna różnica)
    inflection_idx = int(np.argmin(diffs)) + 1
    return inflection_idx
