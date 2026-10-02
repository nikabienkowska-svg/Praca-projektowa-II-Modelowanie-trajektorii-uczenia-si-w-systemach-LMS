"""Moduł ekstrakcji cech tekstowych i semantycznych.
Wykorzystuje sentence-transformers oraz heurystyki lingwistyczne (pewność vs wahanie).
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple


HEDGING_PATTERNS = [
    r"\bperhaps\b",
    r"\bmaybe\b",
    r"\bi admit\b",
    r"\byou have a point\b",
    r"\bi had not considered\b",
    r"\bi might be wrong\b",
    r"\bfair point\b",
    r"\bconcede\b",
    r"\bpossibly\b",
    r"\bcould be\b",
]

CERTAINTY_PATTERNS = [
    r"\bdefinitely\b",
    r"\bobviously\b",
    r"\bundeniably\b",
    r"\bimpossible\b",
    r"\bnever\b",
    r"\balways\b",
    r"\bfact\b",
    r"\bcertainly\b",
    r"\bclearly\b",
]


def count_pattern_matches(text: str, patterns: List[str]) -> int:
    """Zlicza wystąpienia zadanych wyrażeń regularnych w tekście."""
    text_lower = text.lower()
    return sum(len(re.findall(p, text_lower)) for p in patterns)


def extract_linguistic_markers(text: str) -> Dict[str, float]:
    """Ekstrahuje statystyki lingwistyczne: liczbę słów, gęstość wahań (hedging) oraz wskaźnik kategoryczności."""
    words = text.split()
    word_count = max(len(words), 1)

    hedging_count = count_pattern_matches(text, HEDGING_PATTERNS)
    certainty_count = count_pattern_matches(text, CERTAINTY_PATTERNS)

    return {
        "word_count": word_count,
        "hedging_count": hedging_count,
        "certainty_count": certainty_count,
        "hedging_density": hedging_count / word_count,
        "certainty_density": certainty_count / word_count,
        "certainty_to_hedging_ratio": (certainty_count + 1) / (hedging_count + 1),
    }


def compute_semantic_similarity(
    texts_a: List[str], texts_b: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> np.ndarray:
    """Oblicza podobieństwo kosinusowe między parami tekstów za pomocą Sentence-Transformers.
    (Wykorzystanie podejścia z poprzedniego projektu rekomendacji).
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import paired_cosine_distances

    model = SentenceTransformer(model_name)
    emb_a = model.encode(texts_a, show_progress_bar=False, batch_size=32)
    emb_b = model.encode(texts_b, show_progress_bar=False, batch_size=32)

    # 1 - cosine distance = cosine similarity
    similarities = 1.0 - paired_cosine_distances(emb_a, emb_b)
    return similarities
