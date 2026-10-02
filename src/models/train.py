"""Skrypt treningowy dla modeli perswazji i ewolucji postaw."""

import argparse
from typing import Dict, Any
from src.data.loader import load_cmv_sample
from src.models.baselines import build_tabular_features, train_baseline_persuasion_models


def run_baseline_training():
    """Wczytuje próbkę danych i trenuje klasyczne modele bazowe."""
    print("Wczytywanie przetworzonych par ze zbioru CMV...")
    data = load_cmv_sample()
    print(f"Załadowano {len(data)} par.")

    X, y = build_tabular_features(data)
    print(f"Kształt macierzy cech: {X.shape}, rozkład etykiet: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split train/val
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("Trening modeli bazowych (Logistic Regression, Random Forest)...")
    results = train_baseline_persuasion_models(X_train, y_train, X_val, y_val)

    for name, res in results.items():
        print(f"\n--- Wyniki dla: {name} ---")
        for m_name, m_val in res["metrics"].items():
            if isinstance(m_val, float):
                print(f"  {m_name}: {m_val:.4f}")
            else:
                print(f"  {m_name}: {m_val}")


if __name__ == "__main__":
    import numpy as np
    run_baseline_training()
