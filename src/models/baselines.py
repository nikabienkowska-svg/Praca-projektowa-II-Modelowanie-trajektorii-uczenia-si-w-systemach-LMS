"""Klasyczne modele bazowe (Baseline) do porównania z sieciami RNN/LSTM."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.utils.metrics import compute_classification_metrics


def flatten_sequences_to_tabular_features(X_seq: np.ndarray, feature_names: list) -> np.ndarray:
    """Konwertuje tensor sekwencji (N, T, F) do zagregowanej tabeli cech (N, F_agg)
    dla klasycznych modeli ML (średnie, sumy, odchylenia standardowe, wartości końcowe).
    """
    N, T, F = X_seq.shape
    features_list = []

    # 1. Suma po całym okresie
    features_list.append(X_seq.sum(axis=1))
    # 2. Średnia
    features_list.append(X_seq.mean(axis=1))
    # 3. Odchylenie standardowe
    features_list.append(X_seq.std(axis=1))
    # 4. Stan w ostatnim dniu obserwacji
    features_list.append(X_seq[:, -1, :])
    # 5. Stan w ostatnich 7 dniach obserwacji (średnia z ostatniego tygodnia)
    if T >= 7:
        features_list.append(X_seq[:, -7:, :].mean(axis=1))

    X_tabular = np.concatenate(features_list, axis=1)
    return X_tabular


def train_baseline_models(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
) -> Dict[str, Dict[str, Any]]:
    """Trenuje Regresję Logistyczną i Random Forest na zagregowanych cechach sekwencji."""
    X_train_tab = flatten_sequences_to_tabular_features(X_train_seq, feature_names)
    X_val_tab = flatten_sequences_to_tabular_features(X_val_seq, feature_names)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_tab)
    X_val_scaled = scaler.transform(X_val_tab)

    results = {}

    # 1. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_val_scaled)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros_like(y_val)
    results["LogisticRegression"] = {
        "model": lr,
        "metrics": compute_classification_metrics(y_val, lr_probs) if len(np.unique(y_val)) > 1 else {},
    }

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(X_train_tab, y_train)
    rf_probs = rf.predict_proba(X_val_tab)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros_like(y_val)
    results["RandomForest"] = {
        "model": rf,
        "metrics": compute_classification_metrics(y_val, rf_probs) if len(np.unique(y_val)) > 1 else {},
    }

    return results
