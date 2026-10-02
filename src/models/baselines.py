"""Klasyczne modele bazowe do predykcji perswazyjności argumentów (TF-IDF + reg. logistyczna / RF)."""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from src.utils.metrics import compute_classification_metrics
from src.features.text_features import extract_linguistic_markers


def build_tabular_features(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Generuje tabelaryczną macierz cech lingwistycznych i długości tekstów."""
    rows = []
    y = []

    for rec in records:
        op = rec["op_text"]
        # Przypadek pozytywny (Delta)
        feat_pos = extract_linguistic_markers(rec["delta_argument"])
        feat_pos["arg_len"] = len(rec["delta_argument"])
        feat_pos["op_len"] = len(op)
        rows.append(list(feat_pos.values()))
        y.append(1)

        # Przypadek negatywny (No Delta)
        feat_neg = extract_linguistic_markers(rec["nodelta_argument"])
        feat_neg["arg_len"] = len(rec["nodelta_argument"])
        feat_neg["op_len"] = len(op)
        rows.append(list(feat_neg.values()))
        y.append(0)

    return np.array(rows, dtype=np.float32), np.array(y, dtype=np.int64)


def train_baseline_persuasion_models(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Dict[str, Any]:
    """Trenuje Regresję Logistyczną i Random Forest na cechach lingwistycznych."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 1. Regresja Logistyczna
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_val_scaled)[:, 1]

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_val)[:, 1]

    return {
        "LogisticRegression": {
            "model": lr,
            "metrics": compute_classification_metrics(y_val, lr_probs),
        },
        "RandomForest": {
            "model": rf,
            "metrics": compute_classification_metrics(y_val, rf_probs),
        },
    }
