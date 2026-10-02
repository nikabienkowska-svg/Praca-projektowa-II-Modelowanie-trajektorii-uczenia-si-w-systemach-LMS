"""Test jednostkowy sprawdzający spójność modułów CMV, lingwistyki i baseline'ów."""

import os
import json
from src.features.text_features import extract_linguistic_markers
from src.interpretability.belief_trajectory import compute_belief_trajectory, detect_inflection_point
from src.models.baselines import build_tabular_features, train_baseline_persuasion_models


def test_cmv_nlp_pipeline():
    # 1. Sprawdzenie ekstrakcji markerów lingwistycznych
    text_sure = "This is definitely a proven fact and undeniably true."
    stats_sure = extract_linguistic_markers(text_sure)
    assert stats_sure["certainty_count"] >= 2
    assert stats_sure["hedging_count"] == 0

    text_tentative = "Perhaps you have a point, but maybe there is another side."
    stats_tent = extract_linguistic_markers(text_tentative)
    assert stats_tent["hedging_count"] >= 2

    # 2. Sprawdzenie trajektorii przekonań
    comments_seq = [
        "You are wrong. It is impossible to dispute my initial statement.",
        "I still think I am right, but I might see what you mean.",
        "Fair point. Perhaps I had not considered that perspective. I concede.",
    ]
    traj_df = compute_belief_trajectory(comments_seq)
    assert len(traj_df) == 3
    inflection = detect_inflection_point(traj_df)
    assert inflection in [1, 2]

    # 3. Sprawdzenie budowy cech dla par
    mock_records = [
        {
            "op_text": "I think X is true.",
            "delta_argument": "Consider Y, you might find it convincing.",
            "nodelta_argument": "You are just wrong.",
        }
    ] * 20
    X, y = build_tabular_features(mock_records)
    assert X.shape[0] == 40
    assert len(y) == 40

    # 4. Sprawdzenie modeli bazowych
    res = train_baseline_persuasion_models(X[:30], y[:30], X[30:], y[30:])
    assert "LogisticRegression" in res
    assert "RandomForest" in res

    print("\n✅ Wszystkie testy modułów NLP, trajektorii i modeli bazowych przeszły pomyślnie!")


if __name__ == "__main__":
    test_cmv_nlp_pipeline()
