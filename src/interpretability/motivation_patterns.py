"""Moduł analizy i identyfikacji wzorca kryzysu motywacyjnego przed porzuceniem kursu."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


def extract_inactivity_crisis_patterns(
    student_vle: pd.DataFrame,
    student_reg: pd.DataFrame,
    vle: pd.DataFrame,
    pre_dropout_window_days: int = 14,
) -> pd.DataFrame:
    """Identyfikuje wzorce behawioralne w okresie bezpośrednio poprzedzającym
    rezygnację z kursu (kryzys motywacyjny).
    
    Analizuje:
    - Nagły spadek średniej liczby kliknięć na dobę
    - Wzrost maksymalnej przerwy między logowaniami (days of inactivity)
    - Proporcję zaniechania quizów i sprawdzianów
    """
    withdrawn_students = student_reg[student_reg["date_unregistration"].notna()].copy()
    results = []

    merged = student_vle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")

    for _, row in withdrawn_students.iterrows():
        s_id = row["id_student"]
        unreg_day = int(row["date_unregistration"])
        window_start = unreg_day - pre_dropout_window_days

        s_logs = merged[(merged["id_student"] == s_id) & (merged["date"] <= unreg_day)]
        if s_logs.empty:
            continue

        # Logi przed kryzysem vs w oknie kryzysu
        pre_crisis_logs = s_logs[s_logs["date"] < window_start]
        crisis_logs = s_logs[(s_logs["date"] >= window_start) & (s_logs["date"] <= unreg_day)]

        pre_clicks = pre_crisis_logs["sum_click"].sum() if not pre_crisis_logs.empty else 0
        crisis_clicks = crisis_logs["sum_click"].sum() if not crisis_logs.empty else 0

        pre_days_active = pre_crisis_logs["date"].nunique()
        crisis_days_active = crisis_logs["date"].nunique()

        # Aktywność quizowa w oknie kryzysu
        quiz_clicks_crisis = crisis_logs[crisis_logs["activity_type"] == "quiz"]["sum_click"].sum()

        results.append({
            "id_student": s_id,
            "unreg_day": unreg_day,
            "pre_crisis_total_clicks": pre_clicks,
            "crisis_total_clicks": crisis_clicks,
            "pre_crisis_active_days": pre_days_active,
            "crisis_active_days": crisis_days_active,
            "crisis_quiz_clicks": quiz_clicks_crisis,
            "activity_drop_ratio": (
                (pre_clicks / max(pre_days_active, 1)) - (crisis_clicks / max(crisis_days_active, 1))
            ) if pre_days_active > 0 and crisis_days_active > 0 else np.nan,
        })

    return pd.DataFrame(results)
