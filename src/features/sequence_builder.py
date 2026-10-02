"""Moduł do przekształcania dziennych interakcji studentów w sekwencje czasowe (N, T, F)."""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional


FEATURE_ACTIVITY_COLS = [
    "oucontent",
    "forumng",
    "quiz",
    "resource",
    "subpage",
    "url",
    "other",
]


def extract_daily_activity_matrix(
    student_vle: pd.DataFrame, vle: pd.DataFrame, max_day: int = 60
) -> pd.DataFrame:
    """Łączy logi interakcji ze słownikiem zasobów i agreguje kliknięcia w ujęciu:
    (id_student, date, activity_type).
    """
    merged = student_vle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    merged["activity_type"] = merged["activity_type"].fillna("other")
    merged["activity_type"] = merged["activity_type"].apply(
        lambda x: x if x in FEATURE_ACTIVITY_COLS else "other"
    )

    # Filtrujemy tylko dni do wybranego horyzontu obserwacji
    merged = merged[(merged["date"] >= 0) & (merged["date"] <= max_day)]

    # Pivot: indeks=(id_student, date), kolumny=activity_type, wartości=sum_click
    daily = (
        merged.groupby(["id_student", "date", "activity_type"])["sum_click"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Upewniamy się, że wszystkie kolumny aktywności istnieją
    for col in FEATURE_ACTIVITY_COLS:
        if col not in daily.columns:
            daily[col] = 0

    daily["total_clicks"] = daily[FEATURE_ACTIVITY_COLS].sum(axis=1)
    return daily


def build_student_sequences(
    student_vle: pd.DataFrame,
    vle: pd.DataFrame,
    student_reg: pd.DataFrame,
    observation_window_days: int = 60,
    prediction_horizon_days: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[str]]:
    """Tworzy trójwymiarowy tensor PyTorch (N, T, F) dla każdego studenta w oknie obserwacji
    oraz binarny wektor celu: czy student zrezygnował w horyzoncie [T, T + prediction_horizon_days].

    Returns:
        X (torch.Tensor): Kształt (N_students, observation_window_days, num_features)
        y (torch.Tensor): Kształt (N_students,) z etykietami 0/1
        student_ids (List[int]): Lista identyfikatorów studentów
        feature_names (List[str]): Nazwy cech w wymiarze F
    """
    daily = extract_daily_activity_matrix(student_vle, vle, max_day=observation_window_days)
    student_ids = sorted(student_reg["id_student"].unique())

    # Mapowanie unregistration do słownika
    unreg_dict = dict(zip(student_reg["id_student"], student_reg["date_unregistration"]))

    feature_names = FEATURE_ACTIVITY_COLS + [
        "total_clicks",
        "is_active_day",
        "days_since_last_active",
        "cumulative_clicks",
    ]
    num_features = len(feature_names)
    num_students = len(student_ids)

    X_array = np.zeros((num_students, observation_window_days, num_features), dtype=np.float32)
    y_array = np.zeros(num_students, dtype=np.float32)

    # Indeksowanie po studentach
    grouped = daily.groupby("id_student")
    student_daily_dict = {s_id: group.set_index("date") for s_id, group in grouped}

    for idx, s_id in enumerate(student_ids):
        # Etykieta celu: dropout w oknie [observation_window_days, observation_window_days + prediction_horizon_days]
        unreg_date = unreg_dict.get(s_id, np.nan)
        if pd.notna(unreg_date):
            if observation_window_days <= unreg_date <= (observation_window_days + prediction_horizon_days):
                y_array[idx] = 1.0

        # Wypełnianie sekwencji czasowej dzień po dniu
        if s_id in student_daily_dict:
            s_data = student_daily_dict[s_id]
            last_active = 0
            cum_clicks = 0.0

            for d in range(observation_window_days):
                if d in s_data.index:
                    row = s_data.loc[d]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    act_vals = [float(row[c]) for c in FEATURE_ACTIVITY_COLS]
                    tot = float(row["total_clicks"])
                    is_active = 1.0 if tot > 0 else 0.0
                    if is_active:
                        last_active = 0
                    else:
                        last_active += 1
                    cum_clicks += tot
                else:
                    act_vals = [0.0] * len(FEATURE_ACTIVITY_COLS)
                    tot = 0.0
                    is_active = 0.0
                    last_active += 1

                step_features = act_vals + [tot, is_active, float(last_active), cum_clicks]
                X_array[idx, d, :] = step_features

    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    y_tensor = torch.tensor(y_array, dtype=torch.float32)

    return X_tensor, y_tensor, student_ids, feature_names
