"""Moduł do przekształcania dziennych interakcji studentów w sekwencje czasowe (N, T, F)."""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, List, Optional


def build_daily_student_sequences(
    student_vle: pd.DataFrame,
    vle: pd.DataFrame,
    student_reg: pd.DataFrame,
    max_days: int = 100,
    prediction_horizon: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Generuje trójwymiarowy tensor cech sekwencyjnych (Batch, Time_steps, Features)
    oraz wektor etykiet porzucenia kursu w zadanym horyzoncie czasowym (np. 7 dni).

    Args:
        student_vle: DataFrame z interakcjami (studentVle.csv)
        vle: Metadane zasobów (vle.csv)
        student_reg: Daty rejestracji i wyrejestrowania (studentRegistration.csv)
        max_days: Maksymalna liczba analizowanych dni od startu kursu
        prediction_horizon: Okno predykcji porzucenia (domyślnie 7 dni)

    Returns:
        X (torch.Tensor): Tensor sekwencji (N, T, F)
        y (torch.Tensor): Wektor etykiet binarnych (0 = kontynuacja, 1 = dropout)
        student_ids (List[int]): Lista identyfikatorów studentów odpowiadająca indeksom tensora
    """
    # TODO: Implementacja agregacji pivot / resample do macierzy (N, T, F)
    # F może zawierać: [clicks_content, clicks_forum, clicks_quiz, time_gap, cumsum_clicks]
    pass
