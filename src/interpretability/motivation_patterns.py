"""Moduł analizy i identyfikacji wzorca kryzysu motywacyjnego przed porzuceniem kursu."""

import pandas as pd
import numpy as np
from typing import Dict, List


def detect_inactivity_gaps(
    daily_clicks: pd.DataFrame, gap_threshold_days: int = 5
) -> pd.DataFrame:
    """Wykrywa anomalie w postaci nagłych przerw w aktywności
    (dłuższych niż zadany próg dni) poprzedzających decyzję o wyrejestrowaniu.
    """
    # TODO: Analiza przerw między kolejnymi logowaniami a oknem unregistration
    pass
