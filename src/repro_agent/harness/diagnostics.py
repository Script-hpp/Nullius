"""Zusatzkennzahlen der Vorhersagen (ETTh1). Laeuft im Sandbox-Branch (nur numpy), kopiert wie die anderen Harness-Dateien.

`pred_std` ist die Streuung der VORHERSAGEN des Agenten: darf in die Rueckmeldung an den Agenten (relativ zur Trainingsstreuung, die
er kennt; Daten sind mit der Trainingsstatistik standardisiert, Trainingsstreuung = 1). `target_std` und `corr` beruehren die
Testziele und gehen nur ins Dossier, nie in einen Prompt (sonst koennte der Agent seine Vorhersagen darauf skalieren).
"""

from __future__ import annotations

import numpy as np


def prediction_diagnostics(pred: np.ndarray, truth: np.ndarray) -> dict:
    p, t = pred.reshape(-1).astype(np.float64), truth.reshape(-1).astype(np.float64)
    corr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 and t.std() > 0 else None
    return {"pred_std": float(p.std()), "target_std": float(t.std()), "corr": corr}
