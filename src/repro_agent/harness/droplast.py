"""Detektor "letzter Test-Batch verworfen" (ETTh1). Laeuft im Sandbox-Branch (nur numpy), kopiert wie die anderen Harness-Dateien.

Viele ETTh1-Papers vor 12/2023 wurden mit einem Test-Loader ausgewertet, der den letzten unvollstaendigen Batch verwirft
(`drop_last=True`), siehe benchmark/experiments/b1_candidates.md und b2_negative_case.md. Hier wird dieselbe Metrik zusaetzlich ueber die
Fenster gerechnet, die ein solcher Loader (ohne Shuffle, Testreihenfolge) liefern wuerde. Nur Information: die Hauptmetrik und das Urteil
bleiben auf ALLEN Fenstern. Die Werte beruehren die Testziele und gehen deshalb nur ins Dossier und ins Zeugnis, nie in einen Prompt.

Batch-Groessen und Belege (Stand der Repos, Datei:Zeile):
- 32: cure-lab/LTSF-Linear scripts/EXP-LongForecasting/Linear/etth1.sh:24 (--batch_size 32; Test-Loader der damaligen Version mit drop_last=True,
  im FITS-README als betroffen genannt), zhouhaoyi/Informer2020 exp/exp_informer.py:77 (drop_last = True im Test).
- 64: Beispiel in TFB (arXiv 2403.20150, "batch sizes to be 32, 64, and 128").
- 128: yuqinie98/PatchTST PatchTST_supervised/scripts/PatchTST/etth1.sh:42 und data_provider/data_factory.py:19 (drop_last = True, Stand 204c21efe0),
  plumprc/RTSF script.md:3 und data_provider/data_factory.py:18 (drop_last = True, Stand 0fec00104f).
- 256: lss-1138/SparseTSF scripts/SparseTSF/linear/etth1.sh (--batch_size 256; das Repo selbst setzt heute drop_last = False, data_factory.py:20).
Bei 2785 Testfenstern fallen 1, 33, 97 bzw. 225 Fenster weg.
"""

from __future__ import annotations

import numpy as np

DROPLAST_BATCHES = (32, 64, 128, 256)


def kept_windows(n_windows: int, batch: int) -> int:
    """Fenster, die ein Loader mit drop_last=True liefert: nur volle Batches."""
    return (n_windows // batch) * batch


def metrics_droplast(pred: np.ndarray, truth: np.ndarray, batches: tuple[int, ...] = DROPLAST_BATCHES) -> dict:
    """pred, truth: [W, H, C]. Rueckgabe {"128": {"mse", "mae", "windows", "dropped"}, ...}; Batchgroessen > W entfallen (nichts bliebe uebrig)."""
    n = pred.shape[0]
    out: dict[str, dict] = {}
    for b in batches:
        k = kept_windows(n, b)
        if k <= 0:
            continue
        diff = pred[:k].astype(np.float64) - truth[:k].astype(np.float64)
        out[str(b)] = {"mse": float((diff ** 2).mean()), "mae": float(np.abs(diff).mean()), "windows": k, "dropped": n - k}
    return out
