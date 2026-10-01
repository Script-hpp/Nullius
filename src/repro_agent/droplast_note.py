"""Hinweis "Paperzahl passt zur drop_last-Rechnung" (ETTh1). Deterministisch, kein LLM, nur Bericht: Urteil und Hauptmetrik bleiben auf allen
Fenstern. Die Zahlen kommen aus harness/droplast.py (Attempt.metrics_droplast), gehen nur ins Dossier und Zeugnis und nie in einen Prompt.

Regel (Reviewer, 2026-10-03): Hinweis, wenn es eine Batchgroesse b gibt mit
    |Behauptung - droplast(b)| <= NEAR_FRACTION * Toleranz   UND   |Behauptung - droplast(b)| < |Behauptung - Vollwert|.
Die Paperzahl trifft die drop_last-Rechnung also fast genau und besser als die volle Auswertung. Zwei Texte (Web/Zeugnis):
(a) Vollwert ausserhalb der Toleranz: "may explain the difference"; (b) Vollwert innerhalb der Toleranz: "matches almost exactly ... still within tolerance".
"""

from __future__ import annotations

import statistics

# tol/4: die Paperzahl muss den drop_last-Wert fast genau treffen. Bei Toleranz 0,01 sind das 0,0025, etwa zwei Drittel des ganzen Effekts
# (RLinear: Vollwert 0,3704, drop_last(128) 0,3665, Abstand 0,0039). Eine groessere Schwelle wuerde bei zwei Zahlen feuern, die nur zufaellig
# nah beieinander liegen (die drop_last-Werte zu 32/64/128/256 liegen oft nur 0,001 bis 0,005 auseinander).
NEAR_FRACTION = 0.25

SOURCES = [
    {"label": "FITS README, Important Notice (2023-12-25)", "url": "https://github.com/VEWOXIC/FITS"},
    {"label": "SparseTSF, Appendix D.1 (arXiv 2405.00946)", "url": "https://arxiv.org/abs/2405.00946"},
    {"label": "TFB: Towards Comprehensive and Fair Benchmarking (arXiv 2403.20150)", "url": "https://arxiv.org/abs/2403.20150"},
]
BATCH_CONTEXT = {
    "32": "the batch size of the original LTSF-Linear and Informer ETTh1 scripts",
    "64": "a batch size discussed in the TFB study",
    "128": "the batch size of the PatchTST and RLinear ETTh1 scripts",
    "256": "the batch size of the original SparseTSF ETTh1 script",
}


def droplast_note(metric: str, claimed: float, tolerance: float, full: float, per_seed: list[dict]) -> dict | None:
    """per_seed: metrics_droplast je gueltigem Seed ({"128": {"mse", "mae", "windows", "dropped"}, ...}). None = kein Hinweis."""
    if metric not in ("mse", "mae") or not per_seed or tolerance <= 0:
        return None
    sizes = sorted({b for d in per_seed for b in d}, key=int)
    best = None
    for b in sizes:
        rows = [d[b] for d in per_seed if b in d and metric in d[b]]
        if not rows:
            continue
        value = statistics.mean(r[metric] for r in rows)
        dist = abs(claimed - value)
        if dist <= NEAR_FRACTION * tolerance and dist < abs(claimed - full) and (best is None or dist < best[0]):
            best = (dist, b, value, rows[0])
    if best is None:
        return None
    dist, b, value, row = best
    return {
        "metric": metric, "claimed": claimed, "tolerance": tolerance, "full": full, "droplast": value,
        "batch_size": int(b), "windows_dropped": row["dropped"], "windows_total": row["windows"] + row["dropped"],
        "distance_full": abs(claimed - full), "distance_droplast": dist,
        "kind": "within_tolerance" if abs(claimed - full) <= tolerance else "outside_tolerance",
        "batch_context": BATCH_CONTEXT.get(b), "n_seeds": len(per_seed), "sources": SOURCES,
    }
