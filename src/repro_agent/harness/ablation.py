"""Auswertung der Graph-Ablation (Regel: benchmark/experiments/ablation_gcn.md, vorab festgelegt)."""

from __future__ import annotations

USED_DROP = 0.10      # Abfall >= : Graph genutzt
UNUSED_DROP = 0.02    # Abfall <= : Graph nicht genutzt


def classify_ablation(acc_full: float | None, acc_ablated: float | None) -> str:
    """Graph genutzt | Graph nicht genutzt | unklar | nicht messbar."""
    if acc_full is None or acc_ablated is None:
        return "nicht messbar"
    drop = round(acc_full - acc_ablated, 9)   # Gleitkomma: 0,80 - 0,78 soll genau 0,02 sein
    if drop >= USED_DROP:
        return "Graph genutzt"
    if drop <= UNUSED_DROP:
        return "Graph nicht genutzt"
    return "unklar"
