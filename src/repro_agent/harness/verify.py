"""Unabhaengiger Pruef-Harness. Agent liefert Vorhersagen, keine Zahlen.

Vertrag: Agent schreibt /output/predictions.csv mit Spalten id,prediction. Nichts sonst zaehlt.

Ablauf im Branch, nach Agenten-Lauf:
1. Harness frisch aus Repo in Branch kopieren (Agent kann ihn vorher nicht veraendern)
2. Testlabels selbst aus Original-Datensatz laden, nie aus Dateien des Agenten
3. Plausibilitaet: IDs vollstaendig und eindeutig, nicht nur eine Klasse
4. Metrik mit festem Code berechnen
5. Mit Toleranz vergleichen; 3-5 Seeds fuer Stabilitaet
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Callable, Literal

Urteil = Literal["reproduziert", "teilweise", "nicht reproduziert", "ungueltig"]


@dataclass(frozen=True)
class VerifyResult:
    urteil: Urteil
    measured: float | None
    reason: str


def load_predictions(predictions_csv: str) -> dict[str, str]:
    with open(predictions_csv, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "prediction"]:
            raise ValueError(
                f"predictions.csv muss Spalten id,prediction haben, gefunden: {reader.fieldnames}"
            )
        return {row["id"]: row["prediction"] for row in reader}


def check_plausibility(predictions: dict[str, str], expected_ids: set[str]) -> str | None:
    """Gibt Fehlermeldung zurueck, wenn Plausibilitaets-Check fehlschlaegt, sonst None."""
    if set(predictions.keys()) != expected_ids:
        missing = expected_ids - predictions.keys()
        extra = predictions.keys() - expected_ids
        return f"IDs unvollstaendig oder falsch. Fehlend: {len(missing)}, ueberschuessig: {len(extra)}"
    if len(set(predictions.values())) <= 1:
        return "Nur eine einzige Vorhersage-Klasse ueber alle IDs -- verdaechtig"
    return None


def judge(measured: float, target: float, tolerance: float) -> Urteil:
    """Abstandstabelle aus CLAUDE.md: <= Toleranz, <= 3x Toleranz, darueber."""
    distance = abs(measured - target)
    if distance <= tolerance:
        return "reproduziert"
    if distance <= 3 * tolerance:
        return "teilweise"
    return "nicht reproduziert"


LOWER_IS_BETTER = {"mse", "mae"}


def _best_baseline(metric: str, baselines: dict | None) -> tuple[str, float] | None:
    values = {name: b[metric] for name, b in (baselines or {}).items() if metric in b}
    if not values:
        return None
    return (min if metric in LOWER_IS_BETTER else max)(values.items(), key=lambda kv: kv[1])


def _worse(metric: str, measured: float, reference: float) -> bool:
    return measured > reference if metric in LOWER_IS_BETTER else measured < reference


def agent_fault_suspected(
    metric: str, measured: float, target: float, baselines: dict | None,
    all_metrics: dict | None = None,
) -> str | None:
    """Deterministische Plausibilitaet: schlechter als die beste triviale Baseline auf der Metrik des
    Claims ODER auf einer anderen Harness-Metrik mit Baseline, obwohl der Paperwert die Baseline auf
    der Claim-Metrik schlaegt. Dann ist ein Fehler im Agenten-Code wahrscheinlicher als ein Befund
    ueber das Paper. Schlaegt schon der Paperwert die Baseline nicht, gilt der Schutz nicht (sonst
    wuerde ein Paper mit wirklich schwacher Methode dem Agenten angelastet).

    `all_metrics`: Mittelwerte aller Harness-Metriken derselben Laeufe (z. B. mse und mae).
    """
    best = _best_baseline(metric, baselines)
    if best is None:
        return None
    name, ref = best
    if not _worse(metric, ref, target):
        return None   # Paperwert schlaegt die Baseline nicht: kein Schutz
    checks = {metric: measured, **{m: v for m, v in (all_metrics or {}).items() if m != metric}}
    for m, value in checks.items():
        b = _best_baseline(m, baselines)
        if b and _worse(m, value, b[1]):
            return (f"{m.upper()} {value:.4f} ist schlechter als die triviale Baseline '{b[0]}' "
                    f"({b[1]:.4f}), der Paperwert {target} ({metric.upper()}) schlaegt sie")
    return None


def verify(
    predictions_csv: str,
    labels: dict[str, str | float],
    metric_fn: Callable[[list, list], float],
    target: float,
    tolerance: float,
    label_caster: Callable[[str], object] = float,
) -> VerifyResult:
    predictions = load_predictions(predictions_csv)

    plausibility_error = check_plausibility(predictions, set(labels.keys()))
    if plausibility_error:
        return VerifyResult(urteil="ungueltig", measured=None, reason=plausibility_error)

    ordered_ids = sorted(labels.keys())
    preds = [label_caster(predictions[i]) for i in ordered_ids]
    truth = [labels[i] for i in ordered_ids]

    measured = metric_fn(preds, truth)
    distance = abs(measured - target)

    return VerifyResult(urteil=judge(measured, target, tolerance), measured=measured, reason=f"Abstand zu Ziel: {distance:.4f}")
