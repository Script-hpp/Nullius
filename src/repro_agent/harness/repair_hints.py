"""Feste Rueckmeldung an den Agenten, wenn sein Ergebnis schlechter als eine triviale Baseline ist (kein LLM-Text).

Enthaelt nur: gemessenen Wert, Baselines (vom Harness, schon immer im Dossier), Paperwert und die Streuung der eigenen Vorhersagen
im Verhaeltnis zur Trainingsstreuung. Nie Statistik der Testziele (Streuung, Korrelation): sonst koennte der Agent seine Vorhersagen
darauf skalieren (Test tests/test_fault_repair.py).
"""

from __future__ import annotations

CHECKLIST_FORECAST = (
    "- The id is '{window}_{step}_{channel}' and the prediction array is ordered [window, step, channel]; do not swap step and channel.\n"
    "- Inputs and outputs stay in the standardised units of the data file; do not standardise again or undo the standardisation.\n"
    "- Window i uses only test_inputs[i]: its last L steps in time order, and predicts the 96 steps right after them.\n"
    "- Levels shift between training and test: if the plan or the paper uses input normalisation (for example RevIN, subtracting the last value or the window mean), normalise each input window and add it back to the output.\n"
    "- If you keep the best epoch, copy the weights with copy.deepcopy(model.state_dict()); state_dict() alone only references the live tensors and the restore does nothing.\n"
    "- Choose a learning rate that fits the model you actually train; a very large rate on a wide linear layer overfits and predicts worse than zero.\n"
)
CHECKLIST_CLASSIFICATION = (
    "- Predictions must be one class id per test node id, in the order of the test ids; prediction i belongs to node i.\n"
    "- Train only on the training labels. Features, labels and node indices must stay aligned; do not shuffle one without the other.\n"
    "- Use the graph edges as the plan describes (adjacency with self loops, symmetric normalisation), not just the node features.\n"
    "- If you keep the best epoch, copy the weights with copy.deepcopy(model.state_dict()); state_dict() alone only references the live tensors and the restore does nothing.\n"
    "- Check dropout, weight decay and the number of epochs against the plan.\n"
)


def fault_message(metric: str, measured: float, target: float, baselines: dict | None, pred_std: float | None = None) -> str:
    """Deterministische Rueckmeldung fuer repair_code. `baselines`: {name: {metric: wert}}."""
    # Nur Namen der Baselines, keine Zahlen: Baseline-Werte sind Aggregate der Testziele (z. B. MSE von Null = mittleres Quadrat).
    trivial = ", ".join(name for name, vals in (baselines or {}).items() if metric in vals)
    forecast = metric in ("mse", "mae")
    lines = [
        "Your script ran, but its result is worse than a trivial baseline, so the script is most likely wrong, not the paper.",
        f"Measured {metric.upper()}: {measured:.4f}. It is worse than these trivial baselines: {trivial or 'not available'}. The paper reports {target}.",
    ]
    if forecast and pred_std is not None:
        lines.append(f"The standard deviation of your predictions is {pred_std:.3f}; the training data (standardised) has 1.0.")
    lines.append("Check these points and return the corrected full script:")
    lines.append((CHECKLIST_FORECAST if forecast else CHECKLIST_CLASSIFICATION).rstrip())
    lines.append("Keep the method described in the plan: fix the implementation, do not replace the method by a simpler one.")
    return "\n".join(lines)
