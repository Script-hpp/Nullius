"""Handpruefung der extrahierten Bausteinlisten und Einfrieren (reviewed=true).

Die Modell-Extraktion verwirft bei PDF-Formeln zu Recht Zitate (Symbole werden bereinigt); die
tragenden Bausteine stehen dann nicht in der Liste. Hier werden sie ergaenzt, Zitate werden als
TEILSTRING aus dem Paper-Text gezogen (also woertlich) und gegen den Text geprueft.
  uv run python scripts/freeze_components.py
"""

from __future__ import annotations

import json
import time

from repro_agent.claims_source import _norm
from repro_agent.dossier import REPO_ROOT
from repro_agent.fidelity import COMPONENT_DIR, Component, load_components, save_components

FIX = REPO_ROOT / "benchmark" / "fixtures"
REVIEW = ("Handpruefung durch Claude Code gegen den Paper-Text (Zitate programmatisch als Teilstring "
          "gezogen), noch nicht vom Projektteam gegengelesen")


def slice_between(text: str, start: str, end: str | None = None, length: int | None = None) -> str:
    i = text.index(start)
    j = text.index(end, i) if end else i + length
    return text[i:j].replace("\n", " ").strip()


def main() -> None:
    # ---- DLinear
    t = json.loads((FIX / "2205.13504.json").read_text())["text"]
    comps = load_components("zeng_dlinear_2023")
    for c in comps:
        if c.id == "shared_weights_across_variates":
            c.critical = False        # Variante 'individual' ist ebenfalls DLinear
    comps = [c for c in comps if c.id != "moving_average_kernel_25"]   # idempotent
    comps.append(Component(
        id="moving_average_kernel_25", name="Moving-Average-Kernel 25",
        description="The moving average used for the decomposition has kernel size 25.",
        quote=slice_between(t, "For DLinear , the moving average kernel size", "which is the same as Autoformer"),
        critical=True, value="25"))
    for c in comps:
        c.reviewed = True
        assert _norm(c.quote) in _norm(t), c.id
    meta = json.loads((COMPONENT_DIR / "zeng_dlinear_2023.json").read_text())["meta"]
    meta.update(reviewed_by=REVIEW, reviewed_on=time.strftime("%Y-%m-%d"))
    save_components("zeng_dlinear_2023", comps, meta)

    # ---- GCN
    t = json.loads((FIX / "1609.02907.json").read_text())["text"]
    llm_ids = {"full_batch_gradient_descent", "dropout_regularization", "precomputed_normalized_adjacency"}
    comps = load_components("kipf_gcn_2017")
    for c in comps:
        if c.id in llm_ids:
            c.critical = False        # Trainingsdetails, nicht definierend
    added = {"renormalized_adjacency", "self_loops", "propagation_rule", "two_graph_conv_layers"}
    comps = [c for c in comps if c.id not in added]                    # idempotent
    comps += [
        Component("renormalized_adjacency", "Renormalisierte Adjazenz",
                  "The propagation matrix is D~^-1/2 A~ D~^-1/2 with A~ = A + I (renormalization trick).",
                  slice_between(t, "renormalization trick :", ", with"), True),
        Component("self_loops", "Selbstschleifen",
                  "Self-loops are added to the adjacency matrix: A~ = A + I.",
                  slice_between(t, "˜A = A + IN and", length=25), True),
        Component("propagation_rule", "Propagationsregel",
                  "Each layer multiplies the normalised adjacency with the layer input before the weight matrix: H' = sigma(A_hat H W).",
                  slice_between(t, "layer-wise propagation rule", length=120), True),
        Component("two_graph_conv_layers", "Zwei Graph-Faltungsschichten",
                  "The model has two graph convolution layers (the adjacency is applied in both).",
                  slice_between(t, "we consider a two-layer GCN for semi-supervised node classification", length=68), True),
    ]
    for c in comps:
        c.reviewed = True
        assert _norm(c.quote) in _norm(t), c.id
    meta = json.loads((COMPONENT_DIR / "kipf_gcn_2017.json").read_text())["meta"]
    meta.update(reviewed_by=REVIEW, reviewed_on=time.strftime("%Y-%m-%d"))
    save_components("kipf_gcn_2017", comps, meta)
    for pid in ("zeng_dlinear_2023", "kipf_gcn_2017"):
        print(pid)
        for c in load_components(pid):
            print(f"  [{'kritisch' if c.critical else 'klein'}] {c.id} | Wert {c.value} | {c.quote[:90]!r}")


if __name__ == "__main__":
    main()
