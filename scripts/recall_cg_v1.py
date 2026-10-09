"""recall cg v1: Konstanten, Laufplan, Stufen-Gate, Zaehlregel, Budget-Hilfen. Reine Logik, kein Modell-, Tavily- oder Sandbox-Aufruf.

Vorab-Festlegung: benchmark/experiments/recall_cg_v1.md. Dieses Modul wird VOR dem ersten Modellaufruf committet; danach aendern sich Einheiten,
Regeln, Schwellen und Deckel nicht mehr. Laufskript: scripts/run_recall_cg_v1.py, Auswertung: scripts/eval_recall_cg_v1.py.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "benchmark" / "experiments" / "recall_cg_v1.md"
STORE = ROOT / "runs" / "recall_cg_v1.jsonl"
CONFIG = "recall_cg_v1"
MODE = "Nachbau aus Paper und Autorencode"

DEV_UNITS = [   # Reihenfolge = Startreihenfolge in Stufe 1 (langsame Einheiten zuerst, damit sie mit den schnellen ueberlappen)
    "shchur_gat_2018", "shchur_pitfalls_2018", "velickovic_gat_2018", "velickovic_dgi_2019",
    "kipf_gcn_2017", "wu_sgc_2019", "zeng_nlinear_2023", "zeng_dlinear_2023",
]
HELDOUT = "rlinear_2023"
ALL_UNITS = DEV_UNITS + [HELDOUT]
CLAIMS = {   # echter Claim, Placebo-Claim (benchmark/papers.yaml)
    "kipf_gcn_2017": ("gcn_cora_acc", "gcn_cora_acc_placebo"),
    "shchur_pitfalls_2018": ("shchur_gcn_cora_acc", "shchur_gcn_cora_acc_placebo"),
    "zeng_dlinear_2023": ("dlinear_etth1_96_mse", "dlinear_etth1_96_mse_placebo"),
    "wu_sgc_2019": ("sgc_cora_acc", "sgc_cora_acc_placebo"),
    "velickovic_gat_2018": ("gat_cora_acc", "gat_cora_acc_placebo"),
    "velickovic_dgi_2019": ("dgi_cora_acc", "dgi_cora_acc_placebo"),
    "shchur_gat_2018": ("shchur_gat_cora_acc", "shchur_gat_cora_acc_placebo"),
    "zeng_nlinear_2023": ("nlinear_etth1_96_mse", "nlinear_etth1_96_mse_placebo"),
    "rlinear_2023": ("rlinear_etth1_96_mse", "rlinear_etth1_96_mse_placebo"),
}
# Produktkonfiguration von v2 bis v4 (graph_repair, fixture, Cache aus) plus code_grounded und io_helper (die Fixes F1 bis F4 und F8 sind im Code)
ORCH_KWARGS = dict(use_tavily=True, fresh=True, setup_details=True, tavily_mode="fixture", graph_repair=True,
                   code_grounded=True, io_helper=True, llm_cache=False, error_search_live=False)
TRAIN_TIMEOUTS = {"velickovic_gat_2018": 660, "shchur_pitfalls_2018": 900}   # recall_cg_gate_g1_timeouts.json (Regel timeouts.train_timeout); sonst 600

# ---- Budget ------------------------------------------------------------------------------------------------------------------
VAT = 0.19
CAP_GROSS_USD = 9.00          # Kostendeckel (Entscheidung 2026-10-09)
STOP_GROSS_USD = float(__import__("os").environ.get("RECALL_CG_V1_STOP_GROSS_USD", "8.75"))   # Verlaengerung nur nach Entscheidung der Nutzerin/des Nutzers, hoechstens 10.25 (Nachtrag 1)
# harter Stopp auf Aufruf-Ebene (LLM-Client verweigert jeden Aufruf ab hier)
ABORT_FRACTION = 0.95         # costs.ABORT_FRACTION
BUDGET_TOTAL_FOR_STOP = STOP_GROSS_USD / ABORT_FRACTION   # BUDGET_TOTAL_USD fuer die Worker, zusammen mit BUDGET_SINCE = Start des Laufs
MAX_COST_PER_RUN_USD = 0.75   # je Lauf (brutto); v2 bis v4: 0.50. Gen 0 enthaelt zwei Claims, cg-Prompts sind laenger
RESERVE_GROSS_USD = {"placebo": 0.60, "real": 0.40}   # je laufendem Job: Reserve beim Start eines weiteren (konservativ gegen v4: Mittel 0.24, Max 0.36 mit Placebo)
WORKERS = 5                   # parallele Laeufe; MAX_PARALLEL_SEEDS=3 je Lauf => hoechstens 15 Branches (16 getestet)
REASONING_MODEL = "nvidia/nemotron-3-super-120b-a12b"   # Nachtrag 1: Ultra von Token Factory entfernt (404 ab ca. 01:52Z), Super fuer die Rolle reasoning wie in G0 und G1
EX_OK, EX_FIXTURE, EX_BUDGET, EX_BILLING, EX_INFRA = 0, 2, 4, 3, 75


def worker_env(since_iso: str) -> dict[str, str]:
    return {"LLM_PROVIDER": "tokenfactory", "TF_MODEL_REASONING": REASONING_MODEL, "LLM_CACHE": "off", "SANDBOX_BACKEND": "nebius", "TAVILY_MODE": "fixture",
            "MAX_PARALLEL_SEEDS": "3", "MAX_COST_PER_RUN_USD": str(MAX_COST_PER_RUN_USD),
            "BUDGET_TOTAL_USD": f"{BUDGET_TOTAL_FOR_STOP:.4f}", "BUDGET_SINCE": since_iso}


# ---- Referenzurteile -----------------------------------------------------------------------------------------------------------
def load_refs(doc: Path = DOC) -> dict:
    for block in re.findall(r"```yaml\n(.*?)```", doc.read_text(), flags=re.S):
        data = yaml.safe_load(block)
        if isinstance(data, dict) and "reference_verdicts" in data:
            return data["reference_verdicts"]
    raise SystemExit(f"Kein YAML-Block `reference_verdicts` in {doc}")


def reference_verdicts(doc: Path = DOC) -> dict[str, dict]:
    """Je Einheit: Zielwert, Placebo-Zielwert, Referenzmittel, Referenzurteil echt und Placebo (judge mit Toleranz aus dem Block)."""
    from repro_agent.harness.verify import judge
    refs = load_refs(doc)
    tol = refs["tolerance"]
    return {u: {**v, "ref": judge(v["reference_mean"], v["target"], tol), "ref_placebo": judge(v["reference_mean"], v["placebo_target"], tol)}
            for u, v in refs["units"].items()}


# ---- Klassen und Mehrheit ---------------------------------------------------------------------------------------------------
VALID = ("reproduziert", "teilweise", "nicht reproduziert")
AUSFALL, NO_MAJORITY, INCOMPLETE = "Ausfall", "keine Mehrheit", "unvollstaendig"


def klass(urteil: str) -> str:
    return urteil if urteil in VALID else AUSFALL


def real_classes(rows: list[dict], unit: str) -> dict[int, str]:
    """Generation -> Klasse des echten Claims (Zeilen dieser Konfiguration). Abbrueche durch Budget oder Billing stehen nie im Store."""
    return {r["generation"]: klass(r["urteil"]) for r in rows if r["paper"] == unit and r.get("config") == CONFIG and r["origin"] == "paper"}


def unit_majority(classes: dict[int, str]) -> tuple[str, bool]:
    """(Klasse der Einheit, braucht Generation 2). Gen 0 und 1 gleich: die Klasse steht. Verschieden: Generation 2 entscheidet (mindestens 2 von 3
    gleich), fehlt sie, ist die Einheit unvollstaendig. Fehlt Gen 0 oder 1: unvollstaendig, keine Folgegeneration ableitbar."""
    if 0 not in classes or 1 not in classes:
        return INCOMPLETE, False
    if classes[0] == classes[1]:
        return classes[0], False
    if 2 not in classes:
        return INCOMPLETE, True
    top, n = Counter(classes.values()).most_common(1)[0]
    return (top if n >= 2 else NO_MAJORITY), False


def matches(majority: str, ref: str) -> bool:
    return majority == ref   # Ausfall, keine Mehrheit, unvollstaendig sind nie ein Treffer


# ---- Stufen-Gate (recall_plan.md Abschnitt 6) ----------------------------------------------------------------------------------
def gate(rows: list[dict], ref_units: dict) -> dict:
    """F = Entwicklungseinheiten, bei denen beide Generationen die Referenzklasse treffen; U = genau eine. Weiter nur bei F >= 2 und F + U >= 4.
    `complete` ist falsch, solange eine Stufe-1-Zelle fehlt (dann entscheidet das Gate nicht)."""
    ref = {u: ref_units[u]["ref"] for u in DEV_UNITS}
    f = u_ = 0
    complete = True
    for u in DEV_UNITS:
        c = real_classes(rows, u)
        if 0 not in c or 1 not in c:
            complete = False
            continue
        hits = (c[0] == ref[u]) + (c[1] == ref[u])
        f += hits == 2
        u_ += hits == 1
    return {"F": f, "U": u_, "complete": complete, "continue": complete and f >= 2 and f + u_ >= 4}


# ---- Laufplan -------------------------------------------------------------------------------------------------------------------
Job = tuple[str, int, bool]   # (Einheit, Generation, mit Placebo)


def stage1_jobs() -> list[Job]:
    """8 Entwicklungseinheiten x Generation 0 (mit Placebo) und 1, echter Claim. Erst alle Generationen 0 (Placebo-Ergebnisse frueh), dann 1."""
    return [(u, g, g == 0) for g in (0, 1) for u in DEV_UNITS]


def stage2_jobs(rows: list[dict]) -> list[Job]:
    """Reihenfolge nach recall_plan.md: RLinear Gen 0 (Placebo) und 1, danach Entwicklungs-Generation 2 dort, wo Gen 0 und 1 in der Klasse abweichen,
    zuletzt RLinear Gen 2, wenn Gen 0 und 1 abweichen. Der Aufrufer wiederholt den Aufruf, bis nichts Neues kommt."""
    jobs: list[Job] = [(HELDOUT, 0, True), (HELDOUT, 1, False)]
    for u in DEV_UNITS:
        c = real_classes(rows, u)
        if 0 in c and 1 in c and c[0] != c[1]:
            jobs.append((u, 2, False))
    c = real_classes(rows, HELDOUT)
    if 0 in c and 1 in c and c[0] != c[1]:
        jobs.append((HELDOUT, 2, False))
    return jobs


def done(rows: list[dict], job: Job) -> bool:
    unit, gen, _ = job
    return any(r["paper"] == unit and r["generation"] == gen and r.get("config") == CONFIG for r in rows)


def claim_ids(job: Job) -> list[str]:
    real, placebo = CLAIMS[job[0]]
    return [real, placebo] if job[2] else [real]


def reserve(job: Job) -> float:
    return RESERVE_GROSS_USD["placebo" if job[2] else "real"]


def can_start(spent_gross: float, running: list[Job], nxt: Job) -> bool:
    """Reservierung: ein Lauf startet nur, wenn Verbrauch seit Start + Reserve aller laufenden Jobs + Reserve des neuen <= Stopp."""
    return spent_gross + sum(reserve(j) for j in running) + reserve(nxt) <= STOP_GROSS_USD


def load_rows(store: Path = STORE) -> list[dict]:
    rows: list[dict] = []
    if store.exists():
        for line in store.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
