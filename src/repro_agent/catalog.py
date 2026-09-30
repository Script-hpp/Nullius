"""Katalog-Adapter: Discovery-Kandidaten als Loop-Eingabe (benchmark/catalog.yaml, papers.yaml bleibt unberuehrt).

Ein Kandidat ist gelesen, nicht nachgerechnet. Der Adapter rechnet ihn in dieselbe Form um, die der Loop aus
papers.yaml kennt, und haelt Status und Urteil nach dem Lauf fest. Keine Netz-, LLM- oder Sandbox-Aufrufe hier.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .dossier import REPO_ROOT, Claim

CATALOG_YAML = REPO_ROOT / "benchmark" / "catalog.yaml"
DISCOVERY_CACHE = REPO_ROOT / "logs" / "discovery_cache.json"
ACCEPTED_CACHE_VERSIONS = (2, 3)   # Discovery schreibt 3; 2 hat kein Feld input_length (= null)

UNCHECKED, CHECKED, FAILED, NOT_RUNNABLE = "ungeprueft", "geprueft", "lauf_fehlgeschlagen", "nicht_laufbar"
TOLERANCE = 0.01   # fest, vorab: +-1 Prozentpunkt (Accuracy), +-0.01 absolut (MSE/MAE). Paper-Streuung aendert sie nicht.
ETTH1_HORIZON = 96
# Seeds je Task, wenn der Eintrag keine nennt (Kosten: GCN ~25 s pro Seed, siehe docs/next_step_catalog_pipeline.md)
TASK_SEEDS = {"Cora": 5, "Cora-LCC": 5, "ETTh1": 3}
TASK_METRICS = {"Cora": {"accuracy"}, "Cora-LCC": {"accuracy"}, "ETTh1": {"mse", "mae"}}
TASK_SUFFIX = {
    "Cora": "Standard-Split",
    "Cora-LCC": "groesste Komponente, zufaellige 20/30/Rest-Splits",
    "ETTh1": "Horizont 96",
}
METRIC_LABEL = {"accuracy": "Accuracy", "mse": "MSE", "mae": "MAE"}
STD_RE = re.compile(r"(?:±|\\pm|\+-|\+/-)\s*(\d+(?:\.\d+)?)")


# Gesamtaussagen, die ein belegtes Protokoll voraussetzen (E1 deckelt sie auf "vorlaeufig")
UNCAPPED_STRONG = ("Paper reproduziert", "nicht reproduziert", "teilweise reproduziert", "Zielwert erreicht mit anderem Modell")


class NotRunnable(Exception):
    """Kandidat kann der Loop nicht nachbauen; Text = Grund fuer catalog.yaml."""


def catalog_id(arxiv_id: str) -> str:
    return "disc_" + arxiv_id.replace(".", "_")


# ---- Datei -----------------------------------------------------------------
def load_catalog(path: Path = CATALOG_YAML) -> list[dict]:
    if not Path(path).exists():
        return []
    return (yaml.safe_load(Path(path).read_text()) or {}).get("candidates") or []


def save_catalog(entries: list[dict], path: Path = CATALOG_YAML) -> None:
    text = yaml.safe_dump({"candidates": entries}, sort_keys=False, allow_unicode=True, width=120)
    Path(path).write_text(
        "# Discovery-Kandidaten und ihr Pruefstatus (scripts/curate_catalog.py). papers.yaml bleibt unberuehrt.\n" + text)


def find_entry(entries: list[dict], paper_id: str) -> dict:
    for e in entries:
        if e["id"] == paper_id:
            return e
    raise KeyError(f"{paper_id} nicht in benchmark/catalog.yaml")


# ---- Import ----------------------------------------------------------------
def read_cache_candidates(path: Path = DISCOVERY_CACHE) -> list[tuple[dict, str]]:
    """Ungepruefte Kandidaten aus logs/discovery_cache.json (Version 2 und 3). Rueckgabe: (Kandidat, Quelle)."""
    data = json.loads(Path(path).read_text())
    return [(v, f"cache:{Path(path).name}") for v in data.values()
            if isinstance(v, dict) and v.get("status") == "ungeprueft"
            and v.get("cache_version") in ACCEPTED_CACHE_VERSIONS and isinstance(v.get("claim"), dict)]


def read_event_candidates(path: Path) -> list[tuple[dict, str]]:
    """Kandidaten aus einer Discovery-Aufnahme (Ereignisse type=candidate, z. B. web/public/data/discovery/replay.jsonl)."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("type") == "candidate" and isinstance(ev.get("claim"), dict):
            out.append((ev, f"events:{Path(path).name}"))
    return out


def entry_from_candidate(c: dict, source: str) -> dict:
    keep = ("dataset", "claim", "github_url", "source_url", "protocol", "found_by", "reason", "community")
    return {
        "id": catalog_id(c["arxiv_id"]), "arxiv_id": c["arxiv_id"], "title": c["title"],
        "status": UNCHECKED, "imported_from": source,
        "candidate": {k: c[k] for k in keep if k in c},
    }


def import_candidates(entries: list[dict], candidates: list[tuple[dict, str]]) -> int:
    """Idempotent: bekannte IDs bleiben unveraendert (Status, Urteil, Quelle). Gibt Anzahl neuer Eintraege zurueck."""
    known = {e["id"] for e in entries}
    added = 0
    for cand, source in candidates:
        entry = entry_from_candidate(cand, source)
        if entry["id"] not in known:
            entries.append(entry)
            known.add(entry["id"])
            added += 1
    return added


# ---- Adapter ---------------------------------------------------------------
def _task_of(cand: dict) -> str:
    ds, claim = cand.get("dataset"), cand.get("claim") or {}
    if ds == "Cora":
        task = claim.get("matched_task")
        if task not in ("Cora", "Cora-LCC"):
            raise NotRunnable("Cora ohne matched_task: Protokoll nicht belegt, zwei Tasks moeglich (Planetoid, Shchur-LCC)")
        return task
    if ds == "ETTh1":
        try:
            horizon = int(claim.get("horizon"))
        except (TypeError, ValueError):
            raise NotRunnable(f"ETTh1: Horizont fehlt oder nicht lesbar ({claim.get('horizon')!r}), Task misst nur {ETTH1_HORIZON}")
        if horizon != ETTH1_HORIZON:
            raise NotRunnable(f"ETTh1: Horizont {horizon}, Task misst nur {ETTH1_HORIZON}")
        return "ETTh1"
    raise NotRunnable(f"Datensatz {ds!r} ohne Task")


def _fmt(x: float) -> str:
    return f"{x:g}"


def build_paper(entry: dict) -> tuple[dict, list[Claim]]:
    """Loop-Eingabe aus einem Katalogeintrag. Wirft NotRunnable mit Grund, wenn kein Lauf moeglich ist."""
    cand = entry["candidate"]
    claim = cand["claim"]
    task = _task_of(cand)
    metric = claim.get("metric")
    if metric not in TASK_METRICS[task]:
        raise NotRunnable(f"Metrik {metric!r} auf Task {task} nicht messbar")
    try:
        target = float(claim["target"])
    except (KeyError, TypeError, ValueError):
        raise NotRunnable("Zielwert fehlt oder nicht lesbar")
    if metric == "accuracy" and target > 1:
        target = target / 100   # Discovery speichert Prozent, Loop und Harness rechnen mit Anteilen

    quote = " ".join(str(claim.get("quote") or "").split())
    url = cand.get("source_url") or f"https://arxiv.org/abs/{entry['arxiv_id']}"
    method = claim.get("method") or "?"
    shown = f"{target * 100:g} %" if metric == "accuracy" else _fmt(target)
    text = f"{method} erreicht {shown} {METRIC_LABEL[metric]} auf {entry['candidate']['dataset']} ({TASK_SUFFIX[task]})"

    warnings: list[str] = []
    if cand.get("protocol") != "passt":
        warnings.append("Protokoll im Paper nicht belegt: Der Task wurde angenommen, nicht aus dem Paper belegt. "
                        "Die Gesamtaussage bleibt hoechstens vorlaeufig.")
    if claim.get("column_check") != "ok":
        warnings.append("Tabellenspalte nicht geprueft, Zahl koennte zu einer Vergleichsmethode gehoeren.")
    std = STD_RE.search(quote)
    if std:
        warnings.append(f"Info: Das Paper nennt Streuung +-{std.group(1)} (ueber Splits/Laeufe, nicht Toleranz des Mittelwerts); "
                        f"die Toleranz +-{TOLERANCE} bleibt fest.")
    input_length = claim.get("input_length") if task == "ETTh1" else None
    if task == "ETTh1" and not input_length:
        warnings.append("Eingabelaenge im Paper nicht belegt, der Agent waehlt sie frei (bis 720).")

    claim_id = f"{entry['id']}_{metric}"
    extract_metric = {"accuracy": "accuracy", "mse": "MSE for prediction length 96",
                      "mae": "MAE for prediction length 96"}[metric]
    paper = {
        "id": entry["id"], "title": entry["title"], "arxiv_url": f"https://arxiv.org/abs/{entry['arxiv_id']}",
        "dataset": task, "seeds": entry.get("seeds") or TASK_SEEDS[task], "source": "catalog",
        "warnings": warnings, "input_length": input_length, "discovery_quote": quote,
        "claims": [{"id": claim_id, "keep_target": True,   # bei Abweichung Rolle 2 gegen Discovery bleibt der Discovery-Wert
                    "extract": {"method": method, "dataset": entry["candidate"]["dataset"], "metric": extract_metric,
                                "percent": metric == "accuracy"}}],
    }
    claims = [Claim(id=claim_id, text=text, metric=metric, target=target, tolerance=TOLERANCE,
                    source=f"discovery: {url}, Zitat: {quote}", origin="paper", evidence_urls=[url])]
    return paper, claims


def load_catalog_claims(paper_id: str, path: Path = CATALOG_YAML) -> tuple[dict, list[Claim]]:
    return build_paper(find_entry(load_catalog(path), paper_id))


def discovery_details(paper: dict):
    """Setup-Detail Eingabelaenge, Quelle = Discovery-Zitat. Leer fuer Benchmark-Papers."""
    from .setup_details import SetupDetail

    if not paper.get("input_length"):
        return []
    return [SetupDetail(topic="input length", value=str(paper["input_length"]), quote=paper.get("discovery_quote", ""),
                        source="discovery")]


# ---- Reihenfolge (E5) und Status -----------------------------------------------
def order_key(entry: dict) -> tuple[int, str]:
    """1: Protokoll belegt + Spalte geprueft + Code, 2: Protokoll belegt ohne Code, 3: Rest."""
    cand = entry["candidate"]
    strong = cand.get("protocol") == "passt"
    if strong and (cand.get("claim") or {}).get("column_check") == "ok":
        tier = 1 if cand.get("github_url") else 2
    else:
        tier = 3
    return tier, entry["arxiv_id"]


NOT_JURY_READY = ("nicht bewertbar (Methode nicht umgesetzt)", "Agenten-Fehler vermutet", "nicht reproduzierbar (Setup)")


def jury_ready(entry: dict) -> tuple[bool, str]:
    """Jury-Katalog: Status geprueft heisst nur "Lauf ohne Absturz"; zusaetzlich zaehlt die Gesamtaussage."""
    if entry.get("status") != CHECKED:
        return False, f"Status {entry.get('status')}"
    gesamt = (entry.get("verdict") or {}).get("gesamtaussage")
    if gesamt is None:
        return False, "kein Urteil"
    if gesamt in NOT_JURY_READY:
        return False, f"Gesamtaussage: {gesamt}"
    return True, f"Gesamtaussage: {gesamt}"


def verdict_dict(entry: dict, dossier, run_dir: Path | None = None) -> dict:
    """Urteil Achse 1 + Gesamtaussage (bei unbelegtem Protokoll auf vorlaeufig gedeckelt) fuer catalog.yaml und die UI."""
    v = dossier.verdicts[0]
    claim = dossier.claims[0]

    def num(x):
        return None if x != x else round(float(x), 6)   # NaN (kein gueltiger Lauf) -> null

    gesamt = v.gesamtaussage
    if entry["candidate"].get("protocol") != "passt" and gesamt in UNCAPPED_STRONG:
        gesamt = "vorlaeufig"   # E1: Protokoll im Paper nicht belegt, nie mehr als vorlaeufig
    return {
        "urteil": v.urteil, "measured_mean": num(v.measured_mean), "measured_std": num(v.measured_std),
        "seeds": len({a.seed for a in dossier.attempts if a.claim_id == claim.id and a.measured is not None}),
        "target": claim.target, "tolerance": claim.tolerance, "gesamtaussage": gesamt,
        "graph_repairs": dossier.graph_repairs.get(claim.id, 0),
        "run_dir": str(run_dir.relative_to(REPO_ROOT)) if run_dir and run_dir.is_relative_to(REPO_ROOT) else str(run_dir),
    }


def record_result(entry: dict, status: str, reason: str | None = None, dossier=None, run_dir: Path | None = None,
                  cost_gross_usd: float | None = None) -> None:
    """Status (und bei Lauf-Ende das Urteil Achse 1) in den Eintrag schreiben."""
    entry["status"] = status
    entry["checked_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry.pop("reason", None)
    entry.pop("verdict", None)
    entry.pop("jury", None)
    if reason:
        entry["reason"] = reason
    if cost_gross_usd is not None:
        entry["cost_gross_usd"] = round(cost_gross_usd, 4)
    if dossier is not None and dossier.verdicts:
        entry["verdict"] = verdict_dict(entry, dossier, run_dir)
    ready, why = jury_ready(entry)
    entry["jury"] = {"ready": ready, "reason": why}


@dataclass
class CandidateOutcome:
    """Ergebnis eines Laufs fuer einen Kandidaten, ohne catalog.yaml zu beruehren."""
    status: str                 # geprueft | lauf_fehlgeschlagen | nicht_laufbar | ungeprueft (Abbruch Abrechnung/Budget)
    reason: str | None = None
    dossier: object = None
    run_dir: Path | None = None
    cost_gross_usd: float = 0.0
    abort: bool = False         # BillingBlocked/BudgetExceeded: weitere Kandidaten nicht starten


def run_candidate(orch, entry: dict, source: str = "catalog") -> CandidateOutcome:
    """Ein Kandidat durch den Loop (gemeinsam fuer scripts/curate_catalog.py und die API). Schreibt nichts in catalog.yaml."""
    from .costs import BudgetExceeded
    from .llm import BillingBlocked

    try:
        build_paper(entry)
    except NotRunnable as exc:
        return CandidateOutcome(NOT_RUNNABLE, reason=str(exc))
    try:
        dossier = orch.run(entry["id"], source=source)
    except (BillingBlocked, BudgetExceeded) as exc:   # Infrastruktur, kein Befund ueber das Paper: bleibt wiederholbar
        return CandidateOutcome(UNCHECKED, reason=f"Abbruch: {type(exc).__name__}: {exc}"[:300], abort=True)
    except Exception as exc:   # Absturz, Kostenlimit pro Lauf, Sandbox
        run_dir = getattr(orch, "last_run_dir", None)
        where = f" (Lauf: {run_dir})" if run_dir else ""
        return CandidateOutcome(FAILED, reason=f"{type(exc).__name__}: {exc}{where}"[:300], run_dir=run_dir,
                                cost_gross_usd=getattr(orch, "spent_gross_usd", 0.0))
    return CandidateOutcome(CHECKED, dossier=dossier, run_dir=orch.last_run_dir, cost_gross_usd=orch.spent_gross_usd)


def apply_outcome(entry: dict, outcome: CandidateOutcome) -> None:
    """Ergebnis in den Katalogeintrag schreiben (nur Kuratierung, nicht die API)."""
    cost = outcome.cost_gross_usd if outcome.status in (CHECKED, FAILED) else None
    record_result(entry, outcome.status, reason=outcome.reason, dossier=outcome.dossier, run_dir=outcome.run_dir,
                  cost_gross_usd=cost)
