"""Export fuer die statische UI (web/public/data): Laeufe, index.json, catalog.json. Gemeinsam fuer scripts/export_site_data.py und die API.

Zielordner: Umgebungsvariable REPRO_EXPORT_DIR, Standard web/public/data. index.json und catalog.json werden per Temp-Datei
und Umbenennen geschrieben (ein Leser sieht nie eine halbe Datei).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path

import yaml

from .badge import write_badge
from .dossier import REPO_ROOT, Dossier

OUT = Path(os.environ.get("REPRO_EXPORT_DIR") or REPO_ROOT / "web" / "public" / "data")
CATALOG_YAML = REPO_ROOT / "benchmark" / "catalog.yaml"
BENCHMARK_JSON = REPO_ROOT / "web" / "public" / "data" / "benchmark.json"
BENCHMARK_ARM = "B"   # Arm mit Tavily
_index_lock = threading.Lock()


def write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def catalog_entries(catalog_yaml: Path) -> list[dict]:
    if not Path(catalog_yaml).exists():
        return []
    return (yaml.safe_load(Path(catalog_yaml).read_text()) or {}).get("candidates") or []


def run_id_of(run_dir: str | None) -> str | None:
    """runs/<paper_id>/<ts> -> "<paper_id>--<ts>", die ID des Laufs in index.json (Link #/run/<id>)."""
    if not run_dir:
        return None
    p = Path(run_dir)
    return f"{p.parent.name}--{p.name}"


def read_benchmark(out: Path | None = None) -> dict | None:
    """benchmark.json aus dem Exportordner, sonst die eingecheckte Datei. None, wenn keine lesbar ist."""
    for path in ([Path(out) / "benchmark.json"] if out else []) + [BENCHMARK_JSON]:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return None


def reference_benchmark(entry: dict, benchmark: dict | None) -> dict | None:
    """Kennzahl eines Referenz-Eintrags im Benchmark (Arm mit Tavily), berechnet aus den Einzelurteilen in benchmark.json.
    None, wenn der Eintrag keine benchmark_unit nennt oder die Quelle Version/Einheit/Arm nicht enthaelt (nichts erfunden)."""
    ref = entry.get("benchmark_unit")
    if not ref or not benchmark:
        return None
    version = next((v for v in benchmark.get("versions") or [] if v.get("id") == ref.get("version")), None)
    if not version:
        return None
    units = version.get("units") or []
    unit = next((u for u in units if u.get("paper") == ref.get("paper") and u.get("claim") == ref.get("claim")), None)
    verdicts = ((unit or {}).get("arms", {}).get(BENCHMARK_ARM) or {}).get("verdicts")
    if not verdicts:
        return None
    matched = sum(1 for u in units if (u.get("arms", {}).get(BENCHMARK_ARM) or {}).get("majority") == u.get("reference_verdict"))
    return {"version": version["id"], "arm": "with Tavily", "reached": sum(1 for v in verdicts if v == "reproduziert"),
            "runs": len(verdicts), "units_matched": matched, "units": len(units)}


HEAD_FIELDS = ("text", "authors", "authors_source", "year", "venue", "institutions", "institutions_source", "source", "generated_by")


def head_of(run_id: str | None, out_dir: Path | None) -> dict | None:
    """Kompakter Papierkopf aus dossier.summary des exportierten Laufs (kein LLM-, kein Tavily-Aufruf). None, wenn nichts da ist."""
    if not run_id or out_dir is None or "--" not in run_id:
        return None
    paper_id, ts = run_id.split("--", 1)
    try:
        summary = json.loads((Path(out_dir) / paper_id / ts / "dossier.json").read_text()).get("summary")
    except (OSError, ValueError):
        return None
    if not isinstance(summary, dict):
        return None
    head = {k: summary[k] for k in HEAD_FIELDS if summary.get(k)}
    return head if head.get("text") or head.get("authors") or head.get("institutions") else None


def catalog_json(entries: list[dict], benchmark: dict | None = None, out_dir: Path | None = None) -> list[dict]:
    """Nur Felder, die die UI braucht. Mit out_dir: Papierkopf (head) aus dem exportierten Dossier des Laufs."""
    from .claims_source import clean_latex_quote
    rows = []
    for e in entries:
        cand = e.get("candidate") or {}
        claim = cand.get("claim") or {}
        verdict = e.get("verdict")
        clean_claim = {k: claim.get(k) for k in ("method", "metric", "target", "horizon", "quote", "input_length")}
        if clean_claim.get("quote"):
            clean_claim["quote"] = clean_latex_quote(clean_claim["quote"])
        run_id = run_id_of((verdict or {}).get("run_dir"))
        item = {
            "id": e["id"], "arxiv_id": e["arxiv_id"], "title": e["title"],
            "dataset": cand.get("dataset"), "task": claim.get("matched_task") or cand.get("dataset"),
            "claim": clean_claim,
            "protocol": cand.get("protocol"), "column_check": claim.get("column_check"),
            "github_url": cand.get("github_url"), "source_url": cand.get("source_url"),
            "origin": e.get("origin") or "discovery", "benchmark": reference_benchmark(e, benchmark),
            "status": e.get("status"), "reason": e.get("reason"), "checked_at": e.get("checked_at"),
            "cost_gross_usd": e.get("cost_gross_usd"),
            "verdict": verdict, "run_id": run_id, "jury": e.get("jury"),
        }
        head = head_of(run_id, out_dir)
        if head:
            item["head"] = head
        rows.append(item)
    return rows


def write_catalog(catalog_yaml: Path, out: Path, extra_entries: list[dict] | None = None) -> int:
    data = catalog_json(catalog_entries(catalog_yaml) + (extra_entries or []), read_benchmark(out), out)
    write_json_atomic(out / "catalog.json", data)
    return len(data)


def refresh_catalog_metadata(catalog_yaml: Path, out: Path) -> None:
    """Redeploy corrected definitions without erasing persistent run outcomes."""
    path = out / "catalog.json"
    stored = json.loads(path.read_text()) if path.exists() else []
    by_id = {entry["id"]: entry for entry in stored}
    definitions = catalog_json(catalog_entries(catalog_yaml), read_benchmark(out), out)
    metadata = ("arxiv_id", "title", "dataset", "task", "claim", "protocol", "column_check",
                "github_url", "source_url", "origin", "benchmark")
    outcome = ("status", "reason", "checked_at", "cost_gross_usd", "verdict", "run_id")
    for entry in definitions:
        if entry["id"] in by_id:
            row = by_id[entry["id"]]
            row.update({key: entry[key] for key in metadata})
            # A curated check newer than the stored one replaces it; an older catalog never erases newer evidence.
            if entry.get("verdict") and (entry.get("checked_at") or "") > (row.get("checked_at") or ""):
                row.update({key: entry[key] for key in outcome})
                if "head" in entry:
                    row["head"] = entry["head"]
            # The jury flag is a curation decision of catalog.yaml, never a run outcome.
            row["jury"] = entry["jury"]
        else:
            stored.append(entry)
            by_id[entry["id"]] = entry
    write_json_atomic(path, stored)


def refresh_index(index: list[dict], catalog: list[dict], out: Path) -> list[dict]:
    """disc_*-Laeufe: Titel aus catalog.yaml, catalog_id und Gesamtaussage (aus dem exportierten Dossier berechnet)."""
    titles = {e["id"]: e["title"] for e in catalog}
    for ref in index:
        if not ref["paper_id"].startswith("disc_"):
            continue
        ref["title"] = titles.get(ref["paper_id"], ref["title"])
        ref["catalog_id"] = ref["paper_id"]
        dossier = out / ref["paper_id"] / ref["timestamp"] / "dossier.json"
        if dossier.exists():
            verdicts = Dossier.load(dossier).verdicts
            if verdicts:
                ref["gesamtaussage"] = verdicts[0].gesamtaussage   # roh, ohne Deckelung des Katalogs (E1)
    return index


def export_run(run: Path, out: Path, titles: dict[str, str]) -> dict:
    paper_id, ts = run.parent.name, run.name
    dossier = json.loads((run / "dossier.json").read_text())
    dest = out / paper_id / ts
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("dossier.json", "events.jsonl", "zeugnis.md"):
        if (run / name).exists():
            shutil.copy(run / name, dest / name)
    # plan.md is the plan of the paper claim the page shows (the first one with origin "paper"), never a placebo's or an objection's
    claims = dossier.get("claims") or []
    first = next((c for c in claims if c.get("origin") == "paper"), claims[0] if claims else None)
    plans = list(run.glob("*_plan.md"))
    own = run / f"{first['id']}_plan.md" if first else None
    chosen = own if own is not None and own.exists() else (plans[0] if len(plans) == 1 else None)   # several plans and none is the claim's: none
    (dest / "plan.md").unlink(missing_ok=True)
    if chosen:
        shutil.copy(chosen, dest / "plan.md")
    for version in run.glob("*.py"):   # jede Code-Version: <claim>_<hash>.py -> code_<hash>.py (Replay zeigt Version 1, 2, ...)
        shutil.copy(version, dest / f"code_{version.stem.rsplit('_', 1)[-1]}.py")
    try:   # the badge is optional: an incomplete or older dossier must never stop the export of the run
        write_badge(dest, Dossier.load(dest / "dossier.json"), ts, catalog=paper_id.startswith("disc_"))
    except Exception as exc:   # noqa: BLE001
        print(f"Badge skipped for {paper_id}/{ts}: {type(exc).__name__}: {exc}")
    own_attempts = [a for a in dossier['attempts'] if first is not None and a.get('claim_id') == first['id']]
    selected = any(isinstance(a.get('chosen'), bool) for a in own_attempts)
    att = [a for a in own_attempts if a.get('measured') is not None and (not selected or a.get('chosen') is True)]
    (dest / 'code.py').unlink(missing_ok=True)
    code = next(iter(run.glob(f"*_{att[-1]['code_hash']}.py")), None) if att else None
    if code:
        shutil.copy(code, dest / "code.py")
    return {"id": f"{paper_id}--{ts}", "paper_id": paper_id, "title": titles.get(paper_id, paper_id), "timestamp": ts,
            "mode": dossier.get("mode") or "nachbau"}   # Modus des Laufs (Dossier ist die Quelle); alte Laeufe ohne Feld = reiner Paper-Nachbau


def read_index(out: Path) -> list[dict]:
    return json.loads((out / "index.json").read_text()) if (out / "index.json").exists() else []


def add_to_index(ref: dict, catalog: list[dict], out: Path) -> None:
    """Eintrag ersetzen/anhaengen, disc_*-Felder auffrischen, atomar schreiben."""
    with _index_lock:
        index = [r for r in read_index(out) if r["id"] != ref["id"]] + [ref]
        write_json_atomic(out / "index.json", refresh_index(index, catalog, out))


def export_live_run(run_dir: Path, catalog_yaml: Path, out: Path, note: str, seeds: int | None,
                    extra_entries: list[dict] | None = None) -> dict:
    """Live-Lauf der API: Lauf nach out kopieren, index.json-Eintrag mit note und live=true. catalog.yaml bleibt unberuehrt."""
    catalog = catalog_entries(catalog_yaml) + (extra_entries or [])
    ref = export_run(run_dir, out, {e["id"]: e["title"] for e in catalog})
    from . import web_guard   # oeffentlich ausgelieferte Textdateien: keine Geheimnisse, keine Pfade mit Nutzernamen
    web_guard.redact_tree(out / ref["paper_id"] / ref["timestamp"], str(REPO_ROOT))
    ref.update({"note": note, "live": True, "seeds": seeds})
    add_to_index(ref, catalog, out)
    if extra_entries or (out / "catalog.json").exists():   # include completed live discoveries and their evidence
        write_catalog(catalog_yaml, out, extra_entries)
    return next(r for r in read_index(out) if r["id"] == ref["id"])
