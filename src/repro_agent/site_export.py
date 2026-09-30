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

from .dossier import REPO_ROOT, Dossier

OUT = Path(os.environ.get("REPRO_EXPORT_DIR") or REPO_ROOT / "web" / "public" / "data")
CATALOG_YAML = REPO_ROOT / "benchmark" / "catalog.yaml"
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


def catalog_json(entries: list[dict]) -> list[dict]:
    """Nur Felder, die die UI braucht."""
    out = []
    for e in entries:
        cand = e.get("candidate") or {}
        claim = cand.get("claim") or {}
        verdict = e.get("verdict")
        out.append({
            "id": e["id"], "arxiv_id": e["arxiv_id"], "title": e["title"],
            "dataset": cand.get("dataset"), "task": claim.get("matched_task") or cand.get("dataset"),
            "claim": {k: claim.get(k) for k in ("method", "metric", "target", "horizon", "quote", "input_length")},
            "protocol": cand.get("protocol"), "column_check": claim.get("column_check"),
            "github_url": cand.get("github_url"), "source_url": cand.get("source_url"),
            "status": e.get("status"), "reason": e.get("reason"), "checked_at": e.get("checked_at"),
            "cost_gross_usd": e.get("cost_gross_usd"),
            "verdict": verdict, "run_id": run_id_of((verdict or {}).get("run_dir")), "jury": e.get("jury"),
        })
    return out


def write_catalog(catalog_yaml: Path, out: Path) -> int:
    data = catalog_json(catalog_entries(catalog_yaml))
    write_json_atomic(out / "catalog.json", data)
    return len(data)


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
    for plan in run.glob("*_plan.md"):
        shutil.copy(plan, dest / "plan.md")
    for version in run.glob("*.py"):   # jede Code-Version: <claim>_<hash>.py -> code_<hash>.py (Replay zeigt Version 1, 2, ...)
        shutil.copy(version, dest / f"code_{version.stem.rsplit('_', 1)[-1]}.py")
    att = [a for a in dossier["attempts"] if a["measured"] is not None] or dossier["attempts"]
    code = next(iter(run.glob(f"*_{att[-1]['code_hash']}.py")), None) if att else None
    if code:
        shutil.copy(code, dest / "code.py")
    return {"id": f"{paper_id}--{ts}", "paper_id": paper_id, "title": titles.get(paper_id, paper_id), "timestamp": ts}


def read_index(out: Path) -> list[dict]:
    return json.loads((out / "index.json").read_text()) if (out / "index.json").exists() else []


def add_to_index(ref: dict, catalog: list[dict], out: Path) -> None:
    """Eintrag ersetzen/anhaengen, disc_*-Felder auffrischen, atomar schreiben."""
    with _index_lock:
        index = [r for r in read_index(out) if r["id"] != ref["id"]] + [ref]
        write_json_atomic(out / "index.json", refresh_index(index, catalog, out))


def export_live_run(run_dir: Path, catalog_yaml: Path, out: Path, note: str, seeds: int | None) -> dict:
    """Live-Lauf der API: Lauf nach out kopieren, index.json-Eintrag mit note und live=true. catalog.yaml bleibt unberuehrt."""
    catalog = catalog_entries(catalog_yaml)
    ref = export_run(run_dir, out, {e["id"]: e["title"] for e in catalog})
    ref.update({"note": note, "live": True, "seeds": seeds})
    add_to_index(ref, catalog, out)
    return next(r for r in read_index(out) if r["id"] == ref["id"])
