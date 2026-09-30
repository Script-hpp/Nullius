"""Kopiert ausgewaehlte Laeufe fuer die Evidence-Ansicht nach web/public/data (statisch, ohne Backend).

  uv run python scripts/export_site_data.py runs/wu_sgc_2019/20260929-124736 [runs/... ...]
  uv run python scripts/export_site_data.py --catalog-only     # nur web/public/data/catalog.json neu schreiben
Pro Lauf: dossier.json, events.jsonl (falls vorhanden), zeugnis.md, plan.md, code.py (Code der ersten Seed-Versuche, letzte Version).
Schreibt web/public/data/index.json und catalog.json (aus benchmark/catalog.yaml). Nur Laeufe uebergeben, die veroeffentlicht werden
duerfen (keine Prototyp-Laeufe auf OpenRouter).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from repro_agent.dossier import Dossier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "data"
CATALOG_YAML = ROOT / "benchmark" / "catalog.yaml"


def catalog_entries(catalog_yaml: Path) -> list[dict]:
    if not catalog_yaml.exists():
        return []
    return (yaml.safe_load(catalog_yaml.read_text()) or {}).get("candidates") or []


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
    out.mkdir(parents=True, exist_ok=True)
    data = catalog_json(catalog_entries(catalog_yaml))
    (out / "catalog.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
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
    att = [a for a in dossier["attempts"] if a["measured"] is not None] or dossier["attempts"]
    code = next(iter(run.glob(f"*_{att[-1]['code_hash']}.py")), None) if att else None
    if code:
        shutil.copy(code, dest / "code.py")
    return {"id": f"{paper_id}--{ts}", "paper_id": paper_id, "title": titles.get(paper_id, paper_id), "timestamp": ts}


def main(argv: list[str] | None = None, out: Path = OUT, catalog_yaml: Path = CATALOG_YAML) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="runs/<paper_id>/<timestamp>")
    ap.add_argument("--catalog-only", action="store_true", help="nur catalog.json neu schreiben")
    args = ap.parse_args(argv)
    n = write_catalog(catalog_yaml, out)
    print(f"catalog.json: {n} Eintraege")
    if args.catalog_only:
        return
    papers = {p["id"]: p for p in yaml.safe_load((ROOT / "benchmark" / "papers.yaml").read_text())["papers"]}
    titles = {k: v["title"] for k, v in papers.items()} | {e["id"]: e["title"] for e in catalog_entries(catalog_yaml)}
    index = json.loads((out / "index.json").read_text()) if (out / "index.json").exists() else []
    for arg in args.runs:
        run = (ROOT / arg).resolve() if not Path(arg).is_absolute() else Path(arg)
        ref = export_run(run, out, titles)
        index = [r for r in index if r["id"] != ref["id"]] + [ref]
        print("exportiert:", ref["id"])
    index = refresh_index(index, catalog_entries(catalog_yaml), out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
