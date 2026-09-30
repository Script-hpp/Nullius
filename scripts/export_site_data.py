"""Kopiert ausgewaehlte Laeufe fuer die Evidence-Ansicht nach web/public/data (statisch, ohne Backend).

  uv run python scripts/export_site_data.py runs/wu_sgc_2019/20260929-124736 [runs/... ...]
  uv run python scripts/export_site_data.py --catalog-only     # nur web/public/data/catalog.json neu schreiben
Pro Lauf: dossier.json, events.jsonl (falls vorhanden), zeugnis.md, plan.md, code.py (Code der ersten Seed-Versuche, letzte Version).
Schreibt index.json und catalog.json (aus benchmark/catalog.yaml). Zielordner: REPRO_EXPORT_DIR, Standard web/public/data.
Nur Laeufe uebergeben, die veroeffentlicht werden duerfen (keine Prototyp-Laeufe auf OpenRouter).
Die Logik liegt in src/repro_agent/site_export.py (gemeinsam mit der API).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from repro_agent import site_export as se

ROOT = Path(__file__).resolve().parents[1]
OUT = se.OUT
CATALOG_YAML = se.CATALOG_YAML


def main(argv: list[str] | None = None, out: Path = OUT, catalog_yaml: Path = CATALOG_YAML) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="runs/<paper_id>/<timestamp>")
    ap.add_argument("--catalog-only", action="store_true", help="nur catalog.json neu schreiben")
    args = ap.parse_args(argv)
    n = se.write_catalog(catalog_yaml, out)
    print(f"catalog.json: {n} Eintraege")
    if args.catalog_only:
        return
    papers = {p["id"]: p for p in yaml.safe_load((ROOT / "benchmark" / "papers.yaml").read_text())["papers"]}
    catalog = se.catalog_entries(catalog_yaml)
    titles = {k: v["title"] for k, v in papers.items()} | {e["id"]: e["title"] for e in catalog}
    index = se.read_index(out)
    for arg in args.runs:
        run = (ROOT / arg).resolve() if not Path(arg).is_absolute() else Path(arg)
        ref = se.export_run(run, out, titles)
        index = [r for r in index if r["id"] != ref["id"]] + [ref]
        print("exportiert:", ref["id"])
    se.write_json_atomic(out / "index.json", se.refresh_index(index, catalog, out))


if __name__ == "__main__":
    main()
