"""Kopiert ausgewaehlte Laeufe fuer die Evidence-Ansicht nach web/public/data (statisch, ohne Backend).

  uv run python scripts/export_site_data.py runs/wu_sgc_2019/20260929-124736 [runs/... ...]
Pro Lauf: dossier.json, events.jsonl (falls vorhanden), zeugnis.md, plan.md, code.py (Code der ersten Seed-Versuche, letzte Version).
Schreibt web/public/data/index.json. Nur Laeufe uebergeben, die veroeffentlicht werden duerfen (keine Prototyp-Laeufe auf OpenRouter).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "data"


def main() -> None:
    papers = {p["id"]: p for p in yaml.safe_load((ROOT / "benchmark" / "papers.yaml").read_text())["papers"]}
    index = json.loads((OUT / "index.json").read_text()) if (OUT / "index.json").exists() else []
    for arg in sys.argv[1:]:
        run = (ROOT / arg).resolve() if not Path(arg).is_absolute() else Path(arg)
        paper_id, ts = run.parent.name, run.name
        dossier = json.loads((run / "dossier.json").read_text())
        dest = OUT / paper_id / ts
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
        ref = {"id": f"{paper_id}--{ts}", "paper_id": paper_id, "title": papers.get(paper_id, {}).get("title", paper_id), "timestamp": ts}
        index = [r for r in index if r["id"] != ref["id"]] + [ref]
        print("exportiert:", ref["id"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
