"""Offline-Pruefung der Repo-Wahl und der Steckbrief-Eingaben auf gespeicherten Daten (kein Netz, kein LLM, keine Tavily-Aufrufe).

Liest `logs/discovery_cache.json` und je Kandidat den gespeicherten Paper-Text (`runs/disc_<id>/*/paper_text.json`, neuester) und zeigt:
(1) Code-Links im Paper-Text in der Reihenfolge der Wahl (find_repo_links), (2) fuer TSMixer: Hyperparameter-Tabelle (hparam_table),
verify_entries mit den Zitaten aus dem Anhang und die Modellwahl-Regel. Das README der Kandidaten ist offline nicht lesbar: ein Link,
den das Paper selbst setzt, gilt dann als "nicht widerlegt" (wie im Lauf bei nicht lesbarem README).

    uv run python scripts/offline_repo_check.py --root /home/onuralp/Desktop/hackathon/research
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repro_agent.hparam_table import table_details  # noqa: E402
from repro_agent.repo_source import find_repo_links, repo_evidence  # noqa: E402
from repro_agent.setup_details import Doc  # noqa: E402
from repro_agent.spec import other_model_selected, verify_entries  # noqa: E402

TSMIXER_ID = "2303.06053"
EPOCH_QUOTE = "We train each model with a maximum 100 epochs and do early stopping if the validation loss is not improved after 5 epochs."


def latest_text(root: Path, arxiv_id: str) -> str | None:
    hits = sorted(root.glob(f"runs/disc_{arxiv_id.replace('.', '_')}/*/paper_text.json"))
    return json.loads(hits[-1].read_text())["text"] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--only-no-repo", action="store_true", help="nur Kandidaten ohne github_url im Cache (plus TSMixer)")
    args = ap.parse_args()
    cache = json.loads((args.root / "logs" / "discovery_cache.json").read_text())
    for arxiv_id, c in cache.items():
        if "title" not in c or (c.get("github_url") and arxiv_id != TSMIXER_ID):
            continue
        print(f"\n== {arxiv_id}  {c['title'][:80]}  (Cache: github_url={c.get('github_url')}, reason={c.get('reason')})")
        text = latest_text(args.root, arxiv_id)
        if text is None:
            print("  kein gespeicherter Paper-Text: Wahl haengt allein von der Tavily-Suche ab (offline nicht simulierbar)")
            continue
        links = find_repo_links(text)
        for i, l in enumerate(links):
            kind, _ = repo_evidence("", l.target, c["title"], arxiv_id, (c.get("claim") or {}).get("method", ""))
            print(f"  Paper-Link {i + 1}: {l.target}  [{l.strength}]  Pfad-Beleg: {kind or '-'}")
        print(f"  Wahl: {links[0].target + '  (offiziell, vom Paper verlinkt)' if links else 'kein Link im Paper -> Tavily-Suche mit mehreren Treffern + README-Pruefung'}")
        if arxiv_id != TSMIXER_ID:
            continue
        docs = [Doc("paper", text)]
        details = table_details(docs, "TSMixer", "ETTh1")
        print("  Tabelle (ETTh1, TSMixer, T=96):")
        for d in details:
            print(f"    {d.topic}: {d.value}   Zitat: {d.quote}")
        raw = [{"key": k, "value": v.value, "quote": v.quote, "source": "paper"} for k, v in
               zip(["lr", "layers", "dropout", "hidden"], details)]
        raw += [{"key": "epochs", "value": "100", "quote": EPOCH_QUOTE, "source": "paper"},
                {"key": "patience", "value": "5", "quote": EPOCH_QUOTE, "source": "paper"}]
        kept, dropped = verify_entries(raw, docs, "TSMixer", "forecast")
        print(f"  verify_entries: behalten {[(k.key, k.value) for k in kept]}  verworfen {[(d['item']['key'], d['reason']) for d in dropped]}")
        print(f"  Modellwahl-Regel auf das Epochen-Zitat: {other_model_selected(EPOCH_QUOTE, 'TSMixer')!r}  (vorher 'with')")


if __name__ == "__main__":
    main()
