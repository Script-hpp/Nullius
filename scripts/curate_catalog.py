"""Kuratierung: Discovery-Kandidaten durch den Reproduktions-Loop schicken (docs/next_step_catalog_pipeline.md).

  LLM_PROVIDER=tokenfactory uv run python scripts/curate_catalog.py --max-usd 0.50 [--only 2511.08094] [--seeds 10]
  uv run python scripts/curate_catalog.py --dry-run --import-events web/public/data/discovery/replay.jsonl --only 2511.08094 --seeds 10

Kandidaten kommen aus logs/discovery_cache.json (ungeprueft, Version 2/3) und optional aus einer Discovery-Aufnahme
(--import-events, Ereignisse type=candidate). Import ist idempotent, bestehender Status/Urteil bleibt. Reihenfolge nach E5.
--dry-run: liest nur, schreibt nichts, null Aufrufe (kein LLM, Tavily, Sandbox), zeigt die Adapter-Ausgabe.
Nur Token-Factory-Laeufe veroeffentlichen (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repro_agent import catalog as cat
from repro_agent.dossier import REPO_ROOT


def select(entries: list[dict], only: str | None) -> list[dict]:
    """--only: genau dieser Kandidat (jeder Status, Wiederholung moeglich). Sonst alle ungepruften, nach E5 sortiert."""
    chosen = [e for e in entries if (e["arxiv_id"] == only if only else e["status"] == cat.UNCHECKED)]
    return sorted(chosen, key=cat.order_key)


def describe(entry: dict, seeds: int | None) -> dict:
    """Adapter-Ausgabe fuer --dry-run. Enthaelt `nicht_laufbar` mit Grund statt einer Loop-Eingabe."""
    tier = cat.order_key(entry)[0]
    base = {"id": entry["id"], "arxiv_id": entry["arxiv_id"], "status": entry["status"], "tier": tier,
            "imported_from": entry.get("imported_from")}
    try:
        paper, claims = cat.build_paper(entry)
    except cat.NotRunnable as exc:
        return {**base, "runnable": False, "reason": str(exc)}
    c = claims[0]
    return {**base, "runnable": True, "task": paper["dataset"], "seeds": seeds or paper["seeds"],
            "claim": {"id": c.id, "text": c.text, "metric": c.metric, "target": c.target, "tolerance": c.tolerance,
                      "source": c.source, "origin": c.origin},
            "rolle2_extract": paper["claims"][0]["extract"], "input_length": paper["input_length"],
            "warnings": paper["warnings"]}


def overview(entries: list[dict]) -> str:
    """Je Kandidat Status, Urteil, Gesamtaussage, Jury-Eignung mit Grund (liest nur, berechnet jury_ready frisch)."""
    lines = []
    for e in entries:
        v = e.get("verdict") or {}
        ready, why = cat.jury_ready(e)
        lines.append(f"{e['arxiv_id']}  {cat.origin_of(e):<9} {e['status']:<19} urteil={v.get('urteil', '-')}  gesamt={v.get('gesamtaussage', '-')}  "
                     f"jury={'ja' if ready else 'nein'} ({why})")
    return "\n".join(lines)


def curate(entries: list[dict], chosen: list[dict], path: Path, max_usd: float, seeds: int | None, offline: bool,
           orchestrator_factory=None) -> int:
    """Laeuft die Kandidaten nacheinander. Rueckgabe: Exit-Code (0 ok, 2 Abrechnung/Budget/Kostengrenze)."""
    if orchestrator_factory is None:
        from repro_agent.loop import Orchestrator

        def orchestrator_factory():
            return Orchestrator(seeds=seeds, tavily_mode="fixture" if offline else None, graph_repair=True)
    orch = orchestrator_factory()
    spent = 0.0
    for entry in chosen:
        if spent >= max_usd:   # harte Grenze: kein weiterer Kandidat
            print(f"ABBRUCH: --max-usd {max_usd:.2f} $ erreicht ({spent:.4f} $ brutto), {entry['arxiv_id']} und Rest nicht gelaufen")
            print("\nUebersicht:\n" + overview(chosen))
            return 2
        print(f"== {entry['arxiv_id']} {entry['title']}", flush=True)
        outcome = cat.run_candidate(orch, entry)
        cat.apply_outcome(entry, outcome)
        cat.save_catalog(entries, path)
        if outcome.abort:
            print(f"ABBRUCH: {outcome.reason}")
            return 2
        spent += outcome.cost_gross_usd
        if outcome.status == cat.NOT_RUNNABLE:
            print(f"{entry['arxiv_id']}: nicht_laufbar ({outcome.reason})")
        elif outcome.status == cat.FAILED:
            print(f"{entry['arxiv_id']}: lauf_fehlgeschlagen ({outcome.reason})")
        else:
            v = entry.get("verdict") or {}
            print(f"{entry['arxiv_id']}: geprueft, {v.get('urteil')} (Mittel {v.get('measured_mean')}, "
                  f"std {v.get('measured_std')}), {outcome.cost_gross_usd:.4f} $ brutto, gesamt {spent:.4f} $")
    print("\nUebersicht:\n" + overview(chosen))
    return 0


def main(argv: list[str] | None = None, orchestrator_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="nur Uebersicht aus catalog.yaml (Status, Urteil, Jury-Eignung), keine Aufrufe")
    ap.add_argument("--dry-run", action="store_true", help="nur Adapter-Ausgabe, schreibt nichts, null Aufrufe")
    ap.add_argument("--only", metavar="ARXIV_ID", help="nur dieser Kandidat (jeder Status)")
    ap.add_argument("--seeds", type=int, help="Seeds pro Kandidat (Standard: Wert des Katalogeintrags oder des Tasks)")
    ap.add_argument("--max-usd", type=float, default=1.0, help="harte Grenze brutto je Skriptlauf, Abbruch vor dem naechsten Kandidaten")
    ap.add_argument("--import-events", type=Path, help="Discovery-Aufnahme (jsonl, Ereignisse type=candidate) zusaetzlich importieren")
    ap.add_argument("--cache", type=Path, default=cat.DISCOVERY_CACHE, help="Discovery-Cache (Standard logs/discovery_cache.json)")
    ap.add_argument("--catalog", type=Path, default=cat.CATALOG_YAML)
    ap.add_argument("--offline", action="store_true", help="Tavily-Fixtures statt Live-Aufrufe (nur Tests)")
    args = ap.parse_args(argv)

    if args.list:
        print(overview(cat.load_catalog(args.catalog)))
        return 0
    entries = cat.load_catalog(args.catalog)
    candidates = cat.read_cache_candidates(args.cache) if args.cache.exists() else []
    if args.import_events:
        candidates += cat.read_event_candidates(args.import_events)
    added = cat.import_candidates(entries, candidates)
    print(f"Import: {added} neu, {len(entries)} im Katalog ({len(candidates)} Kandidaten gelesen)")
    if not args.dry_run and added:
        cat.save_catalog(entries, args.catalog)

    chosen = select(entries, args.only)
    if args.only and not chosen:
        print(f"{args.only} nicht im Katalog (Cache/Import enthaelt ihn nicht)")
        return 1
    if args.dry_run:
        for e in chosen:
            print(json.dumps(describe(e, args.seeds), indent=2, ensure_ascii=False))
        print(f"DRY-RUN: {len(chosen)} Kandidat(en), keine Aufrufe, nichts geschrieben.")
        return 0
    return curate(entries, chosen, args.catalog, args.max_usd, args.seeds, args.offline, orchestrator_factory)


if __name__ == "__main__":
    sys.exit(main())
