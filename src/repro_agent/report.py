"""Reproduktions-Zeugnis aus Dossier generieren. TODO Meilenstein 4: UI mit Evidence-Ansicht."""

from __future__ import annotations

from .dossier import Dossier
from .droplast_note import droplast_text


def describe_check(o: dict) -> str:
    """Ein Satz: was wurde fuer diesen Einwand gemessen (oder warum nicht)."""
    m, t = o.get("measured") or {}, o["type"]
    if t == "not_measurable" or o["status"] == "nicht pruefbar":
        return f"not measurable: {o.get('note') or o.get('type_reason') or 'stays text'}"
    if t == "reported_number":
        sup = {"paper": "supports the paper's value", "reported": "supports the reported value", "both": "supports both values",
               "neither": "supports neither value"}.get(m.get("supports", ""), "no comparison")
        return f"reported number {o.get('reported_value')}; our measurement {m.get('mean', float('nan')):.4f} {sup}" + (f" ({o['note']})" if o.get("note") else "")
    if t == "seed_variance" and "mean" in m:
        return (f"{m.get('n_extra')} extra seeds: mean {m['mean']:.4f}, std over all seeds {m['std_pool']:.4f} "
                f"(main mean {m['main_mean']:.4f})")
    if t == "parameter_change" and "delta" in m:
        return (f"{o.get('param')} = {o.get('to_value')}: mean {m['mean']:.4f} vs {m['main_mean']:.4f} for the main code "
                f"(delta {m['delta']:+.4f}, {m['n']} paired seeds)")
    return f"{t}: {o.get('note') or 'no valid measurement'}"


def render_markdown(dossier: Dossier) -> str:
    lines = [f"# Reproduktions-Zeugnis: {dossier.paper_id}", ""]
    if dossier.warnings:
        lines.append("## Hinweise zur Belegbarkeit")
        lines.extend(f"- {w}" for w in dossier.warnings)
        lines.append("")
    for verdict in dossier.verdicts:
        lines.append(f"## Claim {verdict.claim_id}: {verdict.urteil}")
        lines.append(
            f"Gemessen: {verdict.measured_mean:.4f} (std {verdict.measured_std:.4f})"
        )
        note = " (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert)" if verdict.methode == "nicht geprueft" else ""
        lines.append(f"Methodentreue: {verdict.methode}{note}. Gesamtaussage: {verdict.gesamtaussage}")
        ab = dossier.ablations.get(verdict.claim_id)
        if ab:
            abl = "n/a" if ab["accuracy_ablated"] is None else f"{ab['accuracy_ablated']:.4f}"
            lines.append(f"Graph-Ablation (Daten ohne Kanten, Seed 0): {ab['verdict']} "
                         f"(Genauigkeit {ab['accuracy_full']:.4f} mit Graph, {abl} ohne Kanten)")
        n = dossier.graph_repairs.get(verdict.claim_id)
        if n:
            lines.append(f"Graph-Reparaturen: {n} (Ablation meldete Graph nicht genutzt, Code wurde neu geschrieben)")
        nf = dossier.fault_repairs.get(verdict.claim_id)
        if nf:
            lines.append(f"Fehler-Reparaturen: {nf} (Ergebnis schlechter als eine triviale Baseline, Code wurde neu geschrieben)")
        dn = dossier.droplast_notes.get(verdict.claim_id)
        if dn:
            lines.append(droplast_text(dn))
        lines.append(verdict.explanation)
        claim = next((c for c in dossier.claims if c.id == verdict.claim_id), None)
        ev = claim.target_evidence if claim else None
        if ev:
            lines.append(
                f"Zielwert {claim.target} belegt: \"{ev['quote']}\" ({ev['table']}, "
                f"{ev['url']}, Version {ev['arxiv_version']}, abgerufen {ev['retrieved_at']}, "
                f"SHA-256 {ev['sha256'][:12]})"
            )
            if ev.get("column_check") == "ok":
                lines.append("Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).")
            if ev.get("caveat"):
                lines.append(f"ACHTUNG Zielwert: {ev['caveat']}")
        elif claim:
            lines.append(f"Zielwert {claim.target}: {claim.source}")
        if verdict.sources:
            lines.append("Quellen:")
            for src in verdict.sources:
                lines.append(f"- {src}")
        lines.append("")
    c = dossier.community
    if c is not None:
        lines.append("## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)")
        if c["findings"]:
            lines.extend(f"- [{f['kind']}] {f['statement']} — \"{f['quote']}\" ({f['url']})"
                         + (f" [Repo-Abgleich: {f['repo_check']}]" if f.get("repo_check") not in (None, "nicht pruefbar") else "")
                         for f in c["findings"])
        else:
            lines.append("No public discussion found.")
        lines.append(f"Abfrage: {c.get('queried_at', '?')} ({c.get('mode', '?')}), {len(c.get('pages', []))} Seiten gelesen, "
                     f"{c.get('dropped', 0)} Aussagen verworfen (Zitat nicht im Text).")
        lines.append("")
    if dossier.objection_checks:
        lines.append("## Objections checked (Tavily role 3 -> own experiments, nur Bericht, nicht Teil des Urteils)")
        for o in dossier.objection_checks:
            lines.append(f"- \"{o['quote']}\" ({o['url']}) -> {describe_check(o)} -> {o['status']}")
        lines.append("bestaetigt/widerlegt gilt nach der vorab festgelegten Regel auf diesem Task; die Autorin des Einwands kann ein anderes Setup benutzt haben.")
        lines.append("")
    for cid, ex in dossier.explanations.items():
        lines.append(f"## Possible explanations für {cid} (Tavily Rolle 6, ungeprüfte Hypothesen, nicht Teil des Urteils)")
        lines.extend(f"- {i['explanation']} — \"{i['quote']}\" ({i['url']})" for i in ex["items"]) if ex["items"] else lines.append("None found.")
        lines.append("")
    lines.append(f"Modus: {dossier.mode} (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)")
    if dossier.repo:
        lines.append(f"Repo gelesen: {dossier.repo['url']} @ {dossier.repo['commit'][:10]}")
    lines.append("")
    if dossier.setup_details:
        lines.append("## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)")
        lines.extend(
            f"- [{d.get('source', 'paper')}] {d['topic']}: {d['value']} — \"{d['quote']}\""
            + (f" (not in the plan: {d['excluded']})" if d.get("excluded") else "")
            for d in dossier.setup_details
        )
        if getattr(dossier, "setup_missing", None):
            lines.append("- Not found in the sources: " + "; ".join(dossier.setup_missing))
        lines.append("")
    if dossier.isolation:
        lines.append("## Isolation (aktive Schutzschichten)")
        lines.extend(f"- {layer}" for layer in dossier.isolation)
    return "\n".join(lines)
