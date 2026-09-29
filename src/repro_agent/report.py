"""Reproduktions-Zeugnis aus Dossier generieren. TODO Meilenstein 4: UI mit Evidence-Ansicht."""

from __future__ import annotations

from .dossier import Dossier


def render_markdown(dossier: Dossier) -> str:
    lines = [f"# Reproduktions-Zeugnis: {dossier.paper_id}", ""]
    for verdict in dossier.verdicts:
        lines.append(f"## Claim {verdict.claim_id}: {verdict.urteil}")
        lines.append(
            f"Gemessen: {verdict.measured_mean:.4f} (std {verdict.measured_std:.4f})"
        )
        note = " (Methodentreue-Pruefer nicht geeicht, Eichung v1 gescheitert)" if verdict.methode == "nicht geprueft" else ""
        lines.append(f"Methodentreue: {verdict.methode}{note}. Gesamtaussage: {verdict.gesamtaussage}")
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
    lines.append(f"Modus: {dossier.mode} (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)")
    if dossier.repo:
        lines.append(f"Repo gelesen: {dossier.repo['url']} @ {dossier.repo['commit'][:10]}")
    lines.append("")
    if dossier.setup_details:
        lines.append("## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)")
        lines.extend(
            f"- [{d.get('source', 'paper')}] {d['topic']}: {d['value']} — \"{d['quote']}\""
            for d in dossier.setup_details
        )
        lines.append("")
    if dossier.isolation:
        lines.append("## Isolation (aktive Schutzschichten)")
        lines.extend(f"- {layer}" for layer in dossier.isolation)
    return "\n".join(lines)
