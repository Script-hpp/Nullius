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
        lines.append(verdict.explanation)
        if verdict.sources:
            lines.append("Quellen:")
            for src in verdict.sources:
                lines.append(f"- {src}")
        lines.append("")
    return "\n".join(lines)
