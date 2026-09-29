"""Forschungsakte: Gedaechtnis des Agenten. `reasoning` bekommt nur verdichtete Fassung."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Claim:
    id: str
    text: str                  # "GCN erreicht 81,5 % auf Cora"
    metric: str                # "accuracy"
    target: float
    tolerance: float
    source: str                # "Tabelle 2, arXiv v1"
    origin: Literal["paper", "einwand", "placebo"]
    evidence_urls: list[str] = field(default_factory=list)
    target_evidence: dict | None = None   # Beleg fuer Zielwert (URL, Hash, Zitat), None = manuell


@dataclass
class Attempt:
    claim_id: str
    branch_id: str
    parent_snapshot: str
    code_hash: str
    seed: int
    status: Literal["fehler", "abweichung", "stabil", "ungueltig", "timeout"]
    measured: float | None
    error_summary: str | None  # verdichtet von `fast`
    tavily_queries: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    metrics: dict | None = None   # alle Metriken des Harness (nicht nur die des Claims)
    # Zeilenabdeckung des Agenten-Codes (nur erster vollstaendiger Seed-Lauf): {"executed": [...], "missing": [...]}
    coverage: dict | None = None


@dataclass
class Verdict:
    claim_id: str
    urteil: Literal[
        "reproduziert", "teilweise", "nicht reproduziert", "nicht reproduzierbar (Setup)",
        "Agenten-Fehler vermutet",
    ]
    measured_mean: float
    measured_std: float
    explanation: str
    sources: list[str] = field(default_factory=list)
    # Achse 2: setzt der Code die Methode des Papers um? Solange der Pruefer nicht geeicht ist
    # (Eichung v1 gescheitert, benchmark/experiments/results/calibration_fidelity_v1.md): "nicht geprueft".
    methode: Literal["ja", "teilweise", "nein", "nicht geprueft"] = "nicht geprueft"

    @property
    def gesamtaussage(self) -> str:
        return gesamtaussage(self.urteil, self.methode)


# Benchmark zaehlt nur diese beiden Gesamtaussagen (Methode umgesetzt).
BENCHMARK_AUSSAGEN = ("Paper reproduziert", "nicht reproduziert")


def gesamtaussage(urteil: str, methode: str) -> str:
    """Achse 1 (Zahl, inkl. 'Agenten-Fehler vermutet') x Achse 2 (Methode). Achse 1 bleibt unberuehrt."""
    if urteil in ("Agenten-Fehler vermutet", "nicht reproduzierbar (Setup)"):
        return urteil
    if methode == "nicht geprueft":
        return "vorlaeufig"
    if urteil == "reproduziert":
        return "Paper reproduziert" if methode == "ja" else "Zielwert erreicht mit anderem Modell"
    if methode != "ja":
        return "nicht bewertbar (Methode nicht umgesetzt)"
    return "nicht reproduziert" if urteil == "nicht reproduziert" else "teilweise reproduziert"


@dataclass
class Dossier:
    paper_id: str
    claims: list[Claim] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    isolation: list[str] = field(default_factory=list)  # aktive Schutzschichten gegen Leaks
    setup_details: list[dict] = field(default_factory=list)  # Tavily-Rolle 4: belegte Angaben im Prompt
    repo: dict | None = None      # {url, commit}: Repo, aus dem Setup-Details gelesen wurden
    baselines: dict | None = None  # triviale Vergleichswerte des Harness (ohne Training)
    ablations: dict = field(default_factory=dict)   # claim_id -> Graph-Ablation (Seed 0), siehe ablation_gcn.md
    mode: str = "nachbau"         # nachbau: Agent schreibt eigenen Code; Repo nie im Branch

    def save(self, timestamp: str) -> Path:
        run_dir = REPO_ROOT / "runs" / self.paper_id / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "dossier.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: Path) -> "Dossier":
        data = json.loads(path.read_text())
        return cls(
            paper_id=data["paper_id"],
            claims=[Claim(**c) for c in data["claims"]],
            attempts=[Attempt(**a) for a in data["attempts"]],
            verdicts=[Verdict(**v) for v in data["verdicts"]],
            isolation=data.get("isolation", []),
            setup_details=data.get("setup_details", []),
            repo=data.get("repo"),
            mode=data.get("mode", "nachbau"),
            ablations=data.get("ablations", {}),
            baselines=data.get("baselines"),
        )

    def condensed_for_reasoning(self) -> str:
        """Verdichtete Fassung fuer `reasoning`-Rolle. Nie alle Logs weitergeben."""
        lines = [f"Paper: {self.paper_id}", ""]
        for claim in self.claims:
            claim_attempts = [a for a in self.attempts if a.claim_id == claim.id]
            lines.append(f"Claim {claim.id}: {claim.text} (Ziel {claim.target} +/- {claim.tolerance})")
            for a in claim_attempts[-3:]:
                lines.append(f"  - Versuch {a.branch_id}: {a.status}, gemessen={a.measured}")
        return "\n".join(lines)
