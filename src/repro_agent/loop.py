"""Orchestrator: der Loop pro Claim (siehe CLAUDE.md).

TODO Meilenstein 1: erster Durchlauf GCN/Cora end-to-end.
TODO Meilenstein 2: Fehler-Rollback, Tavily-Rollen 1-5 verdrahten.
TODO Meilenstein 3: Abweichungs-Loop, Einwaende als Branches, Seeds.
"""

from __future__ import annotations

from .config import load_settings
from .dossier import Dossier
from .llm import LLMClient
from .sandbox.base import get_sandbox
from .tavily_tools import TavilyTools


class Orchestrator:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.llm = LLMClient(self.settings)
        self.sandbox = get_sandbox(self.settings.sandbox_backend)
        self.tavily = TavilyTools(self.settings)

    def run(self, arxiv_url: str) -> Dossier:
        raise NotImplementedError("Loop noch nicht implementiert -- Meilenstein 1")
