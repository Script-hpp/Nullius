"""Sandbox-Schnittstelle. Zwei Implementierungen: local.py (Docker-Fallback), nebius.py (Standard)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

SnapshotId = str
Branch = str


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class SandboxUnavailable(RuntimeError):
    """Infrastrukturfehler der Sandbox-API (z. B. Lese-Timeout beim Statusabfragen). Der Branch ist danach unbrauchbar
    (die SDK-Session steht auf FAILED); der Aufrufer wiederholt den Versuch auf einem NEUEN Branch."""


class SandboxSetupError(RuntimeError):
    """Ein Setup-Befehl des Basis-Snapshots scheitert auch nach einer Wiederholung (z. B. Daten-Download). Ein Snapshot ohne Daten waere
    still unbrauchbar: der Lauf endet als Infrastrukturfehler, nie als Urteil ueber den Agenten."""


def setup_tail(stderr: str, stdout: str = "", n: int = 1500) -> str:
    """Ende der Fehlerausgabe fuer die Meldung (stderr, sonst stdout)."""
    return ((stderr or "").strip() or (stdout or "").strip())[-n:]


def setup_failure(cmd: str, exit_code: int, stderr: str, stdout: str = "") -> SandboxSetupError:
    return SandboxSetupError(f"Setup-Befehl fehlgeschlagen (Exit-Code {exit_code}): {cmd[:200]}\n{setup_tail(stderr, stdout)}")


class Sandbox(Protocol):
    def base_snapshot(
        self, image: str, setup_cmds: list[str], label: str | None = None
    ) -> SnapshotId: ...
    def branch(self, snapshot: SnapshotId) -> Branch: ...
    def run(self, branch: Branch, cmd: str, timeout_s: int) -> RunResult: ...
    def put_file(self, branch: Branch, path: str, content: bytes) -> None: ...
    def put_files(self, branch: Branch, files: dict[str, bytes]) -> None: ...
    def get_file(self, branch: Branch, path: str) -> bytes: ...


def get_sandbox(backend: str) -> Sandbox:
    if backend == "nebius":
        from .nebius import NebiusSandbox

        return NebiusSandbox()
    if backend == "local":
        from .local import LocalSandbox

        return LocalSandbox()
    raise RuntimeError(f"Unbekanntes SANDBOX_BACKEND: {backend}")
