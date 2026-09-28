"""Nebius-Sandbox ueber contree-sdk. Echtes Branching. Standard-Backend (Beta genehmigt, kostenlos).

Doku: docs.tokenfactory.nebius.com/sandboxes/sdk/python_sdk/branching

TODO: exakte contree-sdk-API gegen Doku pruefen, sobald SDK verfuegbar ist.
Methodennamen unten sind Platzhalter (`client.snapshots.create`, `client.branches.create`, ...) --
vor erstem echten Lauf mit SDK-Referenz abgleichen.
"""

from __future__ import annotations

from .base import Branch, RunResult, SnapshotId


class NebiusSandbox:
    def __init__(self) -> None:
        try:
            import contree_sdk  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "contree-sdk nicht installiert. `uv add contree-sdk` und Zugangsdaten in .env "
                "(NEBIUS_API_KEY, NEBIUS_PROJECT_ID) pruefen."
            ) from exc
        self._client = contree_sdk.Client()

    def base_snapshot(self, image: str, setup_cmds: list[str]) -> SnapshotId:
        snapshot = self._client.snapshots.create(image=image, setup_cmds=setup_cmds)
        return snapshot.id

    def branch(self, snapshot: SnapshotId) -> Branch:
        branch = self._client.branches.create(snapshot_id=snapshot)
        return branch.id

    def run(self, branch: Branch, cmd: str, timeout_s: int) -> RunResult:
        result = self._client.branches.exec(branch_id=branch, cmd=cmd, timeout_s=timeout_s)
        return RunResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
        )

    def put_file(self, branch: Branch, path: str, content: bytes) -> None:
        self._client.branches.put_file(branch_id=branch, path=path, content=content)

    def get_file(self, branch: Branch, path: str) -> bytes:
        return self._client.branches.get_file(branch_id=branch, path=path)
