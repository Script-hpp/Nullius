"""Nebius-Sandbox ueber contree-sdk. Standard-Backend (Beta genehmigt, kostenlos).

Doku: docs.tokenfactory.nebius.com/sandboxes/sdk/python_sdk/branching

API gegen contree-sdk 0.3.6 (PyPI) verifiziert, nicht nur geraten:
- `ContreeSync()` liest Auth ohne Zusatzcode aus NEBIUS_API_KEY / NEBIUS_PROJECT_ID
  (IAMAuth-Defaults), Base-URL default = Token-Factory-Sandboxes-Endpoint.
- `client.images.oci(image)` importiert/aufloest ein Image (ContreeImageSync).
- `image.run(shell=cmd, disposable=False).wait()` fuehrt Setup-Kommandos aus und
  gibt ein NEUES Image zurueck (Copy-Semantik) -- Basis fuer echtes Branching:
  mehrfaches `.run()` auf demselben Image aendert das Original nicht.
- `image.session()` gibt ContreeSessionSync zurueck: im Gegensatz zu Image mutiert
  `.run().wait()` hier dieselbe Session (State/Dateisystem bleibt zwischen Aufrufen
  erhalten) -- das ist unser "Branch".
- `session.apply_files({path: bytes})` ist direkte SDK-Methode. `session.read()` taugt
  NICHT fuer Branch-Stand (liest Ausgangsimage), daher get_file ueber base64 in `run`.
"""

from __future__ import annotations

import base64
import shlex
import uuid as uuid_lib

from contree_sdk import ContreeSync
from contree_sdk.sdk.exceptions import OperationTimedOutError

from .base import Branch, RunResult, SnapshotId


class NebiusSandbox:
    def __init__(self) -> None:
        self._client = ContreeSync()
        self._sessions: dict[Branch, object] = {}

    def base_snapshot(self, image: str, setup_cmds: list[str]) -> SnapshotId:
        img = self._client.images.oci(image)
        for cmd in setup_cmds:
            img = img.run(shell=cmd, disposable=False).wait()
        tag = f"repro-snapshot-{uuid_lib.uuid4().hex[:12]}"
        img = img.tag_as(tag)
        return tag

    def branch(self, snapshot: SnapshotId) -> Branch:
        img = self._client.images.use(snapshot)
        session = img.session()
        branch_id = f"repro-branch-{uuid_lib.uuid4().hex[:8]}"
        self._sessions[branch_id] = session
        return branch_id

    def run(self, branch: Branch, cmd: str, timeout_s: int) -> RunResult:
        session = self._sessions[branch]
        try:
            session = session.run(shell=cmd, timeout=timeout_s, disposable=False).wait()
        except OperationTimedOutError:
            return RunResult(exit_code=-1, stdout="", stderr="", timed_out=True)
        self._sessions[branch] = session
        result = session.result
        # Timeout wirft hier nicht, sondern kommt als state.timed_out (exit_code -1, signal 9).
        timed_out = bool(result._raw.result.state.timed_out)
        return RunResult(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            timed_out=timed_out,
        )

    def put_file(self, branch: Branch, path: str, content: bytes) -> None:
        session = self._sessions[branch]
        self._sessions[branch] = session.apply_files({path: content})

    def get_file(self, branch: Branch, path: str) -> bytes:
        # `session.read()` liest das Ausgangsimage (Session-uuid wird nie aktualisiert),
        # nicht den Branch-Stand. Daher ueber `run` + base64.
        result = self.run(branch, f"base64 -w0 {shlex.quote(path)}", timeout_s=60)
        if result.exit_code != 0:
            raise FileNotFoundError(f"{path}: {result.stderr.strip()}")
        return base64.b64decode(result.stdout)
