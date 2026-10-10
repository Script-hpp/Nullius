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
import gzip
import hashlib
import shlex
import uuid as uuid_lib

from contree_sdk import ContreeSync
from contree_sdk.config import ContreeConfig
from contree_sdk.sdk.exceptions import OperationTimedOutError
from contree_sdk.sdk.exceptions.api import ApiTimeoutError

from .base import Branch, RunResult, SandboxSetupError, SandboxUnavailable, SnapshotId, setup_failure
from .. import spend_guard

# SDK-Default `transport_timeout` = 10 s (HTTP-Lesetimeout je Anfrage, auch fuers Abfragen des Operationsstatus). Unter
# paralleler Last antwortete die Beta-API teils nicht in 10 s: ApiTimeoutError beendete ganze Benchmark-Laeufe (3 Abbrueche).
TRANSPORT_TIMEOUT_S = 60.0

# SDK-Default (64 KiB) schneidet stdout still ab, Server-Maximum ist < 10 MiB.
SNAPSHOT_PREFIX = "repro-snapshot-"
_MAX_OUTPUT = 10 * 1024 * 1024 - 1
_CHUNK = 7 * 1024 * 1024  # base64 -> ~9.3 MiB


class NebiusSandbox:
    def __init__(self) -> None:
        self._budget = spend_guard.current_budget.get()
        self._client = ContreeSync(config=ContreeConfig(transport_timeout=TRANSPORT_TIMEOUT_S))
        self._sessions: dict[Branch, object] = {}

    def _check_cancelled(self):
        budget = getattr(self, "_budget", None)
        if budget:
            budget.check()

    @staticmethod
    def _guard(what: str, call):
        """ApiTimeoutError -> SandboxUnavailable. Ein erneutes `.run()` auf derselben Session geht nicht (Session ist nach dem
        Fehler FAILED, `ContreeImageStateError`), die Wiederholung braucht einen neuen Branch (Aufrufer)."""
        try:
            return call()
        except ApiTimeoutError as exc:
            raise SandboxUnavailable(f"Sandbox-API-Timeout bei {what}: {str(exc)[:200]}") from exc

    def base_snapshot(
        self, image: str, setup_cmds: list[str], label: str | None = None
    ) -> SnapshotId:
        self._check_cancelled()
        img = self._client.images.oci(image)
        for cmd in setup_cmds:
            img = self._run_setup(img, cmd)
        # label = Lauf-ID/Claim, damit cleanup_snapshots.py gezielt aufraeumen kann.
        tag = f"{SNAPSHOT_PREFIX}{label or uuid_lib.uuid4().hex[:12]}"
        self._check_cancelled()
        img = img.tag_as(tag)
        return tag

    def _run_setup(self, img, cmd: str):
        """Ein Setup-Befehl auf dem Image. Der Exit-Code zaehlt: `.wait()` wirft bei Exit-Code != 0 nicht, ein gescheiterter Download haette sonst still
        einen Snapshot ohne Daten ergeben. Einmal wiederholen (Netzfehler), dann SandboxSetupError mit dem Ende von stderr. Ein Image-Run hat sein
        Ergebnis wie ein Session-Run unter `.result` (exit_code, stdout, stderr)."""
        failure: SandboxSetupError | None = None
        for _ in range(2):
            self._check_cancelled()
            try:
                new = self._guard("setup", lambda: img.run(shell=cmd, disposable=False).wait())
                result = new.result
            except (SandboxUnavailable, spend_guard.BudgetExceeded):
                raise
            except Exception as exc:   # z. B. Image im Zustand FAILED
                failure = SandboxSetupError(f"Setup-Befehl fehlgeschlagen ({type(exc).__name__}): {cmd[:200]}\n{str(exc)[-1500:]}")
                continue
            if result.exit_code == 0:
                return new
            failure = setup_failure(cmd, result.exit_code, result.stderr or "", result.stdout or "")
        raise failure

    def branch(self, snapshot: SnapshotId) -> Branch:
        self._check_cancelled()
        img = self._client.images.use(snapshot)
        self._check_cancelled()
        session = img.session()
        branch_id = f"repro-branch-{uuid_lib.uuid4().hex[:8]}"
        self._sessions[branch_id] = session
        return branch_id

    def run(
        self, branch: Branch, cmd: str, timeout_s: int, *, truncate_output_at: int | None = None
    ) -> RunResult:
        self._check_cancelled()
        session = self._sessions[branch]
        try:
            session = self._guard("run", lambda: session.run(
                shell=cmd, timeout=timeout_s, disposable=False, truncate_output_at=truncate_output_at
            ).wait())
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
        self._check_cancelled()
        session = self._sessions[branch]
        self._sessions[branch] = self._guard("put_file", lambda: session.apply_files({path: content}))

    def put_files(self, branch: Branch, files: dict[str, bytes]) -> None:
        self._check_cancelled()
        # Ein apply_files-Aufruf = eine Sandbox-Operation statt einer pro Datei.
        session = self._sessions[branch]
        self._sessions[branch] = self._guard("put_files", lambda: session.apply_files(files))

    def get_file(self, branch: Branch, path: str) -> bytes:
        # `session.read()` liest das Ausgangsimage (Session-uuid wird nie aktualisiert),
        # nicht den Branch-Stand. Daher ueber `run`: gzip nach /tmp, Hash und Groesse im
        # Branch, dann Chunks per base64 (stdout ist hart auf < 10 MiB begrenzt).
        q = shlex.quote(path)
        stage = "/tmp/.repro_xfer.gz"
        meta = self.run(
            branch,
            f"test -f {q} && sha256sum < {q} | cut -d' ' -f1 && gzip -c < {q} > {stage} "
            f"&& stat -c %s {stage}",
            timeout_s=300,
        )
        if meta.exit_code != 0:
            raise FileNotFoundError(f"{path}: {meta.stderr.strip() or 'nicht gefunden'}")
        digest, size_s = meta.stdout.split()
        size = int(size_s)
        parts: list[bytes] = []
        for off in range(0, size, _CHUNK):
            r = self.run(
                branch,
                f"tail -c +{off + 1} {stage} | head -c {_CHUNK} | base64 -w0",
                timeout_s=300,
                truncate_output_at=_MAX_OUTPUT,
            )
            if r.exit_code != 0:
                raise OSError(f"{path}: Chunk bei {off} fehlgeschlagen: {r.stderr.strip()}")
            parts.append(base64.b64decode(r.stdout))
        blob = b"".join(parts)
        if len(blob) != size:
            raise OSError(f"{path}: Transfer unvollstaendig ({len(blob)} von {size} Bytes)")
        try:
            data = gzip.decompress(blob)
        except (OSError, EOFError) as exc:
            raise OSError(f"{path}: gzip-Daten beschaedigt ({exc})") from exc
        if hashlib.sha256(data).hexdigest() != digest:
            raise OSError(f"{path}: SHA-256 stimmt nicht (Transfer beschaedigt)")
        return data
