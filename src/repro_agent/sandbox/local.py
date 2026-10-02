"""Docker-Fallback-Sandbox. Snapshots per `docker commit`. KI-Code nie direkt auf Host ausfuehren."""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path

from .base import Branch, RunResult, SnapshotId, setup_failure


class LocalSandbox:
    def base_snapshot(
        self, image: str, setup_cmds: list[str], label: str | None = None
    ) -> SnapshotId:
        container_name = f"repro-base-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            ["docker", "run", "-d", "--name", container_name, image, "sleep", "infinity"],
            check=True,
        )
        try:
            for cmd in setup_cmds:   # Exit-Code pruefen, einmal wiederholen, dann SandboxSetupError (wie NebiusSandbox)
                for attempt in (1, 2):
                    r = subprocess.run(["docker", "exec", container_name, "sh", "-c", cmd], capture_output=True, text=True)
                    if r.returncode == 0:
                        break
                    if attempt == 2:
                        raise setup_failure(cmd, r.returncode, r.stderr, r.stdout)
            snapshot_tag = f"repro-snapshot:{label or uuid.uuid4().hex[:12]}"
            subprocess.run(["docker", "commit", container_name, snapshot_tag], check=True)
            return snapshot_tag
        finally:
            subprocess.run(["docker", "rm", "-f", container_name], check=True)

    def branch(self, snapshot: SnapshotId) -> Branch:
        container_name = f"repro-branch-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            ["docker", "run", "-d", "--name", container_name, snapshot, "sleep", "infinity"],
            check=True,
        )
        return container_name

    def run(self, branch: Branch, cmd: str, timeout_s: int) -> RunResult:
        try:
            result = subprocess.run(
                ["docker", "exec", branch, "sh", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return RunResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return RunResult(
                exit_code=-1,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timed_out=True,
            )

    def put_files(self, branch: Branch, files: dict[str, bytes]) -> None:
        for path, content in files.items():
            self.put_file(branch, path, content)

    def put_file(self, branch: Branch, path: str, content: bytes) -> None:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(content)
            tmp.flush()
            subprocess.run(["docker", "cp", tmp.name, f"{branch}:{path}"], check=True)

    def get_file(self, branch: Branch, path: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / "out"
            subprocess.run(["docker", "cp", f"{branch}:{path}", str(local_path)], check=True)
            return local_path.read_bytes()
