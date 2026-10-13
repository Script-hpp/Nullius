"""Docker-Fallback-Sandbox. Snapshots per `docker commit`. KI-Code nie direkt auf Host ausfuehren."""

from __future__ import annotations

import subprocess
import tempfile
import tarfile
import threading
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
        if not hasattr(self, '_branches'):
            self._branches = set()
        self._branches.add(container_name)
        return container_name

    def close(self) -> None:
        for name in tuple(getattr(self, '_branches', ())):
            subprocess.run(['docker', 'rm', '-f', name], check=True, capture_output=True)
            self._branches.discard(name)

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
            # `docker exec` beenden toetet nur den Client, nicht den Prozess im Container: ohne dies rechnet ein Timeout-Lauf weiter
            # und stiehlt den naechsten Laeufen die Kerne. `kill -9 -1` beendet alle Prozesse ausser PID 1 (`sleep infinity`).
            subprocess.run(["docker", "exec", branch, "sh", "-c", "kill -9 -1"], capture_output=True)
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
        subprocess.run(["docker", "exec", branch, "mkdir", "-p", str(Path(path).parent)], check=True)
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(content)
            tmp.flush()
            subprocess.run(["docker", "cp", tmp.name, f"{branch}:{path}"], check=True)

    def run_agent(self, branch: Branch, cmd: str, timeout_s: int) -> RunResult:
        """Docker isolates networking without requiring CAP_SYS_ADMIN for unshare.

        Disconnect the whole agent container before executing any generated code.
        Evaluation uses another branch and never reconnects this one.
        """
        result = subprocess.run(["docker", "inspect", "--format", "{{json .NetworkSettings.Networks}}", branch],
                                capture_output=True, text=True, check=True)
        import json
        for network in json.loads(result.stdout):
            if network not in {"none", "host"}:
                subprocess.run(["docker", "network", "disconnect", network, branch], check=True)
            elif network == "host":
                return RunResult(1, "", "Agent container must not use host networking", False)
        # This wrapper is constructed by the orchestrator, never by generated code.
        if "unshare -n " not in cmd:
            return RunResult(1, "", "Missing agent network-isolation wrapper", False)
        return self.run(branch, cmd.replace("unshare -n ", "", 1), timeout_s)

    def get_file(self, branch: Branch, path: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / "out"
            subprocess.run(["docker", "cp", f"{branch}:{path}", str(local_path)], check=True)
            return local_path.read_bytes()

    def get_file_limited(self, branch: Branch, path: str, max_bytes: int) -> bytes:
        """Docker's archive API, bounded before allocating prediction contents.

        No size report or Python executable from the agent container is trusted.
        Never extract archive paths onto the host filesystem.
        """
        proc = subprocess.Popen(['docker', 'cp', f'{branch}:{path}', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timer = threading.Timer(60, proc.kill)
        timer.start()
        try:
            with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
                member = archive.next()
                if member is None or not member.isfile() or member.size > max_bytes:
                    raise ValueError('Prediction artifact must be a regular file within its size limit')
                stream = archive.extractfile(member)
                data = stream.read(max_bytes + 1)
                if len(data) != member.size or len(data) > max_bytes:
                    raise ValueError('Prediction artifact transfer incomplete or too large')
                return data
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stdout.close()
            proc.stderr.close()
