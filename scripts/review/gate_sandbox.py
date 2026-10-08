"""Lokale Docker-Sandbox fuer Gate-Messungen (G0): Kernzuteilung ueber einen festen Pool, Netz nur ausserhalb des Agenten-Codes.

Wie `NoNetLocalSandbox` aus cg_pilot.py (Container ohne Netz waehrend `unshare -n`-Kommandos, `unshare -n` braucht im Container Rechte),
aber mit einem Pool aus CPU_SLOTS Kerngruppen zu je CORES_PER_SLOT Kernen: jedes `run` haelt waehrend der Ausfuehrung genau eine Gruppe
(`docker update --cpuset-cpus`), hoechstens CPU_SLOTS Kommandos laufen gleichzeitig, die Threads summieren sich damit auf <= 16.
Der Pilot rotierte nur reihum; bei mehr als vier gleichzeitigen Containern ueberbucht das die Kerne. Timeout: LocalSandbox.run beendet
die Prozesse im Container (sonst rechnet ein Timeout-Lauf weiter).
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

CPU_SLOTS, CORES_PER_SLOT = 4, 4
MEMORY = "12g"   # je Container (5 g und 8 g liessen eine dichte GAT-Aufmerksamkeit schon im Smoke-Lauf sterben, Spitze ueber 7 g); der Host teilt sich Speicher mit anderen Sitzungen, vier Spitzen gleichzeitig sind unwahrscheinlich


class GateSandbox:
    NET = "bridge"

    def __init__(self, slots: int = CPU_SLOTS, cores: int = CORES_PER_SLOT):
        from repro_agent.sandbox.local import LocalSandbox
        self.inner, self.branches, self._lock = LocalSandbox(), [], threading.Lock()
        self.times: dict[str, float] = {}   # Branch -> Laufzeit des letzten vollen Trainingslaufs (ohne Wartezeit auf eine Kerngruppe)
        self.slots: queue.Queue[str] = queue.Queue()
        for k in range(slots):
            self.slots.put(f"{k * cores}-{k * cores + cores - 1}")

    def base_snapshot(self, image, setup_cmds, label=None):
        tag = f"repro-snapshot:{label}" if label else None
        if tag and subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
            return tag   # Wiederverwendung desselben Basis-Snapshots (Daten + Pakete) zwischen Prozessen
        return self.inner.base_snapshot(image, setup_cmds, label=label)

    def branch(self, snapshot):
        name = f"gate-{uuid.uuid4().hex[:8]}"
        subprocess.run(["docker", "run", "-d", "--name", name, "--memory", MEMORY, "--memory-swap", MEMORY, snapshot, "sleep", "infinity"],
                       check=True, capture_output=True)
        subprocess.run(["docker", "exec", name, "mkdir", "-p", "/work", "/output"], check=True, capture_output=True)
        with self._lock:
            self.branches.append(name)
        return name

    def run(self, branch, cmd, timeout_s):
        cores = self.slots.get()
        try:
            subprocess.run(["docker", "update", "--cpuset-cpus", cores, branch], check=True, capture_output=True)
            t0 = time.time()
            try:
                if "unshare -n " not in cmd:
                    return self.inner.run(branch, cmd, timeout_s)
                subprocess.run(["docker", "network", "disconnect", self.NET, branch], check=True, capture_output=True)
                try:
                    return self.inner.run(branch, cmd.replace("unshare -n ", ""), timeout_s)
                finally:
                    subprocess.run(["docker", "network", "connect", self.NET, branch], capture_output=True)
            finally:
                if "train.py" in cmd and "SMOKE=1" not in cmd:
                    self.times[branch] = time.time() - t0
        finally:
            self.slots.put(cores)

    def put_file(self, branch, path, content):   # LocalSandbox.put_file legt das Zielverzeichnis nicht an
        subprocess.run(["docker", "exec", branch, "mkdir", "-p", str(Path(path).parent)], check=True, capture_output=True)
        self.inner.put_file(branch, path, content)

    def put_files(self, branch, files):
        for path, content in files.items():
            self.put_file(branch, path, content)

    def get_file(self, branch, path):
        return self.inner.get_file(branch, path)

    def release(self, branch):
        subprocess.run(["docker", "rm", "-f", branch], capture_output=True)
        with self._lock:
            if branch in self.branches:
                self.branches.remove(branch)

    def cleanup(self):
        with self._lock:
            names, self.branches = list(self.branches), []
        for name in names:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
