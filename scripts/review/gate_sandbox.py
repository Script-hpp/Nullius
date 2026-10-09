"""Lokale Docker-Sandbox fuer Gate-Messungen (G0): Kernzuteilung ueber einen festen Pool, Netz nur ausserhalb des Agenten-Codes.

Wie `NoNetLocalSandbox` aus cg_pilot.py (Container ohne Netz waehrend `unshare -n`-Kommandos, `unshare -n` braucht im Container Rechte),
aber mit einem Pool aus CPU_SLOTS Kerngruppen zu je CORES_PER_SLOT Kernen: jedes `run` haelt waehrend der Ausfuehrung genau eine Gruppe
(`docker update --cpuset-cpus`), hoechstens CPU_SLOTS Kommandos laufen gleichzeitig, die Threads summieren sich damit auf <= 16.
Der Pilot rotierte nur reihum; bei mehr als vier gleichzeitigen Containern ueberbucht das die Kerne. Timeout: LocalSandbox.run beendet
die Prozesse im Container (sonst rechnet ein Timeout-Lauf weiter).
"""

from __future__ import annotations

import copy
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

CPU_SLOTS, CORES_PER_SLOT = 4, 4
# Speicher: Limit je Container, aber die Summe der Limits aller GLEICHZEITIG laufenden Kommandos ist hart auf MEM_BUDGET_GB begrenzt
# (Speicher-Semaphor). Host-OOM am 2026-10-08 19:25: vier Container zu je 12 g (bis 48 g) auf einem 30-g-Host, der ~10 g fuer andere Dienste braucht.
# Schwer = GAT-Einheiten (dichte Aufmerksamkeit, 5 g und 8 g liessen sie sterben, Spitze ~10,6 g); alle anderen leichter.
MEM_BUDGET_GB = 20
HEAVY_GB, LIGHT_GB = 12, 6
HEAVY_UNITS = {"velickovic_gat_2018", "shchur_gat_2018"}


def memory_for(unit: str | None) -> int:
    return HEAVY_GB if unit in HEAVY_UNITS else LIGHT_GB


class MemoryBudget:
    """Zaehlsemaphor in GB: `acquire(gb)` blockiert, bis die Summe der gehaltenen Mengen + gb <= total ist."""

    def __init__(self, total: int):
        self.total, self.used, self.cond = total, 0, threading.Condition()

    def acquire(self, gb: int) -> None:
        assert gb <= self.total, (gb, self.total)
        with self.cond:
            while self.used + gb > self.total:
                self.cond.wait()
            self.used += gb

    def release(self, gb: int) -> None:
        with self.cond:
            self.used -= gb
            self.cond.notify_all()


class GateSandbox:
    NET = "bridge"

    def __init__(self, slots: int = CPU_SLOTS, cores: int = CORES_PER_SLOT, memory_gb: int = LIGHT_GB, budget_gb: int = MEM_BUDGET_GB):
        from repro_agent.sandbox.local import LocalSandbox
        self.inner, self.branches, self._lock = LocalSandbox(), [], threading.Lock()
        self.memory_gb, self.mem = memory_gb, MemoryBudget(budget_gb)
        self.created = threading.local()   # Branches, die der aktuelle Thread seit `begin_attempt` angelegt hat (fuer Freigabe auch bei Exception)
        self.times: dict[str, float] = {}   # Branch -> Laufzeit des letzten vollen Trainingslaufs (ohne Wartezeit auf eine Kerngruppe)
        self.slots: queue.Queue[str] = queue.Queue()
        for k in range(slots):
            self.slots.put(f"{k * cores}-{k * cores + cores - 1}")

    def for_unit(self, unit: str | None) -> "GateSandbox":
        """Ansicht mit dem Speicherlimit der Einheit; teilt Kerngruppen, Speicher-Semaphor, Branch-Liste und Zeiten mit dem Original."""
        view = copy.copy(self)
        view.memory_gb = memory_for(unit)
        return view

    def base_snapshot(self, image, setup_cmds, label=None):
        tag = f"repro-snapshot:{label}" if label else None
        if tag and subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
            return tag   # Wiederverwendung desselben Basis-Snapshots (Daten + Pakete) zwischen Prozessen
        return self.inner.base_snapshot(image, setup_cmds, label=label)

    def branch(self, snapshot):
        name = f"gate-{uuid.uuid4().hex[:8]}"
        mem = f"{self.memory_gb}g"
        getattr(self.created, "names", []).append(name)
        with self._lock:
            self.branches.append(name)   # vor `docker run`: schlaegt ein Schritt fehl, entfernt cleanup() den Container trotzdem
        try:
            subprocess.run(["docker", "run", "-d", "--name", name, "--memory", mem, "--memory-swap", mem, snapshot, "sleep", "infinity"],
                           check=True, capture_output=True)
            subprocess.run(["docker", "exec", name, "mkdir", "-p", "/work", "/output"], check=True, capture_output=True)
        except BaseException:
            self.release(name)
            raise
        return name

    def run(self, branch, cmd, timeout_s):
        self.mem.acquire(self.memory_gb)   # immer zuerst Speicher, dann Kerngruppe (feste Reihenfolge, keine Verklemmung)
        try:
            cores = self.slots.get()
        except BaseException:
            self.mem.release(self.memory_gb)
            raise
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
            self.mem.release(self.memory_gb)

    def put_file(self, branch, path, content):   # LocalSandbox.put_file legt das Zielverzeichnis nicht an
        subprocess.run(["docker", "exec", branch, "mkdir", "-p", str(Path(path).parent)], check=True, capture_output=True)
        self.inner.put_file(branch, path, content)

    def put_files(self, branch, files):
        for path, content in files.items():
            self.put_file(branch, path, content)

    def get_file(self, branch, path):
        return self.inner.get_file(branch, path)

    def begin_attempt(self) -> None:
        self.created.names = []

    def release_attempt(self) -> None:
        """Entfernt alle Branches, die dieser Thread seit `begin_attempt` angelegt hat (auch wenn der Versuch mit Exception endete)."""
        for name in getattr(self.created, "names", []):
            self.release(name)
        self.created.names = []

    def release(self, branch):
        subprocess.run(["docker", "rm", "-f", branch], capture_output=True)
        with self._lock:
            if branch in self.branches:
                self.branches.remove(branch)

    def cleanup(self):
        with self._lock:
            names = list(self.branches)
            self.branches.clear()   # dieselbe Liste wie in den Ansichten (for_unit)
        for name in names:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)


# ---- Nebius-Messweg (G1) -----------------------------------------------------------------------------------------------------
NEBIUS_MAX_PARALLEL_RUNS = 16   # gleichzeitige Kommandos (Sondierung 2026-10-09: 16 Branches ohne Fehler, gleicher Durchsatz je Branch)


class NebiusGateSandbox:
    """Duenne Huelle um `NebiusSandbox` mit der Schnittstelle von `GateSandbox` (`for_unit`, `begin_attempt`, `release_attempt`, `times`, `cleanup`).

    Unterschiede zum lokalen Weg: Netzsperre wie im Produkt (`unshare -n` im Kommando bleibt stehen), keine Kerngruppen und kein Speicher-Semaphor
    (jeder Branch hat eigene 4 vCPU und ca. 4 GB RAM, Sondierung), Wanduhr-Zeit um `run` fuer die Trainingszeit. Der Basis-Snapshot wird ueber
    sein Tag wiederverwendet, falls er schon existiert. Branches haben keine Freigabe-API: die Sitzung wird nur clientseitig vergessen.
    """

    def __init__(self, max_parallel: int = NEBIUS_MAX_PARALLEL_RUNS):
        from repro_agent.sandbox.nebius import NebiusSandbox
        self.inner, self._lock = NebiusSandbox(), threading.Lock()
        self.created = threading.local()
        self.times: dict[str, float] = {}
        self.slots = threading.BoundedSemaphore(max_parallel)
        self.tags: dict[str, str] = {}

    def for_unit(self, unit: str | None) -> "NebiusGateSandbox":
        return self

    def base_snapshot(self, image, setup_cmds, label=None):
        from repro_agent.sandbox.nebius import SNAPSHOT_PREFIX
        tag = f"{SNAPSHOT_PREFIX}{label}" if label else None
        if tag:
            try:   # `images.use` ist lazy: erst ein Branch mit einem Kommando zeigt, ob das Tag existiert
                probe = self.inner.branch(tag)
                if self.inner.run(probe, "true", 60).exit_code == 0:
                    self.inner._sessions.pop(probe, None)
                    return tag   # Wiederverwendung desselben Basis-Snapshots (Daten + Pakete) zwischen Prozessen
            except Exception:   # noqa: BLE001 - nicht vorhanden oder nicht lesbar: neu bauen
                pass
        return self.inner.base_snapshot(image, setup_cmds, label=label)

    def branch(self, snapshot):
        name = self.inner.branch(snapshot)
        getattr(self.created, "names", []).append(name)
        return name

    def run(self, branch, cmd, timeout_s):
        with self.slots:
            t0 = time.time()
            try:
                return self.inner.run(branch, cmd, timeout_s)
            finally:
                if "train.py" in cmd and "SMOKE=1" not in cmd:
                    self.times[branch] = time.time() - t0

    def put_file(self, branch, path, content):
        self.inner.put_file(branch, path, content)

    def put_files(self, branch, files):
        self.inner.put_files(branch, files)

    def get_file(self, branch, path):
        return self.inner.get_file(branch, path)

    def begin_attempt(self) -> None:
        self.created.names = []

    def release_attempt(self) -> None:
        for name in getattr(self.created, "names", []):
            self.release(name)
        self.created.names = []

    def release(self, branch):
        self.inner._sessions.pop(branch, None)

    def cleanup(self):
        self.inner._sessions.clear()
