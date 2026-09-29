"""Orchestrator: der Loop pro Claim (siehe CLAUDE.md).

Meilenstein 1: GCN/Cora end-to-end, sequentiell, ohne Tavily.
TODO Meilenstein 2: Fehler-Rollback, Tavily-Rollen 1-5 verdrahten.
  Timeout (RunResult.timed_out) ist eigener Status "timeout", KEIN "fehler": keine
  Tavily-Fehlersuche, kein Code-Reparaturversuch; stattdessen Datenmenge verkleinern
  oder Experiment verwerfen.
TODO Meilenstein 3: Abweichungs-Loop, Einwaende als Branches, Seeds parallel (Semaphor ~4).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import os
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from .claims_source import PaperText, extract_target_cached, fetch_paper_text
from .config import load_settings
from .dossier import REPO_ROOT, Attempt, Claim, Dossier, Verdict
from .harness.code_check import find_violations
from .events import EventLog
from .harness.ablation import classify_ablation
from .harness.coverage_check import Coverage
from .harness.smoke import check_smoke_output
from .harness.verify import agent_fault_suspected, judge
from .llm import LLMClient, LLMResponse
from .report import render_markdown
from .sandbox.base import Branch, SandboxUnavailable, get_sandbox
from .repo_source import fetch_readme, find_repo_url, read_repo
from .setup_details import Doc, SetupDetail, extract_setup_details, format_for_prompt
from .tasks import HARNESS_DIR, SANDBOX_HARNESS_DIR, TASKS, Task

DEFAULT_SEEDS = 5
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL_SEEDS", "4"))  # gleichzeitige Branches; 20 liefen im Test ohne Fehler
TRAIN_TIMEOUT_S = 600
EVAL_TIMEOUT_S = 300
SMOKE_TIMEOUT_S = 90
INFRA_RETRIES = 3       # Wiederholungen eines Versuchs bei Sandbox-Infrastrukturfehler (neuer Branch je Versuch)
INFRA_BACKOFF_S = 5.0
TRIAGE_CHARS = 1500
WORKDIR = "/work"
PAPERS_YAML = REPO_ROOT / "benchmark" / "papers.yaml"

PLAN_SYSTEM = (
    "You are a careful ML researcher planning a reproduction of a published result. "
    "Be concise and concrete."
)
CODE_SYSTEM = (
    "You write a single self-contained Python script for an ML reproduction. "
    "Reply with exactly one ```python code block and no other prose."
)


ISOLATION = [
    "Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot",
    "Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)",
    "Testlabels dem Agenten vorenthalten, Harness laedt sie frisch",
]


class CostLimitExceeded(RuntimeError):
    pass


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        raise ValueError("Keine Code-Block in der Antwort des Modells")
    return max(blocks, key=len)


def claim_specs(paper: dict) -> dict[str, dict]:
    return {c["id"]: c for c in paper.get("claims", [])}


def load_claims(paper_id: str) -> tuple[dict, list[Claim]]:
    papers = yaml.safe_load(PAPERS_YAML.read_text())["papers"]
    paper = next(p for p in papers if p["id"] == paper_id)
    claims = [
        Claim(
            id=c["id"], text=c["text"], metric=c["metric"], target=c["target"],
            tolerance=c["tolerance"], source=c["source"], origin=c.get("origin", "paper"),
        )
        for c in paper.get("claims", [])
    ]
    if not claims:
        raise RuntimeError(f"Paper {paper_id} hat keine Claims in benchmark/papers.yaml")
    return paper, claims


class Orchestrator:
    def __init__(
        self, use_tavily: bool = True, fresh: bool = False, setup_details: bool = True,
        seeds: int | None = None, claim_ids: list[str] | None = None, paper_only: bool = False,
    ) -> None:
        self.use_tavily = use_tavily
        self.use_setup_details = use_tavily and setup_details   # Rolle 4 haengt am Paper-Text
        self.paper_only = paper_only   # Rolle 4 liest nur den Paper-Text, kein Repo/README (Shchur-Experiment)
        self.seeds = seeds
        self.claim_ids = claim_ids   # nur diese Claims des Papers laufen lassen (None = alle)
        self.details: list[SetupDetail] = []
        self.paper_text: PaperText | None = None
        self.last_run_dir: Path | None = None
        self._cost_base = 0.0
        self.events = EventLog(None)       # ab run(): runs/<paper>/<ts>/events.jsonl
        self._smoke_ok: set[str] = set()   # Codes, die die Formpruefung bestanden haben
        self._snapshots: dict[str, str] = {}   # Basis-Snapshot pro Task im Prozess wiederverwenden
        self.settings = load_settings()
        if fresh:  # --fresh: LLM-Cache weder lesen noch schreiben (Stabilitaetsbeleg, Benchmark)
            self.settings = dataclasses.replace(self.settings, llm_cache=False)
        self.llm = LLMClient(self.settings)
        self.sandbox = get_sandbox(self.settings.sandbox_backend)
        self._lock = threading.Lock()  # LLM-Aufrufe/Kostenzaehler/Dossier aus Seed-Threads

    @property
    def spent_usd(self) -> float:
        return self.llm.total_cost_usd - self._cost_base   # Kosten seit Start des aktuellen Laufs

    def _emit(self, type: str, **data) -> None:
        ev = getattr(self, "events", None)
        if ev is not None:
            ev.write(type, **data)

    def _emit_tavily(self, role: int, endpoint: str, target, n_results: int) -> None:
        self._emit("tavily_call", role=role, endpoint=endpoint, target=target, n_results=n_results)

    @property
    def spent_gross_usd(self) -> float:
        return self.llm.gross(self.spent_usd)

    # ---- LLM ----------------------------------------------------------------
    def _ask(
        self, role, system: str, user: str, max_tokens: int | None = None, temperature: float = 0.2
    ) -> LLMResponse:
        if self.spent_gross_usd >= self.settings.max_cost_per_run_usd:   # Limit gilt brutto
            raise CostLimitExceeded(
                f"MAX_COST_PER_RUN_USD erreicht ({self.spent_gross_usd:.4f} $ brutto, "
                f"{self.spent_usd:.4f} $ netto)"
            )
        with self._lock:
            resp = self.llm.complete(
                role,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature,
            )
        return resp

    def plan(self, paper: dict, claim: Claim, task: Task) -> str:
        return self._ask(
            "reasoning", PLAN_SYSTEM,
            f"Paper: {paper['title']}\nClaim: {claim.text} (target {claim.target}, {claim.metric})\n\n"
            f"{task.data_description}\n\n{format_for_prompt(self.details)}\n\n"
            "Write a short reproduction plan: model architecture, all hyperparameters and training "
            "protocol as described in the paper. Mark every value you are NOT sure about with "
            "'(uncertain)'. Max 15 lines.",
        ).content

    def _code_from_llm(self, system: str, user: str) -> str:
        """Code-Block aus der Antwort; bei fehlendem Block bis zu 3 Versuche mit hoeherer
        Temperatur (gleicher Prompt + gleiche Temperatur gaebe aus dem Cache dieselbe Antwort)."""
        last: ValueError | None = None
        for i in range(3):
            resp = self._ask("code", system, user, temperature=0.2 + 0.3 * i)
            try:
                return extract_code(resp.content)
            except ValueError as exc:
                last = exc
        raise last  # type: ignore[misc]

    def write_code(self, plan: str, task: Task) -> str:
        return self._code_from_llm(
            CODE_SYSTEM,
            f"{task.data_description}\n\n{format_for_prompt(self.details)}\n\n"
            f"Reproduction plan:\n{plan}\n\n"
            "The script runs WITHOUT network access and must not download anything, spawn "
            f"processes or use eval/exec. {task.code_notes} Write train.py implementing this plan.",
        )

    def repair_code(self, plan: str, task: Task, code: str, error_summary: str) -> str:
        return self._code_from_llm(
            CODE_SYSTEM,
            f"{task.data_description}\n\n{format_for_prompt(self.details)}\n\n"
            f"Reproduction plan:\n{plan}\n\nThis script failed:\n"
            f"```python\n{code}\n```\nProblem: {error_summary}\n\nReturn the corrected full script.",
        )

    def triage(self, text: str) -> str:
        """Fehlertext fuer die Reparatur: Ende von stderr/stdout (dort steht der Traceback).

        Eine LLM-Zusammenfassung durch `fast` taugt nicht: Die Free- und TF-Modelle schreiben
        ihren Denkprozess in die Antwort, abgeschnitten bei max_tokens kam kein Fehler an.
        """
        return text.strip()[-TRIAGE_CHARS:]

    # ---- Sandbox ------------------------------------------------------------
    def _prepare_data(self, branch: Branch, seed: int, task: Task) -> str | None:
        """Daten fuer diesen Lauf (Split) im Branch erzeugen. Das Skript liegt nur waehrend der Vorbereitung dort."""
        sb = self.sandbox
        d = f"/opt/prep-{uuid.uuid4().hex[:10]}"
        sb.put_files(branch, {f"{d}/{n}": (HARNESS_DIR / n).read_bytes() for n in task.prepare_files})
        r = sb.run(branch, task.prepare_cmd.format(dir=d, split=task.split_for(seed)) + f"; rc=$?; rm -rf {d}; exit $rc", 300)
        if r.timed_out or r.exit_code != 0:
            return f"Datenvorbereitung fehlgeschlagen: {(r.stderr or r.stdout)[-300:]}"
        return None

    def _smoke_check(self, branch: Branch, code: str, task: Task, seed: int = 0) -> str | None:
        """Formpruefung: Code mit SMOKE=1 auf winziger Datenmenge, Ausgabe gegen die Form des Tasks.
        Findet Tensorform-Fehler in Sekunden statt nach dem Training. None = bestanden."""
        if not (task.smoke_ids or task.smoke_ids_cmd):
            return None
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        if code_hash in self._smoke_ok:
            return None
        sb = self.sandbox
        expected = task.smoke_ids
        if task.smoke_ids_cmd:
            expected = tuple(sb.run(branch, task.smoke_ids_cmd, 60).stdout.split())
        r = sb.run(
            branch,
            f"mkdir -p /output && cd {WORKDIR} && {task.env_for(seed)} SMOKE=1 unshare -n python train.py",
            SMOKE_TIMEOUT_S,
        )
        if r.timed_out:
            return (f"Smoke run (SMOKE=1) exceeded {SMOKE_TIMEOUT_S}s: the script ignores SMOKE=1. "
                    "With SMOKE=1 use the tiny subset and 1 epoch as described.")
        if r.exit_code != 0:
            return "Smoke run (SMOKE=1) crashed:\n" + (r.stderr or r.stdout).strip()[-TRIAGE_CHARS:]
        try:
            out = sb.get_file(branch, "/output/predictions.csv")
        except Exception as exc:   # Sandbox-Adapter melden fehlende Datei unterschiedlich
            return f"Smoke run wrote no /output/predictions.csv ({exc})"
        error = check_smoke_output(out, expected, task.smoke_dims, task.smoke_dim_names)
        if error:
            return "Smoke run (SMOKE=1) output has the wrong form: " + error
        sb.run(branch, "rm -f /output/predictions.csv", 30)
        self._smoke_ok.add(code_hash)
        return None

    def _attempt(self, snapshot: str, code: str, seed: int, task: Task, metric: str, **kw) -> tuple[str, dict]:
        """Ein Versuch auf einem frischen Branch. Bei Infrastrukturfehler der Sandbox (SandboxUnavailable, z. B. API-Timeout)
        wird der GANZE Versuch auf einem neuen Branch wiederholt (der alte ist unbrauchbar); nach INFRA_RETRIES Fehlern
        steigt der Fehler auf und der Lauf bricht ab (kein Urteil ueber den Agenten). Rueckgabe: (Branch-ID, Ergebnis)."""
        for k in range(INFRA_RETRIES):
            branch = self.sandbox.branch(snapshot)
            try:
                return branch, self._run_and_eval(branch, code, seed, task, metric, **kw)
            except SandboxUnavailable as exc:
                self._emit("infra_retry", seed=seed, attempt=k + 1, error=str(exc)[:200])
                print(f"WARNUNG: {exc} (Infrastruktur-Wiederholung {k + 1}/{INFRA_RETRIES})", file=sys.stderr, flush=True)
                if k == INFRA_RETRIES - 1:
                    raise
                time.sleep(INFRA_BACKOFF_S * 2**k)
        raise AssertionError("unreachable")

    def ablation_check(self, snapshot: str, code: str, seed: int, task: Task, metric: str, acc_full: float) -> dict:
        """Zweiter Lauf desselben Codes auf Daten ohne Kanten (ablation_gcn.md). Deterministisch, kein LLM."""
        _, res = self._attempt(snapshot, code, seed, task, metric, ablate=True)
        acc_ablated = res.get("measured") if res["status"] == "ok" else None
        return {
            "accuracy_full": acc_full, "accuracy_ablated": acc_ablated,
            "verdict": classify_ablation(acc_full, acc_ablated),
            "detail": None if acc_ablated is not None else str(res.get("detail", ""))[-300:],
        }

    def _measure_coverage(self, branch: Branch) -> dict | None:
        """Zeilenabdeckung des gerade gelaufenen train.py (coverage im Basis-Snapshot). None = nicht verfuegbar."""
        r = self.sandbox.run(
            branch, f"cd {WORKDIR} && python -m coverage json --data-file=/tmp/.cov -o /tmp/cov.json", 60
        )
        if r.exit_code != 0:
            return None
        cov = Coverage.from_coverage_json(json.loads(self.sandbox.get_file(branch, "/tmp/cov.json")))
        if cov is None:
            return None
        return {"executed": sorted(cov.executed), "missing": sorted(cov.missing)}

    def _run_and_eval(
        self, branch: Branch, code: str, seed: int, task: Task, metric: str, measure_coverage: bool = False,
        ablate: bool = False,
    ) -> dict:
        """Ein Versuch im Branch. Rueckgabe: {"status": timeout|fehler|ungueltig|ok, ...}."""
        sb = self.sandbox
        violations = find_violations(code)
        if violations:
            return {"status": "ungueltig", "detail": "Statische Pruefung: " + "; ".join(violations),
                    "static": True}
        sb.put_file(branch, f"{WORKDIR}/train.py", code.encode())
        if task.prepare_cmd:
            prep_error = self._prepare_data(branch, seed, task)
            if prep_error:
                return {"status": "ungueltig", "detail": prep_error}
        if ablate:   # Nutzungspruefung: veraenderte Daten, kein Smoke-Lauf, keine Abdeckung
            sb.run(branch, task.ablation_cmd, 120)
            measure_coverage = False
        else:
            smoke_error = self._smoke_check(branch, code, task, seed)
            if smoke_error:
                return {"status": "fehler", "detail": smoke_error}
        # `unshare -n`: eigener Netz-Namespace ohne Interfaces -> kein Internet waehrend des
        # Agenten-Codes (Setup ist vorher im Basis-Snapshot abgeschlossen).
        # Abdeckung nur im ersten vollstaendigen Seed-Lauf (Smoke-Lauf ist zu kurz: Epochen-/Early-Stopping-Zweige)
        with_cov = measure_coverage and sb.run(branch, "python -c 'import coverage'", 30).exit_code == 0
        runner = "python -m coverage run --data-file=/tmp/.cov train.py" if with_cov else "python train.py"
        r = sb.run(
            branch,
            f"mkdir -p /output && cd {WORKDIR} && {task.env_for(seed)} unshare -n {runner}",
            TRAIN_TIMEOUT_S,
        )
        if r.timed_out:
            return {"status": "timeout", "detail": f"Training > {TRAIN_TIMEOUT_S}s"}
        if r.exit_code != 0:
            return {"status": "fehler", "detail": (r.stderr or r.stdout)}
        coverage = self._measure_coverage(branch) if with_cov else None

        # Harness frisch aus unserem Repo kopieren, in ein Verzeichnis mit Zufallsnamen: der
        # Agent-Code ist schon gelaufen und kannte den Pfad nicht, kann ihn also nicht vorbelegt haben.
        harness_dir = f"{SANDBOX_HARNESS_DIR}-{uuid.uuid4().hex[:10]}"
        sb.put_files(branch, {
            f"{harness_dir}/{name}": (HARNESS_DIR / name).read_bytes() for name in task.harness_files
        })
        e = sb.run(branch, task.eval_cmd.format(harness_dir=harness_dir, split=task.split_for(seed)), EVAL_TIMEOUT_S)
        if e.timed_out or e.exit_code != 0:
            return {"status": "ungueltig", "detail": f"Harness-Lauf fehlgeschlagen: {e.stderr[-500:]}"}
        try:
            out = json.loads(e.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return {"status": "ungueltig", "detail": f"Harness-Ausgabe unlesbar: {e.stdout[-300:]}"}
        if out["status"] != "ok":
            return {"status": "ungueltig", "detail": out.get("reason", "")}
        if metric not in out.get("metrics", {}):
            return {"status": "ungueltig", "detail": f"Harness lieferte Metrik {metric} nicht"}
        return {
            "status": "ok", "measured": float(out["metrics"][metric]), "metrics": out["metrics"],
            "baselines": out.get("baselines"), "coverage": coverage,
        }

    # ---- Tavily Rolle 2: Zielwerte aus dem Paper -------------------------------
    def resolve_targets(self, paper: dict, claims: list[Claim]) -> None:
        """Ersetzt den manuellen Zielwert durch den belegten. Ungueltige Extraktion = Abbruch."""
        arxiv_id = paper["arxiv_url"].rstrip("/").split("/abs/")[-1]
        tavily = None

        def get_tavily():
            nonlocal tavily
            if tavily is None:
                from .tavily_tools import TavilyTools

                tavily = TavilyTools(self.settings)
                tavily.on_call = self._emit_tavily
            return tavily

        for claim in claims:
            spec = claim_specs(paper).get(claim.id, {}).get("extract")
            if not spec:
                continue
            text = fetch_paper_text(arxiv_id, get_tavily)
            self.paper_text = text
            ev = extract_target_cached(
                self.llm, text, spec["method"], spec["dataset"], spec["metric"],
                percent=spec.get("percent", True),
            )
            claim.target = ev.value
            claim.source = f"{ev.table or 'Tabelle'}, arXiv {ev.arxiv_version or '?'} ({ev.source_kind})"
            claim.evidence_urls = [ev.url]
            claim.target_evidence = dataclasses.asdict(ev)
            print(f"[{claim.id}] Zielwert {ev.value} belegt: {ev.quote!r}")
            self._emit("target_resolved", claim_id=claim.id, value=ev.value, quote=ev.quote, table=ev.table,
                       url=ev.url, arxiv_version=ev.arxiv_version, column_check=ev.column_check, caveat=ev.caveat)

    def _gather_docs(self, claim_text: str, dataset: str) -> tuple[list[Doc], dict | None]:
        """Paper-Text + (falls im Paper verlinkt) README (Tavily) und Repo-Dateien (Sandbox-Klon)."""
        docs = [Doc("paper", self.paper_text.text)]
        if self.paper_only:
            return docs, None
        url = find_repo_url(self.paper_text.text)
        if not url:
            return docs, None
        get_tavily = self._tavily_getter()
        readme = fetch_readme(url, get_tavily)
        if readme:
            docs.append(Doc("repo-readme", readme))
        dump = read_repo(self.sandbox, self.llm, url, claim_text, dataset)
        for path, text in dump.files.items():
            docs.append(Doc(f"repo:{path}@{dump.commit[:7]}", text))
        return docs, {"url": url, "commit": dump.commit, "files": sorted(dump.files)}

    def _tavily_getter(self):
        if not hasattr(self, "_tavily"):
            self._tavily = None

        def get():
            if self._tavily is None:
                from .tavily_tools import TavilyTools

                self._tavily = TavilyTools(self.settings)
                self._tavily.on_call = self._emit_tavily
            return self._tavily

        return get

    # ---- Loop pro Claim -----------------------------------------------------
    def run_claim(
        self, paper: dict, claim: Claim, task: Task, snapshot: str, dossier: Dossier, run_dir: Path
    ) -> None:
        self._emit("claim_start", claim_id=claim.id, text=claim.text, metric=claim.metric, target=claim.target,
                   tolerance=claim.tolerance)
        plan = self.plan(paper, claim, task)
        (run_dir / f"{claim.id}_plan.md").write_text(plan)
        self._emit("plan", claim_id=claim.id, text=plan[:1500])
        code = self.write_code(plan, task)
        self._emit("code_written", claim_id=claim.id, code_hash=hashlib.sha256(code.encode()).hexdigest()[:12])
        seeds = list(range(self.seeds or paper.get("seeds", DEFAULT_SEEDS)))
        # Seed 0 zuerst mit Reparaturschleife -> ein lauffaehiger Code pro Generierung. Die uebrigen
        # Seeds laufen parallel mit genau diesem Code (Streuung = Seed, nicht Code-Variante).
        v0, note0, code = self._run_seed(claim, task, snapshot, dossier, run_dir, plan, code, seeds[0])
        measured = [] if v0 is None else [v0]
        notes = [] if note0 is None else [note0]
        if v0 is None:
            notes.append("Restliche Seeds nicht gestartet: kein lauffaehiger Code")
        else:
            rest = seeds[1:]
            if rest:
                with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(rest))) as pool:
                    results = list(pool.map(
                        lambda sd: self._run_seed(claim, task, snapshot, dossier, run_dir, plan, code, sd),
                        rest,
                    ))
                measured += [v for v, _, _ in results if v is not None]
                notes += [n for _, n, _ in results if n]
        per_metric: dict[str, list[float]] = {}
        for a in dossier.attempts:
            if a.claim_id == claim.id and a.metrics and a.measured is not None:
                for name, value in a.metrics.items():
                    per_metric.setdefault(name, []).append(value)
        all_metrics = {name: statistics.mean(v) for name, v in per_metric.items()}
        verdict = self._verdict(claim, measured, len(seeds), notes, dossier.baselines, all_metrics)
        if v0 is not None and task.ablation_cmd:
            ab = self.ablation_check(snapshot, code, seeds[0], task, claim.metric, v0)
            dossier.ablations[claim.id] = ab
            self._emit("ablation", claim_id=claim.id, **ab)
            print(f"[{claim.id}] Graph-Ablation: {ab['verdict']} (mit Graph {v0:.4f}, ohne Kanten {ab['accuracy_ablated']})", flush=True)
            if ab["verdict"] == "Graph nicht genutzt":   # deterministischer Beleg, nur herabstufend
                verdict.methode = "nein"
        dossier.verdicts.append(verdict)
        self._emit("verdict", claim_id=claim.id, urteil=verdict.urteil, methode=verdict.methode,
                   gesamt=verdict.gesamtaussage, mean=verdict.measured_mean, std=verdict.measured_std,
                   explanation=verdict.explanation)

    def _run_seed(
        self, claim: Claim, task: Task, snapshot: str, dossier: Dossier, run_dir: Path,
        plan: str, code: str, seed: int,
    ) -> tuple[float | None, str | None, str]:
        """Ein Seed mit Reparaturschleife. Rueckgabe: (Messwert oder None, Hinweis, letzter Code)."""
        summary = None
        for n in range(1, self.settings.max_attempts_per_claim + 1):
            code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
            (run_dir / f"{claim.id}_{code_hash}.py").write_text(code)
            t0 = time.time()
            branch, res = self._attempt(snapshot, code, seed, task, claim.metric, measure_coverage=seed == 0)
            status, value = res["status"], res.get("measured")
            summary = None
            if status == "ok":
                status = "stabil" if abs(value - claim.target) <= claim.tolerance else "abweichung"
            else:
                summary = (
                    res["detail"] if res.get("static") or status == "timeout"
                    else self.triage(res["detail"])
                )
            with self._lock:
                dossier.attempts.append(Attempt(
                    claim_id=claim.id, branch_id=branch, parent_snapshot=snapshot, code_hash=code_hash,
                    seed=seed, status=status, measured=value, error_summary=summary,
                    cost_usd=self.spent_usd, metrics=res.get("metrics"), coverage=res.get("coverage"),
                ))
            print(f"[{claim.id}] seed={seed} versuch={n} {status} measured={value} ({time.time()-t0:.0f}s)",
                  flush=True)
            self._emit("attempt", claim_id=claim.id, seed=seed, n=n, branch_id=branch, parent_snapshot=snapshot,
                       code_hash=code_hash, status=status, measured=value, seconds=round(time.time() - t0, 1),
                       error=(summary or "")[-400:] or None)
            if res.get("baselines"):
                dossier.baselines = res["baselines"]
            if value is not None:
                return value, None, code
            if status == "timeout":
                return None, f"Seed {seed}: Timeout, kein Reparaturversuch", code
            if n < self.settings.max_attempts_per_claim:
                try:
                    self._emit("repair", claim_id=claim.id, seed=seed, from_hash=code_hash, error=(summary or "")[-400:])
                    code = self.repair_code(plan, task, code, summary or "")
                except ValueError as exc:   # Modell lieferte trotz Wiederholung keinen Code
                    return None, f"Seed {seed}: Reparatur ohne Code-Block ({exc}), Abbruch nach {n} Versuchen", code
        last_line = (summary or "").strip().splitlines()[-1][:200] if summary else ""
        return None, f"Seed {seed}: kein gueltiger Lauf nach {n} Versuchen (letzter Fehler: {last_line})", code

    def _verdict(
        self, claim: Claim, measured: list[float], n_seeds: int, notes: list[str],
        baselines: dict | None = None, all_metrics: dict | None = None,
    ) -> Verdict:
        tried = "; ".join(notes)
        if not measured:
            return Verdict(
                claim_id=claim.id, urteil="nicht reproduzierbar (Setup)", measured_mean=float("nan"),
                measured_std=float("nan"),
                explanation=f"Mit unserem Setup kein gueltiger Lauf. {tried}",
            )
        mean = statistics.mean(measured)
        std = statistics.stdev(measured) if len(measured) > 1 else 0.0
        urteil = judge(mean, claim.target, claim.tolerance)
        fault = agent_fault_suspected(claim.metric, mean, claim.target, baselines, all_metrics)
        if fault:
            urteil = "Agenten-Fehler vermutet"
        text = (
            f"Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) {std:.4f}. "
            f"Mit unserem Setup gemessen ({len(measured)} von {n_seeds} Seeds, Zielwert "
            f"{claim.target} +/- {claim.tolerance}, {claim.source}): Mittel {mean:.4f}."
        )
        if fault:
            text += f" Agenten-Fehler vermutet, kein Befund ueber das Paper: {fault}."
        if len(measured) > 1 and std == 0.0:
            text += (" WARNUNG: alle Seeds liefern denselben Wert, der Lauf ist vermutlich "
                     "deterministisch, die Seeds tragen keine Information.")
        return Verdict(
            claim_id=claim.id, urteil=urteil, measured_mean=mean, measured_std=std,
            sources=list(claim.evidence_urls),
            explanation=text + (f" {tried}" if tried else ""),
        )

    # ---- Einstieg -----------------------------------------------------------
    def run(self, paper_id: str) -> Dossier:
        paper, claims = load_claims(paper_id)
        if self.claim_ids:
            claims = [c for c in claims if c.id in self.claim_ids]
        task = TASKS[paper["dataset"]]()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = REPO_ROOT / "runs" / paper_id / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        self.last_run_dir = run_dir
        self.events = EventLog(run_dir / "events.jsonl")
        self._emit("run_start", paper_id=paper_id, title=paper["title"], mode="nachbau", use_tavily=self.use_tavily,
                   use_setup_details=self.use_setup_details, paper_only=self.paper_only,
                   models={"fast": self.settings.llm.model_fast, "code": self.settings.llm.model_code,
                           "reasoning": self.settings.llm.model_reasoning},
                   seeds=self.seeds or paper.get("seeds", DEFAULT_SEEDS), claims=[c.id for c in claims])
        self._cost_base = self.llm.total_cost_usd
        if self.use_tavily:
            self.resolve_targets(paper, claims)
        repo_info = None
        if self.use_setup_details and self.paper_text is not None:
            docs, repo_info = self._gather_docs(claims[0].text, paper["dataset"])
            self.details, dropped = extract_setup_details(
                self.llm, docs, claims[0].text, paper["dataset"]
            )
            by_src: dict[str, int] = {}
            for d in self.details:
                by_src[d.source.split("@")[0]] = by_src.get(d.source.split("@")[0], 0) + 1
            print(f"Setup-Details (Rolle 4): {len(self.details)} belegt, {dropped} verworfen, "
                  f"nach Quelle {by_src}", flush=True)
            self._emit("setup_details", n=len(self.details), dropped=dropped, by_source=by_src,
                       repo=repo_info, details=[dataclasses.asdict(d) for d in self.details])
        dossier = Dossier(
            paper_id=paper_id, claims=claims, isolation=list(ISOLATION), repo=repo_info,
            setup_details=[dataclasses.asdict(d) for d in self.details],
        )

        label = re.sub(r"[^a-z0-9-]", "-", f"{paper_id}-{ts}".lower())
        print(f"Basis-Snapshot bauen ({label}) ...")
        key = paper["dataset"]
        if key not in self._snapshots:
            self._snapshots[key] = self.sandbox.base_snapshot(task.image, task.setup_cmds, label=label)
        snapshot = self._snapshots[key]
        try:
            for claim in claims:
                self.run_claim(paper, claim, task, snapshot, dossier, run_dir)
        finally:
            dossier.save(ts)
            (run_dir / "zeugnis.md").write_text(render_markdown(dossier))
        self._emit("run_end", cost_net_usd=round(self.spent_usd, 4), cost_gross_usd=round(self.spent_gross_usd, 4))
        print(f"Fertig. Bericht: {run_dir / 'zeugnis.md'} | Kosten: {self.spent_usd:.4f} $", flush=True)
        return dossier


def main() -> None:
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paper_id = args[0] if args else "kipf_gcn_2017"
    from .llm import BillingBlocked

    try:
        Orchestrator(
            use_tavily="--no-tavily" not in sys.argv,
            fresh="--fresh" in sys.argv,
            setup_details="--no-setup-details" not in sys.argv,
            paper_only="--paper-only" in sys.argv,
        ).run(paper_id)
    except BillingBlocked as exc:
        sys.exit(f"ABBRUCH: {exc}")


if __name__ == "__main__":
    main()
