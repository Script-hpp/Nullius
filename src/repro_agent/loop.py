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
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml

from .claims_source import extract_target, fetch_paper_text
from .config import load_settings
from .dossier import REPO_ROOT, Attempt, Claim, Dossier, Verdict
from .harness.code_check import find_violations
from .harness.verify import judge
from .llm import LLMClient, LLMResponse
from .report import render_markdown
from .sandbox.base import Branch, get_sandbox
from .tasks import HARNESS_DIR, SANDBOX_HARNESS_DIR, TASKS, Task

DEFAULT_SEEDS = 5
TRAIN_TIMEOUT_S = 600
EVAL_TIMEOUT_S = 300
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
TRIAGE_SYSTEM = (
    "Summarise this Python crash in at most 3 lines: root cause and the line/module involved. "
    "No fix suggestions."
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
            tolerance=c["tolerance"], source=c["source"], origin="paper",
        )
        for c in paper.get("claims", [])
    ]
    if not claims:
        raise RuntimeError(f"Paper {paper_id} hat keine Claims in benchmark/papers.yaml")
    return paper, claims


class Orchestrator:
    def __init__(self, use_tavily: bool = True) -> None:
        self.use_tavily = use_tavily
        self.settings = load_settings()
        self.llm = LLMClient(self.settings)
        self.sandbox = get_sandbox(self.settings.sandbox_backend)
        self.spent_usd = 0.0

    # ---- LLM ----------------------------------------------------------------
    def _ask(self, role, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        if self.spent_usd >= self.settings.max_cost_per_run_usd:
            raise CostLimitExceeded(f"MAX_COST_PER_RUN_USD erreicht ({self.spent_usd:.4f} $)")
        resp = self.llm.complete(
            role,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        self.spent_usd += resp.cost_usd
        return resp

    def plan(self, paper: dict, claim: Claim, task: Task) -> str:
        return self._ask(
            "reasoning", PLAN_SYSTEM,
            f"Paper: {paper['title']}\nClaim: {claim.text} (target {claim.target}, {claim.metric})\n\n"
            f"{task.data_description}\n\n"
            "Write a short reproduction plan: model architecture, all hyperparameters and training "
            "protocol as described in the paper. Mark every value you are NOT sure about with "
            "'(uncertain)'. Max 15 lines.",
        ).content

    def write_code(self, plan: str, task: Task) -> str:
        resp = self._ask(
            "code", CODE_SYSTEM,
            f"{task.data_description}\n\nReproduction plan:\n{plan}\n\n"
            "The script runs WITHOUT network access and must not download anything, spawn "
            "processes or use eval/exec; use only /data/cora_public.npz. "
            "Write train.py implementing this plan. Implement the model yourself in PyTorch "
            "(no torch_geometric available). Print the final validation accuracy to stdout.",
        )
        return extract_code(resp.content)

    def repair_code(self, plan: str, task: Task, code: str, error_summary: str) -> str:
        resp = self._ask(
            "code", CODE_SYSTEM,
            f"{task.data_description}\n\nReproduction plan:\n{plan}\n\nThis script failed:\n"
            f"```python\n{code}\n```\nProblem: {error_summary}\n\nReturn the corrected full script.",
        )
        return extract_code(resp.content)

    def triage(self, text: str) -> str:
        return self._ask("fast", TRIAGE_SYSTEM, text[-4000:], max_tokens=300).content.strip()

    # ---- Sandbox ------------------------------------------------------------
    def _run_and_eval(self, branch: Branch, code: str, seed: int, task: Task) -> dict:
        """Ein Versuch im Branch. Rueckgabe: {"status": timeout|fehler|ungueltig|ok, ...}."""
        sb = self.sandbox
        violations = find_violations(code)
        if violations:
            return {"status": "ungueltig", "detail": "Statische Pruefung: " + "; ".join(violations),
                    "static": True}
        sb.put_file(branch, f"{WORKDIR}/train.py", code.encode())
        # `unshare -n`: eigener Netz-Namespace ohne Interfaces -> kein Internet waehrend des
        # Agenten-Codes (Setup ist vorher im Basis-Snapshot abgeschlossen).
        r = sb.run(
            branch,
            f"mkdir -p /output && cd {WORKDIR} && SEED={seed} unshare -n python train.py",
            TRAIN_TIMEOUT_S,
        )
        if r.timed_out:
            return {"status": "timeout", "detail": f"Training > {TRAIN_TIMEOUT_S}s"}
        if r.exit_code != 0:
            return {"status": "fehler", "detail": (r.stderr or r.stdout)}

        # Harness frisch aus unserem Repo kopieren -- Agent konnte ihn vorher nicht aendern.
        sb.run(branch, f"rm -rf {SANDBOX_HARNESS_DIR} && mkdir -p {SANDBOX_HARNESS_DIR}", 60)
        for name in task.harness_files:
            sb.put_file(branch, f"{SANDBOX_HARNESS_DIR}/{name}", (HARNESS_DIR / name).read_bytes())
        e = sb.run(branch, task.eval_cmd, EVAL_TIMEOUT_S)
        if e.timed_out or e.exit_code != 0:
            return {"status": "ungueltig", "detail": f"Harness-Lauf fehlgeschlagen: {e.stderr[-500:]}"}
        try:
            out = json.loads(e.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return {"status": "ungueltig", "detail": f"Harness-Ausgabe unlesbar: {e.stdout[-300:]}"}
        if out["status"] != "ok":
            return {"status": "ungueltig", "detail": out.get("reason", "")}
        return {"status": "ok", "measured": float(out["measured"])}

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
            return tavily

        for claim in claims:
            spec = claim_specs(paper).get(claim.id, {}).get("extract")
            if not spec:
                continue
            text = fetch_paper_text(arxiv_id, get_tavily)
            ev = extract_target(self.llm, text, spec["method"], spec["dataset"], spec["metric"])
            claim.target = ev.value
            claim.source = f"{ev.table or 'Tabelle'}, arXiv {ev.arxiv_version or '?'} ({ev.source_kind})"
            claim.evidence_urls = [ev.url]
            claim.target_evidence = dataclasses.asdict(ev)
            print(f"[{claim.id}] Zielwert {ev.value} belegt: {ev.quote!r}")

    # ---- Loop pro Claim -----------------------------------------------------
    def run_claim(
        self, paper: dict, claim: Claim, task: Task, snapshot: str, dossier: Dossier, run_dir: Path
    ) -> None:
        plan = self.plan(paper, claim, task)
        (run_dir / f"{claim.id}_plan.md").write_text(plan)
        code = self.write_code(plan, task)
        measured: list[float] = []
        notes: list[str] = []

        seeds = list(range(paper.get("seeds", DEFAULT_SEEDS)))
        for seed in seeds:
            for n in range(1, self.settings.max_attempts_per_claim + 1):
                code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
                (run_dir / f"{claim.id}_{code_hash}.py").write_text(code)
                branch = self.sandbox.branch(snapshot)
                t0 = time.time()
                res = self._run_and_eval(branch, code, seed, task)
                status, value = res["status"], res.get("measured")
                summary = None
                if status == "ok":
                    status = "stabil" if abs(value - claim.target) <= claim.tolerance else "abweichung"
                else:
                    summary = (
                        res["detail"] if res.get("static") or status == "timeout"
                        else self.triage(res["detail"])
                    )
                dossier.attempts.append(Attempt(
                    claim_id=claim.id, branch_id=branch, parent_snapshot=snapshot, code_hash=code_hash,
                    seed=seed, status=status, measured=value, error_summary=summary,
                    cost_usd=self.spent_usd,
                ))
                print(f"[{claim.id}] seed={seed} versuch={n} {status} measured={value} ({time.time()-t0:.0f}s)")
                if value is not None:
                    measured.append(value)
                    break
                if status == "timeout":
                    notes.append(f"Seed {seed}: Timeout, kein Reparaturversuch")
                    break
                if n < self.settings.max_attempts_per_claim:
                    code = self.repair_code(plan, task, code, summary or "")
            else:
                notes.append(f"Seed {seed}: kein gueltiger Lauf nach {n} Versuchen ({summary})")

        dossier.verdicts.append(self._verdict(claim, measured, len(seeds), notes))

    def _verdict(
        self, claim: Claim, measured: list[float], n_seeds: int, notes: list[str]
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
        text = (
            f"Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) {std:.4f}. "
            f"Mit unserem Setup gemessen ({len(measured)} von {n_seeds} Seeds, Zielwert "
            f"{claim.target} +/- {claim.tolerance}, {claim.source}): Mittel {mean:.4f}."
        )
        return Verdict(
            claim_id=claim.id, urteil=urteil, measured_mean=mean, measured_std=std,
            sources=list(claim.evidence_urls),
            explanation=text + (f" {tried}" if tried else ""),
        )

    # ---- Einstieg -----------------------------------------------------------
    def run(self, paper_id: str) -> Dossier:
        paper, claims = load_claims(paper_id)
        task = TASKS[paper["dataset"]]()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = REPO_ROOT / "runs" / paper_id / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.use_tavily:
            self.resolve_targets(paper, claims)
        dossier = Dossier(paper_id=paper_id, claims=claims, isolation=list(ISOLATION))

        label = re.sub(r"[^a-z0-9-]", "-", f"{paper_id}-{ts}".lower())
        print(f"Basis-Snapshot bauen ({label}) ...")
        snapshot = self.sandbox.base_snapshot(task.image, task.setup_cmds, label=label)
        try:
            for claim in claims:
                self.run_claim(paper, claim, task, snapshot, dossier, run_dir)
        finally:
            dossier.save(ts)
            (run_dir / "zeugnis.md").write_text(render_markdown(dossier))
        print(f"Fertig. Bericht: {run_dir / 'zeugnis.md'}")
        return dossier


def main() -> None:
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paper_id = args[0] if args else "kipf_gcn_2017"
    Orchestrator(use_tavily="--no-tavily" not in sys.argv).run(paper_id)


if __name__ == "__main__":
    main()
