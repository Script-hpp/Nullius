"""Orchestrator: der Loop pro Claim (siehe CLAUDE.md).

Meilenstein 1: GCN/Cora end-to-end, sequentiell, ohne Tavily.
TODO Meilenstein 2: Fehler-Rollback, Tavily-Rollen 1-5 verdrahten.
  Timeout (RunResult.timed_out) ist eigener Status "timeout", KEIN "fehler": keine
  Tavily-Fehlersuche, kein Code-Reparaturversuch; stattdessen Datenmenge verkleinern
  oder Experiment verwerfen.
TODO Meilenstein 3: Abweichungs-Loop, Einwaende als Branches, Seeds parallel (Semaphor ~4).
"""

from __future__ import annotations

from .spend_guard import context_map
from . import spend_guard

import dataclasses
import hashlib
import io
import json
import math
import re
import shlex
import os
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .catalog import discovery_details, load_catalog_claims
from . import community, objection_experiments
from .claims_source import ExtractionInvalid, TargetVerificationBlocked, PaperText, extract_target_cached, fetch_paper_text, verify_selected_target, _source_table_target
from .config import load_settings
from .costs import BudgetExceeded
from .dossier import REPO_ROOT, Attempt, Claim, Dossier, Verdict
from .harness.code_check import find_violations
from .droplast_note import droplast_note
from .events import EventLog
from .paper_head import abstract as fetch_abstract, arxiv_meta as fetch_arxiv_meta, build_head, claim_for_prompt
from .harness.ablation import classify_ablation
from .harness.edge_direction import ONE as EDGE_ONE, classify_edge_direction, make_probe, probe_bytes, score_probe
from .harness.val_use_check import classify_val_mask, static_val_use
from .harness.coverage_check import Coverage
from .harness.smoke import check_smoke_output
from .harness.repair_hints import fault_message, val_fault_message
from .harness.verify import agent_fault_suspected, judge
from .llm import BillingBlocked, LLMClient, LLMResponse
from .report import render_markdown
from .sandbox.base import Branch, SandboxSetupError, SandboxUnavailable, get_sandbox
from .repo_diff import find_deviations, repair_message, as_dicts
from .repo_source import (
    clone_plan, fetch_readme, find_repo_links, find_repo_url, read_repo, repo_evidence, repo_status,
    search_repo_candidates, search_repo_url,
)
from .setup_details import (
    Doc, SetupDetail, extract_setup_details, extract_setup_details_v2, format_conflicts, format_conflicts_v2, format_for_prompt,
    format_for_prompt_v2)
from .tasks import HARNESS_DIR, SANDBOX_HARNESS_DIR, TASKS, Task
from . import spec_check
from .method_guard import MethodImplementationBlocked, placeholder_issues
from .spec import (
    Spec, as_dict as spec_as_dict, build_spec, config_note, format_code_grounded, format_spec, model_files, model_name_of)

DEFAULT_SEEDS = 5
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL_SEEDS", "4"))  # gleichzeitige Branches; 20 liefen im Test ohne Fehler
MAX_GRAPH_REPAIRS = 2   # Reparaturrunden pro Claim, wenn die Ablation "Graph nicht genutzt" meldet
FIDELITY_NOTE = (
    "\n\nImplement the core mechanism described in the plan. Do not replace the method by a simpler stand-in "
    "(for example a plain linear model, an MLP or a standard GCN for a different architecture). If the full model is too large "
    "for 5 minutes on CPU, preserve every core mechanism and optimize batching or caching. Never replace retrieval, attention "
    "or message passing with constants, zeros, pass, or a stub. If it cannot fit, report that limitation rather than inventing "
    "a substitute. Start the script with a "
    "comment block in the form '# Plan step -> implemented in: function or class name', one line per plan step."
)
PLAN_RETRY_NOTE = (
    "\n\nYour previous answer was code or far too long. Reply with a numbered plan in plain prose, at most 15 lines, "
    "with no code block."
)


def plan_needs_retry(text: str) -> bool:
    """Ein Plan ist kurzer Text. Code-Fence am Anfang, mehrere import-Zeilen oder > 40 Zeilen = Modell hat Code statt Plan geliefert."""
    t = text.strip()
    return t.startswith("```") or sum(1 for ln in t.splitlines() if ln.lstrip().startswith(("import ", "from "))) >= 2 or t.count("\n") > 40


MAX_FAULT_REPAIRS = 2   # Reparaturrunden pro Claim, wenn das Ergebnis schlechter als eine triviale Baseline ist (Agenten-Fehler)
MAX_SEED0_REPAIRS = 3   # Gesamtgrenze beider Arten: jede Runde kostet einen Seed-0-Sandboxlauf (Minuten), Live-Demo bleibt kurz
EDGE_DIRECTION_MESSAGE = (   # B2: fester Hinweis ohne Zahlen
    "Your script ran, but it uses the graph edges in one direction only: information flows along each edge from one node to the other "
    "but not back. The graph must be used in both directions of every edge (messages from j to i and from i to j). "
    "Do not reduce the edge list to pairs with u < v without adding the reverse edges again.\n"
    "Keep the method described in the plan: fix the implementation, do not replace the method by a simpler one. Return the corrected full script."
)
GRAPH_REPAIR_MESSAGE = (
    "Your code does not use the graph: accuracy with edges {full:.4f} equals accuracy without edges {ablated:.4f} "
    "(the code ignores the adjacency). Implement the propagation over the adjacency described in the plan."
)
CANDIDATES = 3          # Code-Kandidaten pro Claim im Produktpfad (A1); Temperatur 0.2, 0.5, 0.7, Auswahl nur ueber Validierung
CANDIDATE_TEMP_STEP = 0.3
CODE_RETRY_TEMP_STEP = 0.15
# Obergrenze fuer Code-Antworten. FITS-Live-Lauf 35758c32 (2026-10-13): Nemotron Super brach bei 0.5-1.1 sechsmal
# mitten im Skript ab (finish_reason nicht "length"), lieferte bei 1.1 einen SyntaxError und bei 1.4 Wortsalat.
CODE_MAX_TEMPERATURE = 0.7
LOWER_IS_BETTER_VAL = ("mse", "mae")
PRODUCT_LIB_NOTE = (
    " Libraries such as torch_geometric, torch_scatter, torch_sparse, dgl and ogb are NOT installed and are forbidden: "
    "write every layer in plain PyTorch (dense or torch.sparse operations)."
)
TRAIN_TIMEOUT_S = 600
ABRT_GRACE_S = 15       # F2: `timeout -s ABRT` greift diese Sekunden vor dem harten Sandbox-Timeout (der Branch und seine Dateien sind danach weg)
SOFT_TIMEOUT_EXIT = 124   # Exit-Code von `timeout`
AGENT_OUTPUT_TAIL_BYTES = 30000   # je Strom; Nebius kappt stdout/stderr sonst nach 65535 Bytes am KOPF (Ausnahme und Broker-Zusammenfassung stehen am Ende)
EVAL_TIMEOUT_S = 300
SMOKE_TIMEOUT_S = 90
INFRA_RETRIES = 3       # Wiederholungen eines Versuchs bei Sandbox-Infrastrukturfehler (neuer Branch je Versuch)
INFRA_BACKOFF_S = 5.0
TRIAGE_CHARS = 1500
WORKDIR = "/work"
REPO_DIFF_MAX_TOKENS = 16000   # A2: ~16k Prompt-Token; mit 6000 zehrte Super das Limit mit Denken auf (v4-Abbruch)
CODE_RETRY_MAX_TOKENS = 24000   # Wiederholung nach fehlendem Code-Block oder abgeschnittener Antwort
MAX_SPEC_REPAIRS = 2   # Steckbrief (S3): gezielte Reparaturrunden je Claim fuer den gewaehlten Code
SPEC_MAX_TOKENS = 12000   # Steckbrief-Aufruf (S1) und Einzelfragen (S3): Super denkt im Antworttext
CODE_ONLY_NOTE = (
    "\n\nIMPORTANT: Reply with the complete script in one ```python block and nothing else. "
    "Do not reason at length before the code; the answer must not be cut off."
)
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


def select_candidate(vals: list[dict | None]) -> tuple[int, str]:
    """Waehlt den Kandidaten mit der besten VALIDIERUNGS-Metrik (Accuracy max, MSE/MAE min). Eingabe sind nur Val-Ergebnisse
    ({"metric", "value"}) lauffaehiger Kandidaten oder None; Testwerte und Zielwert des Papers gehen nie hinein.
    Gleichstand: kleinerer Index. Ohne jeden Val-Wert: erster Kandidat mit Eintrag, Grund "fallback"."""
    scored = [(i, v) for i, v in enumerate(vals) if isinstance(v, dict) and
              isinstance(v.get('value'), (int, float)) and not isinstance(v['value'], bool) and math.isfinite(v['value'])]
    if not scored:
        return 0, "fallback"
    if len({v.get('metric') for _, v in scored}) != 1:
        raise ValueError('Candidate validation metrics differ; they cannot be compared.')
    lower = scored[0][1].get("metric") in LOWER_IS_BETTER_VAL
    best = min(scored, key=lambda iv: (iv[1]["value"] if lower else -iv[1]["value"], iv[0]))
    return best[0], "val"


# Schlusszeilen aus Tags (`</output>`, `<|im_end|>`) oder einzelnen Schlusszeichen (`}`, FITS live 2026-10-14)
_TRAILING_TAGS = re.compile(r"(?:\n[ \t]*(?:<\|?/?\|?[A-Za-z_][\w|]*\|?>|[\]})\"'`]+)[ \t]*)+\s*\Z")


def extract_code(text: str, lenient: bool = False) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    if lenient:   # auch im geschlossenen Block: Schlusszeile vor ``` (Tag oder einzelnes Schlusszeichen) entfernen
        blocks = [b if _is_python(b) else _TRAILING_TAGS.sub("\n", b) for b in blocks]
    if not blocks and lenient:   # nur Produktpfad (graph_repair): andere Sprachmarke, offener Block, Code ohne Block
        blocks = re.findall(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)(?:```|\Z)", text, flags=re.S)
        # Nemotron schliesst den Block manchmal mit einem Tag statt mit ``` (`</|tool_call|>`, `</output>`; SARAF 2026-10-14)
        blocks = [b if _is_python(b) else _TRAILING_TAGS.sub("\n", b) for b in blocks]   # gueltiger Code bleibt unberuehrt
        blocks = [b for b in blocks if _is_python(b)]
        if not blocks and _is_python(text) and re.search(r"^(import|from) ", text, flags=re.M):
            blocks = [text]
    if not blocks:
        raise ValueError("Keine Code-Block in der Antwort des Modells")
    return max(blocks, key=len)


def keep_output_tail(cmd: str) -> str:
    """Agentenlauf mit Ausgabe in Dateien; danach nur das Ende beider Stroeme, Exit-Code unveraendert.

    Das Sandbox-SDK schneidet jeden Strom nach 65535 Bytes ab und behaelt den Anfang. Bei langen Trainingslogs
    gingen so genau Traceback, `WorkerFailure`-Zusammenfassung und faulthandler-Stacks verloren. `tr` ersetzt
    Nicht-ASCII-Bytes, damit ein mitten in einem UTF-8-Zeichen abgeschnittenes Ende die strikte Dekodierung des SDK
    nicht bricht.
    """
    n = AGENT_OUTPUT_TAIL_BYTES
    ascii_only = "LC_ALL=C tr -c '\\11\\12\\15\\40-\\176' '?'"
    # mktemp -d: frisches 0700-Verzeichnis, damit vorher gelaufener Agentencode keinen Symlink unterschieben kann
    return (f'_nd=$(mktemp -d) || exit 125; {cmd} >"$_nd/o" 2>"$_nd/e"; rc=$?; '
            f'tail -c {n} "$_nd/o" | {ascii_only}; tail -c {n} "$_nd/e" | {ascii_only} >&2; rm -rf "$_nd"; exit $rc')


def _is_python(src: str) -> bool:
    import ast
    try:
        ast.parse(src)
        return bool(src.strip())
    except (SyntaxError, ValueError):
        return False


MAX_ERROR_SEARCHES = 4   # Rolle 5: Tavily-Suchen pro Lauf (Kontingent), gleiche Fehlerzeile wird nur einmal gesucht
ERROR_HINT_CHARS = 400   # Auszug je Treffer im Reparatur-Prompt
OWN_RUNTIME_FILES = ("nullius_io.py", "forecast_worker.py", "forecast_runner.py")   # unser Datenhelfer und Vorhersage-Runner
_ERROR_LINE = re.compile(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Warning)\b.*)$", re.M)


def error_query(summary: str) -> str | None:
    """Letzte Ausnahmezeile des Tracebacks als Suchanfrage; Pfade und lange Zahlen entfernt."""
    lines = _ERROR_LINE.findall(summary or "")
    if not lines:
        return None
    q = re.sub(r"(/[\w.-]+)+", "", lines[-1])
    q = re.sub(r"\d{3,}", "N", q)
    return q.strip()[:200] or None


STATIC_LIB_HINT = (
    "\n\nThis library is not installed in the sandbox and is not allowed. Load the data only as described above "
    "and implement every layer in plain PyTorch/NumPy."
)
ONE_CLASS_HINT = "\n\nAll predictions are the same class, so the model did not learn. Check:\n"
OOM_HINT = (
    "\n\nThe process was killed or could not allocate memory: it ran out of memory (the sandbox has a few GB). Do not build dense tensors "
    "of size nodes x nodes x heads (or windows x channels x length x length); use sparse or edge-list operations, or process in chunks, and "
    "free intermediate tensors. Keep the method described in the plan."
)


EXIT_CODE_RE = re.compile(r"\(exit code (-?\d+)\)\s*$")
OOM_EXIT_CODES = {137, -9}


def describe_exit(exit_code: int, text: str) -> str:
    """F3: Fehlertext eines Laufs mit Exit-Code am ENDE (das Ende bleibt im Triage-Fenster); leerer Text nie als leeres Problem."""
    body = (text or "").rstrip()
    if not body.strip():
        body = "The script printed nothing to stderr or stdout before it ended."
    return f"{body}\n(exit code {exit_code})"


def repair_hint(summary: str | None) -> str:
    """Feste Zusatzhinweise (nur Produktpfad) fuer Fehler, an denen die Reparatur in Benchmark v2 viermal scheiterte."""
    from .harness.repair_hints import CHECKLIST_CLASSIFICATION
    text = summary or ""
    m = EXIT_CODE_RE.search(text.strip())
    if "Dataset-Bibliothek" in text:
        return STATIC_LIB_HINT
    if "einzige Vorhersage-Klasse" in text:
        return ONE_CLASS_HINT + CHECKLIST_CLASSIFICATION
    if (m and int(m.group(1)) in OOM_EXIT_CODES) or ("Traceback" not in text and "Killed" in text) or "can't allocate memory" in text or "out of memory" in text.lower():
        return OOM_HINT
    return ""


TIMEOUT_REPAIR_MESSAGE = (
    "The script exceeded the training time limit of {limit} s on CPU. Keep the method, architecture and "
    "hyperparameters, but make it faster: vectorize, no Python loops over nodes, edges or samples, "
    "use sparse or batched tensor operations."
)
STAY_FAST_NOTE = (
    "\n\nAn earlier version of this script already exceeded the training time limit. The corrected script must stay fast: no Python loops "
    "over nodes, edges or samples, vectorised/sparse/batched tensor operations only, in every part of the model."
)
STACK_CHARS = 1200
STDOUT_CHARS = 400


def trim_stack(dump: str) -> str:
    """faulthandler listet die innersten Frames ZUERST: vom Anfang kuerzen (nie vom Ende), Frames der Abdeckungsmessung und des Modul-Starters weglassen."""
    lines = [ln for ln in dump.splitlines() if "/site-packages/coverage/" not in ln and "<frozen runpy>" not in ln]
    return "\n".join(lines)[:STACK_CHARS]


_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+) in (\S+)')


def timeout_report(res: dict, code: str, limit: int) -> str:
    """F2: Repariertext nach einem Timeout mit Messung: wo das Skript beim Zeitlimit stand (faulthandler-Stack des Haupt-Threads, dazu die
    Quellzeile aus dem Agenten-Code) und die letzten Ausgaben des Skripts. Kein LLM."""
    parts = [TIMEOUT_REPAIR_MESSAGE.format(limit=limit)]
    stack = (res.get("stack") or "").strip()
    if stack:
        lines = code.splitlines()
        inner = next(((int(n), fn) for f, n, fn in _FRAME_RE.findall(stack) if f.endswith("train.py")), None)
        if inner and 1 <= inner[0] <= len(lines):
            parts.append(f"When the limit hit, the script was in `{inner[1]}` at train.py line {inner[0]}: `{lines[inner[0] - 1].strip()[:160]}`.")
        parts.append("Python stack at the time limit (most recent call first):\n" + stack[:STACK_CHARS])
    tail = (res.get("stdout_tail") or "").strip()
    parts.append(("Last output of the script before the limit:\n" + tail[-STDOUT_CHARS:]) if tail
                 else "The script printed nothing before the limit (print progress per epoch to see the speed).")
    return "\n\n".join(parts)


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


MODE_PAPER = "nachbau"                    # Nachbau aus Paper: der Agent kennt nur belegte Angaben aus Paper, README und Repo-Zitaten
MAX_REPO_CANDIDATES = 4   # Lesezugriffe (Tavily Extract) auf Repo-Kandidaten je Lauf
MODE_AUTHOR_CODE = "nachbau_autorencode"  # Nachbau aus Paper und Autorencode: roher Repo-Ausschnitt im Prompt, nie ausgefuehrt


AUTHOR_CODE_KWARGS = dict(code_grounded=True, io_helper=True)   # inkl. Fixes F1 bis F4 und F8 (recall cg v1)


def product_kwargs(author_code: bool = True) -> dict:
    """Konfiguration des Produktpfads (API, Katalog-Reproduktion). Standard seit 2026-10-09: Nachbau aus Paper und Autorencode
    (code_grounded + io_helper, Entscheidung der Nutzerin/des Nutzers auf Grundlage von recall cg v1). `author_code=False` ist der
    Schalter zurueck auf den reinen Paper-Nachbau (die Produktkonfiguration von v4).
    Gewollt KEIN Default des Orchestrators: Benchmark- und Experiment-Skripte legen ihre Konfiguration vorab fest und bauen
    `Orchestrator(...)` ohne diese Funktion, damit ihre Laeufe reproduzierbar bleiben."""
    kw: dict = dict(graph_repair=True, setup_v2=True, repo_verify=True, independent_selection=True)
    if author_code:
        kw.update(AUTHOR_CODE_KWARGS)
    return kw


class Orchestrator:
    def __init__(
        self, use_tavily: bool = True, fresh: bool = False, setup_details: bool = True,
        seeds: int | None = None, claim_ids: list[str] | None = None, paper_only: bool = False,
        tavily_mode: str | None = None, graph_repair: bool = False, llm_cache: bool | None = None,
        error_search_live: bool = False, n_candidates: int | None = None, setup_v2: bool = False,
        objection_experiments: bool = False, objection_quick: bool = False,
        spec_mode: bool = False, code_grounded: bool = False, io_helper: bool = False, repo_verify: bool = False,
        allow_test_feedback: bool = False,
        independent_selection: bool = False,
    ) -> None:
        # repo_verify (nur Produktpfad, product_kwargs): Repo-Wahl mit Pruefung. Paper-Link zuerst, dann mehrere Suchtreffer; ein Repo zaehlt nur,
        # wenn sein README Modellname, Titel oder arXiv-ID nennt; inoffizielle Repos werden nie als Autorencode gelesen. Aus: alte Wahl
        # (erster Link, erster Treffer), damit Benchmark- und Experiment-Skripte unveraendert bleiben.
        self.repo_verify = repo_verify
        # graph_repair: Ablation nach Seed 0; meldet sie "Graph nicht genutzt", wird der Code (max. MAX_GRAPH_REPAIRS Mal)
        # repariert. Nur Katalog-Laeufe (curate_catalog.py); Benchmark-Laeufe bleiben unveraendert und vergleichbar.
        self.graph_repair = graph_repair
        # Legacy diagnostic experiments may opt in explicitly. Product/default
        # code repair and candidate acceptance must not depend on test scores.
        self.allow_test_feedback = allow_test_feedback
        self.independent_selection = independent_selection
        # Einwand-Experimente (objection_experiments.py): Funde der Community-Suche werden eingeordnet und, wo messbar, im eigenen
        # Branch geprueft. Nur Bericht. Default aus, damit Benchmark und vorab festgelegte Experimente unveraendert bleiben (API: an).
        self.objection_experiments = objection_experiments
        self.objection_quick = objection_quick   # Schnellmodus: weniger Experimente und Seeds (Live-Demo bleibt kurz)
        self._claim_ctx: dict[str, dict] = {}
        # v5 Schritt A: Pflichtliste, Quellenrang und Geltungsbereich in Rolle 4 (setup_details.py); Default aus, damit v2 bis v4 reproduzierbar bleiben
        # Steckbrief (docs/recall_steckbrief_plan.md): S1 Steckbrief statt Zitatliste, S2 CONFIG erzwingen, S3 Einzelpruefung,
        # S4 Repo-Auswahl je Einheit. Braucht Quellenrang und Geltungsbereich, setzt daher setup_v2. Default aus (v4 reproduzierbar).
        # code_grounded (docs/recall_analysis_3.md Hebel 3): der Repo-Ausschnitt (S4) geht ROH in Plan-, Code- und Reparatur-Prompt,
        # Zahlen erzwingt S2 (nur Zahlenschluessel). Modus: Nachbau aus Paper UND Autorencode, unabhaengig gemessen. Default aus.
        self.code_grounded = code_grounded
        self._base_modes = {"code_grounded": code_grounded, "io_helper": io_helper, "spec_mode": spec_mode}   # wie angefordert; ein Lauf kann zurueckfallen
        # io_helper (docs/recall_analysis_3.md Hebel 4): unser Ausgabevertrag laeuft ueber harness/nullius_io.py. Der Agent liefert Arrays,
        # der Helfer schreibt predictions.csv/val_predictions.csv und die SMOKE-Teilmengen. Nur Code-/Reparatur-Prompt (nicht der Plan). Default aus.
        self.io_helper = io_helper
        self.train_timeout_s = TRAIN_TIMEOUT_S   # je Seed; Gate-Laeufe setzen es aus der Laufzeit der Referenzloesung (timeouts.train_timeout)
        self.spec_mode = spec_mode or code_grounded
        self.spec: Spec | None = None
        self.setup_v2 = setup_v2 or self.spec_mode
        self.mode = MODE_AUTHOR_CODE if code_grounded else MODE_PAPER   # angeforderter Modus; im Lauf nach dem Repo-Zugriff festgestellt (_resolve_mode)
        self.n_candidates = n_candidates if n_candidates is not None else CANDIDATES   # wirkt nur mit graph_repair (A1)
        # "live" (Standard): Paper-Text und README werden bei jedem Lauf per Tavily geholt. "fixture": gespeicherte Fassungen
        # (Tests, vorab festgelegte Experimente: alle Generierungen sehen denselben Text). Env TAVILY_MODE oder --offline.
        self.tavily_mode = tavily_mode or os.getenv("TAVILY_MODE", "live")
        self.error_search_live = error_search_live   # Benchmark v4: Rolle 5 live, Rollen 2/4 aus Fixtures
        if self.tavily_mode not in ("live", "fixture"):
            raise ValueError(f"TAVILY_MODE muss live oder fixture sein, nicht {self.tavily_mode!r}")
        self._tavily_calls = 0
        self._sources: list[dict] = []
        self.use_tavily = use_tavily
        self.use_setup_details = use_tavily and setup_details   # Rolle 4 haengt am Paper-Text
        self.paper_only = paper_only   # Rolle 4 liest nur den Paper-Text, kein Repo/README (Shchur-Experiment)
        self.seeds = seeds
        self.claim_ids = claim_ids   # nur diese Claims des Papers laufen lassen (None = alle)
        self.details: list[SetupDetail] = []
        self.paper_text: PaperText | None = None
        self.last_run_dir: Path | None = None
        self._run_dir: Path | None = None
        self._cost_base = 0.0
        self.events = EventLog(None)       # ab run(): runs/<paper>/<ts>/events.jsonl
        self._smoke_ok: set[tuple[str, str, int]] = set()   # Code, Task und Seed der bestandenen Formpruefung
        self._snapshots: dict[str, str] = {}   # Basis-Snapshot pro Task im Prozess wiederverwenden
        self.settings = load_settings()
        if llm_cache is not None:   # ueberschreibt LLM_CACHE aus .env (die API setzt False: Live-Laeufe rufen das Modell wirklich auf)
            self.settings = dataclasses.replace(self.settings, llm_cache=llm_cache)
        if fresh:  # --fresh: LLM-Cache weder lesen noch schreiben (Stabilitaetsbeleg, Benchmark)
            self.settings = dataclasses.replace(self.settings, llm_cache=False)
        self.llm = LLMClient(self.settings)
        self.llm.on_call = self._llm_completed
        self.sandbox = get_sandbox(self.settings.sandbox_backend)
        self._lock = threading.Lock()  # LLM-Aufrufe/Kostenzaehler/Dossier aus Seed-Threads

    @property
    def spent_usd(self) -> float:
        return self.llm.total_cost_usd - self._cost_base   # Kosten seit Start des aktuellen Laufs

    def close(self) -> None:
        try:
            if hasattr(self.sandbox, 'close'):
                self.sandbox.close()
        finally:
            if hasattr(self.llm, 'close'):
                self.llm.close()

    def _infrastructure_failure(self, dossier: Dossier, ts: str, run_dir: Path, exc: SandboxSetupError) -> None:
        """Sandbox-Setup gescheitert (z. B. Daten-Download): Warnung und Ereignis, Dossier und Zeugnis ohne Urteile speichern. Kein "nicht reproduzierbar (Setup)"."""
        self._warn(f"Sandbox-Setup fehlgeschlagen (Infrastruktur, kein Urteil ueber das Paper): {str(exc).splitlines()[0]}")
        dossier.warnings = list(self._warnings)
        self._emit("infrastructure_error", stage="sandbox_setup", message=str(exc)[-600:])
        dossier.save(ts)
        (run_dir / "zeugnis.md").write_text(render_markdown(dossier))
        self._emit("run_end", cost_net_usd=round(self.spent_usd, 4), cost_gross_usd=round(self.spent_gross_usd, 4), error="sandbox_setup")

    def _set_author_code(self, active: bool) -> None:
        """Schaltet Autorencode-Modus (code_grounded + io_helper) fuer diesen Lauf an oder auf den angeforderten Stand zurueck.
        Aus = exakt die Prompts des reinen Paper-Nachbaus (Steckbrief nur, wenn spec_mode angefordert war)."""
        base = getattr(self, "_base_modes", None)
        if base is None:
            return
        self.code_grounded = base["code_grounded"] and active
        self.io_helper = base["io_helper"] and active
        self.spec_mode = base["spec_mode"] or self.code_grounded

    def _resolve_mode(self, repo_info: dict | None) -> tuple[str, str]:
        """Modus dieses Laufs: "nachbau_autorencode" nur, wenn ein Repo-Ausschnitt wirklich im Steckbrief steht; sonst Rueckfall auf
        "nachbau" (nie falsch etikettiert). Rueckgabe (Modus, Grund des Rueckfalls oder "")."""
        if not (getattr(self, "_base_modes", None) or {}).get("code_grounded"):
            return MODE_PAPER, ""
        spec = getattr(self, "spec", None)
        if not (getattr(self, "use_setup_details", False) and self.paper_text is not None):
            reason = "setup details switched off or paper text not read"
        elif repo_info is None:
            reason = getattr(self, "_repo_fallback_reason", None) or "no repository found or clone failed"
        elif spec is None or not spec.excerpt or not spec.model_code_present:
            reason = "no model code found in the repository"
        else:
            return MODE_AUTHOR_CODE, ""
        self._set_author_code(False)
        self.spec = None
        return MODE_PAPER, reason

    def _warn(self, text: str) -> None:
        if not hasattr(self, "_warnings"):
            self._warnings = []
        self._warnings.append(text)
        self._emit("warning", text=text)

    def _emit(self, type: str, **data) -> None:
        ev = getattr(self, "events", None)
        record = ev.write(type, **data) if ev is not None else None
        cb = getattr(self, "on_event", None)   # nur die API setzt es (Live-Stream); Benchmark/CLI unveraendert
        if cb is not None:
            cb(record if record is not None else {"ts": time.time(), "type": type, **data})

    def _announce_code(self, claim: Claim, code: str) -> None:
        """Code-Version einmal an den Live-Hook schicken (nie in events.jsonl; im Export liegt jede Version als code_<hash>.py)."""
        if getattr(self, "on_event", None) is None:
            return
        h = hashlib.sha256(code.encode()).hexdigest()[:12]
        sent = self.__dict__.setdefault("_code_sent", set())
        if h not in sent:
            sent.add(h)
            self._emit_live("code_text", claim_id=claim.id, code_hash=h, code=code)

    def _emit_live(self, type: str, **data) -> None:
        """Nur an den Live-Hook, nie in events.jsonl (Kosten laufend, kein Beleg des Laufs)."""
        cb = getattr(self, "on_event", None)
        if cb is not None:
            cb({"ts": time.time(), "type": type, **data})

    def _emit_tavily(self, role: int, endpoint: str, target, n_results: int) -> None:
        self._tavily_calls += 1
        self._emit("tavily_call", role=role, endpoint=endpoint, target=target, n_results=n_results)

    def _record_source(self, kind: str, url: str, text: str, live: bool, retrieved_at: str | None = None,
                       sha256: str | None = None, from_cache: bool = False) -> None:
        """Web-Quelle im Dossier und im Ereignisprotokoll festhalten: live oder aus Fixture, Zeitstempel, Hash."""
        rec = {"kind": kind, "url": url, "mode": "live" if live else "fixture", "n_chars": len(text),
               "retrieved_at": retrieved_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "sha256": sha256 or hashlib.sha256(text.encode()).hexdigest()}
        if from_cache:
            rec["from_cache"] = True
        self._sources.append(rec)
        self._emit("source", **rec)

    @property
    def spent_gross_usd(self) -> float:
        return self.llm.gross(self.spent_usd)

    # ---- LLM ----------------------------------------------------------------
    def _llm_completed(self, role, resp) -> None:
        self._emit_live("llm_call", role=str(role), model=getattr(resp, "model", None), cached=bool(getattr(resp, "cached", False)),
                        cost_net_usd=round(resp.cost_usd, 5), total_net_usd=round(self.spent_usd, 5))

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
        if getattr(self.llm, "on_call", None) is None:
            self._llm_completed(role, resp)  # compatibility with injected test/experiment clients
        return resp

    def plan(self, paper: dict, claim: Claim, task: Task) -> str:
        user = (
            f"Paper: {paper['title']}\nClaim: {claim.text} (target {claim.target}, {claim.metric})\n\n"
            f"{task.data_description}\n\n{self._details_prompt()}\n\n"
            f"{self._conflicts_prompt()}"
            "Write a short reproduction plan: model architecture, all hyperparameters and training "
            "protocol as described in the paper. Mark every value you are NOT sure about with "
            "'(uncertain)'. Max 15 lines."
        )
        text = self._ask("reasoning", PLAN_SYSTEM, user).content
        if getattr(self, "graph_repair", False) and plan_needs_retry(text):   # Katalog/API: Plan darf kein Code sein
            self._emit("plan_retry", reason="code" if text.lstrip().startswith("```") or "import " in text else "too_long")
            text = self._ask("reasoning", PLAN_SYSTEM, user + PLAN_RETRY_NOTE, temperature=0.4).content
        return text

    def _active_spec(self) -> Spec | None:
        return getattr(self, "spec", None) if getattr(self, "spec_mode", False) else None

    def _details_prompt(self) -> str:
        """Setup-Angaben im Prompt; im Produktpfad sind Repo-Defaults (argparse) gekennzeichnet (M7).
        Steckbrief-Modus: der Steckbrief ersetzt die Zitatliste (die volle Liste bleibt im Dossier)."""
        spec = self._active_spec()
        if spec is not None and getattr(self, "code_grounded", False) and (spec.items or spec.excerpt):
            return format_code_grounded(spec, self.details)
        if spec is not None and spec.items:
            return format_spec(spec)
        if getattr(self, "setup_v2", False):
            return format_for_prompt_v2(self.details)
        return format_for_prompt(self.details, mark_defaults=getattr(self, "graph_repair", False))

    def _conflicts_prompt(self) -> str:
        """Nur Produktpfad: Paper-Wert und Repo-Wert fuer dasselbe Thema im Plan-Prompt, beide mit Quelle (M7).
        Steckbrief-Modus: Widersprueche sind schon per Rangregel geloest, der Plan sieht sie nicht."""
        spec = self._active_spec()
        if spec is not None and (spec.items or (getattr(self, "code_grounded", False) and spec.excerpt)):
            return ""
        if getattr(self, "setup_v2", False):
            text = format_conflicts_v2(self.details)
        else:
            text = format_conflicts(self.details) if getattr(self, "graph_repair", False) else ""
        return text + "\n\n" if text else ""

    def _fidelity_note(self) -> str:
        """Nur Katalog/API (Flag): Kernmechanismus des Plans umsetzen, Kommentarblock Plan step -> function. Anleitung, kein Pruefer."""
        return FIDELITY_NOTE if getattr(self, "graph_repair", False) else ""

    def _code_from_llm(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Code-Block aus der Antwort; bis zu 3 Versuche. Fehlt der Block oder ist die Antwort abgeschnitten
        (finish_reason "length": bei langem Code zehrt das Denken des Modells das Token-Limit auf), folgen
        hoehere Temperatur (gleicher Prompt + gleiche Temperatur gaebe aus dem Cache dieselbe Antwort),
        hoeheres max_tokens und der Hinweis, nur Code zurueckzugeben."""
        last: ValueError | None = None
        broken: str | None = None
        for i in range(3):
            first_max = CODE_RETRY_MAX_TOKENS if self._active_spec() is not None else None   # Steckbrief: Limit auch beim ersten Aufruf
            # Temperatur gedeckelt; die Versuchsnummer im Hinweis haelt den Cache-Schluessel auch an der Obergrenze verschieden
            note = CODE_ONLY_NOTE + (f" (Retry {i}.)" if i > 1 else "") if i else ""
            resp = self._ask(
                "code", system, user + note,
                max_tokens=CODE_RETRY_MAX_TOKENS if i else first_max,
                temperature=min(temperature + CODE_RETRY_TEMP_STEP * i, max(temperature, CODE_MAX_TEMPERATURE)),
            )
            if getattr(resp, "finish_reason", None) == "length":
                last = ValueError("Antwort abgeschnitten (finish_reason length)")
                continue
            lenient = getattr(self, "graph_repair", False)
            try:
                code = extract_code(resp.content, lenient=lenient)
            except ValueError as exc:
                last = exc
                self._emit("code_missing", try_n=i + 1, content_tail=(resp.content or "")[-1500:])   # Diagnose
                continue
            if lenient and not _is_python(code):
                # Produktpfad: Syntaxfehler kostet sonst Sandbox-Lauf, Tavily-Fehlersuche und eine Reparatur (FITS live 2026-10-14);
                # neu schreiben ist billiger. Bleibt jeder Versuch fehlerhaft, geht der erste wie bisher in die Reparatur.
                broken = broken or code
                last = ValueError("Code mit Syntaxfehler")
                self._emit("code_missing", try_n=i + 1, reason="syntax", content_tail=code[-1500:])
                continue
            return code
        if broken is not None:
            return broken
        raise last  # type: ignore[misc]

    def _product_notes(self, task: Task) -> str:
        """Nur Produktpfad: verbotene Bibliotheken schon beim ersten Schreiben (M3), Validierungsvorhersagen fuer die Kandidatenauswahl (A1)."""
        if not getattr(self, "graph_repair", False):
            return ""
        val_notes = getattr(task, "val_notes", "")
        return (PRODUCT_LIB_NOTE + ' If retaining the best epoch, copy its parameter tensors independently (copy.deepcopy or tensor clones). '
                'Before merging nonadjacent tensor axes for a projection, explicitly permute them into the required order; preserve the task output axes.'
                + (f" {val_notes}" if val_notes else ""))

    def _spec_note(self) -> str:
        spec = self._active_spec()
        return config_note(spec) if spec is not None else ""

    def _enforce_spec(self, code: str) -> str:
        """S2 (deterministisch): abweichende oder fehlende CONFIG-Werte per Override-Zeile auf den Steckbrief setzen."""
        spec = self._active_spec()
        if spec is None or not spec.enforced():
            return code
        new, report = spec_check.config_check(code, spec)
        if report["overrides"]:
            self._emit("spec_override", code_hash=hashlib.sha256(code.encode()).hexdigest()[:12],
                       to_hash=hashlib.sha256(new.encode()).hexdigest()[:12], overrides=report["overrides"])
        return new

    def _prompt_task(self, task: Task) -> Task:
        """Task fuer Code- und Reparatur-Prompt: mit `io_helper` der Ausgabevertrag ueber den Helfer, sonst unveraendert (Prompts byte-gleich)."""
        if getattr(self, 'independent_selection', False):
            task = task.with_independent_selection()
        return task.with_io_helper() if getattr(self, "io_helper", False) else task

    def write_code(self, plan: str, task: Task, temperature: float = 0.2) -> str:
        task = self._prompt_task(task)
        return self._enforce_spec(self._code_from_llm(
            CODE_SYSTEM,
            f"{task.data_description}\n\n{self._details_prompt()}\n\n"
            f"Reproduction plan:\n{plan}\n\n"
            "The script runs WITHOUT network access and must not download anything, spawn "
            f"processes or use eval/exec. {task.code_notes} Write train.py implementing this plan."
            f"{self._product_notes(task)}{self._fidelity_note()}{self._spec_note()}",
            **({"temperature": temperature} if temperature != 0.2 else {}),
        ))

    def repair_code(self, plan: str, task: Task, code: str, error_summary: str) -> str:
        task = self._prompt_task(task)
        return self._enforce_spec(self._code_from_llm(
            CODE_SYSTEM,
            f"{task.data_description}\n\n{self._details_prompt()}\n\n"
            f"Reproduction plan:\n{plan}\n\nThis script failed:\n"
            f"```python\n{code}\n```\nProblem: {error_summary}\n\nReturn the corrected full script."
            f"{self._product_notes(task)}{self._fidelity_note()}{self._spec_note()}",
        ))

    def _error_hints(self, claim: Claim, summary: str | None) -> str:
        """Rolle 5 (nur Produktpfad, Tavily live): Websuche mit der Fehlerzeile, Treffer als Hinweis an `code`.
        Kein LLM, keine Teststatistik (nur der Fehlertext des Agenten-Codes geht hinaus)."""
        live = getattr(self, "tavily_mode", None) == "live" or getattr(self, "error_search_live", False)
        if not (getattr(self, "use_tavily", False) and live):
            return ""
        if any(f"{name} line" in (summary or "") for name in OWN_RUNTIME_FILES):
            return ""   # Fehler aus unserem Helfer/Runner: der Vertrag im Prompt erklaert ihn, Websuche lieferte fremde Repos (FITS 35758c32)
        q = error_query(summary or "")
        with self._lock:
            done = self.__dict__.setdefault("_error_searches", {})
            if q is None or (q not in done and len(done) >= MAX_ERROR_SEARCHES):
                return ""
            cached = q in done
            if not cached:
                done[q] = None   # reservieren, parallele Seeds suchen nicht doppelt
        if cached:
            results = done[q] or []
        else:
            try:
                res = self._tavily_getter()().resolve_error(q)
                results = [r for r in (res or {}).get("results", []) if r.get("content")][:3]
            except Exception as exc:   # Websuche ist Hilfe, kein Muss
                self._warn(f"Rolle 5 (Fehlersuche) fehlgeschlagen: {exc}")
                results = []
            done[q] = results
            self._emit("error_search", claim_id=claim.id, query=q, urls=[r.get("url") for r in results])
        if not results:
            return ""
        parts = [f"- {r.get('title', '')} ({r.get('url', '')}): {r['content'][:ERROR_HINT_CHARS]}" for r in results]
        return ("\n\nWeb search results for this error (Tavily; may be irrelevant, use only what fits):\n"
                + "\n".join(parts))

    def triage(self, text: str) -> str:
        """Fehlertext fuer die Reparatur: Ende von stderr/stdout (dort steht der Traceback).

        Eine LLM-Zusammenfassung durch `fast` taugt nicht: Die Free- und TF-Modelle schreiben
        ihren Denkprozess in die Antwort, abgeschnitten bei max_tokens kam kein Fehler an.
        """
        return text.strip()[-TRIAGE_CHARS:]

    # ---- Sandbox ------------------------------------------------------------
    def _prediction_file(self, branch: Branch, task: Task, val: bool = False) -> bytes:
        path = '/output/val_predictions.csv' if val else '/output/predictions.csv'
        cfg = json.loads(task.io_config) if task.io_config else {}
        limit = 2_000_000
        if cfg.get('kind') == 'windows':
            windows = cfg['val_windows'] if val else cfg['test_windows']
            limit = 64 * windows * cfg['steps'] * cfg['channels'] + 32
        getter = getattr(self.sandbox, 'get_file_limited', None)
        data = getter(branch, path, limit) if getter else self.sandbox.get_file(branch, path)
        if len(data) > limit:
            raise ValueError('Prediction artifact exceeds its size limit')
        return data
    def _prepare_selection_contract(self, branch: Branch, task: Task) -> str | None:
        if not task.selection_contract:
            return None
        import numpy as np
        from .harness.selection_contract import graph_public
        sb = self.sandbox
        cfg = json.loads(task.io_config)
        try:
            if cfg['kind'] == 'nodes':
                original = sb.get_file(branch, cfg['data'])
                with np.load(io.BytesIO(original), allow_pickle=False) as raw:
                    arrays = graph_public({k: raw[k] for k in raw.files})
                buf = io.BytesIO()
                np.savez_compressed(buf, **arrays)
                sb.put_file(branch, cfg['data'], buf.getvalue())
            else:
                private = '/opt/forecast-runtime/source.npz'
                sb.put_files(branch, {f'/opt/forecast-runtime/{name}': (HARNESS_DIR / name).read_bytes()
                                      for name in ('forecast_runner.py', 'forecast_worker.py', 'selection_contract.py')})
                sb.put_file(branch, '/opt/forecast-runtime/config.json', json.dumps(dict(cfg, private_data=private)).encode())
                sb.put_file(branch, f'{WORKDIR}/nullius_io.py', task.io_helper_source().encode())
                setup = sb.run(branch, f'mv {shlex.quote(cfg["data"])} {private} && '
                               'chmod 755 /opt/forecast-runtime /work && '
                               'chmod 444 /opt/forecast-runtime/*.py /work/*.py && '
                               'chmod 400 /opt/forecast-runtime/source.npz /opt/forecast-runtime/config.json && '
                               'setpriv --reuid=65534 --regid=65534 --clear-groups --bounding-set=-all '
                               '--inh-caps=-all --ambient-caps=-all --no-new-privs '
                               'python -I -c "import os; assert os.getuid() == 65534; '
                               'assert not os.access(\'/opt/forecast-runtime/source.npz\', os.R_OK)"', 30)
                if setup.exit_code != 0 or setup.timed_out:
                    return 'Forecast isolation requires setpriv and a runtime able to drop privileges.'
        except Exception as exc:
            return f'Independent selection data could not be prepared: {type(exc).__name__}'
        return None

    def _agent_runner(self, task: Task) -> str:
        if task.selection_contract and json.loads(task.io_config)['kind'] == 'windows':
            return 'python -I /opt/forecast-runtime/forecast_runner.py /opt/forecast-runtime/config.json'
        return 'python train.py'

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
        code_hash = (hashlib.sha256(code.encode()).hexdigest(), repr(task), seed)
        if code_hash in self._smoke_ok:
            return None
        sb = self.sandbox
        expected = task.smoke_ids
        if task.smoke_ids_cmd:
            expected = tuple(sb.run(branch, task.smoke_ids_cmd, 60).stdout.split())
        r = getattr(sb, "run_agent", sb.run)(
            branch,
            keep_output_tail(f"mkdir -p /output && cd {WORKDIR} && {task.env_for(seed)} SMOKE=1 unshare -n {self._agent_runner(task)}"),
            SMOKE_TIMEOUT_S,
        )
        if r.timed_out:
            return (f"Smoke run (SMOKE=1) exceeded {SMOKE_TIMEOUT_S}s: the script ignores SMOKE=1. "
                    "With SMOKE=1 use the tiny subset and 1 epoch as described.")
        if r.exit_code != 0:
            return "Smoke run (SMOKE=1) crashed:\n" + describe_exit(r.exit_code, (r.stderr or r.stdout).strip()[-TRIAGE_CHARS:])
        try:
            out = self._prediction_file(branch, task)
        except Exception as exc:   # Sandbox-Adapter melden fehlende Datei unterschiedlich
            return f"Smoke run wrote no /output/predictions.csv ({exc})"
        error = check_smoke_output(out, expected, task.smoke_dims, task.smoke_dim_names)
        if error:
            return "Smoke run (SMOKE=1) output has the wrong form: " + error
        if getattr(self, "graph_repair", False) and getattr(task, "val_eval_cmd", ""):   # Produktpfad: auch val_predictions.csv pruefen
            val_expected = task.val_smoke_ids
            if task.val_smoke_ids_cmd:
                val_expected = tuple(sb.run(branch, task.val_smoke_ids_cmd, 60).stdout.split())
            try:
                val_out = self._prediction_file(branch, task, val=True)
            except Exception as exc:
                return f"Smoke run wrote no /output/val_predictions.csv ({exc}); the task requires it"
            error = check_smoke_output(val_out, val_expected, task.val_smoke_dims, task.smoke_dim_names)
            if error:
                return "Smoke run (SMOKE=1) val_predictions.csv has the wrong form: " + error
            sb.run(branch, "rm -f /output/val_predictions.csv", 30)
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
                return branch, self._run_and_eval(branch, code, seed, task, metric, eval_snapshot=snapshot, **kw)
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

    def edge_direction_check(self, snapshot: str, code: str, seed: int, task: Task, metric: str) -> dict:
        """B2 (v5): Lauf des Codes auf dem synthetischen Matching-Graphen (harness/edge_direction.py), Test-Accuracy je Seite.
        Deterministisch, kein LLM. Verdict "Kante nur in einer Richtung" = die Seiten unterscheiden sich um >= 0,30."""
        _, res = self._attempt(snapshot, code, seed, task, metric, edge_probe=True)
        acc_a, acc_b = (res.get("acc_a"), res.get("acc_b")) if res["status"] == "ok" else (None, None)
        return {"verdict": classify_edge_direction(acc_a, acc_b), "acc_side_a": acc_a, "acc_side_b": acc_b,
                "detail": None if acc_a is not None else str(res.get("detail", ""))[-300:]}

    def val_label_check(self, snapshot: str, code: str, seed: int, task: Task, metric: str, val: dict) -> dict:
        """Compare a masked-label control with independent validation and inspect static use.

        Product admission rejects detected validation training or an unmeasurable
        control. Early stopping alone remains allowed. Historical test-feedback
        experiments retain their original combined score-drop/static criterion.
        No measurements from this control enter a model prompt.
        """
        static = static_val_use(code)
        _, res = self._attempt(snapshot, code, seed, task, metric, mask_val=True, with_val=True)
        masked = res.get("val") if res["status"] == "ok" else None
        masked_value = masked.get("value") if isinstance(masked, dict) else None
        verdict = classify_val_mask(val.get("value"), masked_value)
        return {
            "verdict": verdict, "val_full": val.get("value"), "val_masked": masked_value,
            "static": static["verdict"], "static_hits": static["hits"][:3],
            "excluded": ((masked_value is None or static["verdict"] == "training")
                         if not getattr(self, 'allow_test_feedback', False)
                         else verdict == "Validierungslabels genutzt" and static["verdict"] == "training"),
            "detail": None if masked_value is not None else str(res.get("detail", ""))[-300:],
        }

    def _selection_check(self, snapshot: str, code: str, seed: int, task: Task, metric: str, val: dict | None) -> dict:
        """Same admission checks for original candidates and every replacement hash."""
        key = (snapshot, hashlib.sha256(code.encode()).hexdigest(), seed, repr(task))
        cache = self.__dict__.setdefault('_selection_checks', {})
        if key in cache:
            return cache[key]
        if task.mask_val_cmd and isinstance(val, dict) and 'value' in val:
            result = self.val_label_check(snapshot, code, seed, task, metric, val)
        else:
            forecast = bool(task.io_config and json.loads(task.io_config).get('kind') == 'windows')
            static = static_val_use(code, forecasting=forecast)
            result = {'static': static['verdict'], 'static_hits': static['hits'][:3],
                      'excluded': static['verdict'] in {'training', 'unklar'},
                      'verdict': 'Validation training detected' if static['verdict'] == 'training' else 'Static validation review only',
                      'scope': 'Known validation-use patterns; not a general independence certificate'}
        cache[key] = result
        return result

    def _admit_final_code(self, claim: Claim, task: Task, snapshot: str, dossier: Dossier, code: str, seed: int) -> dict:
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
        if getattr(self, 'independent_selection', False):
            score = self._current_spec_score(code)
            if score is not None and score[0] != score[1]:
                raise MethodImplementationBlocked('Selected code did not satisfy the mandatory experiment settings. No paper verdict was produced.')
        val = self.__dict__.get('_val', {}).get((code_hash, seed))
        check = self._selection_check(snapshot, code, seed, task, claim.metric, val)
        self._emit('val_label_check', claim_id=claim.id, code_hash=code_hash, stage='final_admission', **check)
        dossier.val_label_check.setdefault(claim.id, {})[code_hash] = {'code_hash': code_hash, **check}
        if check.get('excluded'):
            raise MethodImplementationBlocked('Selected code failed the independent validation controls. No paper verdict was produced.')
        return check

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

    def _train_timeout(self) -> int:
        return int(getattr(self, "train_timeout_s", TRAIN_TIMEOUT_S))

    def _run_and_eval(
        self, branch: Branch, code: str, seed: int, task: Task, metric: str, measure_coverage: bool = False,
        ablate: bool = False, with_val: bool = False, mask_val: bool = False, edge_probe: bool = False,
        eval_snapshot: str | None = None,
    ) -> dict:
        """Ein Versuch im Branch. Rueckgabe: {"status": timeout|fehler|ungueltig|ok, ...}."""
        sb = self.sandbox
        violations = find_violations(code)
        from .harness.training_check import checkpoint_issues
        violations += checkpoint_issues(code)
        spec = self._active_spec()
        if spec is not None and not getattr(self, 'allow_test_feedback', False):
            _, config = spec_check.config_check(code, spec)
            if config['overrides'] or config['conflicts'] or (getattr(self, 'independent_selection', False) and config['unused']):
                keys = sorted({item['key'] for item in config['overrides'] + config['conflicts']})
                keys = sorted(set(keys) | (set(config['unused']) if getattr(self, 'independent_selection', False) else set()))
                violations.append('Configuration contradicts the reproduction spec: ' + ', '.join(keys) +
                                  '. Restore the reference values before training; later module-level writes must not undo them.')
        if violations:
            return {"status": "ungueltig", "detail": "Statische Pruefung: " + "; ".join(violations),
                    "static": True}
        sb.put_file(branch, f"{WORKDIR}/train.py", code.encode())
        if getattr(self, "io_helper", False) and task.io_config:
            sb.put_file(branch, f"{WORKDIR}/nullius_io.py", task.io_helper_source().encode())
        if edge_probe:   # B2: synthetischer Graph statt Datensatz, eigene Auswertung (kein Harness, kein Smoke, keine Abdeckung)
            path, n_nodes, n_val, n_test = task.edge_probe
            npz, truth = make_probe(n_nodes, n_val, n_test, seed=0, n_classes=task.probe_dims[0], n_features=task.probe_dims[1])
            sb.put_file(branch, path, probe_bytes(npz))
            r = getattr(sb, "run_agent", sb.run)(branch, f"mkdir -p /output && cd {WORKDIR} && {task.env_for(seed)} unshare -n python train.py", self._train_timeout())
            if r.timed_out or r.exit_code != 0:
                return {"status": "fehler", "detail": "timeout" if r.timed_out else (r.stderr or r.stdout)}
            try:
                acc_a, acc_b = score_probe(sb.get_file(branch, "/output/predictions.csv"), truth)
            except FileNotFoundError:
                acc_a = acc_b = None
            return {"status": "ok", "acc_a": acc_a, "acc_b": acc_b}
        if task.prepare_cmd:
            prep_error = self._prepare_data(branch, seed, task)
            if prep_error:
                return {"status": "ungueltig", "detail": prep_error}
        prepared_identity = None
        if task.io_config:
            try:
                prepared_identity = json.loads(sb.get_file(branch, json.loads(task.io_config)['data'] + '.source.json'))
                if not isinstance(prepared_identity, dict):
                    prepared_identity = None
            except Exception:
                pass
        selection_error = self._prepare_selection_contract(branch, task)
        if selection_error:
            return {'status': 'ungueltig', 'infrastructure': True, 'detail': selection_error}
        if ablate:   # Nutzungspruefung: veraenderte Daten, kein Smoke-Lauf, keine Abdeckung
            control = sb.run(branch, task.ablation_cmd, 120)
            if control.timed_out or control.exit_code != 0:
                return {"status": "ungueltig", "detail": "Graph control preparation failed: " + (control.stderr or control.stdout)[-300:]}
            measure_coverage = False
        elif mask_val:   # B1: Labels der Validierungsknoten maskiert, kein Smoke-Lauf (lief schon), keine Abdeckung
            control = sb.run(branch, task.mask_val_cmd, 120)
            if control.timed_out or control.exit_code != 0:
                return {"status": "ungueltig", "detail": "Validation control preparation failed: " + (control.stderr or control.stdout)[-300:]}
            measure_coverage = False
        else:
            smoke_error = self._smoke_check(branch, code, task, seed)
            if smoke_error:
                return {"status": "fehler", "detail": smoke_error}
        # `unshare -n`: eigener Netz-Namespace ohne Interfaces -> kein Internet waehrend des
        # Agenten-Codes (Setup ist vorher im Basis-Snapshot abgeschlossen).
        # Abdeckung nur im ersten vollstaendigen Seed-Lauf (Smoke-Lauf ist zu kurz: Epochen-/Early-Stopping-Zweige)
        with_cov = (measure_coverage and self._agent_runner(task) == 'python train.py'
                    and sb.run(branch, "python -c 'import coverage'", 30).exit_code == 0)
        runner = "python -m coverage run --data-file=/tmp/.cov train.py" if with_cov else self._agent_runner(task)
        # F2: `timeout -s ABRT` beendet das Skript kurz VOR dem harten Sandbox-Timeout; `faulthandler` schreibt dabei alle Python-Stacks nach stderr
        # (Stelle des Zeitlimits), stdout (unbuffered) bleibt lesbar. Der harte Timeout toetet auf Nebius den ganzen Branch (Dateien danach weg).
        limit = self._train_timeout()
        soft = max(1, limit - ABRT_GRACE_S)
        r = getattr(sb, "run_agent", sb.run)(
            branch,
            keep_output_tail(f"mkdir -p /output && cd {WORKDIR} && {task.env_for(seed)} PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 "
                             f"unshare -n timeout -s ABRT {soft} {runner}"),
            limit,
        )
        if r.timed_out or r.exit_code == SOFT_TIMEOUT_EXIT:
            out = {"status": "timeout", "detail": f"Training > {limit}s", "exit_code": r.exit_code, "soft_timeout": not r.timed_out}
            if not r.timed_out:
                err = r.stderr or ""
                i = err.find("Fatal Python error")
                out["stack"] = trim_stack(err[i:] if i >= 0 else err)
                out["stdout_tail"] = (r.stdout or "")[-STDOUT_CHARS:]
            return out
        if r.exit_code != 0:
            return {"status": "fehler", "detail": describe_exit(r.exit_code, r.stderr or r.stdout), "exit_code": r.exit_code}
        coverage = self._measure_coverage(branch) if with_cov else None

        # Only prediction bytes cross into a fresh branch of the trusted setup snapshot.
        # A random harness path in the agent's branch cannot protect Python startup/imports.
        if eval_snapshot is None:
            return {"status": "ungueltig", "detail": "Independent evaluation requires the trusted setup snapshot."}
        eval_branch = sb.branch(eval_snapshot)
        prep_error = self._prepare_data(eval_branch, seed, task) if task.prepare_cmd else None
        if prep_error:
            return {"status": "ungueltig", "detail": prep_error}
        sb.run(eval_branch, "mkdir -p /output", 30)
        identity_env = ''
        if task.io_config:
            manifest_path = json.loads(task.io_config)['data'] + '.source.json'
            try:
                # Read only from the pristine evaluation branch, never agent state.
                manifest = json.loads(sb.get_file(eval_branch, manifest_path))
                if not isinstance(manifest, dict) or not manifest or any(not isinstance(k, str) or not isinstance(v, str)
                                                                       or not re.fullmatch('[0-9a-f]{64}', v) for k, v in manifest.items()):
                    raise ValueError('Invalid dataset manifest')
                if prepared_identity != manifest:
                    return {'status': 'ungueltig', 'infrastructure': True, 'detail': 'Raw dataset identity differs between preparation and evaluation.'}
                identity_env = 'NULLIUS_DATA_IDENTITY=' + shlex.quote(json.dumps(manifest, sort_keys=True)) + ' '
                self._emit('data_identity', seed=seed, files=manifest)
            except Exception:
                if task.selection_contract:
                    return {'status': 'ungueltig', 'infrastructure': True, 'detail': 'Trusted raw dataset identity unavailable; evaluation stopped.'}
        try:
            predictions = self._prediction_file(branch, task)
            sb.put_file(eval_branch, "/output/predictions.csv", predictions)
        except Exception as exc:
            return {"status": "ungueltig", "detail": f"Prediction transfer failed: {type(exc).__name__}"}
        if with_val:
            try:
                sb.put_file(eval_branch, "/output/val_predictions.csv", self._prediction_file(branch, task, val=True))
            except Exception:
                pass  # The validation harness reports the missing file; no selection fallback.
        harness_dir = f"{SANDBOX_HARNESS_DIR}-{uuid.uuid4().hex[:10]}"
        sb.put_files(eval_branch, {
            f"{harness_dir}/{name}": (HARNESS_DIR / name).read_bytes() for name in task.harness_files
        })
        e = sb.run(eval_branch, identity_env + task.eval_cmd.format(harness_dir=harness_dir, split=task.split_for(seed)), EVAL_TIMEOUT_S)
        if e.timed_out or e.exit_code != 0:
            return {"status": "ungueltig", "infrastructure": 'Raw dataset changed since preparation' in e.stderr,
                    "detail": f"Harness-Lauf fehlgeschlagen: {e.stderr[-500:]}"}
        try:
            out = json.loads(e.stdout.strip())
        except (IndexError, json.JSONDecodeError):
            return {"status": "ungueltig", "detail": f"Harness-Ausgabe unlesbar: {e.stdout[-300:]}"}
        if not isinstance(out, dict) or out.get("status") != "ok":
            return {"status": "ungueltig", "detail": out.get("reason", "") if isinstance(out, dict) else "Harness output must be an object."}
        if metric not in out.get("metrics", {}):
            return {"status": "ungueltig", "detail": f"Harness lieferte Metrik {metric} nicht"}
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in out['metrics'].values()):
            return {'status': 'ungueltig', 'detail': 'Harness metrics must be finite numbers.'}
        val = self._val_eval(eval_branch, harness_dir, task, seed, identity_env=identity_env) if with_val else None
        return {
            "status": "ok", "measured": float(out["metrics"][metric]), "metrics": out["metrics"],
            "baselines": out.get("baselines"), "coverage": coverage, "val": val,
            "metrics_droplast": out.get("metrics_droplast"),   # nur Dossier/Zeugnis (Testwerte), nie in einen Prompt
            "diagnostics": out.get("diagnostics"),             # nur Dossier/Events; pred_std geht in die Fehler-Rueckmeldung, nie Teststatistik
            "stdout_tail": (r.stdout or "")[-300:],            # Ende der Trainingsausgabe (z. B. Val-Loss) als Evidenz
        }

    def _val_eval(self, branch: Branch, harness_dir: str, task: Task, seed: int, identity_env: str = '') -> dict | None:
        """Validierungs-Metrik des Seed-0-Laufs durch unseren Harness (eigene Val-Labels). Fehler machen den Versuch nie ungueltig:
        ohne Val-Wert nimmt die Auswahl den Kandidaten nur als Rueckfall."""
        sb = self.sandbox
        sb.put_files(branch, {f"{harness_dir}/{n}": (HARNESS_DIR / n).read_bytes() for n in task.val_harness_files})
        e = sb.run(branch, identity_env + task.val_eval_cmd.format(harness_dir=harness_dir, split=task.split_for(seed)), EVAL_TIMEOUT_S)
        if e.timed_out or e.exit_code != 0:
            return {'error': 'Validation harness did not complete successfully.'}
        try:
            out = json.loads(e.stdout.strip())
        except (IndexError, json.JSONDecodeError):
            return {"error": f"Val-Harness-Ausgabe unlesbar: {(e.stderr or e.stdout)[-200:]}"}
        if not isinstance(out, dict):
            return {"error": "Validation harness output must be an object."}
        if out.get("status") != "ok":
            return {"error": str(out.get("reason", ""))[:300]}
        if (isinstance(out.get('value'), bool) or not isinstance(out.get('value'), (int, float))
                or not math.isfinite(out['value']) or not out.get('metric')):
            return {'error': 'Validation metric must be a finite number with a metric name.'}
        return {"metric": out["metric"], "value": float(out["value"]), "baselines": out.get("baselines") or {}}

    # ---- Kurzbeschreibung des Papers (nur Bericht) ----------------------------
    def _paper_summary_round(self, paper: dict, claims: list[Claim]) -> dict | None:
        """Produktpfad (graph_repair): fuer ein Paper ohne Eintrag in summaries.json eine Kurzbeschreibung in einfachen Worten, Autoren und Jahr (arXiv-Abstract-Seite)
        und Universitaeten (nur aus dem schon gelesenen Paper-Text, mit Zitat-Check). Ergebnis nur ins Dossier (Feld summary) und Ereignis `paper_summary`;
        kein Prompt des Agenten bekommt es. Fehler = Warnung, der Lauf laeuft weiter; Abrechnungs-, Budget- und Kostenfehler laufen durch."""
        if not getattr(self, "graph_repair", False) or not claims:
            return None
        arxiv_id = paper["arxiv_url"].rstrip("/").split("/abs/")[-1]
        stored = REPO_ROOT / "web" / "public" / "data" / "discovery" / "summaries.json"
        try:
            if arxiv_id in json.loads(stored.read_text()):
                return None   # stored once already (benchmark and catalog papers): the web app reads that
        except (OSError, json.JSONDecodeError):
            pass
        claim = next((c for c in claims if c.origin == "paper"), claims[0])
        try:
            abs_text = getattr(self, "_fetch_abstract", fetch_abstract)(arxiv_id)
            if not abs_text:
                raise ValueError("kein Abstract auf der arXiv-Seite")
            meta = None
            try:
                meta = getattr(self, "_fetch_meta", fetch_arxiv_meta)(arxiv_id)
            except Exception as exc:   # Autoren/Jahr sind optional
                self._warn(f"Autoren und Jahr nicht gelesen ({type(exc).__name__}); die Kurzbeschreibung entsteht trotzdem.")
            pt = self.paper_text
            doc = {"text": pt.text, "source_kind": pt.source_kind, "arxiv_version": pt.arxiv_version} if pt is not None else None
            head = build_head(lambda system, user, t: self._ask("code", system, user, max_tokens=3000, temperature=t).content,
                              claim_for_prompt(paper, claim), abs_text, meta, doc, arxiv_id)
        except (CostLimitExceeded, BillingBlocked, BudgetExceeded):
            raise
        except Exception as exc:
            self._warn(f"Kurzbeschreibung nicht erstellt ({type(exc).__name__}: {str(exc)[:120]}); der Lauf ist davon nicht betroffen.")
            return None
        if not head:
            self._warn("Kurzbeschreibung nicht erstellt (keine gueltige Antwort); der Lauf ist davon nicht betroffen.")
            return None
        self._emit("paper_summary", **head)
        return head

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
            claim_spec = claim_specs(paper).get(claim.id, {})
            spec = claim_spec.get("extract")
            if not spec:
                continue
            keep = claim_spec.get("keep_target", False)   # Katalog: Discovery-Wert bleibt, Rolle 2 ist Gegenprobe
            strict = claim_spec.get("require_verified_target", False)
            selected_latex = bool(keep and strict and claim.metric in ("mse", "mae") and
                                  (paper.get("discovery_source", {}).get("source_kind") == "latex" or
                                   re.search(r"\\[A-Za-z]+|\s&\s", paper.get("discovery_quote", ""))))
            def blocked(reason):
                self._emit("target_blocked", claim_id=claim.id, reason=reason)
                raise TargetVerificationBlocked(reason)
            if self.paper_text is None:   # ein Abruf je Lauf, nicht je Claim
                calls_before = self._tavily_calls
                try:
                    self.paper_text = fetch_paper_text(arxiv_id, get_tavily, mode=self.tavily_mode,
                                                       require_latex=selected_latex,
                                                       source_version=paper.get("discovery_source", {}).get("arxiv_version"),
                                                       expected_sha256=paper.get("discovery_source", {}).get("sha256"))
                except ExtractionInvalid as exc:
                    if strict:
                        self._emit("source_unavailable", arxiv_id=arxiv_id, reason=str(exc))
                        blocked(f"Claim verification stopped: paper source unavailable ({exc}). No experiment was started.")
                    raise
                live = self._tavily_calls > calls_before
                if self.paper_text.from_cache:
                    unknown = " Current arXiv version could not be determined." if self.paper_text.current_version_unknown else ""
                    failure = f" Last download failure: {self.paper_text.download_error}." if self.paper_text.download_error else ""
                    self._warn(f"arXiv source download unavailable. Using saved LaTeX source "
                               f"{arxiv_id}{self.paper_text.arxiv_version or ' (version unknown)'}, "
                               f"retrieved {self.paper_text.retrieved_at}. The target is checked against this saved source.{unknown}{failure}")
                self._record_source("paper", self.paper_text.url, self.paper_text.text, live,
                                    self.paper_text.retrieved_at, self.paper_text.sha256, self.paper_text.from_cache)
                if self._run_dir is not None:   # gelesener Text mit Zeitstempel und Hash im Lauf, auch wenn sich die Seite aendert
                    (self._run_dir / "paper_text.json").write_text(json.dumps(dataclasses.asdict(self.paper_text), ensure_ascii=False))
            text = self.paper_text
            try:
                ev = verify_selected_target(
                    text, spec["method"], spec["dataset"], spec["metric"], claim.target,
                    paper.get("discovery_quote", ""), spec.get("percent", True),
                ) if keep and strict else None
                if ev is None and selected_latex:
                    # Legacy synthesized quotes can still use a unique, independently
                    # parsed source result. Never hand a missing LaTeX anchor to a model.
                    ev = _source_table_target(text, spec["method"], spec["dataset"], spec["metric"],
                                              spec.get("percent", True))
                    if ev is None:
                        raise ExtractionInvalid("Selected LaTeX source quote is missing or unverifiable; "
                                                "the source may have changed since discovery")
                if ev is None:
                    ev = extract_target_cached(
                        self.llm, text, spec["method"], spec["dataset"], spec["metric"],
                        percent=spec.get("percent", True), mode=self.tavily_mode,
                    )
            except ExtractionInvalid as exc:
                if strict:
                    blocked(f"Claim verification stopped: the target could not be verified ({exc}). No experiment was started.")
                if not keep:
                    raise
                self._warn(f"Rolle 2 konnte den Zielwert im Paper nicht gegenpruefen ({exc}); "
                           f"es gilt der Wert der Discovery {claim.target}.")
                continue
            agrees = abs(ev.value - claim.target) < 1e-6   # 82.92/100 gegen anders gerundetes Ergebnis
            self._emit("target_check", claim_id=claim.id, verified_value=ev.value, configured_value=claim.target,
                       agrees=agrees, url=ev.url, from_store=ev.from_store)   # Gegenprobe: belegter Wert gegen papers.yaml
            if keep and not agrees:
                if strict:
                    blocked(f"Claim verification stopped: discovery selected {claim.target}, but the paper check found {ev.value}. "
                            "Resolve the table column before starting an experiment.")
                self._warn(f"Zielwert weicht ab: Rolle 2 liest {ev.value} ({ev.table or 'Tabelle'}, Zitat: \"{ev.quote}\"), "
                           f"die Discovery {claim.target}. Es gilt der Wert der Discovery; Rolle 2 belegt ihn nicht.")
                continue
            if strict and ev.column_check != "ok":
                blocked("Claim verification stopped: the method, dataset and metric column could not be established. "
                        "No experiment was started.")
            claim.target = ev.value
            claim.source = f"{ev.table or 'Tabelle'}, arXiv {ev.arxiv_version or '?'} ({ev.source_kind})"
            claim.evidence_urls = [ev.url]
            claim.target_evidence = dataclasses.asdict(ev)
            print(f"[{claim.id}] Zielwert {ev.value} belegt: {ev.quote!r}")
            self._emit("target_resolved", claim_id=claim.id, value=ev.value, quote=ev.quote, table=ev.table,
                       url=ev.url, arxiv_version=ev.arxiv_version, column_check=ev.column_check, caveat=ev.caveat,
                       from_store=ev.from_store)

    def _community_round(self, paper: dict, dossier: Dossier) -> None:
        """Community-Check (nur Produktpfad, Tavily live): "What others found" einmal je Lauf, "Possible explanations" nur fuer Claims mit
        Urteil teilweise/nicht reproduziert. Laeuft nach dem letzten Urteil, liest Verdicts nur, schreibt nur dossier.community/explanations
        und Ereignisse. Fehler werden zur Warnung, Abrechnungsfehler laufen durch. Ergebnis je arXiv-ID 7 Tage im Cache (logs/)."""
        if not (getattr(self, "graph_repair", False) and getattr(self, "use_tavily", False) and getattr(self, "tavily_mode", None) == "live"):
            return
        from .llm import BillingBlocked
        title, dataset = paper.get("title", ""), paper.get("dataset", "")
        arxiv_id = paper["arxiv_url"].rstrip("/").split("/abs/")[-1]
        spec = ((paper.get("claims") or [{}])[0].get("extract") or {})
        method, dataset = spec.get("method"), spec.get("dataset") or dataset   # aus dem Claim (Methode, Datensatz des Papers)
        deviating = [v.claim_id for v in dossier.verdicts if v.urteil in community.DEVIATING]
        repo_url = (dossier.repo or {}).get("root") or (dossier.repo or {}).get("url")
        try:
            entry = community.cache_get(arxiv_id)
            mode = "cache" if entry else "live"
            if entry is None:
                entry = {"ts": time.time(), "v": community.CACHE_VERSION}
                calls_before = self._tavily_calls
                entry["community"] = community.what_others_found(
                    self._tavily_getter()(), self.llm, title, arxiv_id, dataset, repo_url, getattr(self, "_repo_files", None),
                    paper.get("community"), method=method)
                entry["live_calls"] = self._tavily_calls - calls_before
                community.cache_put(arxiv_id, entry)   # bezahlte Treffer nie verlieren, auch wenn Rolle 6 danach scheitert
            if deviating and entry.get("explanations") is None:
                entry["explanations"] = community.possible_explanations(self._tavily_getter()(), self.llm, title, arxiv_id, dataset)
                mode = "live" if mode == "live" else "cache+live"
            if mode != "cache":
                community.cache_put(arxiv_id, entry)
        except BillingBlocked:
            raise
        except Exception as exc:   # der Check ist Beigabe: ein Fehler darf Lauf und Urteil nie beruehren
            self._warn(f"Community-Check fehlgeschlagen: {type(exc).__name__}: {exc}")
            return
        found = dict(entry["community"], mode=mode, queried_at=datetime.fromtimestamp(entry["ts"], timezone.utc).isoformat(timespec="seconds"))
        dossier.community = found
        for page in found.get("pages", []):
            rec = {"kind": "community", "url": page["url"], "mode": "live" if mode != "cache" else "cache", "n_chars": page["n_chars"],
                   "retrieved_at": found["queried_at"], "sha256": page["sha256"]}
            dossier.sources.append(rec)
            self._emit("source", **rec)
        self._emit("community", mode=mode, n_findings=len(found["findings"]), dropped=found["dropped"], queries=found["queries"],
                   reused_discovery=found["reused_discovery"], findings=found["findings"],
                   hits=found.get("hits", []), not_read=found.get("not_read", []), not_extracted=found.get("not_extracted", []), dropped_items=found.get("dropped_items", []), candidates=found.get("candidates", []))
        if deviating and entry.get("explanations"):
            expl = entry["explanations"]
            for cid in deviating:
                dossier.explanations[cid] = {"items": expl["items"], "dropped": expl["dropped"], "mode": mode}
            self._emit("explanations", claim_ids=deviating, mode=mode, n_items=len(expl["items"]), dropped=expl["dropped"],
                       items=expl["items"])

    def _gather_docs(self, claim_text: str, dataset: str) -> tuple[list[Doc], dict | None]:
        """Paper-Text + (falls im Paper verlinkt) README (Tavily) und Repo-Dateien (Sandbox-Klon)."""
        from .experiment_settings import active_paper_text
        docs = [Doc("paper", active_paper_text(self.paper_text.text))]
        if self.paper_only:
            return docs, None
        get_tavily = self._tavily_getter()
        self._repo_fallback_reason = None
        status = {}
        if getattr(self, "repo_verify", False):
            url, readme, status = self._resolve_repo(claim_text, get_tavily)
            if not url:
                return docs, None
        else:
            url = find_repo_url(self.paper_text.text)
            if not url and getattr(self, "graph_repair", False) and getattr(self, "_paper_title", None):
                try:   # Rolle 1 (Produktpfad): Paper ohne Code-Link, Repo per Tavily-Suche; Treffer muss Paper nennen
                    calls_before = self._tavily_calls
                    url = search_repo_url(self._paper_title, self.paper_text.arxiv_id, get_tavily, mode=self.tavily_mode)
                    self._emit("repo_search", title=self._paper_title, url=url, live=self._tavily_calls > calls_before)
                except Exception as exc:
                    self._warn(f"Rolle 1 (Repo-Suche) fehlgeschlagen: {exc}")
            if not url:
                return docs, None
            calls_before = self._tavily_calls
            readme = fetch_readme(url, get_tavily, mode=self.tavily_mode)
            if readme:
                self._record_source("readme", url, readme, self._tavily_calls > calls_before)
        if readme:
            docs.append(Doc("repo-readme", readme))
        try:
            unit = getattr(self, "_paper_id", None) if getattr(self, "spec_mode", False) else None   # S4: Fixture je Einheit
            dump = read_repo(self.sandbox, self.llm, url, claim_text, dataset, focus_model=getattr(self, "setup_v2", False),
                             unit_key=unit, model=model_name_of(claim_text) if unit else None, mode=self.tavily_mode, on_note=self._warn)
        except SandboxSetupError as exc:   # toter Link oder Klon scheitert: ohne Repo weiter (README und Paper bleiben), kein Abbruch
            self._warn(f"Repo nicht gelesen (Klon in der Sandbox fehlgeschlagen): {str(exc).splitlines()[0]}")
            return docs, None
        if unit and not model_files([Doc(f"repo:{p}", t) for p, t in dump.files.items()], model_name_of(claim_text), entry_fallback=False):
            self._emit("model_file_missing", model=model_name_of(claim_text), url=url, files=sorted(dump.files))   # S4: nie still ignorieren
            self._warn(f"Steckbrief: Datei des Modells {model_name_of(claim_text)} nicht im Repo-Ausschnitt")
        self._repo_files = dict(dump.files)   # fuer den Abgleich A2 (nur Produktpfad); Reihenfolge wie im Dump
        self._repo_listing = dump.listing or list(dump.files)
        if getattr(self, "code_grounded", False) and dump.truncation:
            from .source_context import SourceContextBlocked
            raise SourceContextBlocked(f"Author repository files were truncated: {', '.join(c['file'] for c in dump.truncation)}. "
                                       "No experiment was started.")
        for path, text in dump.files.items():
            docs.append(Doc(f"repo:{path}@{dump.commit[:7]}", text))
        if getattr(self, "code_grounded", False) and unit:
            # Gleiche deterministische Vollstaendigkeitspruefung wie spaeter im Steckbrief, aber vor den bezahlten
            # Extraktionsaufrufen: SARAF (2026-10-14) zahlte 0.12 $ Rolle 4, bevor der Quellenwaechter stoppte.
            from .source_context import author_excerpt
            author_excerpt(docs, model_name_of(claim_text), dataset, listing=self._repo_listing)
        info = {"url": url, "commit": dump.commit, "files": sorted(dump.files), "n_tracked": dump.n_tracked}
        if dump.subdir:   # Unterordner eines Sammel-Repos: Issues, Sterne usw. gehoeren zur Wurzel
            info["subdir"], info["root"] = dump.subdir, clone_plan(url)[0]
        if status:
            info.update(status)
        return docs, info

    def _resolve_repo(self, claim_text: str, get_tavily) -> tuple[str | None, str, dict]:
        """Repo-Wahl mit Pruefung (repo_verify). Reihenfolge: (1) Code-Link im Paper-Text (starke Hinweise zuerst, Unterordner in Sammel-Repos
        bleiben erhalten), (2) mehrere Tavily-Treffer. Jeder Kandidat wird ueber sein README geprueft (hoechstens MAX_REPO_CANDIDATES Lesezugriffe):
        es muss Modellname, Titel(teil) oder arXiv-ID nennen. Ist das README nicht lesbar, zaehlt nur ein Link, den das Paper selbst setzt.
        Offizielle Repos vor inoffiziellen; ein inoffizielles ist nie Autorencode (Modus bleibt Nachbau aus dem Paper, mit Hinweis).
        Rueckgabe (URL | None, README-Text, Statusangaben fuer Dossier und Ereignis)."""
        text = self.paper_text.text
        title = getattr(self, "_paper_title", None) or ""
        arxiv_id = self.paper_text.arxiv_id
        model = model_name_of(claim_text)
        paper_urls = [u for u in dict.fromkeys([find_repo_url(text), *[l.target for l in find_repo_links(text)][:2]]) if u][:2]
        searched = False
        tried: list[str] = []
        unofficial: dict | None = None

        def consider(url: str, source: str):
            calls_before = self._tavily_calls
            readme = fetch_readme(url, get_tavily, mode=self.tavily_mode)
            live = self._tavily_calls > calls_before
            kind, why = repo_evidence(readme, url, title, arxiv_id, model)
            linked = source == "paper"
            if readme.strip() and not kind:
                verdict, reason = "rejected", "README names neither the model, the paper title nor the arXiv id"
            elif not readme.strip() and not linked:
                verdict, reason = "rejected", "README not readable and the paper does not link this repository"
            else:
                verdict, reason = "accepted", why or "linked in the paper text (README not readable)"
            st, st_why = repo_status(url, readme, linked, kind, text) if verdict == "accepted" else ("", "")
            self._emit("repo_candidate", url=url, source=source, verdict=verdict, reason=reason, status=st, status_reason=st_why,
                       evidence=kind, live=live)
            return verdict, readme, st, st_why, why, live

        queue = [(u, "paper") for u in paper_urls]
        while len(tried) < MAX_REPO_CANDIDATES:
            if not queue and not searched and getattr(self, "graph_repair", False) and title:
                searched = True
                try:   # Rolle 1 (Produktpfad): Suche mit mehreren Treffern
                    calls_before = self._tavily_calls
                    found = search_repo_candidates(title, arxiv_id, get_tavily, mode=self.tavily_mode, model=model)
                    self._emit("repo_search", title=title, url=found[0] if found else None, urls=found, live=self._tavily_calls > calls_before)
                    queue = [(u, "search") for u in found]
                except Exception as exc:
                    self._warn(f"Rolle 1 (Repo-Suche) fehlgeschlagen: {exc}")
            queue = [(u, src) for u, src in queue if u not in tried]
            if not queue:
                break
            url, source = queue.pop(0)
            tried.append(url)
            verdict, readme, st, st_why, why, live = consider(url, source)
            if verdict != "accepted":
                continue
            status = {"status": st, "status_reason": st_why, "source": source, "evidence": why}
            if st == "official":
                if readme:
                    self._record_source("readme", url, readme, live)
                return url, readme, status
            if unofficial is None:
                unofficial = {"url": url, **status}
        if unofficial:
            self._repo_fallback_reason = f"only an unofficial repository was found ({unofficial['url']})"
            self._warn(f"Inoffizielles Repo {unofficial['url']} gefunden, nicht als Autorencode gelesen: {unofficial['status_reason']}. "
                       "Lauf als reiner Nachbau aus dem Paper.")
        elif tried:
            self._repo_fallback_reason = "no repository matching the paper found"
        return None, "", {}

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
    def _objection_round(self, paper: dict, dossier: Dossier) -> None:
        """Einwand-Experimente (Rolle 3, Stufe 2): nur Produktpfad mit Tavily live und eingeschaltetem Schalter, nach Urteilen und
        Community-Check. Liest Funde, schreibt nur dossier.objection_checks und Ereignisse. Fehler werden zur Warnung."""
        if not (getattr(self, "objection_experiments", False) and getattr(self, "graph_repair", False) and getattr(self, "use_tavily", False)
                and getattr(self, "tavily_mode", None) == "live"):
            return
        from .llm import BillingBlocked
        try:
            objection_experiments.run_round(self, dossier, paper, quick=getattr(self, "objection_quick", False))
        except BillingBlocked:
            raise
        except Exception as exc:   # Beigabe: ein Fehler beruehrt Lauf und Urteil nie
            self._warn(f"Einwand-Experimente fehlgeschlagen: {type(exc).__name__}: {exc}")

    def run_claim(
        self, paper: dict, claim: Claim, task: Task, snapshot: str, dossier: Dossier, run_dir: Path
    ) -> None:
        if getattr(self, 'independent_selection', False):
            task = task.with_independent_selection()
        self._emit("claim_start", claim_id=claim.id, text=claim.text, metric=claim.metric, target=claim.target,
                   tolerance=claim.tolerance)
        plan = self.plan(paper, claim, task)
        (run_dir / f"{claim.id}_plan.md").write_text(plan)
        self._emit("plan", claim_id=claim.id, text=plan[:1500])
        self._emit_live("plan_text", claim_id=claim.id, text=plan)   # voller Plan nur fuer den Live-Stream (plan.md im Export)
        seeds = list(range(self.seeds or paper.get("seeds", DEFAULT_SEEDS)))
        n_cand = getattr(self, "n_candidates", 1) if getattr(self, "graph_repair", False) else 1
        cand: int | None = None
        cand_notes: list[str] = []
        if n_cand > 1:   # A1 (nur Produktpfad): mehrere Kandidaten, Seed 0 je Kandidat, Auswahl ueber Validierung
            v0, note0, code, graph_ab, graph_notes, discarded, cand, cand_notes = self._run_candidates(
                claim, task, snapshot, dossier, run_dir, plan, seeds[0], n_cand)
        else:
            code = self.write_code(plan, task)
            self._announce_code(claim, code)
            self._emit("code_written", claim_id=claim.id, code_hash=hashlib.sha256(code.encode()).hexdigest()[:12])
            code, _, spec_disc = self._spec_conform(claim, None, plan, task, code)   # Steckbrief (S2/S3) auch ohne Kandidaten
            # Seed 0 zuerst mit Reparaturschleife -> ein lauffaehiger Code pro Generierung. Die uebrigen
            # Seeds laufen parallel mit genau diesem Code (Streuung = Seed, nicht Code-Variante).
            v0, note0, code = self._run_seed(claim, task, snapshot, dossier, run_dir, plan, code, seeds[0])
            graph_ab, graph_notes, discarded = None, [], set(spec_disc)
            if getattr(self, "graph_repair", False) and v0 is not None:   # Katalog/API: Fehler- und Graph-Check nach Seed 0
                v0, note0, code, graph_ab, graph_notes, discarded, _ = self._seed0_quality_loop(
                    claim, task, snapshot, dossier, run_dir, plan, code, v0, note0, seeds[0])
        notes = ([] if note0 is None else [note0]) + graph_notes + cand_notes
        selected_measurements: list[dict] = []

        def run_rest(code: str, v0: float) -> tuple[list[float], list[str]]:
            nonlocal selected_measurements
            rest, got, rnotes = seeds[1:], [v0], []
            selected_measurements = [{"seed": seeds[0], "code_hash": hashlib.sha256(code.encode()).hexdigest()[:12], "measured": v0}]
            if rest:
                with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(rest))) as pool:
                    results = list(context_map(pool,
                        lambda sd: self._run_seed(claim, task, snapshot, dossier, run_dir, plan, code, sd, cand),
                        rest,
                    ))
                got += [v for v, _, _ in results if v is not None]
                rnotes += [n for _, n, _ in results if n]
                selected_measurements += [{"seed": sd, "code_hash": hashlib.sha256(seed_code.encode()).hexdigest()[:12], "measured": v}
                                          for sd, (v, _, seed_code) in zip(rest, results) if v is not None]
            return got, rnotes

        def selected_attempts():
            return [next((a for a in reversed(dossier.attempts) if a.claim_id == claim.id and a.seed == m['seed']
                          and a.code_hash == m['code_hash'] and a.measured == m['measured']
                          and (cand is None or a.candidate == cand)), None) for m in selected_measurements]

        def metrics_mean() -> dict[str, float]:
            per_metric: dict[str, list[float]] = {}
            for a in selected_attempts():
                if a is not None and a.metrics:
                    for name, value in a.metrics.items():
                        per_metric.setdefault(name, []).append(value)
            return {name: statistics.mean(v) for name, v in per_metric.items()}

        if v0 is not None and getattr(self, "graph_repair", False) and getattr(self, "_repo_files", None):
            v0, code, graph_ab, rd_notes, discarded = self._repo_diff_round(
                claim, task, snapshot, dossier, run_dir, plan, code, v0, seeds[0], cand, graph_ab, discarded,
                paper.get("dataset", ""))
            notes += rd_notes

        if v0 is not None and getattr(self, 'graph_repair', False) and not getattr(self, 'allow_test_feedback', False):
            self._admit_final_code(claim, task, snapshot, dossier, code, seeds[0])
            if task.selection_contract:
                notes.append('Independent selection used the final 20% of validation observations, withheld from fitting; '
                             'checkpoint selection used the remaining visible validation data. Original train/test splits unchanged.')
                if json.loads(task.io_config)['kind'] == 'windows':
                    notes.append('Forecast histories were streamed chronologically to an unprivileged predictor after fitting; '
                                 'each forecast was recorded before the next history was exposed.')

        measured: list[float] = []
        if v0 is None:
            notes.append("Restliche Seeds nicht gestartet: kein lauffaehiger Code")
        else:
            measured, rnotes = run_rest(code, v0)
            notes += rnotes
        all_metrics = metrics_mean()
        if len(measured) != len(seeds) and not getattr(self, 'allow_test_feedback', False):
            raise MethodImplementationBlocked(f'Only {len(measured)} of {len(seeds)} planned seeds produced valid measurements with the selected code. '
                                              'The check stopped without a paper verdict. ' + '; '.join(notes))
        if v0 is None and getattr(self, "graph_repair", False) and placeholder_issues(code, getattr(self, "_repo_files", {})):
            raise MethodImplementationBlocked("Incomplete method: the generated retrieval component is a placeholder after repair attempts. "
                                              "The paper is not assessable; no experiment measurement or paper verdict was produced.")
        verdict = self._verdict(claim, measured, len(seeds), notes, dossier.baselines, all_metrics)
        if (getattr(self, "allow_test_feedback", False) and getattr(self, "graph_repair", False) and verdict.urteil == "Agenten-Fehler vermutet" and v0 is not None
                and dossier.fault_repairs.get(claim.id, 0) < MAX_FAULT_REPAIRS
                and not self._fault_check(claim, dossier, hashlib.sha256(code.encode()).hexdigest()[:12], seeds[0], v0)):
            # Seed 0 lag ueber der Baseline, der Mittelwert ueber alle Seeds nicht (instabiler Code): eine Reparatur, alle Seeds neu
            from_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
            head = f"Fehler-Reparatur nach allen Seeds: Code {from_hash} im Mittel schlechter als triviale Baseline"
            self._emit("fault_repair", claim_id=claim.id, runde="mittelwert", from_hash=from_hash, measured=verdict.measured_mean,
                       reason=agent_fault_suspected(claim.metric, verdict.measured_mean, claim.target, dossier.baselines, all_metrics))
            try:
                new = self.repair_code(plan, task, code, fault_message(claim.metric, verdict.measured_mean, claim.target, dossier.baselines))
                nv, nnote, new = self._run_seed(claim, task, snapshot, dossier, run_dir, plan, new, seeds[0], cand)
            except ValueError as exc:
                nv, nnote = None, f"Reparatur ohne Code-Block ({exc})"
            new_hash = hashlib.sha256(new.encode()).hexdigest()[:12] if nv is not None else None
            if nv is not None and not self._fault_check(claim, dossier, new_hash, seeds[0], nv):
                discarded.add(from_hash)
                code, v0, graph_ab = new, nv, None   # Ablation gilt fuer den alten Code: neu messen
                self._emit("fault_repair_result", claim_id=claim.id, result="adopted", from_hash=from_hash, to_hash=new_hash)
                measured, rnotes = run_rest(code, v0)
                notes += [f"{head}; neuer Code {new_hash}, alle Seeds neu"] + ([nnote] if nnote else []) + rnotes
                dossier.fault_repairs[claim.id] = dossier.fault_repairs.get(claim.id, 0) + 1
                all_metrics = metrics_mean()
                verdict = self._verdict(claim, measured, len(seeds), notes, dossier.baselines, all_metrics)
            else:
                self._emit("fault_repair_result", claim_id=claim.id, result="kept", from_hash=from_hash, to_hash=new_hash,
                           reason=nnote or "Repair still failed the baseline check")
                notes.append(f"{head}; Reparatur ohne Besserung ({nnote or 'weiter schlechter als Baseline'}), Urteil bleibt")
                verdict.explanation += f" {notes[-1]}."
        if v0 is not None and task.ablation_cmd:
            if graph_ab is None:
                ab = self.ablation_check(snapshot, code, seeds[0], task, claim.metric, v0)
                self._emit("ablation", claim_id=claim.id, **ab)
            else:   # Ablation des Codes, mit dem die Seeds liefen, lief schon nach Seed 0
                ab = graph_ab
            dossier.ablations[claim.id] = ab
            print(f"[{claim.id}] Graph-Ablation: {ab['verdict']} (mit Graph {v0:.4f}, ohne Kanten {ab['accuracy_ablated']})", flush=True)
            if ab["verdict"] == "Graph nicht genutzt":   # deterministischer Beleg, nur herabstufend
                verdict.methode = "nein"
        selected_ids = {id(a) for a in selected_attempts() if a is not None}
        for a in dossier.attempts:
            if a.claim_id == claim.id:
                a.chosen = id(a) in selected_ids
        self.__dict__.setdefault("_claim_ctx", {})[claim.id] = {"code": code, "plan": plan, "task": task, "snapshot": snapshot, "n_seeds": len(seeds)}   # fuer Einwand-Experimente
        dossier.verdicts.append(verdict)
        # The live/replay chart must use the exact measurements behind this verdict,
        # including an accepted repair after all seeds. Keep every attempt in history.
        self._emit("measurements_selected", claim_id=claim.id, attempts=selected_measurements)
        self._emit("verdict", claim_id=claim.id, urteil=verdict.urteil, methode=verdict.methode,
                   gesamt=verdict.gesamtaussage, mean=verdict.measured_mean, std=verdict.measured_std,
                   explanation=verdict.explanation)
        self._droplast_round(claim, dossier, discarded, verdict)   # nur Bericht, nach dem Urteil

    def _spec_conform(self, claim: Claim, cand: int | None, plan: str, task: Task, code: str) -> tuple[str, tuple[int, int] | None, set]:
        """S2/S3 nach dem Schreiben: CONFIG erzwingen und pruefen, nicht erzwungene Eintraege einzeln fragen, bei Befunden hoechstens
        MAX_SPEC_REPAIRS gezielte Reparaturen. Rueckgabe (Code, (erfuellt, gesamt) oder None ohne Steckbrief, verworfene Code-Hashes)."""
        spec = self._active_spec()
        h = lambda c: hashlib.sha256(c.encode()).hexdigest()[:12]
        spec_score, disc = None, set()
        if spec is None or not spec.items:
            return code, spec_score, disc
        for rep in range(MAX_SPEC_REPAIRS + 1):
            code, cfg_report = spec_check.config_check(code, spec)
            items_report = spec_check.check_items(
                lambda sys, usr: self._ask("code", sys, usr, max_tokens=SPEC_MAX_TOKENS, temperature=0.0).content,
                code, spec)
            fails = spec_check.failures(cfg_report, items_report)
            spec_score = spec_check.score(spec, cfg_report, items_report)
            self.__dict__.setdefault('_spec_scores', {})[(id(spec), hashlib.sha256(code.encode()).hexdigest())] = spec_score
            self._emit("spec_check", claim_id=claim.id, candidate=cand, round=rep, score=spec_score, fails=fails, code_hash=h(code))
            if not fails or rep == MAX_SPEC_REPAIRS:
                break
            try:
                new_code = self.repair_code(plan, task, code, spec_check.repair_message(fails, spec))
                disc.add(h(code))
                code = new_code
                self._announce_code(claim, code)
            except (CostLimitExceeded, BillingBlocked, BudgetExceeded):
                raise
            except Exception as exc:
                self._warn(f"Steckbrief-Reparatur gescheitert: {exc}")
                break
        return code, spec_score, disc

    def _current_spec_score(self, code: str) -> tuple[int, int] | None:
        spec = self._active_spec()
        if spec is None:
            return None
        key = (id(spec), hashlib.sha256(code.encode()).hexdigest())
        cached = self.__dict__.get('_spec_scores', {}).get(key)
        if cached is not None:
            return cached
        checked_code, config = spec_check.config_check(code, spec)
        # Do not change an already measured program. An override still needed
        # here means this version has not satisfied those keys.
        config['unused'] = list(set(config.get('unused', [])) | {i['key'] for i in config.get('overrides', [])})
        items = [{'key': i.key, 'ok': False} for i in spec.checkable()]
        return spec_check.score(spec, config, items)

    def _run_candidates(
        self, claim: Claim, task: Task, snapshot: str, dossier: Dossier, run_dir: Path, plan: str, seed: int, k: int,
    ):
        """A1: k Code-Kandidaten (Temperatur 0.2 + 0.3 i, gleicher Prompt), je Seed 0 mit Reparatur- und Qualitaetsschleife
        im Produktpfad nacheinander, im historischen Testfeedback-Pfad parallel. Auswahl allein ueber die
        Validierungs-Metrik, die unser Harness rechnet (select_candidate); Testwerte und
        Zielwert gehen nie in die Auswahl. Rueckgabe wie die Seed-0-Phase plus Index des Siegers und Hinweise fuer das Urteil."""
        def h(c: str) -> str:
            return hashlib.sha256(c.encode()).hexdigest()[:12]

        codes: list[str] = []
        idx: list[int] = []
        last_exc: Exception | None = None
        def one(j: int) -> dict:
            i = idx[j]
            code = codes[j]
            code, spec_score, disc = self._spec_conform(claim, i, plan, task, code)

            v0, note0, code = self._run_seed(claim, task, snapshot, dossier, run_dir, plan, code, seed, i)
            res = {"i": i, "v0": v0, "note0": note0, "code": code, "ab": None, "notes": [], "discarded": disc, "faults": 0}
            if spec_score is not None:
                res["spec_score"] = spec_score
            if v0 is not None:   # Kandidatenphase: nur Fehler-Check; Graph-Ablation nur fuer den Sieger
                v0, note0, code, ab, notes, disc2, faults = self._seed0_quality_loop(
                    claim, task, snapshot, dossier, run_dir, plan, code, v0, note0, seed, i, graph=False)
                res.update(v0=v0, note0=note0, code=code, ab=ab, notes=notes, discarded=disc | disc2, faults=faults)
            if self._active_spec() is not None:
                res['spec_score'] = self._current_spec_score(res['code'])
                if (id(self._active_spec()), hashlib.sha256(res['code'].encode()).hexdigest()) not in self.__dict__.get('_spec_scores', {}):
                    self._emit('spec_check', claim_id=claim.id, candidate=i, stage='final_candidate',
                               code_hash=h(res['code']), score=res['spec_score'], fails=[])
            val = self.__dict__.get("_val", {}).get((h(res["code"]), seed)) if res["v0"] is not None else None
            res["val"] = val if isinstance(val, dict) and "value" in val else None
            res["val_check"] = (self._selection_check(snapshot, res["code"], seed, task, claim.metric, res["val"])
                                if res['v0'] is not None and (not getattr(self, 'allow_test_feedback', False) or task.mask_val_cmd)
                                else None)
            return res

        results = []
        for i in range(k):
            try:
                c = self.write_code(plan, task, temperature=min(0.2 + CANDIDATE_TEMP_STEP * i, CODE_MAX_TEMPERATURE))
            except (CostLimitExceeded, BudgetExceeded):
                if not self._candidates_survive_limit(results):
                    raise
                break
            except BillingBlocked:
                raise
            except (RuntimeError, ValueError) as exc:
                last_exc = exc
                self._emit('code_missing', claim_id=claim.id, candidate=i, reason=str(exc)[:300])
                continue
            codes.append(c)
            idx.append(i)
            self._announce_code(claim, c)
            self._emit('code_written', claim_id=claim.id, code_hash=h(c), candidate=i)
            if not getattr(self, 'allow_test_feedback', False):
                try:
                    results.append(one(len(codes)-1))
                except (CostLimitExceeded, BudgetExceeded):
                    # Repairs of a later candidate hit the limit: an earlier, independently validated candidate still counts
                    # (SARAF local 2026-10-14 needed 0.55 $ gross; the live job budget is 0.50 $).
                    if not self._candidates_survive_limit(results):
                        raise
                    break
        if not codes:
            raise RuntimeError(f'Kein Code-Kandidat lieferte Code: {last_exc}')
        if getattr(self, 'allow_test_feedback', False):
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, k)) as pool:
                results = list(context_map(pool, one, range(len(codes))))
        runnable = [r for r in results if r["v0"] is not None]
        val_warning = None
        if runnable:
            # B1: Kandidaten, die Validierungslabels im Training nutzen, scheiden aus der Auswahl aus; sind alle betroffen, gilt die alte Auswahl
            eligible = [r for r in runnable if not (r["val_check"] or {}).get("excluded")]
            if not eligible:
                if not getattr(self, 'allow_test_feedback', False):
                    raise MethodImplementationBlocked('All runnable candidates trained on validation labels. No independent candidate selection or paper verdict is possible.')
                eligible = runnable
                val_warning = "alle lauffaehigen Kandidaten nutzen Validierungslabels im Training (B1), Auswahl wie bisher"
            elif len(eligible) < len(runnable):
                val_warning = f"{len(runnable) - len(eligible)} Kandidat(en) wegen Validierungslabels im Training aus der Auswahl genommen (B1)"
            if not getattr(self, 'allow_test_feedback', False) and task.val_eval_cmd:
                eligible = [r for r in eligible if r['val'] is not None]
                if not eligible:
                    raise MethodImplementationBlocked('Candidate validation did not complete. No candidate was selected using test measurements; no paper verdict was produced.')
            
            if self._active_spec() is not None:
                if getattr(self, 'independent_selection', False):
                    eligible = [r for r in eligible if r.get('spec_score') is not None and r['spec_score'][0] == r['spec_score'][1]]
                    if not eligible:
                        raise MethodImplementationBlocked('No independently validated candidate satisfied the mandatory experiment settings. No paper verdict was produced.')
                max_score = max((r.get("spec_score", (0, 0))[0] for r in eligible), default=0)
                spec_eligible = [r for r in eligible if r.get("spec_score", (0, 0))[0] == max_score]
                pick, reason = select_candidate([r["val"] for r in spec_eligible])
                win = spec_eligible[pick]
                reason = f"spec ({max_score}) + {reason}"
            else:
                pick, reason = select_candidate([r["val"] for r in eligible])
                win = eligible[pick]
        else:
            win, reason = results[0], "fallback"
        self._emit("candidates", claim_id=claim.id, k=len(results), requested=k, chosen=win["i"], chosen_hash=h(win["code"]), reason=reason,
                   candidates=[{"index": r["i"], "code_hash": h(r["code"]), "runnable": r["v0"] is not None,
                                "val": ({"metric": r["val"]["metric"], "value": r["val"]["value"]} if r["val"] else None)}
                               for r in results])
        checks = {str(r["i"]): {"code_hash": h(r["code"]), **r["val_check"]} for r in results if r["val_check"]}
        if checks:
            dossier.val_label_check[claim.id] = checks
            for cand_i, info in checks.items():
                self._emit("val_label_check", claim_id=claim.id, candidate=int(cand_i), **info)
        if val_warning:
            dossier.warnings.append(val_warning)
        loser_hashes = {a.code_hash for a in dossier.attempts
                        if a.claim_id == claim.id and a.candidate is not None and a.candidate != win["i"]}
        v0, note0, code, ab, notes, discarded = win["v0"], win["note0"], win["code"], win["ab"], list(win["notes"]), set(win["discarded"])
        if v0 is not None and getattr(self, "allow_test_feedback", False) and getattr(task, "ablation_cmd", ""):   # legacy diagnostic repair only
            v0, note0, code, ab, notes2, disc2, _ = self._seed0_quality_loop(
                claim, task, snapshot, dossier, run_dir, plan, code, v0, note0, seed, win["i"], graph=True, fault_rounds0=win["faults"])
            notes += notes2
            discarded |= disc2
        discarded = (discarded | loser_hashes) - {h(code)}
        parts = ", ".join(f"{r['i']}: " + (f"{r['val']['metric']} {r['val']['value']:.4f}" if r["val"] else "no val value") for r in results)
        note = (f"Code-Kandidaten: {k}, gewaehlt Kandidat {win['i']} (Code {h(win['code'])}) nach Validierungs-Metrik "
                f"({parts})" + ("" if reason == "val" or reason.endswith(" + val") else "; Rueckfall ohne Val-Wert"))
        return v0, note0, code, ab, notes, discarded, win["i"], [note]

    def _candidates_survive_limit(self, results: list[dict]) -> bool:
        """At a spending limit, continue only with independently validated candidates; never in diagnostic test-feedback mode."""
        if getattr(self, 'allow_test_feedback', False) or not any(
                r['v0'] is not None and r['val'] is not None and not (r['val_check'] or {}).get('excluded') for r in results):
            return False
        budget = spend_guard.current_budget.get()
        if budget is not None:
            budget.check()  # Cancellation/uncertain billing never becomes a fallback.
        self._warn('Additional candidate generation stopped at the spending limit; selecting only independently validated existing candidates.')
        return True

    def _repo_diff_round(
        self, claim: Claim, task: Task, snapshot: str, dossier: Dossier, run_dir: Path, plan: str, code: str, v0: float,
        seed: int, cand: int | None, graph_ab: dict | None, discarded: set[str], dataset: str = "",
    ) -> tuple[float, str, dict | None, list[str], set[str]]:
        """A2 (nur Produktpfad mit Repo-Dateien): unseren Code mit den Repo-Dateien abgleichen, geprueften Abweichungen folgt EINE
        Reparaturrunde. Der neue Code ersetzt den alten nur, wenn er den Fehler-Check, die Graph-Ablation und die Validierung
        (gleich oder besser, Val-Fehlen = alter Code) besteht. Weder Val-/Testzahl noch Zielwert gehen in einen Prompt."""
        def h(c: str) -> str:
            return hashlib.sha256(c.encode()).hexdigest()[:12]

        discarded = set(discarded)
        files = self._repo_files
        try:   # optionaler Schritt: ein Fehlschlag (z. B. Antwort leer, weil das Denken das Token-Limit aufzehrt) beendet nie den Lauf
            devs, dropped, cuts = find_deviations(
                lambda system, user: self._ask("extract", system, user, max_tokens=REPO_DIFF_MAX_TOKENS, temperature=0.0),
                code, files, dataset or "the dataset")
        except (CostLimitExceeded, BudgetExceeded):
            if getattr(self, 'allow_test_feedback', False):
                raise
            budget = spend_guard.current_budget.get()
            if budget is not None:
                budget.check()
            self._emit('repo_diff', claim_id=claim.id, result='incomplete', from_hash=h(code), reason='Optional comparison exceeds the spending limit')
            dossier.repo_diff[claim.id] = {'result': 'incomplete', 'from_hash': h(code), 'reason': 'Optional comparison exceeds the spending limit',
                                         'method_fidelity': 'not_checked'}
            return v0, code, graph_ab, ['Optional repository comparison not completed within budget; validated selected code retained.'], discarded
        except BillingBlocked:
            raise
        except (RuntimeError, ValueError) as exc:
            self._emit("repo_diff", claim_id=claim.id, result="error", reason=str(exc)[:300], from_hash=h(code))
            dossier.repo_diff[claim.id] = {"result": "error", "reason": str(exc)[:300], "from_hash": h(code)}
            return v0, code, graph_ab, ["Repo-Abgleich fehlgeschlagen (LLM-Antwort unbrauchbar), alter Code bleibt"], discarded
        info = {"deviations": as_dicts(devs), "dropped": len(dropped), "truncation": cuts, "from_hash": h(code),
                "method_fidelity": "not_checked", "scope": "selected implementation aspects"}
        if not devs:
            result = "incomplete" if cuts or dropped else "no_deviation"
            self._emit("repo_diff", claim_id=claim.id, result=result, **info)
            dossier.repo_diff[claim.id] = {**info, "result": result}
            note = ("Repo comparison incomplete: files were truncated or findings could not be verified. Method fidelity remains unchecked."
                    if result == "incomplete" else "Repo comparison: no verified differences in the reviewed aspects. Method fidelity remains unchecked.")
            return v0, code, graph_ab, [note], discarded

        def finish(result: str, reason: str, **extra):
            self._emit("repo_diff", claim_id=claim.id, result=result, reason=reason, **info, **extra)
            dossier.repo_diff[claim.id] = {**info, "result": result, "reason": reason, **extra}

        try:
            new_code = self.repair_code(plan, task, code, repair_message(devs))
        except (CostLimitExceeded, BudgetExceeded):
            # Optional wie der Vergleich selbst: die Grenze verhindert weitere Ausgaben, der validierte Code wird weiter gemessen
            # (FITS live 5fdc78fc: Abbruch ohne Urteil nach gewaehltem Kandidaten 0.37199, Seeds brauchen kein LLM).
            if getattr(self, 'allow_test_feedback', False):
                raise
            budget = spend_guard.current_budget.get()
            if budget is not None:
                budget.check()
            finish("kept", "Optional repair exceeds the spending limit")
            return v0, code, graph_ab, ['Optional repository repair not run within budget; validated selected code retained.'], discarded
        except ValueError as exc:
            finish("kept", f"Reparatur ohne Code-Block ({exc})")
            return v0, code, graph_ab, [f"Repo-Abgleich: {len(devs)} Abweichungen, Reparatur ohne Code-Block, alter Code bleibt"], discarded
        nv, nnote, new_code = self._run_seed(claim, task, snapshot, dossier, run_dir, plan, new_code, seed, cand)
        old_h, new_h = h(code), h(new_code)
        reason = None
        new_ab = None
        old_val = self.__dict__.get("_val", {}).get((old_h, seed))
        new_val = self.__dict__.get("_val", {}).get((new_h, seed))
        if nv is None:
            reason = f"neuer Code lief nicht ({nnote})"
        elif new_h == old_h:
            reason = "Reparatur lieferte denselben Code"
        elif (getattr(self, "allow_test_feedback", False) and self._fault_check(claim, dossier, new_h, seed, nv)) or self._val_fault_check(claim, new_h, seed):
            reason = "neuer Code faellt beim Fehler-Check durch"
        elif (not getattr(self, 'allow_test_feedback', False)
              and self._selection_check(snapshot, new_code, seed, task, claim.metric, new_val).get('excluded')):
            reason = "new code failed independent validation controls"
        elif not (isinstance(old_val, dict) and isinstance(new_val, dict) and "value" in old_val and "value" in new_val):
            reason = "Validierungswert fehlt, alter Code bleibt"
        else:
            lower = new_val.get("metric") in LOWER_IS_BETTER_VAL
            if (new_val["value"] > old_val["value"]) if lower else (new_val["value"] < old_val["value"]):
                reason = "Validierung schlechter als beim alten Code"
            elif getattr(self, "allow_test_feedback", False) and getattr(task, "ablation_cmd", ""):
                new_ab = self.ablation_check(snapshot, new_code, seed, task, claim.metric, nv)
                self._emit("ablation", claim_id=claim.id, code_hash=new_h, runde="repo_diff", **new_ab)
                if new_ab["verdict"] == "Graph nicht genutzt":
                    reason = "neuer Code nutzt den Graph nicht (Ablation)"
        val_info = {"val_old": old_val.get("value") if isinstance(old_val, dict) else None,
                    "val_new": new_val.get("value") if isinstance(new_val, dict) else None, "to_hash": new_h}
        if reason is not None:
            if new_h != old_h:
                discarded.add(new_h)
            finish("kept", reason, **val_info)
            return v0, code, graph_ab, [f"Repo-Abgleich: {len(devs)} Abweichungen, Reparatur nicht uebernommen ({reason})"], discarded
        discarded.add(old_h)
        discarded.discard(new_h)
        finish("adopted", "Validation and implementation checks passed" + ("; graph ablation passed" if new_ab is not None else ""), **val_info)
        return nv, new_code, new_ab, [f"Repo-Abgleich: {len(devs)} Abweichungen, Code {old_h} durch {new_h} ersetzt"], discarded

    def _droplast_round(self, claim: Claim, dossier: Dossier, discarded: set[str], verdict: Verdict) -> None:
        """ETTh1: Hinweis, wenn die Behauptung fast genau die drop_last-Rechnung trifft (droplast_note.py). Liest nur, aendert das Urteil nie."""
        if claim.metric not in ("mse", "mae") or not (verdict.measured_mean == verdict.measured_mean):   # NaN = kein Messwert
            return
        per_seed = [a.metrics_droplast for a in dossier.attempts if a.claim_id == claim.id and a.metrics_droplast
                    and a.measured is not None and a.chosen is not False and a.code_hash not in discarded]
        note = droplast_note(claim.metric, claim.target, claim.tolerance, verdict.measured_mean, per_seed)
        if note:
            dossier.droplast_notes[claim.id] = note
            self._emit("droplast_note", claim_id=claim.id, kind=note["kind"], batch_size=note["batch_size"], droplast=note["droplast"], full=note["full"])

    def _val_fault_check(self, claim: Claim, code_hash: str, seed: int) -> str | None:
        """Compare validation predictions with validation-only trivial baselines.
        No test measurements or paper target enter this repair decision.
        """
        val = self.__dict__.get("_val", {}).get((code_hash, seed))
        if not isinstance(val, dict) or not isinstance(val.get("value"), (int, float)) or not math.isfinite(val["value"]):
            return None
        lower = val.get("metric", claim.metric) in LOWER_IS_BETTER_VAL
        bases = [v for v in (val.get("baselines") or {}).values() if isinstance(v, (int, float)) and math.isfinite(v)]
        if bases and ((val['value'] > min(bases)) if lower else (val['value'] < max(bases))):
            return 'Validation predictions are worse than a trivial validation baseline.'
        return None

    def _fault_check(self, claim: Claim, dossier: Dossier, code_hash: str, seed: int, v0: float) -> str | None:
        """Schlechter als eine triviale Baseline? (harness/verify.agent_fault_suspected, deterministisch, kein LLM)"""
        att = next((x for x in reversed(dossier.attempts) if x.claim_id == claim.id and x.seed == seed
                    and x.code_hash == code_hash and x.measured is not None), None)
        metrics = att.metrics if att is not None and att.metrics else {claim.metric: v0}
        return agent_fault_suspected(claim.metric, v0, claim.target, dossier.baselines, metrics)

    def _seed0_quality_loop(
        self, claim: Claim, task: Task, snapshot: str, dossier: Dossier, run_dir: Path, plan: str, code: str,
        v0: float, note0: str | None, seed: int, candidate: int | None = None,
        graph: bool = True, fault_rounds0: int = 0,
    ) -> tuple[float, str | None, str, dict | None, list[str], set[str], int]:
        """Nach Seed 0 (nur Katalog/API): Fehler-Check (schlechter als triviale Baseline) und Graph-Check mit Reparaturrunden.

        Fehler: bis MAX_FAULT_REPAIRS Runden mit fester Rueckmeldung (harness/repair_hints.fault_message). Graph: bis
        MAX_GRAPH_REPAIRS Runden. Zusammen hoechstens MAX_SEED0_REPAIRS (jede Runde kostet einen Seed-0-Sandboxlauf, Minuten).
        Reihenfolge je Runde: erst Fehler-Check, dann Graph-Check des Codes, der den Fehler-Check bestanden hat.
        Rueckgabe: (v0, Hinweis, Code, Ablation des Codes oder None, Hinweise fuer das Urteil, verworfene Code-Hashes,
        Fehler-Runden gesamt). graph=False: nur Fehler-Check (Kandidatenphase, A1); fault_rounds0: schon verbrauchte Fehler-Runden.
        Liefert ein reparierter Code keinen gueltigen Seed 0, gilt der letzte lauffaehige Code. CostLimitExceeded laeuft durch."""
        def h(c: str) -> str:
            return hashlib.sha256(c.encode()).hexdigest()[:12]

        notes: list[str] = []
        discarded: set[str] = set()
        ab: dict | None = None
        fault_rounds, graph_rounds = fault_rounds0, 0

        def budget(kind_rounds: int, kind_max: int) -> bool:
            return kind_rounds < kind_max and fault_rounds + graph_rounds < MAX_SEED0_REPAIRS

        def repair(kind: str, rnd: int, message: str, head: str, ev: dict) -> tuple[bool, str, float, str | None]:
            """Ein Reparaturversuch. Rueckgabe: (ok, code, v0, note). ok=False: Rueckfall auf den bisherigen Code."""
            from_hash = h(code)
            try:
                new_code = self.repair_code(plan, task, code, message)
            except ValueError as exc:   # Modell lieferte keinen Code-Block
                self._emit(f"{kind}_repair", result="no_code", fallback_hash=from_hash, **ev)
                notes.append(f"{head}; Reparatur ohne Code-Block ({exc}), Rueckfall auf Code {from_hash}")
                return False, code, v0, note0
            nv, nnote, new_code = self._run_seed(claim, task, snapshot, dossier, run_dir, plan, new_code, seed, candidate)
            if nv is None:
                self._emit(f"{kind}_repair", result="crash", fallback_hash=from_hash, to_hash=h(new_code), **ev)
                notes.append(f"{head}; neuer Code lief nicht ({nnote}), Rueckfall auf Code {from_hash}")
                return False, code, v0, note0
            discarded.add(from_hash)
            return True, new_code, nv, nnote

        while True:
            legacy_feedback = getattr(self, "allow_test_feedback", False)
            fault = self._fault_check(claim, dossier, h(code), seed, v0) if legacy_feedback else None
            val_fault = None if fault else self._val_fault_check(claim, h(code), seed)   # M1: Zufallsniveau auf Validierung
            if (fault or val_fault) and budget(fault_rounds, MAX_FAULT_REPAIRS):
                fault_rounds += 1
                from_hash = h(code)
                diag = getattr(self, "_diag", {}).get((from_hash, seed)) or {}
                pred_std = diag.get("pred_std") if isinstance(diag, dict) else None
                message = (fault_message(claim.metric, v0, claim.target, dossier.baselines, pred_std) if fault
                           else val_fault_message() if claim.metric == 'accuracy' else
                           'The validation predictions are worse than a trivial validation baseline. Check the implementation against the complete author reference. Keep its method and training protocol; choose any uncertain setting using validation only.')
                fault = fault or val_fault
                ev = {"claim_id": claim.id, "runde": fault_rounds, "from_hash": from_hash, "measured": v0,
                      "basis": 'test' if legacy_feedback else 'validation',
                      "baselines": {k: v.get(claim.metric) for k, v in (dossier.baselines or {}).items() if isinstance(v, dict)}}
                if not legacy_feedback:
                    val = self.__dict__.get('_val', {}).get((from_hash, seed), {})
                    ev.update(measured=val.get('value'), metric=val.get('metric'), baselines=val.get('baselines', {}))
                head = f"Fehler-Reparatur Runde {fault_rounds}: Code {from_hash} schlechter als triviale Baseline ({fault})"
                ok, code, v0, note0 = repair("fault", fault_rounds, message, head, ev)
                if not ok:
                    break
                self._emit("fault_repair", result="repariert", to_hash=h(code), **ev)
                notes.append(f"{head}; neuer Code {h(code)}")
                continue
            if not legacy_feedback or not task.ablation_cmd or not graph:
                break
            ab = self.ablation_check(snapshot, code, seed, task, claim.metric, v0)
            self._emit("ablation", claim_id=claim.id, runde=fault_rounds + graph_rounds, code_hash=h(code), **ab)
            edge = None
            if ab["verdict"] == "Graph genutzt" and getattr(task, "edge_probe", None):   # B2: Kanten in beiden Richtungen?
                edge = self.edge_direction_check(snapshot, code, seed, task, claim.metric)
                self._emit("edge_direction_check", claim_id=claim.id, runde=fault_rounds + graph_rounds, code_hash=h(code), **edge)
                with self._lock:
                    dossier.edge_direction.setdefault(claim.id, []).append({"code_hash": h(code), **edge})
            if edge is not None and edge["verdict"] == EDGE_ONE and budget(graph_rounds, MAX_GRAPH_REPAIRS):
                graph_rounds += 1
                from_hash = h(code)
                ev = {"claim_id": claim.id, "runde": graph_rounds, "from_hash": from_hash, "kind": "edge_direction"}
                head = f"Graph-Reparatur Runde {graph_rounds}: Code {from_hash} nutzt Kanten nur in einer Richtung (Probe)"
                ok, code, v0, note0 = repair("graph", graph_rounds, EDGE_DIRECTION_MESSAGE, head, ev)
                if not ok:
                    break
                self._emit("graph_repair", result="repariert", to_hash=h(code), **ev)
                notes.append(f"{head}; neuer Code {h(code)}")
                continue
            if ab["verdict"] == "Graph nicht genutzt" and budget(graph_rounds, MAX_GRAPH_REPAIRS):
                graph_rounds += 1
                from_hash = h(code)
                ev = {"claim_id": claim.id, "runde": graph_rounds, "from_hash": from_hash,
                      "acc_full": ab["accuracy_full"], "acc_ablated": ab["accuracy_ablated"]}
                head = (f"Graph-Reparatur Runde {graph_rounds}: Code {from_hash} nutzt den Graph nicht "
                        f"(Genauigkeit mit Kanten {ab['accuracy_full']:.4f}, ohne {ab['accuracy_ablated']:.4f})")
                ok, code, v0, note0 = repair("graph", graph_rounds,
                                             GRAPH_REPAIR_MESSAGE.format(full=ab["accuracy_full"], ablated=ab["accuracy_ablated"]), head, ev)
                if not ok:
                    break
                self._emit("graph_repair", result="repariert", to_hash=h(code), **ev)
                notes.append(f"{head}; neuer Code {h(code)}")
                continue
            break
        discarded.discard(h(code))   # Reparatur kann denselben Code liefern; der finale Code zaehlt nie als verworfen
        with self._lock:   # Kandidaten laufen parallel; Zaehler ueber alle Kandidaten summiert
            if fault_rounds - fault_rounds0:
                dossier.fault_repairs[claim.id] = dossier.fault_repairs.get(claim.id, 0) + fault_rounds - fault_rounds0
            if graph_rounds:
                dossier.graph_repairs[claim.id] = dossier.graph_repairs.get(claim.id, 0) + graph_rounds
        return v0, note0, code, ab, notes, discarded, fault_rounds

    def _run_seed(
        self, claim: Claim, task: Task, snapshot: str, dossier: Dossier, run_dir: Path,
        plan: str, code: str, seed: int, candidate: int | None = None,
    ) -> tuple[float | None, str | None, str]:
        """Ein Seed mit Reparaturschleife. Rueckgabe: (Messwert oder None, Hinweis, letzter Code)."""
        summary = None
        timed_out_before = False   # F2 (c): nach einem Timeout haengt jede weitere Reparatur dieses Seeds den Hinweis "schnell bleiben" an
        for n in range(1, self.settings.max_attempts_per_claim + 1):
            code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
            (run_dir / f"{claim.id}_{code_hash}.py").write_text(code)
            self._announce_code(claim, code)
            t0 = time.time()
            product = getattr(self, "graph_repair", False)
            extra_kw = {"with_val": True} if product and seed == 0 and getattr(task, "val_eval_cmd", "") else {}   # Val nur Seed 0, nur Produktpfad
            issues = placeholder_issues(code, getattr(self, "_repo_files", {})) if product else []
            if issues:
                self._emit("method_check", claim_id=claim.id, code_hash=code_hash, result="incomplete", issues=issues)
                branch, res = "not-started", {"status": "ungueltig", "static": True, "method_incomplete": True,
                                              "detail": "Incomplete method: " + " ".join(issues)}
            else:
                branch, res = self._attempt(snapshot, code, seed, task, claim.metric, measure_coverage=seed == 0, **extra_kw)
            status, value = res["status"], res.get("measured")
            if res.get('infrastructure'):
                raise SandboxSetupError(res['detail'])
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
                    diagnostics=res.get("diagnostics"), candidate=candidate, val=res.get("val"), metrics_droplast=res.get("metrics_droplast"),
                ))
                self.__dict__.setdefault("_diag", {})[(code_hash, seed)] = res.get("diagnostics")
                if res.get("val"):
                    self.__dict__.setdefault("_val", {})[(code_hash, seed)] = res["val"]
            print(f"[{claim.id}] seed={seed} versuch={n} {status} measured={value} ({time.time()-t0:.0f}s)",
                  flush=True)
            self._emit("attempt", claim_id=claim.id, seed=seed, n=n, branch_id=branch, parent_snapshot=snapshot,
                       code_hash=code_hash, status=status, measured=value, seconds=round(time.time() - t0, 1),
                       error=(summary or "")[-400:] or None, stdout_tail=res.get("stdout_tail"),
                       diagnostics=res.get("diagnostics"),
                       **({"method_incomplete": True} if issues else {}),
                       **({"candidate": candidate} if candidate is not None else {}),
                       **({"val": res["val"]} if res.get("val") else {}))
            if res.get("baselines"):
                dossier.baselines = res["baselines"]
            if value is not None:
                return value, None, code
            if seed != 0 and not getattr(self, 'allow_test_feedback', False):
                return None, f'Seed {seed}: selected code did not finish; no per-seed rewrite was made, to keep the method identical across seeds.', code
            if status == "timeout":
                if not product:
                    return None, f"Seed {seed}: Timeout, kein Reparaturversuch", code
                summary = timeout_report(res, code, self._train_timeout())   # Produktpfad: schneller machen (F2: mit Stack und Ausgabe)
                timed_out_before = True
            if n < self.settings.max_attempts_per_claim:
                hints = self._error_hints(claim, summary) if status == "fehler" and product else ""
                try:
                    self._emit("repair", claim_id=claim.id, seed=seed, candidate=candidate, from_hash=code_hash, error=(summary or "")[-400:])
                    extra = (repair_hint(summary) + (STAY_FAST_NOTE if timed_out_before and status != "timeout" else "")) if product else ""
                    code = self.repair_code(plan, task, code, (summary or "") + extra + hints)
                    self._emit("repair_result", claim_id=claim.id, seed=seed, candidate=candidate, from_hash=code_hash, to_hash=hashlib.sha256(code.encode()).hexdigest()[:12])
                except ValueError as exc:   # Modell lieferte trotz Wiederholung keinen Code
                    if not product:
                        return None, f"Seed {seed}: Reparatur ohne Code-Block ({exc}), Abbruch nach {n} Versuchen", code
                    try:   # Produktpfad: Code neu aus dem Plan schreiben statt abbrechen
                        self._emit("rewrite", claim_id=claim.id, seed=seed, candidate=candidate, from_hash=code_hash)
                        code = self.write_code(plan, task)
                    except ValueError as exc2:
                        return None, f"Seed {seed}: Reparatur und Neuschreiben ohne Code-Block ({exc2}), Abbruch nach {n} Versuchen", code
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
    def run(self, paper_id: str, source: str = "benchmark") -> Dossier:
        """Load a benchmark, curated catalog entry, or persisted server-owned live claim."""
        if source == "catalog":
            paper, claims = load_catalog_claims(paper_id)
        elif source == "live_catalog":
            from . import live_catalog
            from .catalog import build_paper
            entry = live_catalog.find_entry(paper_id)
            if entry is None:
                raise ValueError("Saved discovery claim not found")
            paper, claims = build_paper(entry)
        elif source == "benchmark":
            paper, claims = load_claims(paper_id)
        else:
            raise ValueError(f"Unknown paper source: {source!r}")
        self._paper_id = paper_id   # S4: Fixture-Schluessel je Einheit (Steckbrief-Modus)
        self._warnings = list(paper.get("warnings", []))
        self.details = []   # pro Lauf neu, sonst erbt ein Paper ohne Rolle 4 die Details des vorigen
        if self.claim_ids:
            claims = [c for c in claims if c.id in self.claim_ids]
        task = TASKS[paper["dataset"]]()
        if getattr(self, 'independent_selection', False):
            task = task.with_independent_selection()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        contract = task.selection_info()
        if contract:
            self._warnings.append(f"Independent candidate selection uses {contract['selection_samples']} of {contract['total_validation_samples']} "
                                  "validation windows/nodes with hidden answers. Checkpoints use the remaining visible validation data. "
                                  "Training/test splits are unchanged; this selection protocol differs from older runs.")
        run_dir = REPO_ROOT / "runs" / paper_id / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        self.last_run_dir = run_dir
        self._run_dir = run_dir
        self.paper_text = None    # pro Lauf neu holen (live) statt den Text des vorigen Laufs weiterzuverwenden
        self._sources = []
        self._code_sent = set()
        self._claim_ctx = {}
        self._repo_files = {}
        self._repo_listing = []
        self._smoke_ok = set()
        self._setup_missing: list[str] = []
        self.events = EventLog(run_dir / "events.jsonl")
        self._set_author_code(True)   # jeder Lauf startet im angeforderten Modus, ein Rueckfall des vorigen Laufs gilt nicht weiter
        self.spec = None
        requested = MODE_AUTHOR_CODE if getattr(self, "code_grounded", False) else MODE_PAPER
        self._emit("run_start", paper_id=paper_id, title=paper["title"], mode_requested=requested, use_tavily=self.use_tavily,
                   use_setup_details=self.use_setup_details, paper_only=self.paper_only,
                   selection_policy='test_feedback_diagnostic' if getattr(self, 'allow_test_feedback', False) else 'validation_only',
                   selection_contract=task.selection_info(),
                   models={"fast": self.settings.llm.model_fast, "code": self.settings.llm.model_code,
                           "reasoning": self.settings.llm.model_reasoning},
                   seeds=self.seeds or paper.get("seeds", DEFAULT_SEEDS), claims=[c.id for c in claims])
        for w in self._warnings:
            self._emit("warning", text=w)
        self._cost_base = self.llm.total_cost_usd
        if hasattr(self.llm, "begin_run"):
            self.llm.begin_run()
        if self.use_tavily:
            self.resolve_targets(paper, claims)
        paper_head = self._paper_summary_round(paper, claims)   # nach Rolle 2; nur Bericht
        repo_info = None
        extra_details = discovery_details(paper, self.paper_text.text if self.paper_text is not None else None)
        if self.use_setup_details and self.paper_text is not None:
            self._paper_title = paper.get("title")
            docs, repo_info = self._gather_docs(claims[0].text, paper["dataset"])
            if getattr(self, "setup_v2", False):
                self.details, dropped, setup_missing, v2_info = extract_setup_details_v2(
                    self.llm, docs, claims[0].text, paper["dataset"], self._paper_title)
                self._setup_missing = setup_missing
            else:
                self.details, dropped = extract_setup_details(
                    self.llm, docs, claims[0].text, paper["dataset"]
                )
                v2_info = None
            self.details = self.details + extra_details
            by_src: dict[str, int] = {}
            for d in self.details:
                by_src[d.source.split("@")[0]] = by_src.get(d.source.split("@")[0], 0) + 1
            print(f"Setup-Details (Rolle 4): {len(self.details)} belegt, {dropped} verworfen, "
                  f"nach Quelle {by_src}", flush=True)
            self._emit("setup_details", n=len(self.details), dropped=dropped, by_source=by_src,
                       repo=repo_info, details=[dataclasses.asdict(d) for d in self.details])
            if v2_info is not None:
                excluded = [d for d in self.details if getattr(d, "excluded", "")]
                self._emit("setup_v2", family=v2_info["family"], missing_first=v2_info["missing_first"], missing=setup_missing,
                           targeted_docs=v2_info["targeted_docs"], added=v2_info["added"],
                           bodies=v2_info["bodies"], literals=v2_info["literals"],
                           excluded=[{"topic": d.topic, "value": d.value, "reason": d.excluded} for d in excluded])
            # Autorencode angefordert, aber kein Repo gelesen: kein Steckbrief-Aufruf ohne Ausschnitt, der Rueckfall (_resolve_mode) folgt
            if getattr(self, "spec_mode", False) and not (getattr(self, "code_grounded", False) and repo_info is None):
                self.spec = build_spec(
                    lambda system, user: self._ask("extract", system, user, max_tokens=SPEC_MAX_TOKENS, temperature=0.0).content,
                    self.details, docs, claims[0].text, paper["dataset"], numeric_only=getattr(self, "code_grounded", False),
                    repo_listing=getattr(self, "_repo_listing", None)
                )
                self._emit("spec", **spec_as_dict(self.spec))
                print(f"Steckbrief (S1): {len(self.spec.items)} Werte erzwungen/pruefbar, {len(self.spec.missing)} fehlend", flush=True)

        elif extra_details:
            self.details = extra_details
        mode, fallback_reason = self._resolve_mode(repo_info)
        if fallback_reason:
            self._warn(f"Autorencode nicht verfügbar ({fallback_reason}): Lauf als reiner Nachbau aus dem Paper.")
        self.mode = mode
        self._emit("mode", mode=mode, requested=requested, fallback_reason=fallback_reason or None,
                   repo=(repo_info or {}).get("url"), commit=(repo_info or {}).get("commit"),
                   excerpt_docs=(self.spec.excerpt_docs if mode == MODE_AUTHOR_CODE else []))
        dossier = Dossier(
            paper_id=paper_id, claims=claims, isolation=list(ISOLATION), repo=repo_info, sources=list(self._sources), mode=mode,
            setup_details=[dataclasses.asdict(d) for d in self.details], warnings=list(self._warnings), summary=paper_head,
            setup_missing=list(self._setup_missing) if getattr(self, "setup_v2", False) else [],
            spec=spec_as_dict(getattr(self, "spec", None)) if getattr(self, "spec", None) is not None else None,
            selection_policy='test_feedback_diagnostic' if getattr(self, 'allow_test_feedback', False) else 'validation_only',
            selection_contract=task.selection_info(),
        )

        label = re.sub(r"[^a-z0-9-]", "-", f"{paper_id}-{ts}".lower())
        print(f"Basis-Snapshot bauen ({label}) ...")
        contract_key = hashlib.sha256(json.dumps([task.image, task.setup_cmds, task.selection_contract], sort_keys=True).encode()).hexdigest()[:12]
        key = paper["dataset"] + '-' + contract_key
        if key not in self._snapshots:
            try:
                self._snapshots[key] = self.sandbox.base_snapshot(task.image, task.setup_cmds, label=label)
            except SandboxSetupError as exc:
                self._infrastructure_failure(dossier, ts, run_dir, exc)   # Infrastrukturfehler, kein Urteil: Dossier und Zeugnis ohne Urteile, dann weiter nach oben
                raise
        snapshot = self._snapshots[key]
        try:
            for claim in claims:
                self.run_claim(paper, claim, task, snapshot, dossier, run_dir)
            self._community_round(paper, dossier)   # erst nach allen Urteilen; nur Bericht
            self._objection_round(paper, dossier)   # nach dem Community-Check; nur Bericht
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

    # Standard: Produktpfad mit Nachbau aus Paper und Autorencode. `--no-author-code` = die bisherige schlichte CLI-Konfiguration
    # (reiner Paper-Nachbau, ohne Produktschalter); `--paper-only`/`--no-tavily` lesen kein Repo, der Lauf faellt dann selbst zurueck.
    kw = {} if "--no-author-code" in sys.argv else product_kwargs(True)
    orchestrator = None
    try:
        orchestrator = Orchestrator(
            use_tavily="--no-tavily" not in sys.argv,
            fresh="--fresh" in sys.argv,
            setup_details="--no-setup-details" not in sys.argv,
            paper_only="--paper-only" in sys.argv,
            tavily_mode="fixture" if "--offline" in sys.argv else None,
            **kw,
        )
        orchestrator.run(paper_id, source="catalog" if "--catalog" in sys.argv else "benchmark")
    except BillingBlocked as exc:
        sys.exit(f"ABBRUCH: {exc}")
    except SandboxSetupError as exc:
        sys.exit(f"ABBRUCH: Sandbox-Setup fehlgeschlagen (Infrastruktur): {exc}")
    finally:
        close = getattr(orchestrator, 'close', None)
        if close:
            close()


if __name__ == "__main__":
    main()
