"""Modus Nachlauf, Meilenstein M3: Nemotron liefert Startbefehl, Seed-Weg, Angaben zum Test-Loader und den Adapter.

Alles Pruefen bleibt deterministisch (nachlauf.py, harness/independence.py): das Modell liest und schreibt, ein Programm prueft.
Regeln: benchmark/experiments/nachlauf_v1.md (final). Bauplan: docs/plan_nachlauf.md.

Ablauf (`NachlaufAgent.run`):
1. Repo auf festem Commit lesen (Lese-Snapshot des Orchestrators), das Modell waehlt Dateien aus der Dateiliste.
2. Aufruf "Startbefehl" (JSON, jede Angabe mit woertlichem Zitat, deterministisch gegen die Datei geprueft; nicht belegt -> einmal neu fragen,
   dann Rueckfall).
3. Erkundung (Repo-Seed): Training mit Reparatur (P1-Pruefung vor jedem Lauf), dann Adapter schreiben und pruefen (statisch, SMOKE=1, voller
   Lauf, P3, P2) mit festen Rueckmeldungen ohne jede Zahl.
4. Der feste Stand (Befehl + Adapter) laeuft ueber alle Seeds mit `nachlauf.run_unit` (8 Versaetze je Seed) und liefert das Dossier.

Prompts sind allgemein: kein Repo-, Modell- oder Dateiname und kein Schalter aus einem bestimmten Repo; Beispiele sind erfunden und so
gekennzeichnet. Der Agent erfaehrt nie eine Zahl aus Harness, P2 oder P3 (nur feste Saetze). tests/test_nachlauf_agent.py bewacht beides.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import nachlauf as nl
from .claims_source import _norm
from .dossier import Attempt, Claim, Dossier
from .events import EventLog
from .harness import independence
from .harness.code_check import find_violations
from .llm import LLMResponse
from .loop import extract_code
from .repo_source import _last_string_list, imported_files
from .setup_details import quote_in_text

MAX_FILES = 14                 # vom Modell gewaehlte Dateien
MAX_FILE_CHARS = 16_000        # je Datei im Prompt
MAX_TOTAL_CHARS = 120_000      # alle Dateien eines Prompts
MAX_LISTING = 600
MAX_DEP_FILES = 6
EXPLORE_OFFSETS = 2            # P2-Versaetze waehrend der Erkundung (die Endlaeufe nutzen alle, Standard 8)
BASE_PIP = ["pandas", "scikit-learn", "matplotlib"]   # Standardpakete der Datenlader dieser Familie (nicht in den Task-Paketen)
LENGTH_RETRY_TOKENS = (None, 12_000, 16_000)
ONLY_JSON_NOTE = "\n\nAnswer with exactly one JSON object and no other text."
ONLY_CODE_NOTE = "\n\nReturn only the complete Python script in one code block, no explanations."
DEPENDENCY_FILE_RE = re.compile(r"(^|/)(requirements[\w.-]*\.txt|environment[\w.-]*\.ya?ml|setup\.py|setup\.cfg|pyproject\.toml|Pipfile)$")
PKG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHELL_META_RE = re.compile(r"[;|&<>`]|\$\(|\$\{|\n")

# ---- Prompts (allgemein; Beispiele erfunden) --------------------------------------------------------------------------------

FILE_SELECT_SYSTEM = "You pick files from a code repository. Keep reasoning brief."
FILE_SELECT_USER = (
    "Task: select_files\n"
    "A reproducer wants to run the original training code of a repository to check this result: {claim} (dataset: {dataset}).\n"
    "From the file list below, pick up to {n} files that are needed to: (1) find the documented command that trains the model of this result "
    "on this dataset (shell scripts, documentation with commands), (2) read the entry script and its argument definitions, "
    "(3) read the file that defines the model of the result, (4) read the code that builds the test data loader, (5) read the training and "
    "test loop (where the weights are saved and where the authors' own test metric is printed). Prefer files for this model and skip files "
    "that only concern other models of the repository. Answer with a JSON list of exact paths from the list and end with it.\n\n{listing}"
)

COMMAND_SYSTEM = (
    "You read a code repository and extract how its authors run one experiment. Every statement about the repository needs a verbatim quote "
    "(copied character for character) from one of the provided files. Reply with exactly one JSON object and no other text."
)
COMMAND_FORMAT = """\
{
  "command": "single line, starts with python, run from the repository root",
  "command_source": {"file": "path of the file the command comes from", "quote": "the verbatim line(s) of that file"},
  "seed": {"kind": "flag | copy | none", "flag": "switch name, only for kind flag", "file": "entry script path, only for kind copy",
           "line": "verbatim line of the entry script that sets the random seed, only for kind copy"},
  "checkpoint_glob": "path pattern relative to the repository root where the training run saves the weights, e.g. out/*/weights.bin",
  "test_loader": {"drop_last": {"value": true, "file": "path", "quote": "verbatim code that builds the test loader and sets drop_last"},
                  "batch_size": {"value": 32, "file": "path", "quote": "verbatim text that fixes the batch size used for testing"}},
  "input_len": 0, "output_len": 0,
  "needs_time_features": {"value": false, "file": "path of the model file", "quote": "verbatim code of the model's forward function that uses time-stamp features, or null"},
  "model_file": "path of the file that defines the model class", "training_file": "path of the file with the training and test loop"
}"""
COMMAND_USER = (
    "Task: start_command\n"
    "Result to check: {claim} (dataset: {dataset}).\n"
    "Find how the authors run this experiment and answer with one JSON object of this shape:\n{format}\n\n"
    "Rules:\n"
    "- The command must be the one the authors document for this model, dataset and forecast horizon. Change nothing except device and path switches. "
    "Never use a switch that trains on all data or uses test data in training.\n"
    "- One line starting with python, no pipes, no && or ;, no output redirection, no environment variable prefix.\n"
    "- Quotes must be copied exactly from the provided files. Use null for anything the repository does not state; do not guess.\n"
    "- seed.kind: flag if the entry script defines a command-line switch for the random seed; copy if the seed is a fixed number in one line of the entry "
    "script (give that exact line); none if neither exists.\n"
    "- input_len and output_len are the input length and forecast length given in your command.\n"
    "- needs_time_features is true only if the model's forward function actually uses the time-stamp features of the batch.\n"
    "Invented example of an answer's style (not from any real repository): command `python -u main_demo.py --task demo --horizon 24`, "
    "quote `python -u main_demo.py --task demo --horizon 24 --lr 0.01`.\n\n"
    "Repository file list (first entries):\n{listing}\n\nFiles:\n{files}"
)
COMMAND_REASK = (
    "Task: start_command_requote\n"
    "Result to check: {claim} (dataset: {dataset}).\n"
    "These items of your previous answer could not be verified as verbatim quotes of the named files: {items}.\n"
    "Reply with one JSON object that contains only these items, in the same structure as before, each with an exact quote copied from the named file, "
    "or null if the repository does not state it.\n\nFiles:\n{files}"
)
REPAIR_COMMAND_USER = (
    "Task: repair_command\n"
    "Result to check: {claim} (dataset: {dataset}).\n"
    "The training run with this command did not produce a usable result.\nCommand: {command}\n\nProblem:\n{problem}\n\n"
    "Answer with one JSON object: {{\"command\": ..., \"command_source\": {{\"file\": ..., \"quote\": ...}}, \"checkpoint_glob\": ..., "
    "\"install\": [package names]}}. You may change the command (still the authors' documented one, with fixes only where the error requires), the "
    "checkpoint path pattern, or ask to install a missing package, but only a package that is named in the repository's dependency files. "
    "Use an empty list for install if none is needed.\n\nFiles:\n{files}"
)

ADAPTER_SYSTEM = (
    "You write a single self-contained Python script that loads a trained model of a code repository and writes its predictions. "
    "Reply with exactly one ```python code block and no other prose."
)
ADAPTER_CONTRACT = """\
Contract of predict.py (run without network access, CPU only, from the repository root; the repository is importable, put its root on sys.path):
- Read /data/etth1_public.npz with numpy.load. Keys: train [8640, 7], val [2880, 7] (not needed) and test_inputs, shape [n, 720, 7], float32, already z-score
  standardised with the mean and standard deviation of the training rows. Take n from the file; predict ALL windows in it. Window i is independent of every
  other window.
- Use only the last L steps of each window as model input, where L is the input length of the model (the input length of the start command).
- For every window predict the next 96 hourly steps of all 7 channels, in the same standardised units.
- Load the model class from the repository and the weights from the file named by the environment variable NACHLAUF_CKPT. The weights were written by the
  repository's own training code; check in the training file whether it saved a state dict or a whole model. Build the model configuration with an
  argparse.Namespace (or a small class) that has every attribute the model reads, set to the values of the start command or the entry script defaults.
- Put the model in evaluation mode and run it under torch.no_grad(). If the model's forward function needs extra inputs that do not change the prediction, pass dummy tensors.
- Write /output/predictions.csv: header `id,prediction`, one row per value, id = "{window}_{step}_{channel}" (window 0..n-1, step 0..95, channel 0..6),
  prediction = float. Write it efficiently (numpy to text), not with per-row Python objects.
- If the environment variable SMOKE equals 1, predict only the first 8 windows and write only those rows.
- Do not read any other data file, do not use the network, and do not use subprocess, exec, eval, compile, importlib or URLs in the code. A static check rejects
  these and also the words pip, git, curl, wget and download anywhere in the file, including comments.
"""
ADAPTER_USER = (
    "Task: write_adapter\n{contract}\nStart command of the authors: {command}\n\n"
    "Model file(s):\n{model_files}\n\nTraining and test loop file:\n{training_file}\n\nEntry script (argument definitions):\n{entry}\n\n"
    "Data description:\n{data}\n\nWrite predict.py."
)
ADAPTER_REPAIR_USER = (
    "Task: repair_adapter\n{contract}\nStart command of the authors: {command}\n\n"
    "Model file(s):\n{model_files}\n\nTraining and test loop file:\n{training_file}\n\n"
    "This script was rejected:\n```python\n{code}\n```\nProblem: {problem}\n\nReturn the corrected full script."
)
DATA_DESCRIPTION = (
    "Dataset ETTh1 (electricity transformer temperature, hourly), 7 channels, multivariate forecasting; input windows of 720 hourly steps, forecast horizon 96 steps; "
    "values standardised with the mean and standard deviation of the first 12 months."
)

# Feste Rueckmeldungen: nie eine Zahl aus Harness, P2 oder P3.
P3_SENTENCE = ("The predictions do not match the authors' own evaluation on the same windows. Check input length, normalisation flags, "
               "model arguments and window order.")
P2_SENTENCE = ("The prediction for a window changed when other windows were removed from the file. Each prediction must depend only on its own window; "
               "put the model in evaluation mode.")
P1_SENTENCE = "The switch {flag} is not allowed: it trains on all data or uses test data in training. Use a command without it."
COMMAND_SHAPE_SENTENCE = ("The command must be a single python command line without pipes, redirects, environment prefixes, && or ;, "
                          "run from the repository root.")
PACKAGE_SENTENCE = ("The package {pkg} is not named in the repository's dependency files, so it must not be installed. "
                    "Only packages named there may be installed.")


class AgentAnswerError(RuntimeError):
    """Die Antwort des Modells blieb nach den Wiederholungen unbrauchbar (kein JSON, kein Code, abgeschnitten)."""


class BudgetStop(RuntimeError):
    """Kostenstopp: vor dem naechsten LLM-Aufruf geprueft (Kosten dieses Prozesses UND Store)."""


# ---- Kosten ------------------------------------------------------------------------------------------------------------------

def store_gross(path: Path | None, vat_rate: float) -> float:
    """Bruttokosten der fertigen Laeufe im Store (Zeilen mit cost_usd_run netto)."""
    if path is None or not Path(path).exists():
        return 0.0
    total = 0.0
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                total += float(json.loads(line).get("cost_usd_run") or 0.0)
            except (ValueError, TypeError):
                continue
    return total * (1.0 + vat_rate)


class CostGuard:
    """Stopp vor jedem LLM-Aufruf: (Kosten dieses Prozesses, soweit noch nicht im Store) + Store >= Grenze, und Kosten dieses Laufs >= MAX_COST_PER_RUN."""

    def __init__(self, llm, store: Path | None = None, limit_gross: float | None = None, vat_rate: float = 0.19, per_run_gross: float | None = None):
        self.llm, self.store, self.limit, self.vat, self.per_run = llm, store, limit_gross, vat_rate, per_run_gross
        self._stored_net = 0.0      # Kosten dieses Prozesses, die schon als Zeile im Store stehen (nicht doppelt zaehlen)
        self._run_base = 0.0

    def start_run(self) -> None:
        self._run_base = self.llm.total_cost_usd

    def note_stored(self, net_usd: float) -> None:
        self._stored_net += net_usd

    def run_net(self) -> float:
        return self.llm.total_cost_usd - self._run_base

    def total_gross(self) -> float:
        return store_gross(self.store, self.vat) + (self.llm.total_cost_usd - self._stored_net) * (1.0 + self.vat)

    def check(self) -> None:
        if self.limit is not None and self.total_gross() >= self.limit:
            raise BudgetStop(f"Kostenstopp: {self.total_gross():.4f} $ brutto (Prozess und Store) >= {self.limit:.2f} $")
        if self.per_run is not None and self.run_net() * (1.0 + self.vat) >= self.per_run:
            raise BudgetStop(f"MAX_COST_PER_RUN erreicht ({self.run_net() * (1.0 + self.vat):.4f} $ brutto)")


# ---- LLM-Zugang ---------------------------------------------------------------------------------------------------------------

def parse_json_object(text: str, required: str | None = None) -> dict | None:
    """Letztes JSON-Objekt der obersten Ebene in der Antwort (optional mit Pflichtschluessel). Das Modell denkt im Antworttext, das Ergebnis steht
    am Ende; verschachtelte Objekte zaehlen nicht als eigene Antwort."""
    dec = json.JSONDecoder()
    text = text or ""
    pos, found = 0, None
    while True:
        i = text.find("{", pos)
        if i < 0:
            break
        try:
            value, end = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            pos = i + 1
            continue
        if isinstance(value, dict) and (required is None or required in value):
            found = value
        pos = i + end
    return found


class Asker:
    """Alle LLM-Aufrufe des Modus: Kostenstopp vorher, abgeschnittene Antworten (finish_reason length) nie verwerten, Protokoll aller Prompts."""

    def __init__(self, llm, guard: CostGuard, events: EventLog | None = None, log=print):
        self.llm, self.guard, self.events, self.log = llm, guard, events, log
        self.calls: list[dict] = []
        self.prompts: list[dict] = []      # {purpose, system, user}: nur fuer Tests und Diagnose, nie im Dossier

    def _once(self, purpose: str, role: str, system: str, user: str, max_tokens, temperature: float, n: int) -> LLMResponse:
        self.guard.check()
        resp = self.llm.complete(role, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 temperature=temperature, max_tokens=max_tokens, timeout_s=400)
        self.prompts.append({"purpose": purpose, "system": system, "user": user})
        self.calls.append({"purpose": purpose, "role": role, "try": n, "cost_usd": round(getattr(resp, "cost_usd", 0.0), 5),
                           "finish_reason": getattr(resp, "finish_reason", None), "cached": bool(getattr(resp, "cached", False))})
        return resp

    def ask(self, purpose: str, role: str, system: str, user: str, parse, note: str, max_tokens: int | None = None, temperature: float = 0.2):
        """Bis zu 3 Versuche. Abgeschnitten oder nicht lesbar: hoehere Temperatur (gleicher Prompt gaebe aus dem Cache dieselbe Antwort), hoehere
        Token-Grenze, Hinweis auf das Format."""
        last = "no answer"
        for i in range(3):
            resp = self._once(purpose, role, system, user + (note if i else ""), LENGTH_RETRY_TOKENS[i] or max_tokens, temperature + 0.3 * i, i + 1)
            if getattr(resp, "finish_reason", None) == "length":
                last = "answer truncated (finish_reason length)"
                continue
            try:
                return parse(resp.content)
            except ValueError as exc:
                last = str(exc)
        raise AgentAnswerError(f"{purpose}: {last}")

    def json(self, purpose: str, system: str, user: str, required: str | None = None, role: str = "code", max_tokens: int = 8000) -> dict:
        def parse(text: str) -> dict:
            obj = parse_json_object(text, required)
            if obj is None:
                raise ValueError("no JSON object in the answer")
            return obj
        return self.ask(purpose, role, system, user, parse, ONLY_JSON_NOTE, max_tokens)

    def code(self, purpose: str, system: str, user: str) -> str:
        return self.ask(purpose, "code", system, user, lambda t: extract_code(t, lenient=True), ONLY_CODE_NOTE, 8000)

    def files(self, purpose: str, system: str, user: str) -> list[str]:
        return self.ask(purpose, "extract", system, user, lambda t: _last_string_list(t), ONLY_JSON_NOTE, 6000, temperature=0.0)


# ---- Repo lesen ---------------------------------------------------------------------------------------------------------------

@dataclass
class RepoReading:
    url: str
    commit: str
    listing: list[str]
    files: dict[str, str] = field(default_factory=dict)
    chosen: list[str] = field(default_factory=list)


def read_repo_at_commit(sandbox, asker: Asker, url: str, commit: str, claim_text: str, dataset: str) -> RepoReading:
    """Lese-Snapshot des Orchestrators (nie ein Agenten-Branch): Klon auf festem Commit, das Modell waehlt Dateien aus der Dateiliste,
    dazu Abhaengigkeitsdateien und lokal importierte Module. Zahl und Groesse sind begrenzt (Kosten)."""
    key = re.sub(r"[^a-z0-9]+", "-", url.lower().split("github.com/")[-1]).strip("-")
    snap = sandbox.base_snapshot(
        "python:3.11-slim",
        [nl.APT, f"git clone -q {url} /repo && git -C /repo checkout -q {commit} && test \"$(git -C /repo rev-parse HEAD)\" = {commit}"],
        label=f"nachlauf-read-{key}",
    )
    b = sandbox.branch(snap)
    listing = [l for l in sandbox.run(b, "cd /repo && git ls-files", 60).stdout.splitlines() if l.strip()]
    raw = asker.files("select_files", FILE_SELECT_SYSTEM, FILE_SELECT_USER.format(
        claim=claim_text, dataset=dataset, n=MAX_FILES, listing="\n".join(listing[:MAX_LISTING])))
    valid = set(listing)
    chosen = [p for p in dict.fromkeys(raw) if p in valid][:MAX_FILES]
    extra = [p for p in listing if DEPENDENCY_FILE_RE.search(p)][:MAX_DEP_FILES]
    files: dict[str, str] = {}

    def read(paths):
        for p in paths:
            if p in files:
                continue
            try:
                files[p] = sandbox.get_file(b, f"/repo/{p}").decode(errors="replace")[:MAX_FILE_CHARS]
            except (OSError, UnicodeDecodeError, FileNotFoundError, KeyError):
                continue
    read(chosen)
    read(extra)
    read(imported_files(files, listing)[:4])
    return RepoReading(url=url, commit=commit, listing=listing, files=files, chosen=chosen)


def _render_files(files: dict[str, str], only: list[str] | None = None, budget: int = MAX_TOTAL_CHARS) -> str:
    out, used = [], 0
    for path, text in files.items():
        if only is not None and path not in only:
            continue
        chunk = f"=== FILE: {path} ===\n{text}\n"
        if used + len(chunk) > budget:
            chunk = chunk[: max(0, budget - used)]
        if not chunk:
            break
        out.append(chunk)
        used += len(chunk)
    return "\n".join(out)


# ---- Angaben pruefen (deterministisch) --------------------------------------------------------------------------------------

def _quote_ok(quote, file, files: dict[str, str], extra_hay: str = "") -> tuple[bool, str]:
    """Zitat steht woertlich (normalisiert, Auslassungen erlaubt) in der genannten Datei (oder in `extra_hay`, z. B. dem Befehl)."""
    if not isinstance(quote, str) or not quote.strip():
        return False, "no quote"
    if file in files:
        if quote_in_text(quote, _norm(files[file])):
            return True, ""
        if extra_hay and quote_in_text(quote, _norm(extra_hay)):
            return True, ""
        return False, "quote not found in the named file"
    if extra_hay and quote_in_text(quote, _norm(extra_hay)):
        return True, ""
    return False, "file not among the provided files"


def sanitize_command(command: str) -> str:
    """Log-Umleitung, nohup und Abschluss-& entfernen (gehoeren nicht zum Befehl der Autoren)."""
    return re.sub(r"\s+", " ", nl._LOG_RE.sub(" ", command or "")).strip()


def command_problem(command: str) -> str | None:
    c = sanitize_command(command)
    if not c or not re.match(r"^python3?\s", c) or SHELL_META_RE.search(c):
        return COMMAND_SHAPE_SENTENCE
    return None


def script_of(command: str) -> str:
    return next((t for t in shlex.split(command) if t.endswith(".py")), "")


def line_number(text: str, quote: str) -> int:
    """Zeile (1-basiert) der ersten Zeile des Zitats im Dateitext, 0 wenn nicht gefunden."""
    first = next((l.strip() for l in quote.splitlines() if l.strip()), "")
    for n, l in enumerate(text.splitlines(), 1):
        if first and first in l:
            return n
    return 0


def _intval(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def verify_answer(ans: dict, files: dict[str, str], only: set[str] | None = None) -> dict:
    """Prueft jede Angabe einzeln. Rueckgabe: {item: {"belegt": bool, "reason": str, "value": ...}} fuer command_source, seed,
    test_loader.drop_last, test_loader.batch_size, needs_time_features (nur `only`, falls gesetzt)."""
    out: dict[str, dict] = {}
    want = lambda k: only is None or k in only  # noqa: E731
    cmd = sanitize_command(str(ans.get("command") or ""))

    if want("command_source"):
        src = ans.get("command_source") or {}
        ok, why = _quote_ok(src.get("quote"), src.get("file"), files)
        out["command_source"] = {"belegt": ok, "reason": why, "file": src.get("file"), "quote": src.get("quote")}

    if want("seed"):
        seed = ans.get("seed") or {}
        kind = seed.get("kind")
        res: dict = {"belegt": False, "reason": "", "kind": kind}
        if kind == "none":
            res.update(belegt=True, kind="none")
        elif kind == "flag":
            flag = str(seed.get("flag") or "")
            defs = [t for t in files.values() if "add_argument" in t or "DEFINE_" in t]
            hit = next((t for t in defs if re.search(rf"""['"]{re.escape(flag)}['"]""", t)), None) if flag.startswith("--") else None
            if hit:
                m = re.search(rf"""['"]{re.escape(flag)}['"][^)]*?default\s*=\s*(-?\d+)""", hit, re.S)
                res.update(belegt=True, flag=flag, repo_seed=int(m.group(1)) if m else None)
            else:
                res["reason"] = "switch not found in an argument definition of the provided files"
        elif kind == "copy":
            path, line = seed.get("file"), str(seed.get("line") or "")
            text = files.get(path or "", "")
            matches = [l for l in text.splitlines() if l.strip() == line.strip() and line.strip()]
            ints = re.findall(r"(?<![\w.])-?\d+(?![\w.])", line)
            if path != script_of(cmd):
                res["reason"] = "seed line is not in the entry script of the command"
            elif len(matches) != 1 or len(ints) != 1:
                res["reason"] = f"seed line must occur exactly once and contain exactly one integer (found {len(matches)} lines, {len(ints)} integers)"
            else:
                actual = matches[0]
                template = actual.replace("{", "{{").replace("}", "}}")
                template = re.sub(r"(?<![\w.])-?\d+(?![\w.])", "{seed}", template, count=1)
                res.update(belegt=True, file=path, regex=re.escape(actual), template=template, repo_seed=int(ints[0]))
        else:
            res["reason"] = "seed.kind must be flag, copy or none"
        out["seed"] = res

    tl = (ans.get("test_loader") or {})
    if want("test_loader.drop_last"):
        d = tl.get("drop_last") or {}
        ok, why = _quote_ok(d.get("quote"), d.get("file"), files)
        if ok and "drop_last" not in str(d.get("quote")):
            ok, why = False, "quote does not mention drop_last"
        if ok and not isinstance(d.get("value"), bool):
            ok, why = False, "value must be true or false"
        out["test_loader.drop_last"] = {"belegt": ok, "reason": why, "value": d.get("value"), "file": d.get("file"), "quote": d.get("quote")}
    if want("test_loader.batch_size"):
        d = tl.get("batch_size") or {}
        v = _intval(d.get("value"))
        ok, why = _quote_ok(d.get("quote"), d.get("file"), files, extra_hay=cmd)
        if ok and (v is None or v <= 0 or not re.search(rf"(?<![\d.]){v}(?![\d.])", str(d.get("quote")))):
            ok, why = False, "batch size value does not appear in the quote"
        out["test_loader.batch_size"] = {"belegt": ok, "reason": why, "value": v, "file": d.get("file"), "quote": d.get("quote")}

    if want("needs_time_features"):
        d = ans.get("needs_time_features") or {}
        val = d.get("value")
        if val is False or val is None:
            out["needs_time_features"] = {"belegt": True, "reason": "", "value": False}
        else:
            ok, why = _quote_ok(d.get("quote"), d.get("file"), files)
            out["needs_time_features"] = {"belegt": ok, "reason": why, "value": bool(val) and ok, "file": d.get("file"), "quote": d.get("quote")}
    return out


def pick_files(files: dict[str, str], *names) -> list[str]:
    return [n for n in names if isinstance(n, str) and n in files]


# ---- Agent --------------------------------------------------------------------------------------------------------------------

@dataclass
class AgentRecord:
    """Alles, was das Modell geliefert hat, und was davon belegt wurde (Dossier: nachlauf[claim]["agent"])."""
    reading: dict = field(default_factory=dict)
    command_answer: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    discarded: list = field(default_factory=list)
    attempts: list = field(default_factory=list)     # Erkundung: Befehl und Adapter je Versuch
    repairs: list = field(default_factory=list)
    adapters: list = field(default_factory=list)     # {hash, status, problems}
    llm_calls: list = field(default_factory=list)
    cost_usd_net: float = 0.0
    warnings: list = field(default_factory=list)
    outcome: str = ""


class NachlaufAgent:
    def __init__(self, sandbox, llm, *, store: Path | None = None, budget_stop: float | None = None, out_dir: Path | None = None, log=print,
                 public_loader=None, rng=None, np_rng=None, max_attempts: int | None = None, parallel: int | None = None,
                 explore_offsets: int | list[int] = EXPLORE_OFFSETS):
        self.sb, self.llm, self.out_dir, self.log = sandbox, llm, Path(out_dir) if out_dir else None, log
        settings = getattr(llm, "settings", None)
        vat = getattr(settings, "vat_rate", 0.19)
        self.max_attempts = max_attempts or getattr(settings, "max_attempts_per_claim", 4)
        per_run = getattr(settings, "max_cost_per_run_usd", None)
        self.guard = CostGuard(llm, store, budget_stop, vat, per_run)
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventLog(self.out_dir / "events.jsonl" if self.out_dir else None)
        self.asker = Asker(llm, self.guard, self.events, log)
        loader = public_loader or nl._load_public_local
        cache: dict = {}

        def memo():
            if "pub" not in cache:
                cache["pub"] = loader()
            return cache["pub"]
        self.public_loader = memo      # einmal je Lauf (Erkundung und Endlauf teilen es)
        self.rng, self.np_rng, self.parallel = rng, np_rng, parallel
        self.explore_offsets = explore_offsets
        self.rec = AgentRecord()

    # -- Hilfen
    def emit(self, type: str, **data) -> None:
        self.events.write(type, **data)
        self.log(f"[{type}] " + json.dumps(data, ensure_ascii=False, default=str)[:300])

    def _save(self, name: str, text: str) -> None:
        if self.out_dir:
            (self.out_dir / name).write_text(text)

    # -- Schritt 2: Startbefehl
    def ask_command(self, claim_text: str, dataset: str, reading: RepoReading) -> tuple[dict, dict]:
        files_txt = _render_files(reading.files)
        ans = self.asker.json("start_command", COMMAND_SYSTEM, COMMAND_USER.format(
            claim=claim_text, dataset=dataset, format=COMMAND_FORMAT, listing="\n".join(reading.listing[:80]), files=files_txt), required="command")
        ver = verify_answer(ans, reading.files)
        bad = [k for k, v in ver.items() if not v["belegt"]]
        if bad:
            detail = "; ".join(f"{k} ({ver[k]['reason']})" for k in bad)
            self.emit("nachlauf_repair", kind="requote", items=bad)
            again = self.asker.json("start_command_requote", COMMAND_SYSTEM, COMMAND_REASK.format(
                claim=claim_text, dataset=dataset, items=detail, files=files_txt))
            merged = {**ans}
            for k in bad:
                top, _, sub = k.partition(".")
                new = again.get(top)
                if new is None:
                    continue
                if sub:
                    base = dict(merged.get(top) or {})
                    base[sub] = new.get(sub) if isinstance(new, dict) else None
                    merged[top] = base
                else:
                    merged[top] = new
            ver2 = verify_answer(merged, reading.files, only=set(bad))
            for k in bad:
                if not ver2.get(k, {"belegt": False})["belegt"]:
                    self.rec.discarded.append({"item": k, "first": ver[k], "second": ver2.get(k)})
            ans, ver = merged, {**ver, **ver2}
        return ans, ver

    def _spec(self, claim: Claim, paper_id: str, url: str, commit: str, ans: dict, ver: dict, reading: RepoReading,
              seed_override: dict | None = None) -> nl.UnitSpec:
        files = reading.files
        cmd = sanitize_command(str(ans.get("command") or ""))
        src = ans.get("command_source") or {}
        src_file = src.get("file") if isinstance(src.get("file"), str) else ""
        line = line_number(files.get(src_file, ""), str(src.get("quote") or "")) if ver.get("command_source", {}).get("belegt") else 0
        seed_v = ver.get("seed", {})
        if seed_v.get("belegt") and seed_v.get("kind") == "copy":
            seed = {"kind": "copy", "repo_seed": seed_v["repo_seed"], "regex": seed_v["regex"], "template": seed_v["template"]}
        elif seed_v.get("belegt") and seed_v.get("kind") == "flag":
            seed = {"kind": "flag", "flag": seed_v["flag"], "repo_seed": seed_v.get("repo_seed")}
        else:
            seed = {"kind": "none", "repo_seed": None}
        dl, bs = ver.get("test_loader.drop_last", {}), ver.get("test_loader.batch_size", {})
        if dl.get("belegt") and bs.get("belegt"):
            test_loader = {"drop_last": bool(dl["value"]), "batch": int(bs["value"]),
                           "source": f"{dl.get('file')}:{line_number(files.get(dl.get('file'), ''), str(dl.get('quote')))}"}
        elif dl.get("belegt") and dl["value"] is False:
            test_loader = {"drop_last": False, "batch": int(bs.get("value") or 0), "source": f"{dl.get('file')}:{line_number(files.get(dl.get('file'), ''), str(dl.get('quote')))}"}
        else:
            test_loader = {"drop_last": False, "batch": 0, "source": None, "belegt": False}   # P3 vergleicht mit allen Fenstern
        script = script_of(cmd)
        argfiles = [p for p, t in files.items() if "add_argument" in t]
        reqs = [p for p in files if re.search(r"(^|/)requirements[\w.-]*\.txt$", p)]
        return nl.UnitSpec(
            unit=re.sub(r"[^a-z0-9]+", "", paper_id.lower())[:24] or "unit", paper_id=f"nachlauf/{paper_id}", claim_text=claim.text, target=claim.target,
            tolerance=claim.tolerance, repo_url=url, commit=commit, command=cmd, command_source={"file": src_file, "line": line}, script=script,
            seed=seed, adapter_source="", test_loader=test_loader, argparse_files=argfiles, requirements_files=reqs, pip_packages=list(BASE_PIP),
            checkpoint_glob=str(ans.get("checkpoint_glob") or "checkpoints/*/*.pth"),
            own_metric_pattern=rf"(?im)^\s*(?:test[ _-]*)?{re.escape(claim.metric)}\s*[:=]\s*([0-9.eE+-]+)", metric=claim.metric, claim_id=claim.id,
            seed_note={"copy": "copy of the entry script, only the seed line changed (diff checked in the branch)",
                       "flag": "command-line switch of the repository", "none": "no seed method: single measurement"}[seed["kind"]],
            claim_source=claim.source,
        )

    # -- Schritt 3a: Training mit Reparatur
    def _repair_command(self, claim: Claim, dataset: str, spec: nl.UnitSpec, reading: RepoReading, problem: str) -> dict:
        ans = self.asker.json("repair_command", COMMAND_SYSTEM, REPAIR_COMMAND_USER.format(
            claim=claim.text, dataset=dataset, command=spec.command, problem=problem[-1500:], files=_render_files(reading.files)), required="command")
        return ans

    def _dependency_text(self, reading: RepoReading) -> str:
        return "\n".join(t for p, t in reading.files.items() if DEPENDENCY_FILE_RE.search(p))

    def _rebuild(self, run: nl._Run, spec: nl.UnitSpec, n: int) -> None:
        run.train_snap = self.sb.base_snapshot(nl.etth1_task().image, nl.training_setup_cmds(spec), label=f"nachlauf-{spec.unit}-r{n}-train")
        run.pred_snap = self.sb.base_snapshot(nl.etth1_task().image, nl.predict_setup_cmds(spec), label=f"nachlauf-{spec.unit}-r{n}-predict")

    def explore_training(self, run: nl._Run, claim: Claim, dataset: str, spec: nl.UnitSpec, reading: RepoReading) -> tuple[nl.UnitSpec, dict | None]:
        seed = spec.repo_seed
        problem = ""
        pending: str | None = None      # abgelehnte Reparatur (Paket): der Versuch ist verbraucht, es laeuft nichts
        for k in range(1, self.max_attempts + 1):
            run.spec = spec
            h = hashlib.sha256(spec.command.encode()).hexdigest()[:12]
            rec = {"kind": "command", "n": k, "hash": h, "command": spec.command}
            shape = command_problem(spec.command)
            p1 = nl.check_command(spec.command, run.repo_files, spec.command_source) if not shape else {"valid": True, "forbidden": [], "documented": False, "diff": []}
            result: dict | None = None
            if pending:
                problem, rec["status"], rec["rejected_by"] = pending, "ungueltig", ["package"]
                pending = None
            elif shape:
                problem, rec["status"] = shape, "ungueltig"
            elif not p1["valid"]:
                flags = sorted({h_.split(":")[0] for h_ in p1["forbidden"]})
                problem, rec["status"] = " ".join(P1_SENTENCE.format(flag=f) for f in flags), "ungueltig"   # P1: kein Trainingslauf mit diesem Befehl
                rec["rejected_by"] = ["P1"]
            else:
                result = run.train(seed)
                rec["status"] = result["status"]
                if result["status"] == "ok":
                    rec["documented"], rec["diff"] = p1["documented"], p1["diff"]
                    self.rec.attempts.append(rec)
                    return spec, result
                problem = result.get("reason") or result["status"]
            rec["reason"] = problem[-300:]
            self.rec.attempts.append(rec)
            self.emit("nachlauf_command", n=k, hash=h, status=rec["status"], documented=p1.get("documented"), rejected_by=rec.get("rejected_by"))
            if k == self.max_attempts:
                break
            self.emit("nachlauf_repair", kind="command", round=k, reason=problem[:200])
            ans = self._repair_command(claim, dataset, spec, reading, problem)
            installs = [str(x) for x in (ans.get("install") or []) if isinstance(x, str)]
            dep = _norm(self._dependency_text(reading))
            ok_pk, rejected_pk = [], []
            for pkg in installs:
                named = PKG_NAME_RE.match(pkg) and re.search(rf"(?<![\w.-]){re.escape(_norm(pkg))}(?![\w.-])", dep)
                (ok_pk if named else rejected_pk).append(pkg)
            new_cmd = sanitize_command(str(ans.get("command") or spec.command))
            ver = verify_answer({**ans, "command": new_cmd}, reading.files, only={"command_source"})
            src = ans.get("command_source") or {}
            sfile = src.get("file") if isinstance(src.get("file"), str) else spec.command_source.get("file", "")
            ln = line_number(reading.files.get(sfile, ""), str(src.get("quote") or "")) if ver.get("command_source", {}).get("belegt") else 0
            spec = dataclasses.replace(
                spec, command=new_cmd, script=script_of(new_cmd) or spec.script, checkpoint_glob=str(ans.get("checkpoint_glob") or spec.checkpoint_glob),
                command_source={"file": sfile, "line": ln} if ln else dict(spec.command_source),
                pip_packages=[*spec.pip_packages, *[p for p in ok_pk if p not in spec.pip_packages]])
            if rejected_pk:       # Paket nicht in den Abhaengigkeitsdateien: fester Satz, der naechste Versuch ist verbraucht, es laeuft nichts
                pending = " ".join(PACKAGE_SENTENCE.format(pkg=p) for p in rejected_pk)
                self.rec.repairs.append({"kind": "package_rejected", "round": k, "packages": rejected_pk})
            if ok_pk:
                self.rec.repairs.append({"kind": "package_install", "round": k, "packages": ok_pk})
                self._rebuild(run, spec, k)
        return spec, None

    # -- Schritt 3b: Adapter
    def _adapter_context(self, reading: RepoReading, ans: dict, spec: nl.UnitSpec) -> dict:
        files = reading.files
        models = pick_files(files, ans.get("model_file")) or [p for p in files if "model" in p.lower() and p.endswith(".py")][:3]
        training = pick_files(files, ans.get("training_file"))
        entry = files.get(spec.script, "")
        entry_args = "\n".join(l for l in entry.splitlines() if "add_argument" in l or "seed" in l.lower())[:8000] or entry[:8000]
        return {
            "model_files": _render_files(files, models, 30_000) or "(not available)",
            "training_file": _render_files(files, training, 30_000) or "(not available)",
            "entry": entry_args or "(not available)",
        }

    def _write_adapter(self, ctx: dict, spec: nl.UnitSpec) -> str:
        return self.asker.code("write_adapter", ADAPTER_SYSTEM, ADAPTER_USER.format(
            contract=ADAPTER_CONTRACT, command=spec.command, data=DATA_DESCRIPTION, **ctx))

    def _repair_adapter(self, ctx: dict, spec: nl.UnitSpec, code: str, problem: str) -> str:
        return self.asker.code("repair_adapter", ADAPTER_SYSTEM, ADAPTER_REPAIR_USER.format(
            contract=ADAPTER_CONTRACT, command=spec.command, code=code, problem=problem[-1500:], model_files=ctx["model_files"], training_file=ctx["training_file"]))

    def explore_adapter(self, run: nl._Run, spec: nl.UnitSpec, ckpt: bytes, own: float | None, ctx: dict) -> tuple[str | None, str]:
        """Rueckgabe (Adapter-Quelltext oder None, Grund). Je Pruefung (P3, P2) hoechstens eine Reparaturrunde mit festem Satz."""
        used = {"P3": False, "P2": False}
        code = self._write_adapter(ctx, spec)
        reason = ""
        for v in range(1, self.max_attempts + 1):
            h = hashlib.sha256(code.encode()).hexdigest()[:12]
            self._save(f"adapter_{h}.py", code)
            spec = dataclasses.replace(spec, adapter_source=code)
            run.spec = spec
            problems: list[tuple[str, str]] = []
            viol = find_violations(code)
            stage = "static"
            if viol:
                problems.append(("static", "The static check rejected the script: " + "; ".join(viol)))
            else:
                stage = "smoke"
                err = run.smoke(ckpt)
                if err:
                    problems.append(("smoke", err))
                else:
                    stage = "run"
                    p = run.predict_and_measure(ckpt, spec.repo_seed)
                    if p["status"] != "ok":
                        problems.append(("run", p.get("reason") or p["status"]))
                    else:
                        stage = "checks"
                        sc = nl.self_check(own, p["metrics"], p.get("metrics_droplast"), spec.test_loader, spec.metric)
                        offs = self.explore_offsets if isinstance(self.explore_offsets, list) else independence.random_offsets(self.explore_offsets, self.rng)
                        ind = run.independence(ckpt, p["full_pred"], offs, spec.repo_seed)
                        if sc["status"] == "ungueltig":
                            problems.append(("P3", P3_SENTENCE))
                        if not ind["ok"]:
                            problems.append(("P2", P2_SENTENCE))
            self.rec.adapters.append({"n": v, "hash": h, "stage": stage, "status": "ok" if not problems else "ungueltig", "problems": [k for k, _ in problems]})
            self.emit("nachlauf_adapter", n=v, hash=h, stage=stage, ok=not problems, problems=[k for k, _ in problems])
            if not problems:
                return code, ""
            reason = "; ".join(f"{k}: {t[:200]}" for k, t in problems)
            repeated = [k for k, _ in problems if k in used and used[k]]
            if repeated:        # nach der einen Runde erneut ungueltig: kein weiterer Reparaturversuch fuer diese Pruefung
                return None, f"{'/'.join(repeated)} still invalid after the repair round"
            for k, _ in problems:
                if k in used:
                    used[k] = True
            if v == self.max_attempts:
                break
            self.emit("nachlauf_repair", kind="adapter", round=v, problems=[k for k, _ in problems])
            self.rec.repairs.append({"kind": "adapter", "round": v, "problems": [k for k, _ in problems]})
            code = self._repair_adapter(ctx, spec, code, " ".join(t for _, t in problems))
        return None, reason or "adapter attempts exhausted"

    # -- Einstieg
    def run(self, paper_id: str, claim: Claim, repo_url: str, commit: str, *, dataset: str = "ETTh1", seeds: list | None = None,
            offsets: int | list[int] = independence.N_OFFSETS) -> Dossier:
        self.guard.start_run()
        t0 = time.time()
        claim_text = claim.text
        reading = read_repo_at_commit(self.sb, self.asker, repo_url, commit, claim_text, dataset)
        self.rec.reading = {"url": repo_url, "commit": commit, "n_tracked": len(reading.listing), "chosen": reading.chosen, "files": sorted(reading.files)}
        ans, ver = self.ask_command(claim_text, dataset, reading)
        self.rec.command_answer, self.rec.verification = ans, ver
        self._save("command_answer.json", json.dumps({"answer": ans, "verification": ver, "discarded": self.rec.discarded}, indent=2, ensure_ascii=False, default=str))
        spec = self._spec(claim, paper_id, repo_url, commit, ans, ver, reading)
        warnings = []
        if not ver.get("command_source", {}).get("belegt"):
            warnings.append("Startbefehl nicht belegt: kein woertliches Zitat in den Dateien des Repos")
        if not ver.get("seed", {}).get("belegt") or spec.seed["kind"] == "none":
            warnings.append("eine Messung, keine Streuung")
        if spec.test_loader.get("belegt") is False:
            warnings.append("Test-Loader nicht belegt: P3 vergleicht mit allen Fenstern")
        self.emit("nachlauf_command", n=0, command=spec.command, belegt={k: v["belegt"] for k, v in ver.items()}, seed_kind=spec.seed["kind"],
                  test_loader=spec.test_loader, discarded=[d["item"] for d in self.rec.discarded])

        def finish_failed(reason: str, status_note: str) -> Dossier:
            claim_, dossier, info = nl.new_dossier(spec)
            self._attach(dossier, info, warnings)
            self.rec.outcome = status_note
            for a in self.rec.attempts:
                dossier.attempts.append(Attempt(claim.id, "", "", a.get("hash", ""), spec.repo_seed or 0, a.get("status") if a.get("status") in ("fehler", "ungueltig", "timeout") else "fehler",
                                                None, (a.get("reason") or "")[:300]))
            from .loop import Orchestrator   # Urteilsregel des Nachbaus (Methode nutzt self nicht), wie nachlauf.run_unit
            dossier.verdicts.append(Orchestrator._verdict(None, claim_, [], len(seeds or [1]), [reason], None, {}))
            info["agent"] = self._agent_info()
            return dossier

        timefeat = ver.get("needs_time_features", {})
        if timefeat.get("belegt") and timefeat.get("value"):
            return finish_failed("model needs time features, not supported in v1", "needs_time_features")

        # Snapshots und Vorpruefungen (wie run_unit)
        run = nl._Run(self.sb, spec, self.events, self.log, self.public_loader, self.rng, self.np_rng)
        run.parallel = self.parallel
        run.train_snap = self.sb.base_snapshot(nl.etth1_task().image, nl.training_setup_cmds(spec), label=f"nachlauf-{spec.unit}-train")
        run.pred_snap = self.sb.base_snapshot(nl.etth1_task().image, nl.predict_setup_cmds(spec), label=f"nachlauf-{spec.unit}-predict")
        raw = run.probe_predict_snapshot()
        if raw:
            return finish_failed(f"raw data in the prediction snapshot: {raw}", "raw_data")
        probe = run.probe_training()
        run.repo_files = {**reading.files, **run.repo_files}
        run.script_text = run.repo_files.get(spec.script, "")
        if probe["head"] != commit:
            return finish_failed(f"repository commit in the snapshot {probe['head']} != {commit}", "commit_mismatch")

        spec, trained = self.explore_training(run, claim, dataset, spec, reading)
        if trained is None:
            return finish_failed("no working training command within the attempt limit", "training_failed")
        ctx = self._adapter_context(reading, ans, spec)
        code, why = self.explore_adapter(run, spec, trained["ckpt_bytes"], trained.get("own_metric"), ctx)
        if code is None:
            return finish_failed(f"no valid adapter: {why}", "adapter_failed")
        spec = dataclasses.replace(spec, adapter_source=code)

        seed_list = [spec.repo_seed if (isinstance(s, str) and s == "repo") else int(s) for s in (seeds if seeds is not None else ["repo", 0, 1, 2, 3])]
        dossier = nl.run_unit(self.sb, spec, seed_list, offsets=offsets, out_dir=self.out_dir, public_loader=run.public_loader, rng=self.rng,
                              np_rng=self.np_rng, log=self.log, parallel=self.parallel, snapshots=(run.train_snap, run.pred_snap), event_log=self.events)
        info = next(iter(dossier.nachlauf.values()))
        self._attach(dossier, info, warnings)
        self.rec.outcome = "ok"
        info["agent"] = self._agent_info()
        info["wall_s"] = round(time.time() - t0, 1)
        # fehlgeschlagene Erkundungsversuche vor den Seeds im Dossier
        dossier.attempts[:0] = [Attempt(claim.id, "", run.train_snap, a.get("hash", ""), spec.repo_seed or 0,
                                        a["status"] if a.get("status") in ("fehler", "ungueltig", "timeout") else "fehler", None, (a.get("reason") or "")[:300])
                                for a in self.rec.attempts if a.get("status") != "ok"]
        return dossier

    def _attach(self, dossier: Dossier, info: dict, warnings: list[str]) -> None:
        for w in warnings:
            if w not in dossier.warnings:
                dossier.warnings.append(w)

    def _agent_info(self) -> dict:
        r = self.rec
        r.llm_calls = list(self.asker.calls)
        r.cost_usd_net = round(self.guard.run_net(), 5)
        d = dataclasses.asdict(r)
        d["n_llm_calls"] = len(r.llm_calls)
        return d
