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
from .costs import BudgetExceeded
from .llm import BillingBlocked, LLMResponse
from .loop import extract_code
from . import nachlauf_candidates as cand
from .repo_source import _last_string_list, imported_files
from .setup_details import quote_in_text

MAX_FILES = 14                 # vom Modell gewaehlte Dateien
MAX_FILE_CHARS = 16_000        # je Datei im Prompt
MAX_TOTAL_CHARS = 120_000      # alle Dateien eines Prompts
MAX_LISTING = 600
MAX_DEP_FILES = 6
MAX_LOADER_FILES = 8
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
    "You read a code repository and decide how its authors run one experiment. You never write a command yourself: you choose a numbered, "
    "documented command from a list that a program extracted from the repository. Reply with exactly one JSON object and no other text."
)
COMMAND_FORMAT = """\
{
  "command_candidate": 3,
  "checkpoint_glob": "path pattern relative to the repository root where the training run saves the weights, e.g. out/*/weights.bin",
  "test_loader": {"drop_last": {"candidate": 1, "value": true},
                  "batch_size": {"switch": "name of the command-line switch that sets the test batch size, e.g. --bs"}},
  "input_len": 0, "output_len": 0,
  "needs_time_features": {"value": false, "file": "path of the model file", "quote": "verbatim code of the model's forward function that uses time-stamp features, or null"},
  "model_file": "path of the file that defines the model class", "training_file": "path of the file with the training and test loop"
}"""
COMMAND_USER = (
    "Task: start_command\n"
    "Result to check: {claim} (dataset: {dataset}).\n"
    "Documented commands found in the repository by a program ({counts}):\n{candidates}\n\n"
    "Lines that set drop_last, found by a program in the files that build data loaders:\n{loader}\n\n"
    "Answer with one JSON object of this shape:\n{format}\n\n"
    "Rules:\n"
    "- All numbers are the numbers in square brackets at the start of a listed line, never line numbers of files.\n"
    "- command_candidate: the number of the documented command that trains the model of this result on this dataset for this forecast horizon. "
    "Use null if none of them matches; never guess and never write a command yourself.\n"
    "- test_loader.drop_last.candidate: the number of the line (second list) that sets drop_last for the loader used when testing; value is true or false as "
    "written on that line; use null if none matches.\n"
    "- test_loader.batch_size: give the switch name when the code takes the test batch size from a command-line argument (the switch of the argument "
    "definitions); otherwise give {{\"file\": ..., \"quote\": ...}} with a verbatim quote that fixes the batch size to a number.\n"
    "- input_len and output_len are the input length and forecast length of the chosen command.\n"
    "- needs_time_features is true only if the model's forward function actually uses the time-stamp features of the batch; then quote that code verbatim.\n"
    "Invented example of the answer's style (not from any real repository): {{\"command_candidate\": 2, \"test_loader\": {{\"drop_last\": {{\"candidate\": 4, "
    "\"value\": false}}, \"batch_size\": {{\"switch\": \"--bs\"}}}}}}.\n\n"
    "Repository file list (first entries):\n{listing}\n\nFiles:\n{files}"
)
COMMAND_REASK = (
    "Task: start_command_requote\n"
    "Result to check: {claim} (dataset: {dataset}).\n"
    "These items of your previous answer could not be verified: {items}.\n"
    "Reply with one JSON object that contains only these items, in the same structure as before, each with an exact quote copied from the named file, "
    "or null if the repository does not state it.\n\nFiles:\n{files}"
)
SEED_USER = (
    "Task: choose_seed\n"
    "The documented command for this result is: {command}\nIt runs the entry script {script}. Lines of that script that set a random seed, found by a program:\n{seeds}\n\n"
    "Answer with one JSON object {{\"seed_candidate\": N}} where N is the number in square brackets (not a line number of the file) of the line that sets the random seed of the training run "
    "(a fixed number in the script, or a command-line switch for the seed), or null if none of these lines is the seed of the training.\n"
    "Invented example: {{\"seed_candidate\": 1}}."
)
REPAIR_COMMAND_USER = (
    "Task: repair_command\n"
    "Result to check: {claim} (dataset: {dataset}).\n"
    "The training run with this command did not produce a usable result.\nCommand: {command} (documented command number {k})\n\nProblem:\n{problem}\n\n"
    "Documented commands found in the repository by a program:\n{candidates}\n\n"
    "Answer with one JSON object: {{\"command_candidate\": N, \"command\": ..., \"checkpoint_glob\": ..., \"install\": [package names]}}. "
    "N is the number of a documented command: keep the current one or choose another by number. \"command\" is optional: a one-line variant of that documented "
    "command with changed, added or removed switches (the same script). A command without a number is rejected. You may also change the checkpoint "
    "path pattern, or ask to install a missing package, but only a package that is named in the repository's dependency files. "
    "Use an empty list for install if none is needed.\n\nFiles:\n{files}"
)

ADAPTER_SYSTEM = (
    "You write a single self-contained Python script that loads a trained model of a code repository and writes its predictions. "
    "Reply with exactly one ```python code block and no other prose."
)
ADAPTER_CONTRACT = """\
Contract of predict.py (run without network access, CPU only). The working directory is the repository root, but the script itself is stored somewhere else:
import os and sys and put os.getcwd() on sys.path (sys.path.insert(0, os.getcwd())) before importing anything from the repository. Never make assumptions about
__file__ or the directory of the script.
- Read /data/etth1_public.npz with numpy.load. Keys: train [8640, 7], val [2880, 7] (not needed) and test_inputs, shape [n, 720, 7], float32, already z-score
  standardised with the mean and standard deviation of the training rows. Take n from the file; predict ALL windows in it. Window i is independent of every
  other window.
- Use only the last L steps of each window as model input, where L is the input length of the model (the input length of the start command).
- For every window predict the next 96 hourly steps of all 7 channels, in the same standardised units.
- Load the model class from the repository (imports relative to the repository root) and the weights from the file named by the environment variable NACHLAUF_CKPT. The weights were written by the
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
NO_CANDIDATE_SENTENCE = ("The command must come from one of the numbered documented commands: give its number in command_candidate. "
                         "You may change switches of that command, but you must not invent a command.")
SAME_SCRIPT_SENTENCE = "The command must run the same script as the documented command with the given number."
NO_COMMAND_REASON = "no documented start command found for this claim"
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

    def __init__(self, llm, guard: CostGuard, events: EventLog | None = None, log=print, out_dir: Path | None = None):
        self.llm, self.guard, self.events, self.log = llm, guard, events, log
        self.out_dir = Path(out_dir) if out_dir else None
        self._counts: dict[str, int] = {}
        self.calls: list[dict] = []
        self.prompts: list[dict] = []      # {purpose, system, user}: nur fuer Tests und Diagnose, nie im Dossier

    def _once(self, purpose: str, role: str, system: str, user: str, max_tokens, temperature: float, n: int) -> LLMResponse:
        self.guard.check()
        c = self._counts[purpose] = self._counts.get(purpose, 0) + 1
        self._write(f"llm_{purpose}_{c}_prompt.txt", f"### system\n{system}\n\n### user\n{user}\n")     # Prompt vor dem Aufruf (nur Laufordner)
        resp = self.llm.complete(role, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 temperature=temperature, max_tokens=max_tokens, timeout_s=400)
        self._write(f"llm_{purpose}_{c}.txt", getattr(resp, "content", "") or "")      # Rohantwort sofort, noch vor dem Parsen
        self.prompts.append({"purpose": purpose, "system": system, "user": user})
        self.calls.append({"purpose": purpose, "role": role, "try": n, "cost_usd": round(getattr(resp, "cost_usd", 0.0), 5),
                           "finish_reason": getattr(resp, "finish_reason", None), "cached": bool(getattr(resp, "cached", False))})
        return resp

    def _write(self, name: str, text: str) -> None:
        if self.out_dir:
            try:
                (self.out_dir / name).write_text(text)
            except OSError:
                pass

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
    files: dict[str, str] = field(default_factory=dict)      # fuer den Prompt (gekuerzt): Auswahl des Modells + Pflichtdateien
    chosen: list[str] = field(default_factory=list)
    full: dict[str, str] = field(default_factory=dict)       # Dateien mit Python-Aufruf eines Skripts (git grep), ungekuerzt (Kandidatensuche)
    loader_files: dict[str, str] = field(default_factory=dict)   # Dateien mit DataLoader/drop_last (git grep), immer mitgegeben
    fetch: object = None                                     # Pfad -> Text oder None (spaetes Lesen, z. B. des Einstiegsskripts)
    imports: set = field(default_factory=set)                # Namen der Module, die Python-Dateien des Repos importieren (git grep)


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
    chosen = [p for p in dict.fromkeys(x for x in raw if isinstance(x, str)) if p in valid][:MAX_FILES]
    if not chosen:      # Auswahl unbrauchbar (leer, falsche Form, keine gueltigen Pfade): deterministischer Rueckfall
        chosen = fallback_files(listing)
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

    def fetch(p: str) -> str | None:
        try:
            data = sandbox.get_file(b, f"/repo/{p}")
        except (OSError, FileNotFoundError, KeyError):
            return None
        return data.decode(errors="replace") if len(data) <= cand.MAX_FILE_BYTES else None

    def grep(ere: str) -> list[str]:
        out = sandbox.run(b, cand.grep_cmd(ere), 120).stdout.splitlines()
        return [p.strip() for p in out if p.strip() in valid]

    full: dict[str, str] = {}
    for p in grep(cand.CMD_GREP_ERE)[:cand.MAX_GREP_FILES]:        # deterministisch: alle Dateien mit Python-Aufruf eines Skripts, unabhaengig von der Modellauswahl
        text = fetch(p)
        if text is not None:
            full[p] = text
    loaders: dict[str, str] = {}
    paths = [p for p in grep(cand.LOADER_GREP_ERE) if p.endswith(".py")]
    paths.sort(key=lambda p: (0 if re.search(r"data|loader|factory", p, re.I) else 1, p.count("/"), p))      # flache Pfade zuerst (Hauptcode vor Kopien in Unterordnern)
    for p in paths[:MAX_LOADER_FILES]:
        text = fetch(p)
        if text is not None:
            loaders[p] = text
            files.setdefault(p, text[:MAX_FILE_CHARS])         # Loader-Dateien gehoeren immer zu den gelesenen Dateien
    raw_imports = sandbox.run(b, cand.IMPORT_GREP_CMD, 120).stdout
    imports = {m.group(1) for m in re.finditer(r"^\s*(?:import|from)\s+([A-Za-z_]\w*)", raw_imports, re.M)}
    return RepoReading(url=url, commit=commit, listing=listing, files=files, chosen=chosen, full=full, loader_files=loaders, fetch=fetch, imports=imports)


FALLBACK_RE = re.compile(r"(^|/)(readme[\w.-]*|[\w.-]*\.sh|(run|train|main|exp)[\w.-]*\.py|[\w/.-]*model[\w.-]*\.py|[\w/.-]*(loader|data_factory|dataset)[\w.-]*\.py)$", re.I)


def fallback_files(listing: list[str]) -> list[str]:
    """Wenn die Dateiauswahl des Modells unbrauchbar ist: Dokumentation, Skripte, Einstieg, Modell- und Ladedateien nach Namen."""
    return [p for p in listing if FALLBACK_RE.search(p)][:MAX_FILES]


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
    if not isinstance(file, str):
        file = ""
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
    """Ganzzahl aus int oder reiner Ziffernfolge ('128'); bool, Fliesskomma und alles andere gelten als nicht lesbar."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v):
        return int(v)
    return None


def _obj(x) -> dict:
    """Objekt oder leer: String, Liste, Zahl, null und Co. gelten als fehlend (nie ein Absturz)."""
    return x if isinstance(x, dict) else {}


def _text(x) -> str:
    return x if isinstance(x, str) else ""


def _num(v) -> int | None:
    """Kandidatennummer: int oder reine Ziffernfolge; bool, Fliesskomma, Listen u. a. gelten als keine Nummer."""
    return _intval(v)


def chosen_candidate(ans: dict, cands: list) -> "cand.Candidate | None":
    k = _num(_obj(ans).get("command_candidate"))
    return next((c for c in cands if c.k == k), None) if k is not None else None


def verify_answer(ans: dict, files: dict[str, str], cands: list | None = None, loader: list | None = None, only: set[str] | None = None) -> dict:
    """Prueft jede Angabe einzeln. Rueckgabe {item: {"belegt": bool, "reason": str, ...}} fuer command_candidate, test_loader.drop_last,
    test_loader.batch_size, needs_time_features (nur `only`, falls gesetzt). Der Befehl und seine Quelle kommen aus der Kandidatenliste des
    Orchestrators (nie aus der Antwort); drop_last-Zeile und Wert ebenso. Jede Angabe mit falschem Typ oder falscher Form (String statt Objekt, Liste,
    null, fehlender Schluessel, nicht lesbare Zahl, unbekannte Nummer) gilt als nicht belegt und geht in denselben Rueckfall."""
    ans, cands, loader = _obj(ans), cands or [], loader or []
    out: dict[str, dict] = {}
    want = lambda k: only is None or k in only  # noqa: E731
    chosen = chosen_candidate(ans, cands)

    if want("command_candidate"):
        raw = ans.get("command_candidate")
        if raw is None:
            out["command_candidate"] = {"belegt": False, "reason": "no candidate chosen", "k": None}
        elif chosen is None:
            out["command_candidate"] = {"belegt": False, "reason": f"{str(raw)[:40]!r} is not the number of a listed command", "k": None}
        else:
            out["command_candidate"] = {"belegt": True, "reason": "", "k": chosen.k, "file": chosen.file, "line": chosen.line, "text": chosen.text,
                                        "command": chosen.command, "note": chosen.note}

    tl = _obj(ans.get("test_loader"))
    if want("test_loader.drop_last"):
        d = _obj(tl.get("drop_last"))
        k = _num(d.get("candidate"))
        line = next((l for l in loader if l.k == k), None) if k is not None else None
        val = cand.drop_last_value(line.text) if line else None
        if not d:
            res = {"belegt": False, "reason": "drop_last must be an object with candidate and value", "value": None}
        elif line is None:
            res = {"belegt": False, "reason": "no listed drop_last line chosen", "value": None}
        elif val is None:
            res = {"belegt": False, "reason": "the chosen line does not set drop_last to a literal True or False", "value": None}
        else:
            res = {"belegt": True, "reason": "", "value": val, "file": line.file, "line": line.line, "text": line.text}
            if isinstance(d.get("value"), bool) and d["value"] != val:
                res["model_value"] = d["value"]          # das Modell widerspricht der Zeile: die Zeile gilt
        out["test_loader.drop_last"] = res

    if want("test_loader.batch_size"):
        d = _obj(tl.get("batch_size"))
        res = {"belegt": False, "reason": "batch_size must be an object with switch, or file and quote", "value": None}
        sw = d.get("switch")
        if isinstance(sw, str) and sw.startswith("-"):
            v = cand.switch_value(chosen.command, sw) if chosen else None
            if v is not None:
                res = {"belegt": True, "reason": "", "value": v, "source": "command", "switch": sw}
            else:
                dflt = cand.argument_default(sw, files)
                if dflt:
                    res = {"belegt": True, "reason": "", "value": dflt[0], "source": f"{dflt[1]}:{dflt[2]}", "switch": sw, "quote": dflt[3]}
                else:
                    res["reason"] = f"no value of {sw} in the command and no default= in an argument definition"
        elif d.get("quote") is not None or d.get("file") is not None:
            ok, why = _quote_ok(d.get("quote"), d.get("file"), files)
            ints = re.findall(r"(?<![\w.])\d+(?![\w.])", _text(d.get("quote")))
            if ok and len(ints) == 1:
                res = {"belegt": True, "reason": "", "value": int(ints[0]), "source": f"{d.get('file')}", "quote": d.get("quote")}
            else:
                res["reason"] = why or "the quote must contain exactly one number"
        out["test_loader.batch_size"] = res

    if want("needs_time_features"):
        raw = ans.get("needs_time_features")
        d = _obj(raw)
        val = raw if isinstance(raw, bool) else d.get("value") if d else raw      # gerettet: nacktes true/false
        if val is False or val is None:
            out["needs_time_features"] = {"belegt": True, "reason": "", "value": False}
        elif val is True or val == "true":
            ok, why = _quote_ok(d.get("quote"), d.get("file"), files) if d else (False, "needs_time_features true needs file and quote")
            out["needs_time_features"] = {"belegt": ok, "reason": why, "value": bool(ok), "file": d.get("file"), "quote": d.get("quote")}
        else:
            out["needs_time_features"] = {"belegt": False, "reason": "needs_time_features must be an object with value true or false", "value": False}
    return out


def pick_files(files: dict[str, str], *names) -> list[str]:
    return [n for n in names if isinstance(n, str) and n in files]


# ---- Agent --------------------------------------------------------------------------------------------------------------------

@dataclass
class AgentRecord:
    """Alles, was das Modell geliefert hat, und was davon belegt wurde (Dossier: nachlauf[claim]["agent"])."""
    reading: dict = field(default_factory=dict)
    candidates: dict = field(default_factory=dict)       # Zahl gefunden/gezeigt/verworfen, Stichprobe der verworfenen, gezeigte Liste, Wahl
    seed_candidates: list = field(default_factory=list)
    seed_choice: dict = field(default_factory=dict)
    loader_lines: list = field(default_factory=list)
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
        self.asker = Asker(llm, self.guard, self.events, log, self.out_dir)
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
        self.cands: list = []        # nummerierte Kandidaten im Prompt
        self.loader: list = []       # drop_last-Zeilen im Prompt
        self.chosen: "cand.Candidate | None" = None

    # -- Hilfen
    def emit(self, type: str, **data) -> None:
        self.events.write(type, **data)
        self.log(f"[{type}] " + json.dumps(data, ensure_ascii=False, default=str)[:300])

    def _save(self, name: str, text: str) -> None:
        if self.out_dir:
            (self.out_dir / name).write_text(text)

    # -- Schritt 2: Startbefehl
    def _cand_listing(self) -> str:
        return "\n".join(c.line_text() for c in self.cands) or "(none)"

    def ask_command(self, claim_text: str, dataset: str, reading: RepoReading) -> tuple[dict, dict]:
        files_txt = _render_files(reading.files)
        ci = self.rec.candidates
        counts = f"{ci['n_shown']} of {ci['n_total']} shown" + (f", {ci['n_dropped']} more not shown" if ci["n_dropped"] else "")
        ans = self.asker.json("start_command", COMMAND_SYSTEM, COMMAND_USER.format(
            claim=claim_text, dataset=dataset, format=COMMAND_FORMAT, counts=counts, candidates=self._cand_listing(),
            loader="\n".join(l.line_text() for l in self.loader) or "(none)", listing="\n".join(reading.listing[:80]), files=files_txt))
        ver = verify_answer(ans, reading.files, self.cands, self.loader)
        # Nummern und Zeilen werden nicht nachgefragt (kein Zitat zu pruefen); nur das Zitat der Zeitmerkmale darf einmal neu erfragt werden
        for k, v in ver.items():
            if not v["belegt"] and k != "needs_time_features":
                self.rec.discarded.append({"item": k, "first": v, "second": None})
        if not ver["needs_time_features"]["belegt"]:
            detail = f"needs_time_features ({ver['needs_time_features']['reason']})"
            self.emit("nachlauf_repair", kind="requote", items=["needs_time_features"])
            again = _obj(self.asker.json("start_command_requote", COMMAND_SYSTEM, COMMAND_REASK.format(
                claim=claim_text, dataset=dataset, items=detail, files=files_txt)))
            merged = {**_obj(ans)}
            if again.get("needs_time_features") is not None:
                merged["needs_time_features"] = again["needs_time_features"]
            ver2 = verify_answer(merged, reading.files, self.cands, self.loader, only={"needs_time_features"})
            if not ver2["needs_time_features"]["belegt"]:
                self.rec.discarded.append({"item": "needs_time_features", "first": ver["needs_time_features"], "second": ver2["needs_time_features"]})
            ans, ver = merged, {**ver, **ver2}
        return _obj(ans), ver

    def ask_seed(self, chosen: "cand.Candidate", reading: RepoReading) -> dict:
        """Seed-Kandidaten im Einstiegsskript des GEWAEHLTEN Befehls (deterministisch), das Modell waehlt per Nummer. Rueckgabe: Spec-Seed."""
        script = chosen.script
        text = reading.full.get(script) or (reading.fetch(script) if callable(reading.fetch) else None) or ""
        if text:
            reading.files.setdefault(script, text[:MAX_FILE_CHARS])
        sc = cand.seed_candidates(script, text)
        self.rec.seed_candidates = [c.line_text() for c in sc]
        none = {"kind": "none", "repo_seed": None}
        if not sc:
            return none
        if len(sc) == 1:      # genau eine nutzbare Zeile: ohne LLM-Aufruf
            pick, how, k = sc[0], "only_candidate", 1
        else:
            ans = _obj(self.asker.json("choose_seed", COMMAND_SYSTEM, SEED_USER.format(
                command=chosen.command, script=script, seeds="\n".join(c.line_text() for c in sc))))
            k = _num(ans.get("seed_candidate"))
            pick, how = cand.resolve_seed_choice(k, sc, text)
        spec_seed = cand.seed_spec(pick, text) if pick else None
        if spec_seed is None:
            self.rec.discarded.append({"item": "seed", "first": {"reason": "no usable seed line chosen" if pick is None else "chosen line is not unique or has not exactly one number", "how": how,
                                                                 "k": k}, "second": None})
            return none
        self.rec.seed_choice = {"k": pick.k, "file": pick.file, "line": pick.line, "text": pick.text, "kind": pick.kind, "how": how, "answered": k}
        return spec_seed

    def _spec(self, claim: Claim, paper_id: str, url: str, commit: str, ans: dict, ver: dict, reading: RepoReading,
              chosen: "cand.Candidate", seed: dict) -> nl.UnitSpec:
        files = reading.files
        ans = _obj(ans)
        dl, bs = ver.get("test_loader.drop_last", {}), ver.get("test_loader.batch_size", {})
        if dl.get("belegt") and bs.get("belegt"):
            test_loader = {"drop_last": bool(dl["value"]), "batch": int(bs["value"]), "source": f"{dl['file']}:{dl['line']}", "batch_source": bs.get("source")}
        elif dl.get("belegt") and dl["value"] is False:
            test_loader = {"drop_last": False, "batch": 0, "source": f"{dl['file']}:{dl['line']}"}       # ohne drop_last zaehlt die Batchgroesse nicht
        else:
            test_loader = {"drop_last": False, "batch": 0, "source": None, "belegt": False}              # P3 vergleicht mit allen Fenstern
        argfiles = [p for p, t in files.items() if "add_argument" in t]
        reqs = [p for p in files if re.search(r"(^|/)requirements[\w.-]*\.txt$", p)]
        return nl.UnitSpec(
            unit=re.sub(r"[^a-z0-9]+", "", paper_id.lower())[:24] or "unit", paper_id=f"nachlauf/{paper_id}", claim_text=claim.text, target=claim.target,
            tolerance=claim.tolerance, repo_url=url, commit=commit, command=chosen.command, command_source={"file": chosen.file, "line": chosen.line},
            script=chosen.script, seed=seed, adapter_source="", test_loader=test_loader, argparse_files=argfiles, requirements_files=reqs,
            pip_packages=list(BASE_PIP), checkpoint_glob=_text(ans.get("checkpoint_glob")).strip() or "checkpoints/*/*.pth",
            own_metric_pattern=rf"(?im)^\s*(?:test[ _-]*)?{re.escape(claim.metric)}\s*[:=]\s*([0-9.eE+-]+)", metric=claim.metric, claim_id=claim.id,
            seed_note={"copy": "copy of the entry script, only the seed line changed (diff checked in the branch)",
                       "flag": "command-line switch of the repository", "none": "no seed method: single measurement"}[seed["kind"]],
            claim_source=claim.source,
        )

    # -- Schritt 3a: Training mit Reparatur
    def _repair_command(self, claim: Claim, dataset: str, spec: nl.UnitSpec, reading: RepoReading, problem: str) -> dict:
        ans = self.asker.json("repair_command", COMMAND_SYSTEM, REPAIR_COMMAND_USER.format(
            claim=claim.text, dataset=dataset, command=spec.command, k=self.chosen.k if self.chosen else "?", problem=problem[-1500:],
            candidates=self._cand_listing(), files=_render_files(reading.files)))
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
            ans = _obj(ans)
            installs = [x for x in (ans.get("install") if isinstance(ans.get("install"), list) else []) if isinstance(x, str)]
            dep = _norm(self._dependency_text(reading))
            ok_pk, rejected_pk = [], []
            for pkg in installs:
                named = PKG_NAME_RE.match(pkg) and re.search(rf"(?<![\w.-]){re.escape(_norm(pkg))}(?![\w.-])", dep)
                (ok_pk if named else rejected_pk).append(pkg)
            k_new = _num(ans.get("command_candidate"))
            cnd = next((c for c in self.cands if c.k == k_new), None) if k_new is not None else None
            mod = sanitize_command(_text(ans.get("command")))
            pend: list[str] = []
            if cnd is None:                       # Befehl ohne Kandidaten-Nummer: nie ausfuehren
                pend.append(NO_CANDIDATE_SENTENCE)
                self.rec.repairs.append({"kind": "command_rejected", "round": k, "reason": "no candidate number"})
            elif mod and script_of(mod).lstrip("./").rsplit("/", 1)[-1] != cnd.script.rsplit("/", 1)[-1]:
                pend.append(SAME_SCRIPT_SENTENCE)
                self.rec.repairs.append({"kind": "command_rejected", "round": k, "reason": "different script", "k": cnd.k})
            else:
                new_cmd = mod or cnd.command
                if cnd.script != spec.script:     # anderer Kandidat mit anderem Einstiegsskript: die Seed-Wahl galt dem alten Skript
                    spec = dataclasses.replace(spec, seed={"kind": "none", "repo_seed": None}, seed_note="no seed method: single measurement")
                    self.rec.repairs.append({"kind": "seed_dropped", "round": k, "reason": "script changed"})
                self.chosen = cnd
                spec = dataclasses.replace(spec, command=new_cmd, script=cnd.script, command_source={"file": cnd.file, "line": cnd.line})
            spec = dataclasses.replace(
                spec, checkpoint_glob=_text(ans.get("checkpoint_glob")).strip() or spec.checkpoint_glob,
                pip_packages=[*spec.pip_packages, *[p for p in ok_pk if p not in spec.pip_packages]])
            if rejected_pk:
                pend.extend(PACKAGE_SENTENCE.format(pkg=p) for p in rejected_pk)
                self.rec.repairs.append({"kind": "package_rejected", "round": k, "packages": rejected_pk})
            if pend:       # abgelehnte Reparatur: fester Satz, der naechste Versuch ist verbraucht, es laeuft nichts
                pending = " ".join(pend)
            if ok_pk:
                self.rec.repairs.append({"kind": "package_install", "round": k, "packages": ok_pk})
                self._rebuild(run, spec, k)
        return spec, None

    # -- Schritt 3b: Adapter
    def _adapter_context(self, reading: RepoReading, ans: dict, spec: nl.UnitSpec) -> dict:
        files, ans = reading.files, _obj(ans)
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
            detail: dict = {}
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
                        detail = {"p3_distance": sc.get("distance"), "p3_compared_with": sc.get("compared_with"), "p2_max_abs_diff": ind.get("max_abs_diff")}
                        if sc["status"] == "ungueltig":
                            problems.append(("P3", P3_SENTENCE))
                        if not ind["ok"]:
                            problems.append(("P2", P2_SENTENCE))
            problem_text = " ".join(t for _, t in problems)[-1500:]       # genau der Text, der in den Reparatur-Prompt geht (keine Zahl aus P2/P3)
            self.rec.adapters.append({"n": v, "hash": h, "stage": stage, "status": "ok" if not problems else "ungueltig", "problems": [k for k, _ in problems],
                                      "problem_text": problem_text, "detail": detail})     # detail: Zahlen von P2/P3, nur Dossier, nie ein Prompt
            self.emit("nachlauf_adapter", n=v, hash=h, stage=stage, ok=not problems, problems=[k for k, _ in problems], problem_text=problem_text)
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
        """Ein unerwarteter Fehler im Agenten-Teil (auch eine unbrauchbare Modell-Antwort nach den Wiederholungen) beendet den Lauf mit dem Urteil
        "nicht reproduzierbar (Setup)", dem Grund im Dossier und der Traceback-Datei `error.txt`; kein Traceback nach aussen. Kosten-/Abrechnungsfehler
        (BudgetStop, BillingBlocked, BudgetExceeded) laufen durch."""
        self._spec_cur = None
        try:
            return self._run(paper_id, claim, repo_url, commit, dataset=dataset, seeds=seeds, offsets=offsets)
        except (BudgetStop, BillingBlocked, BudgetExceeded):
            raise
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self._save("error.txt", tb)
            self.emit("nachlauf_repair", kind="crash", reason=f"{type(exc).__name__}: {str(exc)[:200]}")
            spec = self._spec_cur or self._minimal_spec(claim, paper_id, repo_url, commit)
            claim_, dossier, info = nl.new_dossier(spec)
            reason = f"unexpected error in the agent part: {type(exc).__name__}: {str(exc)[:300]}"
            self.rec.outcome = "error"
            from .loop import Orchestrator
            dossier.verdicts.append(Orchestrator._verdict(None, claim_, [], len(seeds or [1]), [reason], None, {}))
            info["agent"] = self._agent_info()
            info["agent"]["error"] = {"type": type(exc).__name__, "message": str(exc)[:500], "traceback_tail": tb[-1500:]}
            return dossier

    def _run(self, paper_id: str, claim: Claim, repo_url: str, commit: str, *, dataset: str = "ETTh1", seeds: list | None = None,
             offsets: int | list[int] = independence.N_OFFSETS) -> Dossier:
        self.guard.start_run()
        t0 = time.time()
        claim_text = claim.text
        reading = read_repo_at_commit(self.sb, self.asker, repo_url, commit, claim_text, dataset)
        # dokumentierte Befehle deterministisch finden (alle verfolgten Textdateien mit Python-Aufruf), Modell waehlt per Nummer
        all_c, skipped = cand.extract_candidates(reading.full, reading.listing)
        self.cands, cinfo = cand.narrow(all_c, cand.claim_words(claim_text, dataset))
        cinfo["skipped_unresolved"], cinfo["shown"] = skipped, [c.line_text() for c in self.cands]
        self.rec.candidates = cinfo
        for c in self.cands:      # Dateien mit Kandidaten gehoeren immer zu den gelesenen Dateien, unabhaengig von der Auswahl des Modells
            reading.files.setdefault(c.file, reading.full.get(c.file, "")[:MAX_FILE_CHARS])
        self.loader = cand.loader_lines(reading.loader_files)
        self.rec.loader_lines = [l.line_text() for l in self.loader]
        self.rec.reading = {"url": repo_url, "commit": commit, "n_tracked": len(reading.listing), "chosen": reading.chosen, "files": sorted(reading.files)}
        spec = self._minimal_spec(claim, paper_id, repo_url, commit)
        self._spec_cur = spec
        warnings: list[str] = []

        def finish_failed(reason: str, status_note: str) -> Dossier:
            claim_, dossier, info = nl.new_dossier(spec)
            self._attach(dossier, info, warnings)
            self.rec.outcome = status_note
            for a in self.rec.attempts:
                dossier.attempts.append(Attempt(claim.id, "", "", a.get("hash", ""), spec.repo_seed or 0, a.get("status") if a.get("status") in ("fehler", "ungueltig", "timeout") else "fehler",
                                                None, (a.get("reason") or "")[:300]))
            dossier.attempts.extend(self._adapter_attempts(claim, spec))
            from .loop import Orchestrator   # Urteilsregel des Nachbaus (Methode nutzt self nicht), wie nachlauf.run_unit
            dossier.verdicts.append(Orchestrator._verdict(None, claim_, [], len(seeds or [1]), [reason], None, {}))
            info["agent"] = self._agent_info()
            return dossier

        if not self.cands:           # kein dokumentierter Befehl: kein erfundener Befehl
            return finish_failed(NO_COMMAND_REASON, "no_command")
        ans, ver = self.ask_command(claim_text, dataset, reading)
        self.rec.command_answer, self.rec.verification = ans, ver
        self._save("command_answer.json", json.dumps({"answer": ans, "verification": ver, "discarded": self.rec.discarded, "candidates": cinfo},
                                                     indent=2, ensure_ascii=False, default=str))
        if not ver["command_candidate"]["belegt"]:
            return finish_failed(NO_COMMAND_REASON, "no_command")
        self.chosen = chosen = chosen_candidate(ans, self.cands)
        spec = self._spec(claim, paper_id, repo_url, commit, ans, ver, reading, chosen, {"kind": "none", "repo_seed": None})
        self._spec_cur = spec
        timefeat = ver.get("needs_time_features", {})
        if timefeat.get("belegt") and timefeat.get("value"):
            return finish_failed("model needs time features, not supported in v1", "needs_time_features")
        seed = self.ask_seed(chosen, reading)         # Seed-Pruefung NACH der Wahl des Befehls
        spec = dataclasses.replace(spec, seed=seed, seed_note={"copy": "copy of the entry script, only the seed line changed (diff checked in the branch)",
                                                               "flag": "command-line switch of the repository", "none": "no seed method: single measurement"}[seed["kind"]])
        self._spec_cur = spec
        if seed["kind"] == "none":
            warnings.append("eine Messung, keine Streuung")
        if spec.test_loader.get("belegt") is False:
            warnings.append("Test-Loader nicht belegt: P3 vergleicht mit allen Fenstern")
        self.emit("nachlauf_command", n=0, command=spec.command, command_source=spec.command_source, candidate=chosen.k,
                  belegt={k: v["belegt"] for k, v in ver.items()}, seed_kind=spec.seed["kind"], test_loader=spec.test_loader,
                  n_candidates=cinfo["n_total"], discarded=[d["item"] for d in self.rec.discarded])

        # Snapshots und Vorpruefungen (wie run_unit)
        run = nl._Run(self.sb, spec, self.events, self.log, self.public_loader, self.rng, self.np_rng)
        run.parallel = self.parallel
        run.train_snap = self.sb.base_snapshot(nl.etth1_task().image, nl.training_setup_cmds(spec), label=f"nachlauf-{spec.unit}-train")
        run.pred_snap = self.sb.base_snapshot(nl.etth1_task().image, nl.predict_setup_cmds(spec), label=f"nachlauf-{spec.unit}-predict")
        raw = run.probe_predict_snapshot()
        if raw:
            return finish_failed(f"raw data in the prediction snapshot: {raw}", "raw_data")
        probe = run.probe_training()
        run.repo_files = {**reading.files, **reading.full, **run.repo_files}      # ungekuerzte Dateien der Kandidaten fuer P1
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
                              np_rng=self.np_rng, log=self.log, parallel=self.parallel, snapshots=(run.train_snap, run.pred_snap), event_log=self.events,
                              repo_imports=reading.imports or None)
        info = next(iter(dossier.nachlauf.values()))
        self._attach(dossier, info, warnings)
        self.rec.outcome = "ok"
        info["agent"] = self._agent_info()
        info["wall_s"] = round(time.time() - t0, 1)
        # fehlgeschlagene Erkundungsversuche vor den Seeds im Dossier
        dossier.attempts[:0] = [Attempt(claim.id, "", run.train_snap, a.get("hash", ""), spec.repo_seed or 0,
                                        a["status"] if a.get("status") in ("fehler", "ungueltig", "timeout") else "fehler", None, (a.get("reason") or "")[:300])
                                for a in self.rec.attempts if a.get("status") != "ok"] + self._adapter_attempts(claim, spec)
        return dossier

    def _attach(self, dossier: Dossier, info: dict, warnings: list[str]) -> None:
        for w in warnings:
            if w not in dossier.warnings:
                dossier.warnings.append(w)

    def _adapter_attempts(self, claim: Claim, spec: nl.UnitSpec) -> list[Attempt]:
        """Fehlgeschlagene Adapter-Versionen der Erkundung als Versuche; error_summary = Problemtext, der in den Reparatur-Prompt ging."""
        out = []
        for a in self.rec.adapters:
            if a.get("status") != "ok":
                status = "fehler" if set(a.get("problems", [])) & {"static", "smoke", "run"} else "ungueltig"
                out.append(Attempt(claim.id, "", "", a.get("hash", ""), spec.repo_seed or 0, status, None, (a.get("problem_text") or "")[:1500]))
        return out

    def _minimal_spec(self, claim: Claim, paper_id: str, url: str, commit: str) -> nl.UnitSpec:
        return nl.UnitSpec(
            unit=re.sub(r"[^a-z0-9]+", "", paper_id.lower())[:24] or "unit", paper_id=f"nachlauf/{paper_id}", claim_text=claim.text, target=claim.target,
            tolerance=claim.tolerance, repo_url=url, commit=commit, command="", command_source={}, script="", seed={"kind": "none", "repo_seed": None},
            adapter_source="", test_loader={}, claim_id=claim.id, metric=claim.metric, claim_source=claim.source)

    def _agent_info(self) -> dict:
        r = self.rec
        r.llm_calls = list(self.asker.calls)
        r.cost_usd_net = round(self.guard.run_net(), 5)
        d = dataclasses.asdict(r)
        d["n_llm_calls"] = len(r.llm_calls)
        return d
