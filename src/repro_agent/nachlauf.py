"""Modus "Nachlauf" (Meilenstein M1: Harness und Ablauf, ohne LLM). Der Code der Autoren trainiert, unser Harness misst.

Vorab-Festlegung: benchmark/experiments/nachlauf_v1.md (Regeln P1 bis P4). Bauplan: docs/plan_nachlauf.md.
Der Nachbau-Pfad (loop.py) bleibt unberuehrt; dieses Modul nutzt nur dessen Urteilsregel (`Orchestrator._verdict`).

Je Seed drei Arten von Branches, der Agenten-Adapter sieht nie Testlabels und nie den Rohdatensatz:
1. Trainings-Branch (Snapshot mit Repo auf festem Commit und ETTh1.csv): nur der Startbefehl der Autoren, `unshare -n`.
2. Vorhersage-Branches (Snapshot OHNE Rohdaten, nur /data/etth1_public.npz): Formpruefung (SMOKE=1), voller Lauf, P2-Proben.
3. Harness (frisch kopiert, erst nach dem Adapter-Lauf) im Branch des vollen Laufs.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import secrets
import shlex
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .dossier import Attempt, Claim, Dossier
from .events import EventLog
from .harness import independence
from .harness.code_check import find_violations
from .harness.smoke import check_smoke_output
from .harness.verify import judge
from .tasks import HARNESS_DIR, SANDBOX_HARNESS_DIR, etth1_task

REPO_DIR = "/work/repo"
ADAPTER_DIR = "/opt/nachlauf"
ADAPTER_PATH = f"{ADAPTER_DIR}/predict.py"
CKPT_PATH = f"{ADAPTER_DIR}/ckpt/checkpoint.pth"
SEED_DIR = "/opt/seed"
OUTPUT = "/output"
TRAIN_TIMEOUT_S = 600
PREDICT_TIMEOUT_S = 600
SMOKE_TIMEOUT_S = 120
PROBE_TIMEOUT_S = 300
EVAL_TIMEOUT_S = 600
ETTH1_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
SELF_CHECK_TOLERANCE = 0.001   # P3

NACHLAUF_ISOLATION = [
    "Nachlauf: Trainingslauf nur mit dem Startbefehl der Autoren auf festem Commit, kein vom Agenten geschriebener Code im Trainings-Branch (unshare -n); git status und git diff --stat im Klon muessen leer sein",
    "Nachlauf: Vorhersage-Branch ohne Rohdatensatz (Snapshot ohne ETTh1.csv, nur /data/etth1_public.npz), Adapter unter unshare -n, statische Codepruefung (harness/code_check.py) und Formpruefung mit SMOKE=1",
    "Nachlauf P2 (harness/independence.py): Vorhersagen haengen nur vom eigenen Fenster ab; 8 zufaellige Versaetze, Probe ersetzt die oeffentliche Datei, Abweichung <= 1e-4",
    "Nachlauf P3: Eigenausgabe des Autorencodes gegen die Harness-Rechnung auf denselben Fenstern (alle oder drop_last-Teilmenge), Abstand <= 0,001",
    "Testlabels dem Adapter vorenthalten, Harness laedt sie frisch und laeuft erst nach dem Adapter",
]


# ---- Spec (Handarbeit in M1, in M3 liefert der Agent sie) --------------------------------------------------------------

@dataclass
class UnitSpec:
    unit: str
    paper_id: str
    claim_text: str
    target: float
    tolerance: float
    repo_url: str
    commit: str
    command: str                       # Startbefehl, ausgefuehrt im Repo-Ordner (cwd), ohne Log-Umleitung
    command_source: dict               # {"file": Pfad im Repo, "line": Zeilennummer}
    script: str                        # Startskript im Repo (fuer die Seed-Kopie)
    seed: dict                         # {"kind": "copy", "repo_seed": n, "regex": ..., "template": "fix_seed = {seed}"} | {"kind": "flag", "flag": "--seed", "repo_seed": n|None} | {"kind": "none", "repo_seed": n}
    adapter_source: str                # predict.py nach dem Adapter-Vertrag
    test_loader: dict                  # {"drop_last": bool, "batch": int, "source": "datei:zeile"}
    argparse_files: list[str] = field(default_factory=list)       # Repo-Dateien mit add_argument (Hilfetexte fuer P1)
    requirements_files: list[str] = field(default_factory=list)   # Abhaengigkeitsdateien des Repos (env_diff)
    pip_packages: list[str] = field(default_factory=list)         # zusaetzliche Pakete (aus den Abhaengigkeitsdateien des Repos)
    checkpoint_glob: str = "checkpoints/*/checkpoint.pth"         # relativ zum Repo, genau eine Datei
    own_metric_pattern: str = r"^mse:\s*([0-9.eE+-]+)"            # Eigenausgabe des Autorencodes (stdout), letzter Treffer
    metric: str = "mse"
    data_dir: str = "dataset"          # Ordner, in dem das Repo ETTh1.csv erwartet
    claim_id: str = ""
    seed_note: str = ""

    @property
    def repo_seed(self) -> int | None:
        return self.seed.get("repo_seed")


# ---- P1: Startbefehl ----------------------------------------------------------------------------------------------------

# Ohne Meldung erlaubt: Geraet/GPU, Pfade (Log-Umleitung wird vor dem Vergleich entfernt).
ALLOWED_FLAGS = {"--use_gpu", "--gpu", "--devices", "--use_multi_gpu", "--root_path", "--data_path", "--checkpoints"}

# Verboten: Schalter, die laut Name oder Hilfetext auf Training mit allen Daten oder Testdaten im Training zielen. Die Liste ist NICHT
# vollstaendig (Wortlisten sind umgehbar); sie faengt den bekannten Fall `train_only` und typische Hilfetexte. Der Reviewer liest jeden Befehl.
FORBIDDEN_FLAG_NAMES = {"--train_only"}
FORBIDDEN_HELP_PATTERNS = [
    r"train(ing)?\s+(only\s+)?(on|with)\s+(all|the\s+(entire|whole|full|complete))",
    r"entire\s+(data\s*set|data)",
    r"whole\s+(data\s*set|data)",
    r"\ball\s+(of\s+)?the\s+data\b",
    r"including\s+(the\s+)?(test|validation)",
    r"(use|using)\s+(all|the\s+entire|the\s+whole)\s+(the\s+)?data",
    r"train[_\s-]*only",
    r"(on|with)\s+(the\s+)?full\s+(input\s+)?data",      # LTSF-Linear: 'perform training on full input dataset without validation and testing'
]

_LOG_RE = re.compile(r"\d*>&\d+|\d*>>?\s*[^\s&|;]+|\bnohup\b|&\s*$")
_VAR_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_]\w*)=(.*?)\s*$")
_VAR_USE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Fortsetzungszeilen (Backslash) zu logischen Zeilen; (Startzeile 1-basiert, Text)."""
    out, buf, start = [], "", 0
    for n, raw in enumerate(text.splitlines(), 1):
        if not buf:
            start = n
        line = raw.rstrip()
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        out.append((start, buf + line))
        buf = ""
    if buf:
        out.append((start, buf))
    return out


def _shell_vars(text: str) -> dict[str, str]:
    vars_: dict[str, str] = {}
    for _, line in _logical_lines(text):
        m = _VAR_ASSIGN.match(line)
        if m and " " not in m.group(2).strip("'\""):
            vars_[m.group(1)] = m.group(2).strip("'\"")
    return vars_


def _tokens(line: str, vars_: dict[str, str] | None = None) -> list[str]:
    line = _LOG_RE.sub(" ", line)
    if vars_:
        line = _VAR_USE.sub(lambda m: vars_.get(m.group(1) or m.group(2), m.group(0)), line)
    try:
        return shlex.split(line, comments=False)
    except ValueError:
        return line.split()


def _parse(tokens: list[str]) -> tuple[list[str], dict[str, str | None]]:
    """(Praefix bis zum ersten --Schalter, {Schalter: Wert})."""
    i = next((k for k, t in enumerate(tokens) if t.startswith("--")), len(tokens))
    prefix, flags = tokens[:i], {}
    while i < len(tokens):
        flag, vals = tokens[i], []
        i += 1
        while i < len(tokens) and not tokens[i].startswith("--"):
            vals.append(tokens[i])
            i += 1
        flags[flag] = " ".join(vals) if vals else None
    return prefix, flags


def _flag_diff(doc: dict, cmd: dict) -> list[dict]:
    diff = []
    for f in sorted(set(doc) | set(cmd)):
        if f in ALLOWED_FLAGS:
            continue
        if f not in doc:
            diff.append({"kind": "added", "flag": f, "doc": None, "command": cmd[f]})
        elif f not in cmd:
            diff.append({"kind": "removed", "flag": f, "doc": doc[f], "command": None})
        elif doc[f] != cmd[f]:
            diff.append({"kind": "changed", "flag": f, "doc": doc[f], "command": cmd[f]})
    return diff


def forbidden_flags(command: str, repo_files: dict[str, str]) -> list[str]:
    """Schalter im Befehl, die laut Name oder Hilfetext (add_argument(..., help=...) in repo_files) auf Training mit allen Daten zielen."""
    _, flags = _parse(_tokens(command))
    helps: dict[str, str] = {}
    for text in repo_files.values():
        for chunk in text.split("add_argument(")[1:]:
            chunk = re.split(r"\n\s*parser\.|\n\s*args\s*=", chunk, maxsplit=1)[0]
            names = re.findall(r"""['"](--[\w-]+)['"]""", chunk)
            m = re.search(r"""help\s*=\s*(['"])(.*?)\1""", chunk, re.S)
            for n in names:
                helps[n] = m.group(2) if m else ""
    hits = []
    for f in flags:
        if f in FORBIDDEN_FLAG_NAMES:
            hits.append(f"{f}: Schalter zielt auf Training mit allen Daten (Name)")
            continue
        for p in FORBIDDEN_HELP_PATTERNS:
            if re.search(p, helps.get(f, ""), re.I):
                hits.append(f"{f}: Hilfetext '{helps[f][:80]}' zielt auf Training mit allen Daten oder Testdaten")
                break
    return hits


def check_command(command: str, repo_files: dict[str, str], source: dict) -> dict:
    """P1: Der Befehl muss (nach Aufloesen einfacher Shell-Variablen) woertlich in source["file"] stehen.

    Rueckgabe: documented, diff (added/removed/changed je Schalter), source (Datei, Zeile der besten Fundstelle), doc_line,
    forbidden (Liste), valid (kein verbotener Schalter). Abweichungen ausser den erlaubten (Geraet, Pfade, Log) werden gelistet, nicht abgelehnt.
    """
    text = repo_files.get(source["file"], "")
    cmd_prefix, cmd_flags = _parse(_tokens(command))
    keys = [t for t in cmd_prefix if t.endswith(".py")] or cmd_prefix[-1:]
    vars_ = _shell_vars(text)
    best: tuple[int, int, str, list[dict]] | None = None
    for line_no, line in _logical_lines(text):
        toks = _tokens(line, vars_)
        prefix, flags = _parse(toks)
        if not any(k in prefix for k in keys):
            continue
        diff = _flag_diff(flags, cmd_flags)
        if prefix != cmd_prefix:
            diff.insert(0, {"kind": "changed", "flag": "<command>", "doc": " ".join(prefix), "command": " ".join(cmd_prefix)})
        score = (len(diff), 0 if line_no == source.get("line") else 1)
        if best is None or score < (best[0], best[1]):
            best = (score[0], score[1], f"{line_no}\t{line.strip()}", diff)
    forb = forbidden_flags(command, repo_files)
    if best is None:
        return {"documented": False, "diff": [{"kind": "missing", "flag": "<command>", "doc": None, "command": command}],
                "source": dict(source), "doc_line": None, "forbidden": forb, "valid": not forb}
    line_no, doc_line = best[2].split("\t", 1)
    return {"documented": not best[3], "diff": best[3], "source": {"file": source["file"], "line": int(line_no)},
            "doc_line": doc_line, "forbidden": forb, "valid": not forb}


# ---- P3: Selbstpruefung -------------------------------------------------------------------------------------------------

def self_check(own: float | None, harness_metrics: dict, metrics_droplast: dict | None, test_loader: dict, metric: str = "mse") -> dict:
    """Eigenausgabe des Autorencodes gegen den Harness auf denselben Fenstern: drop_last-Teilmenge zur Batchgroesse, sonst alle Fenster."""
    out: dict = {"own": own, "tolerance": SELF_CHECK_TOLERANCE, "test_loader": test_loader}
    if own is None:
        return {**out, "status": "nicht geprueft", "reason": "Autorencode gibt keine Test-Metrik aus", "harness": None, "distance": None, "compared_with": None}
    if test_loader.get("drop_last"):
        entry = (metrics_droplast or {}).get(str(test_loader.get("batch")))
        if not entry or metric not in entry:
            return {**out, "status": "nicht geprueft", "harness": None, "distance": None, "compared_with": None,
                    "reason": f"Harness hat keine drop_last-Auswertung fuer Batch {test_loader.get('batch')}"}
        harness, how = entry[metric], f"drop_last_{test_loader['batch']}"
    else:
        harness, how = harness_metrics[metric], "all_windows"
    d = abs(own - harness)
    return {**out, "status": "geprueft" if d <= SELF_CHECK_TOLERANCE else "ungueltig", "harness": harness, "distance": d, "compared_with": how,
            "reason": None if d <= SELF_CHECK_TOLERANCE else "harness and author code do not measure the same thing"}


# ---- Umgebung -----------------------------------------------------------------------------------------------------------

def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def env_diff(requirements_text: str, freeze_text: str) -> list[dict]:
    """Pakete aus der Abhaengigkeitsdatei des Repos gegen die installierten Versionen (pip freeze). Nur Abweichungen."""
    installed = {_norm(n): v for n, _, v in (l.partition("==") for l in freeze_text.splitlines() if "==" in l)}
    diffs = []
    for line in requirements_text.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*(==|>=|<=|~=|>|<)?\s*([\w.+\-]*)", line.split("#")[0])
        if not m or not m.group(1):
            continue
        name, op, ver = _norm(m.group(1)), m.group(2), m.group(3)
        have = installed.get(name)
        if have is None:
            diffs.append({"package": name, "required": (op or "") + ver or None, "installed": None})
        elif op == "==" and have.split("+")[0] != ver:
            diffs.append({"package": name, "required": f"=={ver}", "installed": have})
    return diffs


# ---- Befehle ------------------------------------------------------------------------------------------------------------

def _timed(cmd: str, marker: str = "__WALL_MS__") -> str:
    return f"S=$(date +%s%N); {cmd}; RC=$?; E=$(date +%s%N); echo \"{marker} $(( (E-S)/1000000 ))\"; exit $RC"


def _marker(text: str, name: str = "__WALL_MS__") -> str | None:
    m = re.search(rf"^{name} (.*)$", text, re.M)
    return m.group(1).strip() if m else None


def _wall_s(text: str) -> float | None:
    ms = _marker(text)
    return round(int(ms) / 1000, 1) if ms else None


def _pip(spec: UnitSpec) -> str:
    # numpy<2 (aeltere Repos nutzen np.Inf); torch wie im Task (CPU)
    return "pip install -q 'numpy<2' " + " ".join(shlex.quote(p) for p in spec.pip_packages) + " torch --extra-index-url https://download.pytorch.org/whl/cpu"


def _clone(spec: UnitSpec) -> str:
    return (f"git clone -q {spec.repo_url} {REPO_DIR} && git -C {REPO_DIR} checkout -q {spec.commit} && "
            f"test \"$(git -C {REPO_DIR} rev-parse HEAD)\" = {spec.commit}")


APT = "apt-get update -qq && apt-get install -y -qq git ca-certificates > /dev/null"
NO_RAW_CHECK = ("if [ -n \"$(find / -xdev -name 'ETTh1*' 2>/dev/null)\" ]; then "
                "echo 'Setup-Check: Rohdatei ETTh1* im Vorhersage-Snapshot' >&2; exit 1; fi")
FIND_RAW = "find / -xdev -name 'ETTh1*' 2>/dev/null; echo __FIND_DONE__"


def training_setup_cmds(spec: UnitSpec) -> list[str]:
    from .harness import etth1_data
    dest = f"{REPO_DIR}/{spec.data_dir}"
    fetch = (f"mkdir -p {dest} && python -c \"import urllib.request; urllib.request.urlretrieve('{etth1_data.URL}', '{dest}/ETTh1.csv')\" && "
             f"echo '{ETTH1_SHA256}  {dest}/ETTh1.csv' | sha256sum -c -")
    return [APT, _pip(spec), _clone(spec), fetch]


def predict_setup_cmds(spec: UnitSpec) -> list[str]:
    """Wie der Task (etth1_task().setup_cmds: laedt die Rohdatei, erzeugt /data/etth1_public.npz, loescht die Rohdatei) plus Repo ohne Daten."""
    return [APT, _pip(spec), *etth1_task().setup_cmds, _clone(spec), NO_RAW_CHECK]


def train_cmd(spec: UnitSpec, script_override: str | None = None, extra_flag: str = "") -> str:
    command = spec.command
    if script_override:
        toks = shlex.split(command)
        toks = [script_override if t == spec.script else t for t in toks]
        command = " ".join(shlex.quote(t) for t in toks)
    py = f"PYTHONPATH={REPO_DIR} unshare -n {command}{extra_flag}"
    return f"mkdir -p {OUTPUT} && cd {REPO_DIR} && " + _timed(py)


def git_cmd() -> str:
    return f"cd {REPO_DIR} && echo \"__GIT_STATUS__[$(git status --short)]\" && echo \"__GIT_DIFFSTAT__[$(git diff --stat)]\""


def _bracket(text: str, name: str) -> str | None:
    m = re.search(rf"^{name}\[(.*?)\]\s*$", text, re.M | re.S)
    return m.group(1) if m else None


def predict_cmd(smoke: bool = False) -> str:
    env = f"NACHLAUF_CKPT={CKPT_PATH}" + (" SMOKE=1" if smoke else "")
    return f"mkdir -p {OUTPUT} && cd {REPO_DIR} && " + _timed(f"{env} unshare -n python {ADAPTER_PATH}")


def make_seed_copy(original: str, regex: str, template: str, seed: int) -> str:
    """Kopie des Startskripts, in der genau die Zeile `regex` ersetzt ist."""
    lines = original.splitlines(keepends=True)
    hits = [i for i, l in enumerate(lines) if re.fullmatch(regex, l.rstrip("\r\n"))]
    if len(hits) != 1:
        raise ValueError(f"Seed-Zeile {regex!r} kommt {len(hits)}-mal im Startskript vor (erwartet: genau 1)")
    new = list(lines)
    new[hits[0]] = template.format(seed=seed) + ("\n" if lines[hits[0]].endswith("\n") else "")
    changed = [l for l in difflib.unified_diff(lines, new, n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    if len(changed) != 2:
        raise ValueError("Seed-Kopie unterscheidet sich nicht in genau einer Zeile")
    return "".join(new)


# ---- Ablauf -------------------------------------------------------------------------------------------------------------

def _load_public_local() -> dict:
    """Die oeffentliche Datei des Tasks lokal neu erzeugen (Orchestrator-Seite, gleiche Funktion wie im Snapshot)."""
    import tempfile

    from .harness import etth1_data
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = etth1_data.download(Path(tmp))
        out = Path(tmp) / "pub.npz"
        etth1_data.prepare_public(csv_path, out)
        return independence.load_npz_bytes(out.read_bytes())


class _Run:
    """Zustand eines Laufs: Sandbox, Spec, Ereignisse, Ausgabe."""

    def __init__(self, sb, spec: UnitSpec, events: EventLog, log, public_loader, rng, np_rng):
        self.sb, self.spec, self.events, self.log = sb, spec, events, log
        self.public_loader, self.rng, self.np_rng = public_loader, rng, np_rng
        self.train_snap = self.pred_snap = ""
        self.repo_files: dict[str, str] = {}
        self.script_text = ""

    def emit(self, type: str, **data) -> None:
        self.events.write(type, unit=self.spec.unit, **data)
        self.log(f"[{type}] " + json.dumps(data, ensure_ascii=False, default=str)[:300])

    # -- einmalig je Einheit
    def probe_training(self) -> dict:
        sb, spec = self.sb, self.spec
        b = sb.branch(self.train_snap)
        want = {spec.command_source["file"], spec.script, *spec.argparse_files, *spec.requirements_files}
        for path in sorted(want):
            r = sb.run(b, f"cat {REPO_DIR}/{path}", 60)
            if r.exit_code == 0:
                self.repo_files[path] = r.stdout
        self.script_text = self.repo_files.get(spec.script, "")
        freeze = sb.run(b, "pip freeze", 120).stdout
        req = "\n".join(self.repo_files.get(p, "") for p in spec.requirements_files)
        head = sb.run(b, f"git -C {REPO_DIR} rev-parse HEAD", 30).stdout.strip()
        return {"freeze": freeze, "env_diff": env_diff(req, freeze) if req else [], "head": head}

    def probe_predict_snapshot(self) -> list[str]:
        r = self.sb.run(self.sb.branch(self.pred_snap), FIND_RAW, PROBE_TIMEOUT_S)
        found = [l for l in r.stdout.splitlines() if l and l != "__FIND_DONE__"]
        self.log(f"find / -name 'ETTh1*' im Vorhersage-Snapshot: {found if found else '(leer)'}")
        return found

    # -- je Seed
    def train(self, seed: int) -> dict:
        sb, spec = self.sb, self.spec
        res: dict = {"seed": seed}
        b = sb.branch(self.train_snap)
        script_override, extra = None, ""
        if seed != spec.repo_seed:
            kind = spec.seed["kind"]
            if kind == "copy":
                copy = make_seed_copy(self.script_text, spec.seed["regex"], spec.seed["template"], seed)
                script_override = f"{SEED_DIR}/{Path(spec.script).name}"
                sb.put_files(b, {script_override: copy.encode()})
                d = sb.run(b, f"diff {REPO_DIR}/{spec.script} {script_override} | grep -c '^[<>]'", 60)
                if d.stdout.strip() != "2":
                    return {**res, "status": "ungueltig", "reason": f"seed copy differs in {d.stdout.strip()} diff lines (expected 2)"}
            elif kind == "flag":
                extra = f" {spec.seed['flag']} {seed}"
            else:
                return {**res, "status": "skipped", "reason": "no seed method"}
        tr = sb.run(b, train_cmd(spec, script_override, extra), TRAIN_TIMEOUT_S)
        res.update(train_exit=tr.exit_code, train_wall_s=_wall_s(tr.stdout), timed_out=tr.timed_out)
        own = re.findall(spec.own_metric_pattern, tr.stdout, re.M)
        res["own_metric"] = float(own[-1]) if own else None
        self.emit("nachlauf_train", seed=seed, exit_code=tr.exit_code, wall_s=res["train_wall_s"], own_metric=res["own_metric"], timed_out=tr.timed_out)
        if tr.timed_out:
            return {**res, "status": "timeout", "reason": f"training > {TRAIN_TIMEOUT_S}s"}
        if tr.exit_code != 0:
            return {**res, "status": "fehler", "reason": (tr.stderr or tr.stdout).strip()[-1500:]}
        g = sb.run(b, git_cmd(), 60)
        status, stat = _bracket(g.stdout, "__GIT_STATUS__"), _bracket(g.stdout, "__GIT_DIFFSTAT__")
        res["git_clean"] = g.exit_code == 0 and status == "" and stat == ""
        if not res["git_clean"]:
            return {**res, "status": "ungueltig", "reason": f"repo files changed by the run: status={status!r} diffstat={stat!r}"}
        ls = sb.run(b, f"ls -1 {REPO_DIR}/{spec.checkpoint_glob}", 30)
        files = [l for l in ls.stdout.splitlines() if l.strip()]
        if ls.exit_code != 0 or len(files) != 1:
            return {**res, "status": "fehler", "reason": f"expected exactly one checkpoint for {spec.checkpoint_glob}, found {len(files)}"}
        res["checkpoint"] = files[0]
        res["ckpt_bytes"] = sb.get_file(b, files[0])
        res["status"] = "ok"
        return res

    def _predict_branch(self, ckpt: bytes, extra_files: dict[str, bytes] | None = None) -> str:
        b = self.sb.branch(self.pred_snap)
        self.sb.put_files(b, {ADAPTER_PATH: self.spec.adapter_source.encode(), CKPT_PATH: ckpt, **(extra_files or {})})
        return b

    def smoke(self, ckpt: bytes) -> str | None:
        """Formpruefung (SMOKE=1, erste 8 Fenster) in eigenem Branch. None = bestanden."""
        sb, task = self.sb, etth1_task()
        b = self._predict_branch(ckpt)
        r = sb.run(b, predict_cmd(smoke=True), SMOKE_TIMEOUT_S)
        if r.timed_out or r.exit_code != 0:
            return "smoke run failed: " + ("timeout" if r.timed_out else (r.stderr or r.stdout).strip()[-1500:])
        try:
            out = sb.get_file(b, f"{OUTPUT}/predictions.csv")
        except Exception as exc:
            return f"smoke run wrote no predictions.csv ({exc})"
        return check_smoke_output(out, task.smoke_ids, task.smoke_dims, task.smoke_dim_names)

    def predict_and_measure(self, ckpt: bytes, seed: int) -> dict:
        sb, spec, task = self.sb, self.spec, etth1_task()
        b = self._predict_branch(ckpt)
        pr = sb.run(b, predict_cmd(), PREDICT_TIMEOUT_S)
        res: dict = {"predict_exit": pr.exit_code, "predict_wall_s": _wall_s(pr.stdout), "branch": b}
        self.emit("nachlauf_predict", seed=seed, exit_code=pr.exit_code, wall_s=res["predict_wall_s"], timed_out=pr.timed_out)
        if pr.timed_out or pr.exit_code != 0:
            return {**res, "status": "timeout" if pr.timed_out else "fehler", "reason": (pr.stderr or pr.stdout).strip()[-1500:]}
        # Harness frisch, erst jetzt, in ein Verzeichnis mit Zufallsnamen; etth1_eval.py unveraendert (mit Netz)
        hdir = f"{SANDBOX_HARNESS_DIR}-{uuid.uuid4().hex[:10]}"
        sb.put_files(b, {f"{hdir}/{n}": (HARNESS_DIR / n).read_bytes() for n in task.harness_files})
        e = sb.run(b, task.eval_cmd.format(harness_dir=hdir), EVAL_TIMEOUT_S)
        try:
            out = json.loads(e.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return {**res, "status": "ungueltig", "reason": f"harness output unreadable: {e.stderr[-300:] or e.stdout[-300:]}"}
        if e.exit_code != 0 or out.get("status") != "ok":
            return {**res, "status": "ungueltig", "reason": out.get("reason") or "harness failed"}
        res.update(metrics=out["metrics"], baselines=out.get("baselines"), metrics_droplast=out.get("metrics_droplast"),
                   diagnostics=out.get("diagnostics"))
        res["full_pred"] = independence.parse_predictions(sb.get_file(b, f"{OUTPUT}/predictions.csv"))
        res["status"] = "ok"
        return res

    def independence(self, ckpt: bytes, full_pred, offsets: list[int], seed: int) -> dict:
        """P2: je Versatz ein frischer Branch, Probe ersetzt /data/etth1_public.npz."""
        sb = self.sb
        public = self.public_loader()
        results = []
        for r in offsets:
            arrays, mapping = independence.make(public, r, self.np_rng)
            m = len(mapping)
            b = self._predict_branch(ckpt, {"/data/etth1_public.npz": independence.to_npz_bytes(arrays)})
            chk = sb.run(b, "python -c \"import numpy as np; print(np.load('/data/etth1_public.npz')['test_inputs'].shape[0])\"; "
                            "find / -xdev -name 'etth1_public*' 2>/dev/null | wc -l", 120)
            parts = chk.stdout.split()
            if parts[:2] != [str(m), "1"]:
                results.append({"offset": r, "ok": False, "max_abs_diff": None, "n_compared": 0, "reason": f"probe file not in place in the branch: {chk.stdout.strip()[:80]}"})
                continue
            run = sb.run(b, predict_cmd(), PREDICT_TIMEOUT_S)
            if run.timed_out or run.exit_code != 0:
                results.append({"offset": r, "ok": False, "max_abs_diff": None, "n_compared": 0, "reason": "adapter failed on probe: " + (run.stderr or run.stdout).strip()[-300:]})
                continue
            probe_pred = independence.parse_predictions(sb.get_file(b, f"{OUTPUT}/predictions.csv"), m)
            cmp = independence.compare(full_pred, probe_pred, mapping)
            results.append({"offset": r, **cmp})
            self.log(f"P2 offset {r}: windows in probe {m} (mapping stays in the orchestrator), max diff {cmp['max_abs_diff']}")
        summary = independence.summarize(results)
        self.emit("nachlauf_independence", seed=seed, ok=summary["ok"], offsets=summary["offsets"], max_abs_diff=summary["max_abs_diff"], reason=summary["reason"])
        return summary


def run_unit(sandbox, spec: UnitSpec, seeds: list[int], *, offsets: int | list[int] = independence.N_OFFSETS, out_dir: Path | None = None,
             public_loader=None, rng=None, np_rng=None, log=print) -> Dossier:
    """Fuehrt eine Einheit aus (Hand-Spec) und liefert das Dossier (mode="nachlauf"). out_dir: Ordner fuer events.jsonl (und dossier.json bei save)."""
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    claim = Claim(id=spec.claim_id or f"{spec.unit}_etth1_96_{spec.metric}", text=spec.claim_text, metric=spec.metric, target=spec.target,
                  tolerance=spec.tolerance, source="Tabelle 2 (Hand-Spec)", origin="paper")
    dossier = Dossier(paper_id=spec.paper_id, claims=[claim], mode="nachlauf", isolation=list(NACHLAUF_ISOLATION),
                      repo={"url": spec.repo_url, "commit": spec.commit})
    info: dict = {"repo": spec.repo_url, "commit": spec.commit, "command": spec.command, "command_source": spec.command_source,
                  "command_documented": None, "command_diff": [], "env_diff": [], "seed_method": spec.seed_note or spec.seed.get("kind"),
                  "own_metric": {}, "self_check": {}, "independence": {}, "rejected_by": {}, "test_loader": spec.test_loader, "predict_snapshot_raw_files": None}
    dossier.nachlauf[claim.id] = info
    notes: list[str] = []
    events = EventLog(Path(out_dir) / "events.jsonl" if out_dir else None)
    run = _Run(sandbox, spec, events, log, public_loader or _load_public_local, rng, np_rng)
    measured: list[float] = []
    all_metrics: dict[str, list[float]] = {}

    def finish() -> Dossier:
        from .loop import Orchestrator   # Urteilsregel des Nachbaus unveraendert wiederverwenden (die Methode nutzt `self` nicht)
        base = None
        verdict = Orchestrator._verdict(None, claim, measured, len(seeds), notes, base, {k: statistics.mean(v) for k, v in all_metrics.items()})
        dossier.verdicts.append(verdict)
        return dossier

    # statische Pruefung des Adapters (einmal, im Orchestrator)
    violations = find_violations(spec.adapter_source)
    if violations:
        notes.append("Adapter abgelehnt (statische Codepruefung): " + "; ".join(violations))
        info["rejected_by"][str(seeds[0] if seeds else 0)] = ["static"]
        dossier.attempts.append(Attempt(claim.id, "", "", "", seeds[0] if seeds else 0, "ungueltig", None, "; ".join(violations)))
        return finish()
    code_hash = hashlib.sha256(spec.adapter_source.encode()).hexdigest()[:12]

    run.train_snap = sandbox.base_snapshot(etth1_task().image, training_setup_cmds(spec), label=f"nachlauf-{spec.unit}-train")
    run.pred_snap = sandbox.base_snapshot(etth1_task().image, predict_setup_cmds(spec), label=f"nachlauf-{spec.unit}-predict")
    raw = run.probe_predict_snapshot()
    info["predict_snapshot_raw_files"] = raw
    if raw:
        notes.append(f"Vorhersage-Snapshot enthaelt Rohdateien {raw}: Abbruch")
        dossier.attempts.append(Attempt(claim.id, "", run.pred_snap, code_hash, seeds[0] if seeds else 0, "ungueltig", None, "raw data in predict snapshot"))
        return finish()
    probe = run.probe_training()
    info["env_diff"] = probe["env_diff"]
    if probe["head"] != spec.commit:
        notes.append(f"Repo-Commit im Snapshot {probe['head']} != {spec.commit}")
        return finish()

    # P1
    p1 = check_command(spec.command, run.repo_files, spec.command_source)
    info.update(command_documented=p1["documented"], command_diff=p1["diff"], command_source=p1["source"])
    if not p1["documented"]:
        dossier.warnings.append("Startbefehl weicht von der Dokumentation ab: " + "; ".join(
            f"{d['flag']}: {d['doc']!r} -> {d['command']!r}" for d in p1["diff"]))
    if not p1["valid"]:
        notes.append("P1: Startbefehl abgelehnt (" + "; ".join(p1["forbidden"]) + ")")
        info["rejected_by"][str(spec.repo_seed or 0)] = ["P1"]
        dossier.attempts.append(Attempt(claim.id, "", run.train_snap, code_hash, spec.repo_seed or 0, "ungueltig", None, "; ".join(p1["forbidden"])))
        return finish()
    if probe["env_diff"]:
        dossier.warnings.append("Umgebung weicht von den Abhaengigkeitsdateien des Repos ab: " + "; ".join(
            f"{d['package']} {d['installed'] or 'fehlt'} statt {d['required']}" for d in probe["env_diff"]))
    usable = [s for s in seeds if s == spec.repo_seed or spec.seed["kind"] in ("copy", "flag")]
    if len(usable) < len(seeds):
        notes.append(f"{len(seeds) - len(usable)} Seeds uebersprungen (kein Seed-Weg ausser dem Repo-Seed)")
    if len(usable) <= 1:
        dossier.warnings.append("eine Messung, keine Streuung")
    offs = offsets if isinstance(offsets, list) else independence.random_offsets(offsets, rng)

    for seed in usable:
        def attempt(status, measured_=None, reason=None, metrics=None, droplast=None, diag=None, branch=""):
            dossier.attempts.append(Attempt(claim.id, branch, run.train_snap, code_hash, seed, status, measured_, reason,
                                            metrics=metrics, metrics_droplast=droplast, diagnostics=diag))
        try:
            t = run.train(seed)
            info["own_metric"][str(seed)] = t.get("own_metric")
            if t["status"] != "ok":
                notes.append(f"Seed {seed}: Training {t['status']} ({(t.get('reason') or '')[:120]})")
                attempt("ungueltig" if t["status"] == "ungueltig" else t["status"], reason=t.get("reason"))
                continue
            ckpt = t["ckpt_bytes"]
            err = run.smoke(ckpt)
            if err:
                notes.append(f"Seed {seed}: Formpruefung des Adapters: {err[:120]}")
                attempt("fehler", reason=err)
                continue
            p = run.predict_and_measure(ckpt, seed)
            if p["status"] != "ok":
                notes.append(f"Seed {seed}: Vorhersage/Harness {p['status']} ({(p.get('reason') or '')[:120]})")
                attempt("ungueltig" if p["status"] == "ungueltig" else p["status"], reason=p.get("reason"), branch=p.get("branch", ""))
                continue
            sc = self_check(t.get("own_metric"), p["metrics"], p.get("metrics_droplast"), spec.test_loader, spec.metric)
            info["self_check"][str(seed)] = sc
            run.emit("nachlauf_self_check", seed=seed, status=sc["status"], distance=sc["distance"], compared_with=sc["compared_with"])
            # P2 laeuft auch bei P3-Verstoss: jede Pruefung meldet ihr eigenes Ergebnis (Negativkontrolle: Abschreiber faellt bei P2 UND P3 auf)
            ind = run.independence(ckpt, p["full_pred"], offs, seed)
            info["independence"][str(seed)] = ind
            failed, reasons = [], []
            if sc["status"] == "ungueltig":
                failed.append("P3")
                reasons.append(f"P3: {sc['reason']} (distance {sc['distance']:.4f})")
            if not ind["ok"]:
                failed.append("P2")
                reasons.append(f"P2: {ind['reason']}")
            if failed:
                info["rejected_by"][str(seed)] = failed
                notes.append(f"Seed {seed}: ungueltig ({', '.join(failed)})")
                attempt("ungueltig", reason="; ".join(reasons), metrics=p["metrics"], droplast=p.get("metrics_droplast"), branch=p["branch"])
                continue
            value = p["metrics"][spec.metric]
            measured.append(value)
            for k, v in p["metrics"].items():
                all_metrics.setdefault(k, []).append(v)
            attempt("stabil" if judge(value, spec.target, spec.tolerance) == "reproduziert" else "abweichung", value,
                    metrics=p["metrics"], droplast=p.get("metrics_droplast"), diag=p.get("diagnostics"), branch=p["branch"])
        except Exception as exc:   # Infrastrukturfehler der Sandbox: Seed als Fehler, Lauf geht weiter
            notes.append(f"Seed {seed}: {type(exc).__name__}: {str(exc)[:150]}")
            attempt("fehler", reason=f"{type(exc).__name__}: {str(exc)[:300]}")
    if any(sc["status"] == "nicht geprueft" for sc in info["self_check"].values()):
        dossier.warnings.append("Zuordnung nicht geprueft (keine Eigenausgabe des Autorencodes)")
    return finish()
