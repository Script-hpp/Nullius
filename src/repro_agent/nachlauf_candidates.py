"""Modus Nachlauf: dokumentierte Befehle, Seed-Zeilen und Test-Loader-Zeilen deterministisch finden (Auftrag 38c).

Das Modell erfindet keinen Befehl. Der Orchestrator durchsucht die verfolgten Textdateien des Repos und legt eine nummerierte Liste
`[k] datei:zeile: text` vor; das Modell waehlt per Nummer. Quelle (Datei, Zeile, Text) und der einzeilige Befehl kommen aus der Liste, nicht
aus der Antwort des Modells, das Zitat ist damit immer woertlich.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from . import nachlauf as nl

# git grep -E (POSIX): Dateien mit einem Python-Aufruf eines Skripts bzw. mit Test-Loader-Zeilen. Datendateien und Binaeres ausgeschlossen.
CMD_GREP_ERE = r"python[0-9.]*[[:space:]].*\.py"
LOADER_GREP_ERE = r"DataLoader\(|drop_last"
EXCLUDE_PATHSPECS = (":!*.csv", ":!*.json", ":!*.ipynb", ":!*.npy", ":!*.npz", ":!*.pkl", ":!*.pdf", ":!*.svg", ":!*.png", ":!*.jpg", ":!*.log")
MAX_FILE_BYTES = 200_000
MAX_GREP_FILES = 120
MAX_CANDIDATES_PROMPT = 30        # N: mehr Kandidaten werden vorher deterministisch eingegrenzt, der Rest wird gezaehlt
MAX_LOADER_LINES = 40
MAX_SEED_LINES = 20

PY_CALL_RE = re.compile(r"(?:^|[\s`(!$>])(python3?(?:\.\d+)?)(?=\s)")
CUT_RE = re.compile(r"\s(?:&&|\|\||;|\|)\s|\s;|`|\)\s*$")
SEED_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*(#.*)?$")
SEED_CALL_RE = re.compile(r"\bseed\s*\(\s*(-?\d+)\s*\)")
ADD_ARG_SEED_RE = re.compile(r"""add_argument\(\s*['"](--[\w-]*seed[\w-]*)['"][^)]*\)""", re.S | re.I)
DROP_LAST_VALUE_RE = re.compile(r"drop_last\s*=\s*(True|False)\b")
STOPWORDS = {"erreicht", "reaches", "achieves", "reach", "achieve", "mse", "mae", "auf", "bei", "on", "at", "the", "with", "und", "and", "for"}


@dataclass
class Candidate:
    k: int
    file: str
    line: int
    text: str            # Zeile(n) wie in der Datei (Zeilenfortsetzungen zu einer Zeile verbunden), nicht aufgeloest
    command: str         # einzeiliger Befehl (Variablen und Schleifenwerte aufgeloest, Umleitungen entfernt)
    script: str          # Skript des Repos, das der Befehl aufruft
    note: str = ""       # z. B. "loop h=96"
    score: int = 0

    def line_text(self) -> str:
        return f"[{self.k}] {self.file}:{self.line}: {self.text[:300]}" + (f"  (command with {self.note}: {self.command[:300]})" if self.note else "")


def grep_cmd(ere: str) -> str:
    return ("cd /repo && git grep -lIE " + shlex.quote(ere) + " -- . " + " ".join(shlex.quote(p) for p in EXCLUDE_PATHSPECS)
            + f" | head -{MAX_GREP_FILES}")


def _resolve_script(token: str, listing: set[str], by_base: dict[str, list[str]]) -> str:
    t = re.sub(r"^\./", "", token)
    if t in listing:
        return t
    hits = by_base.get(t.rsplit("/", 1)[-1], [])
    return hits[0] if len(hits) == 1 else ""


def extract_candidates(files: dict[str, str], listing: list[str]) -> tuple[list[Candidate], int]:
    """Alle Python-Aufrufe eines Skripts des Repos in den Textdateien. Rueckgabe (Kandidaten, Zahl uebersprungener Zeilen mit nicht aufloesbaren
    Variablen). Schleifen (`for v in a b; do`) und Zeilenfortsetzungen werden aufgeloest (nachlauf._lines_with_vars, dieselbe Aufloesung wie P1)."""
    lset = set(listing)
    by_base: dict[str, list[str]] = {}
    for p in listing:
        by_base.setdefault(p.rsplit("/", 1)[-1], []).append(p)
    out: list[Candidate] = []
    skipped = 0
    seen: set[tuple] = set()
    for path in sorted(files):
        for line_no, line, vars_, combo in nl._lines_with_vars(files[path]):
            pm = PY_CALL_RE.search(line)
            if not pm:
                continue
            seg = line[pm.start(1):]
            cut = CUT_RE.search(seg)
            seg = seg[: cut.start()] if cut else seg
            toks = nl._tokens(seg, vars_)
            if not toks or not re.match(r"^python3?(\.\d+)?$", toks[0]):
                continue
            if any("$" in t for t in toks):
                skipped += 1
                continue
            script = next((r for r in (_resolve_script(t, lset, by_base) for t in toks[1:] if t.endswith(".py")) if r), "")
            if not script:
                continue
            cmd = shlex.join(toks)
            key = (path, line_no, cmd)
            if key in seen:
                continue
            seen.add(key)
            out.append(Candidate(0, path, line_no, re.sub(r"\s+", " ", line.strip()), cmd, script, note=", ".join(f"{v}={x}" for v, x in combo.items())))
    return out, skipped


def claim_words(claim_text: str, dataset: str) -> dict:
    """Modell- und Datensatzname aus dem Claim-Text (nichts fest verdrahtet): erstes Wort des Claims, Datensatzname, ganze Zahlen des Claims."""
    m = re.match(r"\s*([A-Za-z][\w.-]*)", claim_text or "")
    model = m.group(1) if m and m.group(1).lower() not in STOPWORDS else ""
    nums = re.findall(r"(?<![\d.,])\d{2,4}(?![\d.,])", claim_text or "")
    return {"model": model, "dataset": dataset or "", "numbers": sorted(set(nums))}


def score(c: Candidate, words: dict) -> int:
    hay = f"{c.file} {c.text} {c.command}".lower()
    s = 0
    if words["model"] and words["model"].lower() in hay:
        s += 3
    if words["dataset"] and words["dataset"].lower() in hay:
        s += 2
    s += sum(1 for n in words["numbers"] if re.search(rf"(?<![\d.]){n}(?![\d.])", hay))
    return s


def narrow(cands: list[Candidate], words: dict, n: int = MAX_CANDIDATES_PROMPT) -> tuple[list[Candidate], dict]:
    """Hoechstens n Kandidaten in den Prompt; bei mehr werden Zeilen mit Modell-, Datensatz- und Zahlenbezug des Claims bevorzugt. Nummern 1..m."""
    for c in cands:
        c.score = score(c, words)
    total = len(cands)
    if total <= n:
        chosen = list(cands)
    else:
        ranked = sorted(cands, key=lambda c: (-c.score, c.file, c.line))
        chosen = ranked[:n]        # zuerst Zeilen mit Bezug zum Claim, danach in Dateireihenfolge aufgefuellt
    for i, c in enumerate(chosen, 1):
        c.k = i
    dropped = [c for c in cands if c not in chosen]
    return chosen, {"n_total": total, "n_shown": len(chosen), "n_dropped": len(dropped), "words": words,
                    "dropped_sample": [f"{c.file}:{c.line}" for c in dropped[:20]]}


# ---- Seed ------------------------------------------------------------------------------------------------------------------------

@dataclass
class SeedCandidate:
    k: int
    kind: str            # copy (Zuweisung/Aufruf mit Zahl) | flag (Schalterdefinition)
    file: str
    line: int
    text: str
    flag: str = ""
    value: int | None = None

    def line_text(self) -> str:
        return f"[{self.k}] {self.file}:{self.line}: {self.text[:240]}  ({'fixed number in the script' if self.kind == 'copy' else 'command-line switch ' + self.flag})"


def seed_candidates(path: str, text: str) -> list[SeedCandidate]:
    """Zeilen im Einstiegsskript, die einen Seed setzen: Zuweisung einer Zahl an einen Namen mit 'seed', Aufruf mit Zahl, Schalterdefinition mit 'seed'."""
    out: list[SeedCandidate] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        m = SEED_ASSIGN_RE.match(line)
        if m and "seed" in m.group(1).lower():
            out.append(SeedCandidate(0, "copy", path, n, line.strip(), value=int(m.group(2))))
        elif SEED_CALL_RE.search(line) and "add_argument" not in line:
            ints = re.findall(r"(?<![\w.])-?\d+(?![\w.])", line)
            if len(ints) == 1:
                out.append(SeedCandidate(0, "copy", path, n, line.strip(), value=int(ints[0])))
    for m in ADD_ARG_SEED_RE.finditer(text):
        d = re.search(r"default\s*=\s*(-?\d+)", m.group(0))
        n = text.count("\n", 0, m.start()) + 1
        out.append(SeedCandidate(0, "flag", path, n, re.sub(r"\s+", " ", m.group(0)).strip(), flag=m.group(1), value=int(d.group(1)) if d else None))
    out.sort(key=lambda c: c.line)
    out = out[:MAX_SEED_LINES]
    for i, c in enumerate(out, 1):
        c.k = i
    return out


def seed_spec(c: SeedCandidate, text: str) -> dict | None:
    """Spec-Seed aus einem Kandidaten. copy: die Zeile muss genau einmal vorkommen und genau eine ganze Zahl enthalten."""
    if c.kind == "flag":
        return {"kind": "flag", "flag": c.flag, "repo_seed": c.value}
    matches = [l for l in text.splitlines() if l.strip() == c.text.strip()]
    ints = re.findall(r"(?<![\w.])-?\d+(?![\w.])", c.text)
    if len(matches) != 1 or len(ints) != 1:
        return None
    actual = matches[0]
    template = actual.replace("{", "{{").replace("}", "}}")
    template = re.sub(r"(?<![\w.])-?\d+(?![\w.])", "{seed}", template, count=1)
    return {"kind": "copy", "repo_seed": int(ints[0]), "regex": re.escape(actual), "template": template}


# ---- Test-Loader -----------------------------------------------------------------------------------------------------------------

@dataclass
class LoaderLine:
    k: int
    file: str
    line: int
    text: str

    def line_text(self) -> str:
        return f"[{self.k}] {self.file}:{self.line}: {self.text[:240]}"


def loader_lines(files: dict[str, str]) -> list[LoaderLine]:
    """Zeilen mit `drop_last` in den Dateien mit DataLoader/drop_last (deterministisch gefunden)."""
    out: list[LoaderLine] = []
    for path in sorted(files):
        for n, raw in enumerate(files[path].splitlines(), 1):
            if "drop_last" in raw and raw.strip() and not raw.strip().startswith("#"):
                out.append(LoaderLine(0, path, n, raw.strip()))
    out = out[:MAX_LOADER_LINES]
    for i, c in enumerate(out, 1):
        c.k = i
    return out


def drop_last_value(text: str) -> bool | None:
    """True/False, wenn die Zeile den Wert woertlich setzt (`drop_last = True` oder `drop_last=False` als Argument), sonst None (z. B. nur ein Verweis)."""
    m = DROP_LAST_VALUE_RE.search(text)
    return None if not m else m.group(1) == "True"


def argument_default(switch: str, texts: dict[str, str]) -> tuple[int, str, int, str] | None:
    """default= der Argumentdefinition des Schalters: (Wert, Datei, Zeile, Zeilentext)."""
    pat = re.compile(rf"""add_argument\(\s*['"]{re.escape(switch)}['"][^)]*?default\s*=\s*(-?\d+)""", re.S)
    for path in sorted(texts):
        m = pat.search(texts[path])
        if m:
            n = texts[path].count("\n", 0, m.start()) + 1
            return int(m.group(1)), path, n, texts[path].splitlines()[n - 1].strip()
    return None


def switch_value(command: str, switch: str) -> int | None:
    toks = nl._tokens(command)
    for i, t in enumerate(toks):
        if t == switch and i + 1 < len(toks) and re.fullmatch(r"\d+", toks[i + 1]):
            return int(toks[i + 1])
        if t.startswith(switch + "=") and re.fullmatch(r"\d+", t.split("=", 1)[1]):
            return int(t.split("=", 1)[1])
    return None
