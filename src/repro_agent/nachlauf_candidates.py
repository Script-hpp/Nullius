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
IMPORT_GREP_CMD = "cd /repo && git grep -hE '^[[:space:]]*(import|from)[[:space:]]+[A-Za-z_]' -- '*.py' | head -20000"
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
        """Aufgeloeste Fassung (Variablen und Schleifenwerte eingesetzt), nicht der Rohtext; Schleifenwerte stehen dahinter."""
        return f"[{self.k}] {self.file}:{self.line}: {self.command[:300]}" + (f"  (loop {self.note})" if self.note else "")


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


HORIZON_RE = re.compile(r"(?:horizont|horizon|prediction length|forecast(?:ing)? length|pred(?:iction)?[_ ]len)\D{0,15}(\d{1,4})", re.I)
INPUT_RE = re.compile(r"(?:eingabefenster|eingabel[a\u00e4]nge|input (?:window|length)|look-?back(?: window| length)?|history)\D{0,15}(\d{1,4})", re.I)
PAGE_SIZE = MAX_CANDIDATES_PROMPT
MAX_MORE = 2     # das Modell darf zweimal die naechsten PAGE_SIZE Kandidaten anfordern


def claim_words(claim_text: str, dataset: str) -> dict:
    """Aus dem Claim-Text (nichts fest verdrahtet): Modellname (Wort(e) vor \"erreicht\"/\"reaches\", sonst erstes Wort; Teile mit Buchstaben, z. B. \"Name/42\" ->
    \"Name\"), Datensatzname, Horizont und Eingabelaenge (Zahl hinter dem Stichwort), alle ganzen Zahlen des Claims."""
    text = claim_text or ""
    m = re.match(r"\s*(.+?)\s+(?:erreicht|reaches|achieves|reach|achieve|erzielt|liefert)\b", text, re.I)
    name = m.group(1).strip() if m else ((re.match(r"\s*([A-Za-z][\w./-]*)", text) or [None, ""])[1])
    if name.lower() in STOPWORDS:
        name = ""
    parts = [p for p in re.split(r"[/\s]+", name) if re.search(r"[A-Za-z]{3,}", p) and p.lower() not in STOPWORDS]
    first = lambda rx: (int(mm.group(1)) if (mm := rx.search(text)) else None)  # noqa: E731
    nums = re.findall(r"(?<![\d.,])\d{2,4}(?![\d.,])", text)
    return {"model": name, "model_parts": parts, "dataset": dataset or "", "horizon": first(HORIZON_RE), "input_len": first(INPUT_RE),
            "numbers": sorted(set(nums))}


def _switch_has_value(command: str, value: int) -> bool:
    toks = nl._tokens(command)
    v = str(value)
    return any((t.startswith("-") and i + 1 < len(toks) and toks[i + 1] == v) or (t.startswith("-") and t.split("=", 1)[-1] == v and "=" in t)
               for i, t in enumerate(toks))


def _has_number(hay: str, value: int) -> bool:
    return bool(re.search(rf"(?<![\d.]){value}(?![\d.])", hay))


def score(c: Candidate, words: dict) -> int:
    """Passung eines aufgeloesten Kandidaten zum Claim: Modellname im Befehl oder Text (+4, im Pfad +1), Datensatz (+3, Pfad +1), Horizont als Wert eines
    Schalters (+3, sonst als Zahl im Befehl +1), Eingabelaenge ebenso (+2/+1). Nennt der Claim weder Horizont noch Eingabelaenge, zaehlen seine ganzen Zahlen +1."""
    path, cmd, text = c.file.lower(), c.command.lower(), c.text.lower()
    s = 0
    parts = [p.lower() for p in words.get("model_parts") or ([words["model"]] if words.get("model") else [])]
    if parts:
        s += 4 if any(p in cmd or p in text for p in parts) else 0
        s += 1 if any(p in path for p in parts) else 0
    d = (words.get("dataset") or "").lower()
    if d:
        s += 3 if (d in cmd or d in text) else 0
        s += 1 if d in path else 0
    for key, strong in (("horizon", 3), ("input_len", 2)):
        v = words.get(key)
        if v is not None:
            s += strong if _switch_has_value(c.command, v) else (1 if _has_number(cmd, v) else 0)
    if words.get("horizon") is None and words.get("input_len") is None:
        s += sum(1 for n in words.get("numbers", []) if _has_number(cmd + " " + text, int(n)))
    return s


def order_candidates(cands: list[Candidate], words: dict) -> list[Candidate]:
    """Punktwert je aufgeloestem Kandidaten, gruppiert nach Datei: Dateien absteigend nach dem besten Punktwert, darin absteigend nach Punktwert und Zeile.
    Nummern 1..n ueber die ganze Liste (stabil, auch fuer spaeter angeforderte Seiten)."""
    for c in cands:
        c.score = score(c, words)
    best: dict[str, int] = {}
    for c in cands:
        best[c.file] = max(best.get(c.file, 0), c.score)
    ordered = sorted(cands, key=lambda c: (-best[c.file], c.file, -c.score, c.line, c.command))      # Befehl als letzter Gleichstand: unabhaengig von der Eingabereihenfolge
    for i, c in enumerate(ordered, 1):
        c.k = i
    return ordered


def narrow(cands: list[Candidate], words: dict, n: int = PAGE_SIZE) -> tuple[list[Candidate], dict]:
    """Geordnete Liste (`order_candidates`); die ersten n kommen in den Prompt, der Rest wird gezaehlt (weitere Seiten auf Anforderung des Modells)."""
    ordered = order_candidates(cands, words)
    shown = ordered[:n]
    return shown, candidate_info(ordered, len(shown), words, pages=1)


def candidate_info(ordered: list[Candidate], n_shown: int, words: dict, pages: int, more_requests: int = 0) -> dict:
    return {"n_total": len(ordered), "n_shown": n_shown, "n_dropped": len(ordered) - n_shown, "words": words, "pages": pages, "more_requests": more_requests,
            "top_scores": [f"{c.file}:{c.line}={c.score}" for c in ordered[:10]], "dropped_sample": [f"{c.file}:{c.line}" for c in ordered[n_shown:n_shown + 20]]}


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
        return (f"[{self.k}] (line {self.line} of {self.file}) {self.text[:240]}  "
                f"({'fixed number in the script' if self.kind == 'copy' else 'command-line switch ' + self.flag})")


def seed_candidates(path: str, text: str) -> list[SeedCandidate]:
    """Nur NUTZBARE Zeilen im Einstiegsskript: Zuweisung einer Zahl an einen Namen mit 'seed', Aufruf mit Zahl, Schalterdefinition mit 'seed'.
    Zeilen, die einen Seed nur benutzen (`random.seed(fix_seed)`), und Zeilen, die nicht genau einmal vorkommen, werden nicht angeboten."""
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
    lines = [l.strip() for l in text.splitlines()]
    out = [c for c in out if c.kind == "flag" or lines.count(c.text.strip()) == 1]      # copy: Zeile eindeutig (seed_spec braucht genau eine)
    out.sort(key=lambda c: c.line)
    out = out[:MAX_SEED_LINES]
    for i, c in enumerate(out, 1):
        c.k = i
    return out


def resolve_seed_choice(k: int | None, sc: list[SeedCandidate], text: str) -> tuple[SeedCandidate | None, str]:
    """Wahl des Modells -> Kandidat. Nummer der Liste; sonst als Zeilennummer des Skripts gelesen (Verwechslung); zeigt diese Zeile nur auf eine Variable
    (`random.seed(fix_seed)`), wird deterministisch auf die Zuweisung dieser Variablen aufgeloest. Rueckgabe (Kandidat oder None, Weg)."""
    if k is None:
        return None, "none"
    hit = next((c for c in sc if c.k == k), None)
    if hit:
        return hit, "number"
    hit = next((c for c in sc if c.line == k), None)
    if hit:
        return hit, "line_number"
    lines = text.splitlines()
    if 1 <= k <= len(lines):
        names = set(re.findall(r"[A-Za-z_]\w*", lines[k - 1]))
        for c in sc:
            m = SEED_ASSIGN_RE.match(c.text) if c.kind == "copy" else None
            if m and m.group(1) in names:
                return c, "resolved_variable"
    return None, "unusable"


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
