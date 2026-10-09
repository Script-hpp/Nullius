"""Quelle fuer Tavily-Rolle 4: das Repo zum Paper.

Aufteilung nach Staerken der Werkzeuge:
- Tavily liest das README (Extract). Das Repo selbst wird in der SANDBOX geklont (`git clone`,
  Commit festgehalten), nicht ueber gerenderte GitHub-Seiten gelesen.
- Der Klon liegt nur in einem eigenen Lese-Snapshot des Orchestrators, NICHT in den Branches
  des Agenten (Modus "Nachbau": der Agent sieht das Original-Repo nie, nur belegte Angaben).
- Ergebnis wird als Fixture gespeichert (Repo + Commit + gelesene Dateien).
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .llm import LLMClient
from .sandbox.base import Sandbox

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "fixtures" / "repos"
MAX_FILES = 12
MAX_FILE_CHARS = 30_000
MAX_IMPORTED = 6   # zusaetzliche, von den gewaehlten Dateien importierte lokale Module
MAX_LISTING = 600
MAX_SUBDIR_FILE_BYTES = 1_000_000   # Unterordner-Klon: groessere Dateien (Daten, Gewichte) kommen nicht in die Dateiliste
MAX_SUBDIR_TOTAL_BYTES = 200_000_000   # Unterordner groesser als das: Klon wird abgelehnt (SandboxSetupError)

GITHUB_URL_RE = re.compile(r"https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+)((?:/[\w.%~+-]+)*)", re.I)   # mit Unterpfad
STRONG_HINT = re.compile(   # PDF-Extraktion trennt Woerter ("avail-able"), daher "-?\s*"
    r"avail-?\s*able\s+(at|on|from|in)|(code|implementation|implementations)\s+(is|are|will be)\s+(publicly\s+)?"
    r"(avail-?\s*able|released|provided)|open-?sourced?\s+(at|on)|code:|(code|implementation)\s+(can be found|link)", re.I)
WEAK_HINT = re.compile(r"code|implementation|source", re.I)
NON_REPO_SEGMENTS = {"issues", "pull", "pulls", "wiki", "releases", "actions", "discussions", "blob", "raw", "commit", "commits", "compare"}
# Infrastruktur-Repos, die in arXiv-HTML-Texten stehen (Fehlerformular, Konverter), nie das Repo des Papers
INFRA_REPOS = {("arxiv", "html_feedback"), ("brucemiller", "latexml")}
GENERIC_NAMES = {"model", "models", "method", "methods", "ours", "our", "proposed", "baseline", "baselines", "network", "networks", "net",
                 "frozen", "this", "paper", "the", "and", "all"}


@dataclass
class RepoDump:
    url: str
    commit: str
    files: dict[str, str] = field(default_factory=dict)   # Pfad -> Inhalt (gelesene Auswahl)
    n_tracked: int = 0
    subdir: str = ""   # gelesener Unterordner eines Sammel-Repos (Pfade in `files` bleiben relativ zur Repo-Wurzel)


@dataclass(frozen=True)
class RepoLink:
    owner: str
    repo: str
    subdir: str      # Unterordner in Sammel-Repos ("tsmixer" bei google-research/google-research/tree/master/tsmixer), sonst ""
    strength: str    # "strong" (Hinweis wie "available at"), "weak" (Hinweiswort davor), "single" (einziges Repo im Text)
    pos: int

    @property
    def root(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"

    @property
    def target(self) -> str:
        return f"{self.root}/tree/HEAD/{self.subdir}" if self.subdir else self.root


def normalize_pdf_text(text: str) -> str:
    """PDF-Extraktion setzt Leerzeichen in URLs ("https: // github. com/x/y", "tree/master/ tsmixer"). Nur fuer die Linksuche."""
    text = re.sub(r"https?:\s*//\s*", lambda m: m.group(0).split(":")[0] + "://", text)
    text = re.sub(r"github\.\s+com", "github.com", text)
    return re.sub(r"(github\.com/[^\s]*?/(?:tree|blob)/[\w.-]+/)\s+(?=[\w.-]+(?:[/\s.,;)]|$))", r"\1", text)


def parse_github_url(url: str, lenient: bool = False) -> tuple[str, str, str] | None:
    """(owner, repo, subdir) einer GitHub-URL; None bei Nicht-Repo-Links (Issue, Blob-Datei, Wiki). `.../tree/<branch>/<pfad>` ergibt den Unterordner.
    lenient (Suchtreffer): Issue-/Blob-/Wiki-Seiten eines Repos zaehlen als das Repo selbst (Wurzel), Links im Paper-Text nicht."""
    m = GITHUB_URL_RE.match((url or "").strip())
    if not m:
        return None
    owner, repo, rest = m.group(1), re.sub(r"\.git$", "", m.group(2)), m.group(3)
    segs = [x for x in rest.split("/") if x]
    subdir = ""
    if segs:
        if segs[0] == "tree" and len(segs) >= 2:
            subdir = "/".join(segs[2:])
        elif segs[0] in NON_REPO_SEGMENTS and not lenient:
            return None
    if not repo or (owner.lower(), repo.lower()) in INFRA_REPOS or subdir.startswith(".") or ".." in subdir.split("/"):
        return None
    return owner, repo, subdir


def find_repo_links(paper_text: str) -> list[RepoLink]:
    """Code-Links im Paper-Text, beste zuerst: (1) starker Hinweis ("The implementation is available at: ...", Reihenfolge im Text,
    Abstract vor Fussnoten), (2) schwacher Hinweis (naechstes Hinweiswort), (3) genau ein Repo im Text. Mehrere Repos ohne Hinweis
    ergeben keinen Link. Unterordner von Sammel-Repos bleiben erhalten; Blob-/Issue-Links und Infrastruktur-Repos fallen weg."""
    text = normalize_pdf_text(paper_text)
    found: list[tuple[int, str, str, str]] = []   # pos, owner, repo, subdir
    for m in GITHUB_URL_RE.finditer(text):
        raw = m.group(0).rstrip(".,;:)>]'\"/")
        parsed = parse_github_url(raw)
        if parsed:
            found.append((m.start(), *parsed))
    out: list[RepoLink] = []
    seen: set[tuple[str, str, str]] = set()
    weak: list[tuple[int, RepoLink]] = []
    for pos, owner, repo, subdir in found:
        key = (owner.lower(), repo.lower(), subdir.lower())
        if key in seen:
            continue
        seen.add(key)
        if STRONG_HINT.search(text[max(0, pos - 40): pos]):
            out.append(RepoLink(owner, repo, subdir, "strong", pos))
            continue
        window = text[max(0, pos - 120): pos]
        hits = [h.start() for h in WEAK_HINT.finditer(window)]
        if hits:
            weak.append((len(window) - max(hits), RepoLink(owner, repo, subdir, "weak", pos)))
    out += [link for _, link in sorted(weak, key=lambda x: x[0])]
    if not out and len({(f[1].lower(), f[2].lower()) for f in found}) == 1:   # genau ein Repo (Fussnote ohne Hinweiswort)
        pos, owner, repo, subdir = found[0]
        out.append(RepoLink(owner, repo, subdir, "single", pos))
    return out


def find_repo_url(paper_text: str) -> str | None:
    """Bestes Code-Ziel aus dem Paper (Repo-URL, bei Unterordner `.../tree/HEAD/<pfad>`), sonst None. Siehe find_repo_links."""
    links = find_repo_links(paper_text)
    return links[0].target if links else None


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def model_tokens(title: str, model: str = "") -> list[str]:
    """Namen, die ein Repo des Papers nennen sollte: Modell des Claims (ohne Klammerzusatz/Variante) und Akronym vor dem Doppelpunkt des Titels."""
    names = []
    for raw in (model, title.split(":")[0] if ":" in title else ""):
        cand = re.sub(r"\([^)]*\)", "", raw).split(",")[0].split("/")[0].strip()
        if len(cand) >= 3 and " " not in cand and _alnum(cand) not in GENERIC_NAMES and cand.lower() not in {n.lower() for n in names}:
            names.append(cand)
    return names


def repo_evidence(readme: str, url: str, title: str, arxiv_id: str, model: str = "") -> tuple[str, str]:
    """Belegt der README-Text (oder der Pfad der URL), dass das Repo zu DIESEM Paper gehoert? Rueckgabe (Staerke, Beschreibung):
    ("paper", ...) bei arXiv-ID oder Titel(teil); ("model", ...) bei Modellname; ("", "") ohne Beleg. Dateinamen/Pfad zaehlen fuer den Modellnamen."""
    hay = _alnum(readme)
    base = re.sub(r"v\d+$", "", arxiv_id or "")
    if base and re.search(rf"(?<!\d){re.escape(base)}(?!\d)", readme or ""):
        return "paper", f"arXiv id {base} in README"
    parts = [title] + [p for p in re.split(r"[:\u2013\u2014]", title) if len(p.strip()) >= 12]
    for part in parts:
        n = _alnum(part)
        if len(n) >= 12 and n in hay:
            return "paper", f"title '{part.strip()[:60]}' in README"
    path = _alnum(" ".join(re.findall(r"[\w.-]+", url.split("github.com/")[-1])))
    for name in model_tokens(title, model):
        n = _alnum(name)
        if n and (re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", hay) or re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", path)):
            return "model", f"model name {name} in README or path"
    return "", ""


UNOFFICIAL_RE = re.compile(r"unofficial|un-official|re-?implementation|reimplementation|not (?:the )?official|community implementation", re.I)
OFFICIAL_RE = re.compile(r"(?<![a-z-])official\s+(?:[\w-]+\s+){0,2}(?:implementation|code|repo|repository|pytorch|tensorflow)|(?<![a-z-])official(?:ly)?\s+released", re.I)


def repo_status(url: str, readme: str, linked_by_paper: bool, evidence: str, paper_text: str = "", authors: list[str] | None = None) -> tuple[str, str]:
    """("official"|"unofficial", Begruendung). Offiziell: der Paper-Text verlinkt das Repo selbst (`linked_by_paper`), ODER Beleg auf das Paper
    (arXiv-ID/Titel im README) plus Besitzer passt zu Autor/Organisation, ODER README nennt sich offiziell und belegt das Paper. Alles andere ist
    "unofficial", ebenso ein README, das sich inoffiziell nennt. Ein inoffizielles Repo ist nie Autorencode."""
    owner = (parse_github_url(url) or ("",))[0].lower()
    if UNOFFICIAL_RE.search(readme or ""):
        return "unofficial", "README calls itself unofficial or a re-implementation"
    if linked_by_paper:
        return "official", "linked in the paper text"
    if evidence == "paper":
        front = paper_text[:2500]
        names = [a.split()[-1] for a in (authors or []) if a.split()] or re.findall(r"\b[A-Z][a-z]{3,}\b", front)
        owner_tokens = [t for t in re.split(r"[^a-z0-9]+", owner) if len(t) >= 4 and t not in {"research", "labs", "group", "team", "official"}]
        if any(len(n) >= 4 and n.lower() in owner for n in names) or any(t in front.lower() for t in owner_tokens):
            return "official", "owner matches an author or organisation and the README names the paper"
        if OFFICIAL_RE.search(readme or ""):
            return "official", "README states it is the official implementation and names the paper"
    return "unofficial", "no link in the paper, no owner/author match, README does not claim to be official"


def search_repo_candidates(title: str, arxiv_id: str, get_tavily, fixture_dir: Path | None = None, mode: str = "fixture",
                           model: str = "") -> list[str]:
    """Rolle 1, wenn das Paper keinen (brauchbaren) Code-Link hat: Tavily-Suche auf github.com, ALLE passenden Treffer (nicht nur der erste).
    Ein Treffer zaehlt nur, wenn sein Text/Titel/URL die arXiv-ID, den vollen Titel oder (mit `model`) den Modellnamen nennt; die eigentliche
    Pruefung (README) folgt im Aufrufer. Treffer mit arXiv-ID oder Titel zuerst. Ergebnis wird als Fixture gespeichert (gleiche Treffer fuer jede
    Generierung); mode="live" schreibt nie nach `fixture_dir`. Rueckgabe: Ziel-URLs (bei Unterordner `.../tree/HEAD/<pfad>`)."""
    fixture_dir = fixture_dir or FIXTURE_DIR
    fixture = fixture_dir / f"search_{arxiv_id}.json"
    if fixture.exists():
        data = json.loads(fixture.read_text())
        return list(data["urls"]) if "urls" in data else ([data["url"]] if data.get("url") else [])
    res = get_tavily().find_material(title, domains=["github.com"])
    norm = _alnum
    names = [n.lower() for n in model_tokens(title, model)] if model else []
    strong, weaker = [], []
    for r in (res or {}).get("results", []):
        parsed = parse_github_url(r.get("url", ""), lenient=True)
        if not parsed:
            continue
        target = f"https://github.com/{parsed[0]}/{parsed[1]}" + (f"/tree/HEAD/{parsed[2]}" if parsed[2] else "")
        raw = r.get("content", "") + " " + r.get("title", "") + " " + r.get("url", "")
        text = norm(raw)
        if (arxiv_id and arxiv_id in raw) or norm(title) in text:
            strong.append(target)
        elif names and any(re.search(rf"(?<![a-z0-9]){re.escape(norm(n))}(?![a-z0-9])", text) for n in names):
            weaker.append(target)
    urls = list(dict.fromkeys(strong + weaker))
    if mode == "live":
        return urls
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps({"title": title, "arxiv_id": arxiv_id, "url": urls[0] if urls else None, "urls": urls, "mode": mode,
                                   "candidates": [r.get("url") for r in (res or {}).get("results", [])]}, indent=1))
    return urls


def search_repo_url(title: str, arxiv_id: str, get_tavily, fixture_dir: Path | None = None, mode: str = "fixture") -> str | None:
    """Erster Treffer von search_repo_candidates (Repo-Wurzel-URL bei Alt-Aufrufern ohne Unterordner)."""
    urls = search_repo_candidates(title, arxiv_id, get_tavily, fixture_dir, mode)
    return urls[0] if urls else None


IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import\s+([\w, ]+)|import\s+([\w.]+))", re.M)
IMPORT_RE_S4 = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import\s+([\w, *]+)|import\s+([\w.]+))", re.M)   # Steckbrief-Modus: auch `from gcn.layers import *`


def imported_files(files: dict[str, str], listing: list[str], relative: bool = False) -> list[str]:
    """Lokale Module, die die gewaehlten Dateien importieren (eine Ebene): dort stehen oft Schichten und Vorverarbeitung.
    relative=True (Steckbrief-Modus, S4): auch relative Imports (`from . import gat`, `from ..layers import GCN`) und
    Pfade ohne Ruecksicht auf Gross-/Kleinschreibung."""
    have, out = set(files), []
    paths = set(listing)
    lower = {p.lower(): p for p in listing} if relative else {}

    def find(cand: str | None) -> str | None:
        if not cand:
            return None
        cand = cand.lstrip("/")
        if cand in paths:
            return cand
        return lower.get(cand.lower()) if relative else None

    for path, text in files.items():
        if not path.endswith(".py"):
            continue
        base = path.rsplit("/", 1)[0] if "/" in path else ""
        for a, names, b in (IMPORT_RE_S4 if relative else IMPORT_RE).findall(text):
            raw = a or b
            if relative and raw.startswith("."):
                dots = len(raw) - len(raw.lstrip("."))
                parts = base.split("/") if base else []
                parts = parts[: max(0, len(parts) - (dots - 1))]
                rest = raw.lstrip(".").replace(".", "/")
                mod = "/".join(p for p in parts + ([rest] if rest else []) if p)
                cands = [f"{mod}.py" if rest else None, f"{mod}/__init__.py" if rest else None]
                cands += [f"{mod}/{n.strip()}.py" if mod else f"{n.strip()}.py" for n in names.split(",") if n.strip()]
            else:
                mod = raw.replace(".", "/")
                cands = [f"{mod}.py", f"{mod}/__init__.py", f"{base}/{mod}.py" if base else None]
                cands += [f"{mod}/{n.strip()}.py" for n in names.split(",") if n.strip()]   # from utils import layers
            for cand in cands:
                hit = find(cand)
                if hit and hit not in have and hit not in out:
                    out.append(hit)
    return out


def model_file_in_listing(listing: list[str], model: str) -> str | None:
    """Datei, die nach dem Claim-Modell heisst (models/NLinear.py, gnnbench/models/gat.py), kuerzester Pfad zuerst."""
    m = model.lower()
    hits = [p for p in listing if p.endswith(".py") and p.rsplit("/", 1)[-1][:-3].lower() == m]
    return min(hits, key=len) if hits else None


def model_config_in_listing(listing: list[str], model: str) -> list[str]:
    """Konfigurationsdateien des Claim-Modells (config/optimized/gat.conf.yaml), kuerzeste zuerst, hoechstens 2."""
    m = model.lower()
    hits = [p for p in listing if re.search(r"\.(ya?ml|json|conf|cfg|ini|toml)$", p, re.I) and m in p.rsplit("/", 1)[-1].lower()]
    return sorted(hits, key=len)[:2]


def _repo_key(url: str) -> str:
    return url.rstrip("/").split("github.com/")[-1].replace("/", "_")


def clone_plan(url: str) -> tuple[str, str]:
    """(Klon-URL der Repo-Wurzel, Unterordner) fuer ein Ziel aus find_repo_url/Suche. Ohne GitHub-Struktur: (url, "")."""
    parsed = parse_github_url(url)
    if not parsed:
        return url, ""
    owner, repo, subdir = parsed
    if subdir and not re.fullmatch(r"[\w./-]+", subdir):
        raise ValueError(f"unsafe subfolder in repository URL: {subdir!r}")
    return f"https://github.com/{owner}/{repo}", subdir


def _last_string_list(content: str) -> list[str]:
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\[", content))):
        try:
            value, _ = dec.raw_decode(content[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return value
    return []


def _resolve_paths(raw: list[str], listing: list[str]) -> list[str]:
    """Antwort-Pfade auf exakte Pfade der Dateiliste abbilden: Anfuehrungszeichen/Backticks, `./`, `/repo/` und Gross-/Kleinschreibung
    toleriert. Unbekannte Pfade fallen weg, Reihenfolge und Eindeutigkeit bleiben."""
    valid, lower, out = set(listing), {p.lower(): p for p in listing}, []
    for r in raw:
        cand = r.strip().strip("`'\"").removeprefix("/repo/").removeprefix("./").lstrip("/")
        hit = cand if cand in valid else lower.get(cand.lower())
        if hit and hit not in out:
            out.append(hit)
    return out


def _mentioned_paths(content: str, listing: list[str]) -> list[str]:
    """Letzter Ausweg bei Antworten ohne JSON-Liste: Pfade der Dateiliste, die im Text vorkommen (Reihenfolge des ersten Auftretens)."""
    found = sorted((content.find(p), p) for p in listing if len(p) > 3 and p in content)
    return [p for _, p in found]


def select_files(llm: LLMClient, listing: list[str], claim_text: str, dataset: str, focus_model: bool = False) -> list[str]:
    def ask(focus: bool, max_tokens: int) -> list[str]:
        resp = llm.complete("extract", [
            {"role": "system", "content": "You pick files from a code repository. Keep reasoning brief."},
            {"role": "user", "content": (
                f"A reproducer wants to re-implement this result: {claim_text} (dataset: {dataset}).\n"
                f"From the file list below, pick up to {MAX_FILES} files that most likely state how the "
                "experiment is set up: run scripts/configs for this dataset, argument definitions with "
                "defaults, the training loop, the data loader/splits, learning-rate or scheduler "
                "utilities, README. "
                + ("Name the model of the goal in your choice: prefer files of that model and skip files that only concern other models "
                   "of the repository. " if focus else "")
                + "Answer with a JSON list of exact paths from the list and end with it.\n\n"
                + "\n".join(listing[:MAX_LISTING])
            )},
        ], temperature=0.0, max_tokens=max_tokens, timeout_s=400)
        return (_resolve_paths(_last_string_list(resp.content), listing) or _resolve_paths(_mentioned_paths(resp.content, listing), listing))[:MAX_FILES]

    chosen = ask(focus_model, 6000)
    # Live-Fund 2026-10-09 (Kipf GCN): eine Antwort ohne einen gueltigen Pfad ergab einen leeren Ausschnitt und den stillen Rueckfall auf
    # "Nachbau aus Paper". Daher einmal neu fragen (schlichter Auftrag, mehr Platz), erst danach gibt read_repo eine Namensheuristik vor.
    return chosen or ask(False, 9000)


_SETUP_STEMS = {"train": 7, "main": 7, "run": 7, "trainer": 7, "experiment": 6, "exp": 6,
                "layer": 4, "layers": 4, "utils": 4, "config": 4, "configs": 4, "args": 4, "options": 4, "metrics": 4, "inits": 4,
                "data": 3, "dataset": 3, "datasets": 3, "loader": 3, "dataloader": 3, "preprocess": 3}
_MODEL_STEMS = ("model", "models", "net", "nets", "network", "networks")


def heuristic_model_files(listing: list[str], model: str) -> list[str]:
    """Python-Dateien, die nach dem Modell oder nach Modellen heissen (gcn/models.py, models/NLinear.py), kuerzester Pfad zuerst."""
    from .spec import DATA_DIR

    m = model.lower()
    hits = []
    for p in listing:
        stem = p.rsplit("/", 1)[-1][:-3].lower() if p.endswith(".py") else None
        if stem is None or DATA_DIR.search(p) or re.search(r"(^|/)(tests?|examples?|docs?)/", p, re.I):
            continue
        if stem in _MODEL_STEMS or stem.endswith(("model", "models")) or (m and (stem == m or m in stem)):   # slmodel.py, base_model.py
            hits.append(p)
    return sorted(hits, key=lambda p: (p.rsplit("/", 1)[-1][:-3].lower() != m, len(p)))


def heuristic_files(listing: list[str], model: str, dataset: str) -> list[str]:
    """Namensheuristik ohne LLM, wenn die Dateiwahl keinen gueltigen Pfad liefert: README, Modelldatei, Trainingsskript, Schichten,
    Hilfsmodule, Konfigurationen des Modells. Nie leer, solange das Repo Python-Dateien oder ein README hat."""
    from .spec import BINARY_EXT, DATA_DIR

    ds, m = dataset.lower(), model.lower()
    score: dict[str, int] = {p: 8 for p in heuristic_model_files(listing, model)}
    for p in listing:
        if BINARY_EXT.search(p) or DATA_DIR.search(p) or re.search(r"(^|/)(tests?|examples?|docs?)/", p, re.I):
            continue
        name = p.rsplit("/", 1)[-1]
        stem = name.rsplit(".", 1)[0].lower()
        if "/" not in p and name.lower().startswith("readme"):
            score[p] = max(score.get(p, 0), 9)
        elif name.lower() == "requirements.txt":
            score[p] = max(score.get(p, 0), 2)
        elif p.endswith(".py"):
            s_ = _SETUP_STEMS.get(stem, 0) + (3 if ds and ds in stem else 0)
            if s_:
                score[p] = max(score.get(p, 0), s_)
        elif re.search(r"\.(ya?ml|json|conf|cfg|ini|toml)$", p, re.I) and ((m and m in name.lower()) or (ds and ds in name.lower())):
            score[p] = max(score.get(p, 0), 3)
    return sorted(score, key=lambda p: (-score[p], len(p)))[:MAX_FILES]


IMPORT_CHAR_BUDGET = 120_000   # Steckbrief-Modus: importierte Dateien bis zu diesem Gesamtumfang statt fester Anzahl


def subdir_listing(ls_tree_out: str) -> list[str]:
    """Dateiliste aus `git ls-tree -r -l` ("<mode> blob <sha> <size>\\t<pfad>"): Dateien ueber MAX_SUBDIR_FILE_BYTES fallen weg; ist der Unterordner
    insgesamt groesser als MAX_SUBDIR_TOTAL_BYTES, scheitert der Klon mit SandboxSetupError (Groessenlimit fuer Sammel-Repos)."""
    from .sandbox.base import SandboxSetupError

    paths, total = [], 0
    for line in ls_tree_out.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 4 or not path:
            continue
        size = int(parts[3]) if parts[3].isdigit() else 0
        total += size
        if size <= MAX_SUBDIR_FILE_BYTES:
            paths.append(path)
    if total > MAX_SUBDIR_TOTAL_BYTES:
        raise SandboxSetupError(f"subfolder too large ({total // 1_000_000} MB > {MAX_SUBDIR_TOTAL_BYTES // 1_000_000} MB)")
    return paths


def unit_fixture_name(url: str, unit_key: str) -> str:
    return f"{_repo_key(url)}__{re.sub(r'[^A-Za-z0-9_.-]+', '_', unit_key)}.json"


def read_repo(
    sandbox: Sandbox, llm: LLMClient, url: str, claim_text: str, dataset: str,
    fixture_dir: Path | None = None, focus_model: bool = False, unit_key: str | None = None, model: str | None = None,
    mode: str = "fixture", on_note=None,
) -> RepoDump:
    """Fixture zuerst; sonst klonen (Sandbox), Dateien waehlen (LLM), lesen, speichern (nicht bei mode="live").

    unit_key (Steckbrief-Modus, S4): Fixture je Einheit (Paper) statt je Repo. Eine fuer einen anderen Claim angelegte Fixture
    (NLinear erbte die DLinear-Auswahl) wird dann nie gelesen. Zusaetzlich: Pflicht-Modelldatei (`model`), relative Imports
    zwei Ebenen tief mit Zeichenbudget, nur Quelltext (keine Pickles/Daten)."""
    from .spec import is_source_text

    fixture_dir = fixture_dir or FIXTURE_DIR
    fixture = fixture_dir / (unit_fixture_name(url, unit_key) if unit_key else f"{_repo_key(url)}.json")
    if fixture.exists():
        dump = RepoDump(**json.loads(fixture.read_text()))
        if unit_key:
            dump.files = {p: t for p, t in dump.files.items() if is_source_text(p, t)}
        return dump

    clone_url, subdir = clone_plan(url)
    if subdir:   # Sammel-Repo (google-research/google-research/tree/.../tsmixer): nur der Unterordner (sparse, ohne Blobs ausserhalb)
        clone_cmd = (f"git clone -q --depth 1 --filter=blob:none --sparse {shlex.quote(clone_url)} /repo && "
                     f"cd /repo && git sparse-checkout set {shlex.quote(subdir)}")
    else:
        clone_cmd = f"git clone -q --depth 1 {shlex.quote(clone_url)} /repo"
    snap = sandbox.base_snapshot(
        "python:3.11-slim",
        [
            "apt-get update -qq && apt-get install -y -qq git > /dev/null",
            clone_cmd,
            "cd /repo && git rev-parse HEAD > /repo/.head_commit",
        ],
        label=f"repo-{_repo_key(url).lower()}"[:60],
    )
    branch = sandbox.branch(snap)
    commit = sandbox.get_file(branch, "/repo/.head_commit").decode().strip()
    if subdir:
        listing = subdir_listing(sandbox.run(branch, f"cd /repo && git ls-tree -r -l HEAD -- {shlex.quote(subdir)}", 60).stdout)
    else:
        listing = sandbox.run(branch, "cd /repo && git ls-files", 60).stdout.splitlines()
    chosen = select_files(llm, listing, claim_text, dataset, focus_model)
    if not chosen:   # beide Antworten ohne gueltigen Pfad: Namensheuristik statt leerem Ausschnitt, sichtbar gemacht
        chosen = heuristic_files(listing, model or "", dataset)
        if on_note is not None:
            on_note(f"Dateiwahl des Modells lieferte keinen gueltigen Pfad ({len(listing)} Dateien im Repo); Rueckfall auf Namensheuristik: {', '.join(chosen) or 'keine Datei'}")
    files = {}

    def read(paths):
        for path in paths:
            try:
                text = sandbox.get_file(branch, f"/repo/{path}").decode(errors="replace")[:MAX_FILE_CHARS]
            except (OSError, UnicodeDecodeError):
                continue
            if unit_key and not is_source_text(path, text):
                continue
            files[path] = text
    if unit_key:
        if model:
            mf = model_file_in_listing(listing, model)
            if mf and mf not in chosen:
                chosen = [mf] + chosen
            chosen = chosen + [c for c in model_config_in_listing(listing, model) if c not in chosen]
        read(chosen)
        if model:   # Modellklasse fehlt in der Auswahl (z. B. nur train.py gewaehlt): Dateien namens models.py/<Modell>.py nachladen
            from .setup_details import Doc
            from .spec import model_files

            if not model_files([Doc(f"repo:{p}", t) for p, t in files.items()], model):
                read([p for p in heuristic_model_files(listing, model) if p not in files][:2])
        for _ in range(2):   # zwei Ebenen
            budget = IMPORT_CHAR_BUDGET - sum(len(t) for t in files.values())
            nxt = []
            for p in imported_files(files, listing, relative=True):
                if budget <= 0:
                    break
                nxt.append(p)
                budget -= MAX_FILE_CHARS // 3   # Schaetzung je Datei vor dem Lesen
            if not nxt:
                break
            read(nxt)
    else:
        read(chosen)
        read(imported_files(files, listing)[:MAX_IMPORTED])
    dump = RepoDump(url=url, commit=commit, files=files, n_tracked=len(listing), subdir=subdir)
    if mode == "live":
        return dump
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps(asdict(dump), ensure_ascii=False, indent=1))
    return dump


def fetch_readme(url: str, get_tavily, fixture_dir: Path | None = None, mode: str = "fixture") -> str:
    """README ueber Tavily Extract (Rolle 4). mode="fixture": Fixture zuerst, spart Credits; mode="live": immer Aufruf, schreibt nie nach `fixture_dir`."""
    fixture_dir = fixture_dir or FIXTURE_DIR
    fixture = fixture_dir / f"{_repo_key(url)}_readme.json"
    if mode == "fixture" and fixture.exists():
        return json.loads(fixture.read_text())["text"]
    result = get_tavily().get_setup_details(url)
    text = next((r.get("raw_content") or "" for r in result.get("results", [])), "")
    if mode == "live":
        return text
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps({"url": url, "text": text}, ensure_ascii=False, indent=1))
    return text
