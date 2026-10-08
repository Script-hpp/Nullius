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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .llm import LLMClient
from .sandbox.base import Sandbox

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "fixtures" / "repos"
MAX_FILES = 12
MAX_FILE_CHARS = 30_000
MAX_IMPORTED = 6   # zusaetzliche, von den gewaehlten Dateien importierte lokale Module
MAX_LISTING = 600

GITHUB_RE = re.compile(r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?=[\s)\]>.,;\"']|$)")
STRONG_HINT = re.compile(   # PDF-Extraktion trennt Woerter ("avail-able"), daher "-?"
    r"avail-?able\s+(at|on|from)|(code|implementation)\s+(is|are|will be)\s+(publicly\s+)?"
    r"(avail-?able|released)|open-?sourced?\s+(at|on)|code:", re.I)
WEAK_HINT = re.compile(r"code|implementation|source", re.I)


@dataclass
class RepoDump:
    url: str
    commit: str
    files: dict[str, str] = field(default_factory=dict)   # Pfad -> Inhalt (gelesene Auswahl)
    n_tracked: int = 0


def find_repo_url(paper_text: str) -> str | None:
    """GitHub-Link zum Code des Papers.

    Zuerst Links direkt hinter einer starken Formulierung ("Code is available at ..."), erster
    Treffer im Text gewinnt (Abstract vor Fussnoten). Sonst schwacher Hinweis, naechster Treffer.
    """
    for m in GITHUB_RE.finditer(paper_text):
        if STRONG_HINT.search(paper_text[max(0, m.start() - 40): m.start()]):
            return f"https://github.com/{m.group(1)}/{m.group(2)}"
    best: tuple[int, str] | None = None
    for m in GITHUB_RE.finditer(paper_text):
        window = paper_text[max(0, m.start() - 120): m.start()]
        hits = [h.start() for h in WEAK_HINT.finditer(window)]
        if hits and (best is None or len(window) - max(hits) < best[0]):
            best = (len(window) - max(hits), f"https://github.com/{m.group(1)}/{m.group(2)}")
    if best:
        return best[1]
    repos = {(m.group(1), m.group(2)) for m in GITHUB_RE.finditer(paper_text)}
    if len(repos) == 1:   # genau ein Repo im Text (z. B. Fussnote ohne Hinweiswort): eindeutig genug
        owner, name = repos.pop()
        return f"https://github.com/{owner}/{name}"
    return None


def search_repo_url(title: str, arxiv_id: str, get_tavily, fixture_dir: Path = FIXTURE_DIR, mode: str = "fixture") -> str | None:
    """Rolle 1, wenn das Paper keinen Code-Link hat: Tavily-Suche auf github.com. Ein Treffer zaehlt nur, wenn sein
    Text die arXiv-ID oder den vollen Titel nennt. Ergebnis wird als Fixture gespeichert (gleicher Treffer fuer jede Generierung)."""
    fixture = fixture_dir / f"search_{arxiv_id}.json"
    if fixture.exists():
        return json.loads(fixture.read_text()).get("url")
    res = get_tavily().find_material(title, domains=["github.com"])
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()
    url = None
    for r in (res or {}).get("results", []):
        m = re.match(r"https?://github\.com/([\w.-]+)/([\w.-]+)", r.get("url", ""))
        text = norm(r.get("content", "") + " " + r.get("title", ""))
        if m and (arxiv_id in (r.get("content", "") + r.get("url", "")) or norm(title) in text):
            url = f"https://github.com/{m.group(1)}/{m.group(2)}"
            break
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps({"title": title, "arxiv_id": arxiv_id, "url": url, "mode": mode,
                                   "candidates": [r.get("url") for r in (res or {}).get("results", [])]}, indent=1))
    return url


IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import\s+([\w, ]+)|import\s+([\w.]+))", re.M)


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
        for a, names, b in IMPORT_RE.findall(text):
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


def _repo_key(url: str) -> str:
    return url.rstrip("/").split("github.com/")[-1].replace("/", "_")


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


def select_files(llm: LLMClient, listing: list[str], claim_text: str, dataset: str, focus_model: bool = False) -> list[str]:
    resp = llm.complete("extract", [
        {"role": "system", "content": "You pick files from a code repository. Keep reasoning brief."},
        {"role": "user", "content": (
            f"A reproducer wants to re-implement this result: {claim_text} (dataset: {dataset}).\n"
            f"From the file list below, pick up to {MAX_FILES} files that most likely state how the "
            "experiment is set up: run scripts/configs for this dataset, argument definitions with "
            "defaults, the training loop, the data loader/splits, learning-rate or scheduler "
            "utilities, README. "
            + ("Name the model of the goal in your choice: prefer files of that model and skip files that only concern other models "
               "of the repository. " if focus_model else "")
            + "Answer with a JSON list of exact paths from the list and end with it.\n\n"
            + "\n".join(listing[:MAX_LISTING])
        )},
    ], temperature=0.0, max_tokens=6000, timeout_s=400)
    valid = set(listing)
    return [p for p in _last_string_list(resp.content) if p in valid][:MAX_FILES]


IMPORT_CHAR_BUDGET = 120_000   # Steckbrief-Modus: importierte Dateien bis zu diesem Gesamtumfang statt fester Anzahl


def unit_fixture_name(url: str, unit_key: str) -> str:
    return f"{_repo_key(url)}__{re.sub(r'[^A-Za-z0-9_.-]+', '_', unit_key)}.json"


def read_repo(
    sandbox: Sandbox, llm: LLMClient, url: str, claim_text: str, dataset: str,
    fixture_dir: Path = FIXTURE_DIR, focus_model: bool = False, unit_key: str | None = None, model: str | None = None,
) -> RepoDump:
    """Fixture zuerst; sonst klonen (Sandbox), Dateien waehlen (LLM), lesen, speichern.

    unit_key (Steckbrief-Modus, S4): Fixture je Einheit (Paper) statt je Repo. Eine fuer einen anderen Claim angelegte Fixture
    (NLinear erbte die DLinear-Auswahl) wird dann nie gelesen. Zusaetzlich: Pflicht-Modelldatei (`model`), relative Imports
    zwei Ebenen tief mit Zeichenbudget, nur Quelltext (keine Pickles/Daten)."""
    from .spec import is_source_text

    fixture = fixture_dir / (unit_fixture_name(url, unit_key) if unit_key else f"{_repo_key(url)}.json")
    if fixture.exists():
        dump = RepoDump(**json.loads(fixture.read_text()))
        if unit_key:
            dump.files = {p: t for p, t in dump.files.items() if is_source_text(p, t)}
        return dump

    snap = sandbox.base_snapshot(
        "python:3.11-slim",
        [
            "apt-get update -qq && apt-get install -y -qq git > /dev/null",
            f"git clone -q --depth 1 {url} /repo",
            "cd /repo && git rev-parse HEAD > /repo/.head_commit",
        ],
        label=f"repo-{_repo_key(url).lower()}",
    )
    branch = sandbox.branch(snap)
    commit = sandbox.get_file(branch, "/repo/.head_commit").decode().strip()
    listing = sandbox.run(branch, "cd /repo && git ls-files", 60).stdout.splitlines()
    chosen = select_files(llm, listing, claim_text, dataset, focus_model)
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
        read(chosen)
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
    dump = RepoDump(url=url, commit=commit, files=files, n_tracked=len(listing))
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps(asdict(dump), ensure_ascii=False, indent=1))
    return dump


def fetch_readme(url: str, get_tavily, fixture_dir: Path = FIXTURE_DIR, mode: str = "fixture") -> str:
    """README ueber Tavily Extract (Rolle 4). mode="fixture": Fixture zuerst, spart Credits; mode="live": immer Aufruf."""
    fixture = fixture_dir / f"{_repo_key(url)}_readme.json"
    if mode == "fixture" and fixture.exists():
        return json.loads(fixture.read_text())["text"]
    result = get_tavily().get_setup_details(url)
    text = next((r.get("raw_content") or "" for r in result.get("results", [])), "")
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps({"url": url, "text": text}, ensure_ascii=False, indent=1))
    return text
