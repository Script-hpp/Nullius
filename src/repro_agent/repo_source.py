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
    return best[1] if best else None


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


def select_files(llm: LLMClient, listing: list[str], claim_text: str, dataset: str) -> list[str]:
    resp = llm.complete("extract", [
        {"role": "system", "content": "You pick files from a code repository. Keep reasoning brief."},
        {"role": "user", "content": (
            f"A reproducer wants to re-implement this result: {claim_text} (dataset: {dataset}).\n"
            f"From the file list below, pick up to {MAX_FILES} files that most likely state how the "
            "experiment is set up: run scripts/configs for this dataset, argument definitions with "
            "defaults, the training loop, the data loader/splits, learning-rate or scheduler "
            "utilities, README. Answer with a JSON list of exact paths from the list and end with it.\n\n"
            + "\n".join(listing[:MAX_LISTING])
        )},
    ], temperature=0.0, max_tokens=6000, timeout_s=400)
    valid = set(listing)
    return [p for p in _last_string_list(resp.content) if p in valid][:MAX_FILES]


def read_repo(
    sandbox: Sandbox, llm: LLMClient, url: str, claim_text: str, dataset: str,
    fixture_dir: Path = FIXTURE_DIR,
) -> RepoDump:
    """Fixture zuerst; sonst klonen (Sandbox), Dateien waehlen (LLM), lesen, speichern."""
    fixture = fixture_dir / f"{_repo_key(url)}.json"
    if fixture.exists():
        return RepoDump(**json.loads(fixture.read_text()))

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
    chosen = select_files(llm, listing, claim_text, dataset)
    files = {}
    for path in chosen:
        try:
            files[path] = sandbox.get_file(branch, f"/repo/{path}").decode(errors="replace")[:MAX_FILE_CHARS]
        except (OSError, UnicodeDecodeError):
            continue
    dump = RepoDump(url=url, commit=commit, files=files, n_tracked=len(listing))
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps(asdict(dump), ensure_ascii=False, indent=1))
    return dump


def fetch_readme(url: str, get_tavily, fixture_dir: Path = FIXTURE_DIR) -> str:
    """README ueber Tavily Extract (Rolle 4). Fixture zuerst, spart Credits."""
    fixture = fixture_dir / f"{_repo_key(url)}_readme.json"
    if fixture.exists():
        return json.loads(fixture.read_text())["text"]
    result = get_tavily().get_setup_details(url)
    text = next((r.get("raw_content") or "" for r in result.get("results", [])), "")
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps({"url": url, "text": text}, ensure_ascii=False, indent=1))
    return text
