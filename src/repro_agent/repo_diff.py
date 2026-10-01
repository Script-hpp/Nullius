"""A2: Abgleich unseres Codes mit den Repo-Dateien des Papers (nur Produktpfad, nur mit Repo).

Ein Aufruf listet nur Abweichungen zu einer festen Aspektliste. Wie bei Rolle 4 gilt: das Modell liest, ein Programm prueft.
Jede Abweichung traegt ein woertliches Zitat aus der genannten Repo-Datei und eines aus unserem Code; sonst wird sie verworfen.
Der Prompt enthaelt unseren Code und oeffentlichen Repo-Code, nie Val-/Testzahlen, Zielwert oder Claim-Text. Modus bleibt "Nachbau":
Das Repo kommt nie in einen Branch, nur kurze Zitate gehen als Hinweis an `code`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .claims_source import _norm
from .setup_details import _parse_list, quote_in_text

MAX_REPO_CHARS = 60_000
ASPECTS = (
    "dropout (which tensors, which rate)", "feature and adjacency normalisation", "weight initialisation",
    "regularisation (which parameters get weight decay, how it is scaled)",
    "stopping criterion (patience, maximum epochs, which weights are restored for testing)",
)
SYSTEM = "You compare a re-implementation with the original code of a paper. Keep reasoning brief."


@dataclass(frozen=True)
class Deviation:
    aspect: str
    file: str
    repo_quote: str
    our_quote: str
    difference: str


def truncate_files(files: dict[str, str], limit: int = MAX_REPO_CHARS) -> tuple[dict[str, str], list[dict]]:
    """Deterministisch in der Reihenfolge von dump.files: ganze Dateien, solange sie passen, die erste ueberlaufende wird
    gekuerzt, alle weiteren entfallen. Rueckgabe: (gezeigte Dateien, Kuerzungen [{file, kept_chars, of_chars}])."""
    shown: dict[str, str] = {}
    cuts: list[dict] = []
    left = limit
    for path, text in files.items():
        if left <= 0:
            cuts.append({"file": path, "kept_chars": 0, "of_chars": len(text)})
        elif len(text) <= left:
            shown[path] = text
            left -= len(text)
        else:
            shown[path] = text[:left]
            cuts.append({"file": path, "kept_chars": left, "of_chars": len(text)})
            left = 0
    return shown, cuts


def build_prompt(code: str, files: dict[str, str], dataset: str) -> str:
    repo = "\n\n".join(f"=== FILE {p} ===\n{t}" for p, t in files.items())
    aspects = "\n".join(f"- {a}" for a in ASPECTS)
    return (
        f"A reproducer re-implemented a paper's method on {dataset}. Below are the reproducer's script and files of the "
        "original repository. List ONLY the places where the script differs from the repository in one of these aspects:\n"
        f"{aspects}\n"
        "Do not list matches, style differences or aspects not named. Answer with a JSON list of "
        '{"aspect": the aspect, "file": exact repository file path, "repo_quote": a VERBATIM excerpt (max 300 chars) from that '
        'repository file, "our_quote": a VERBATIM excerpt (max 300 chars) from the script, "difference": one sentence}. '
        "If there is no difference, answer []. Keep your reasoning short and end with the JSON list.\n\n"
        f"SCRIPT:\n{code}\n\nREPOSITORY FILES:\n{repo}"
    )


def verify_deviations(raw: list, code: str, files: dict[str, str]) -> tuple[list[Deviation], list[dict]]:
    """Zitat der Repo-Datei muss (normalisiert) in genau dieser gezeigten Datei stehen, Zitat aus unserem Code in unserem Code."""
    hay_files = {p: _norm(t) for p, t in files.items()}
    hay_code = _norm(code)
    kept, dropped, seen = [], [], set()
    for item in raw:
        if not isinstance(item, dict):
            dropped.append({"item": item, "reason": "kein Objekt"})
            continue
        f = {k: str(item.get(k, "")).strip() for k in ("aspect", "file", "repo_quote", "our_quote", "difference")}
        if not all(f.values()):
            dropped.append({"item": item, "reason": "Feld fehlt"})
        elif f["file"] not in hay_files:
            dropped.append({"item": item, "reason": "Datei nicht unter den gezeigten Repo-Dateien"})
        elif not quote_in_text(f["repo_quote"], hay_files[f["file"]]):
            dropped.append({"item": item, "reason": "Repo-Zitat nicht in der Datei"})
        elif not quote_in_text(f["our_quote"], hay_code):
            dropped.append({"item": item, "reason": "Zitat nicht in unserem Code"})
        elif (f["file"], _norm(f["repo_quote"])) not in seen:
            seen.add((f["file"], _norm(f["repo_quote"])))
            kept.append(Deviation(**f))
    return kept, dropped


def find_deviations(ask, code: str, files: dict[str, str], dataset: str) -> tuple[list[Deviation], list[dict], list[dict]]:
    """`ask(system, user)` liefert ein Objekt mit .content (Rolle extract, temperature 0). Rueckgabe: (geprueft, verworfen, Kuerzungen)."""
    shown, cuts = truncate_files(files)
    resp = ask(SYSTEM, build_prompt(code, shown, dataset))
    kept, dropped = verify_deviations(_parse_list(resp.content), code, shown)
    return kept, dropped, cuts


def repair_message(devs: list[Deviation]) -> str:
    """Fester Hinweistext (nur Zitate und Aspekte, keine Zahlen aus Messungen)."""
    lines = [f"- {d.aspect}: {d.difference}\n  repository ({d.file}): \"{d.repo_quote}\"\n  your script: \"{d.our_quote}\"" for d in devs]
    return (
        "Your script differs from the paper's original repository in these places:\n" + "\n".join(lines) + "\n"
        "Where the reproduction plan does not state otherwise, align your script with the repository on these points. "
        "Keep the method described in the plan: change only these details and return the corrected full script."
    )


def as_dicts(devs: list[Deviation]) -> list[dict]:
    return [asdict(d) for d in devs]


__all__ = ["Deviation", "find_deviations", "repair_message", "truncate_files", "as_dicts"]
