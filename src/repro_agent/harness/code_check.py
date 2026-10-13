"""Statische Pruefung von Agenten-Code auf Netzzugriff/Downloads/Prozesse.

Zweite Schicht neben dem abgeschalteten Netz im Trainingslauf (`unshare -n`). Wortlisten sind
umgehbar, deshalb nie allein verlassen. Treffer machen den Versuch `ungueltig`.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize

# (Regex, Beschreibung). `(?<![.\w])` verhindert Treffer auf model.eval(), self.exec(...) usw.
FORBIDDEN: list[tuple[str, str]] = [
    (r"://", "URL im Code"),
    (r"\b(urllib|requests|httpx|aiohttp|http\.client|ftplib|socket|paramiko|smtplib)\b", "Netzwerk-Modul"),
    (r"\b(subprocess|pexpect)\b|\bos\.(system|popen|exec\w*|spawn\w*)\b", "Prozessstart"),
    (r"(?<![.\w])(exec|eval|compile)\s*\(", "dynamische Codeausfuehrung"),
    (r"__import__|importlib", "dynamischer Import"),
    (r"\b(torch_geometric|torchvision\.datasets|torchtext|tensorflow_datasets|ogb|dgl)\b", "Dataset-Bibliothek"),
    # fetch_* nur als Datensatz-Lader (sklearn.datasets.fetch_openml usw.), nicht jede lokale Funktion namens fetch_neighbors_from_memory
    (r"\bdatasets\.fetch_\w+|\bfetch_(?:openml|california_housing|covtype|kddcup99|lfw_\w+|olivetti_faces|rcv1|"
     r"species_distributions|20newsgroups\w*|file)\b|\bPlanetoid\b|\bdownload\w*", "Download-Funktion"),
    (r"\b(pip|wget|curl|git)\b", "Installations-/Download-Tool"),
    (r"\bind\.(cora|citeseer|pubmed)\b|planetoid", "Zugriff auf Rohdaten des Datensatzes"),
]


def executable_text(code: str) -> str:
    """Code ohne Kommentare und Docstrings (E19), positionsgleich (gleiche Zeilen, Leerzeichen statt Text). Beschreibender Text fuehrt
    nichts aus; Strings, die der Code benutzt, bleiben stehen. Nicht tokenisierbarer Code bleibt unveraendert (strenger, nie lockerer)."""
    try:
        tree = ast.parse(code)
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
        return code
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add((first.value.lineno, first.value.col_offset))
    lines = code.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    chars = list(code)
    for tok in tokens:
        if tok.type == tokenize.COMMENT or (tok.type == tokenize.STRING and tok.start in docstrings):
            a = starts[tok.start[0] - 1] + tok.start[1]
            b = starts[tok.end[0] - 1] + tok.end[1]
            for i in range(a, b):
                if chars[i] != "\n":
                    chars[i] = " "
    return "".join(chars)


def find_violations(code: str) -> list[str]:
    hits = []
    code = executable_text(code)
    for pattern, desc in FORBIDDEN:
        m = re.search(pattern, code)
        if m:
            line = code.count("\n", 0, m.start()) + 1
            hits.append(f"{desc}: '{m.group(0)}' in Zeile {line}")
    return hits
