"""Hyperparameter-Tabellen im Paper-Anhang, deterministisch gelesen (kein LLM).

Paper-Text aus der PDF-Extraktion liefert Tabellen als eine Zeile: Spaltenkoepfe, danach die Zeilen
("ETTh1 Search space Learing rate Blocks Dropout Hidden size Heads Model T 0.001, 0.0001 ... TSMixer 96 0.0001 6 0.9 512 relu 192 0.001 4 ...").
Markdown-Tabellen (Tavily Extract) haben dieselben Tokens mit "|". Ein LLM ordnet solche Zeilen unzuverlaessig zu und der Text hinter
`MAX_TEXT_CHARS` wurde nie gelesen. Hier: Kopfzeile erkennen (>= 3 bekannte Spaltennamen, auch mit Tippfehler "Learing rate"), die Zeile des
Modells (und des Horizonts) im Block dieses Datensatzes suchen, Spalten und Werte nur zuordnen, wenn ihre Anzahl uebereinstimmt.
Jede Angabe bekommt ein woertliches Zitat (Kopf ... Zeile bis zu ihrem Wert), das im Dokument steht."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .claims_source import _norm
from .setup_details import Doc, SetupDetailV2, quote_in_text

# (Thema der Angabe, Muster des Spaltennamens). Themen passen zu den Pflichtthemen (setup_details.MANDATORY).
COLUMNS: list[tuple[str, str]] = [
    ("learning rate", r"lear(?:n)?(?:ing)?[ -]?rate(?:\s*\(lr\))?|lr"),
    ("batch size", r"batch[ -]?size"),
    ("epochs", r"(?:max(?:imum)?\.?\s+)?epochs?"),
    ("patience", r"patience"),
    ("dropout", r"dropout(?:\s+rate)?"),
    ("weight decay", r"weight[ -]?decay"),
    ("number of blocks / layers", r"(?:num(?:ber)?\.?\s+of\s+)?(?:blocks?|layers?|n[_ ]?layers|e[_ ]?layers)"),
    ("hidden size", r"hidden(?:\s+(?:size|dim(?:ension)?s?|units))?|d[_ ]?model|d[_ ]?ff"),
    ("number of heads", r"(?:num(?:ber)?\.?\s+of\s+)?(?:n[_ ]?)?heads?"),
    ("input length", r"input\s+length|seq(?:uence)?[_ ]?len(?:gth)?|look-?back(?:\s+window)?"),
]
COLUMN_RE = re.compile("|".join(f"(?P<c{i}>{pat})" for i, (_, pat) in enumerate(COLUMNS)), re.I)
SEP = r"[\s|,;:/&]*"
NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
HORIZONS = {96, 192, 336, 720}
TASK_HORIZON = {"ETTh1": 96}   # tasks.py: der ETT-Task misst nur Horizont 96
MIN_HEADER_COLUMNS = 3
HEADER_LOOKBACK = 140          # Zeichen vor der Kopfzeile, in denen der Datensatzname stehen muss
BLOCK_CHARS = 2500             # Zeichen hinter der Kopfzeile, in denen die Zeile des Modells stehen muss
MIN_QUOTE_FRAGMENT = 12


@dataclass(frozen=True)
class Header:
    start: int
    end: int
    columns: list[tuple[str, str]]   # (Thema, Originalschreibweise)


def _headers(text: str) -> list[Header]:
    """Laeufe von mindestens MIN_HEADER_COLUMNS Spaltennamen, nur durch Trennzeichen getrennt."""
    out: list[Header] = []
    run: list[re.Match] = []

    def close() -> None:
        if len(run) >= MIN_HEADER_COLUMNS:
            out.append(Header(run[0].start(), run[-1].end(),
                              [(COLUMNS[int(m.lastgroup[1:])][0], m.group(0)) for m in run]))
        run.clear()

    for m in COLUMN_RE.finditer(text):
        if (m.start() > 0 and text[m.start() - 1].isalnum()) or (m.end() < len(text) and text[m.end()].isalnum()):
            continue   # nur ganze Woerter ("lr" in "clr", "heads" in "headset")
        if run and not re.fullmatch(SEP, text[run[-1].end():m.start()]):
            close()
        run.append(m)
    close()
    return out


def header_windows(text: str, before: int = 300, after: int = BLOCK_CHARS, max_chars: int = 9000) -> str:
    """Ausschnitte um erkannte Kopfzeilen von Hyperparameter-Tabellen (fuer den LLM-Auftrag, wenn der deterministische Leser nichts zuordnet)."""
    spans: list[tuple[int, int]] = []
    for h in _headers(text):
        lo, hi = max(0, h.start - before), min(len(text), h.end + after)
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))
    out, used = [], 0
    for a, b in spans:
        if used + (b - a) > max_chars:
            continue
        out.append(text[a:b])
        used += b - a
    return "\n...\n".join(out)


def _tokens(segment: str, base: int) -> list[tuple[str, int, int]]:
    """(Token, Start, Ende) im Originaltext; Pipes und Kommas am Rand fallen aus dem Token, nicht aus den Positionen."""
    out = []
    for m in re.finditer(r"[^\s|]+", segment):
        tok = m.group(0).strip(",;")
        if tok:
            lo = base + m.start() + m.group(0).index(tok)
            out.append((tok, lo, lo + len(tok)))
    return out


def _word_eq(tok: str, name: str) -> bool:
    return re.sub(r"[^a-z0-9]", "", tok.lower()) == re.sub(r"[^a-z0-9]", "", name.lower())


def _fragment(text: str, lo: int, hi: int) -> str:
    """Originalschnipsel text[lo:hi]; nach vorn verlaengert, bis es MIN_QUOTE_FRAGMENT Zeichen hat (Zitate mit Auslassung brauchen Mindestlaenge)."""
    while hi - lo < MIN_QUOTE_FRAGMENT and lo > 0:
        lo -= 1
    return text[lo:hi].strip()


def table_details(docs: list[Doc], model: str, dataset: str, horizon: int | None = None) -> list[SetupDetailV2]:
    """Angaben aus Hyperparameter-Tabellen des Papers fuer `model` auf `dataset` (bei Horizont-Tabellen die Zeile des Horizonts `horizon`).
    Nur Paper-Dokumente (Name "paper"). Leere Liste, wenn Kopf, Datensatz, Modellzeile oder Spaltenzahl nicht eindeutig passen."""
    horizon = horizon if horizon is not None else TASK_HORIZON.get(dataset)
    out: list[SetupDetailV2] = []
    for doc in docs:
        if doc.name != "paper":
            continue
        text = doc.text
        heads = _headers(text)
        for i, h in enumerate(heads):
            if not re.search(rf"(?<![A-Za-z0-9]){re.escape(dataset)}(?![A-Za-z0-9])", text[max(0, h.start - HEADER_LOOKBACK):h.start], re.I):
                continue   # Datensatz des Blocks nicht belegt: nie raten
            block_end = min(heads[i + 1].start if i + 1 < len(heads) else len(text), h.end + BLOCK_CHARS)
            toks = _tokens(text[h.end:block_end], h.end)
            key = next((k for k, (t, _, _) in enumerate(toks) if _word_eq(t, model)), None)
            if key is None:
                continue
            n_cols, k = len(h.columns), key + 1
            row_h = None
            if k < len(toks) and toks[k][0].isdigit() and int(toks[k][0]) in HORIZONS | ({horizon} if horizon else set()):
                row_h, k = int(toks[k][0]), k + 1
                if horizon is not None and row_h != horizon:
                    continue   # erste Zeile des Modells gehoert zu einem anderen Horizont (Zeile des gesuchten Horizonts steht dann nicht direkt hinter dem Namen)
            elif horizon is not None and any(t.isdigit() and int(t) in HORIZONS for t, _, _ in toks[k:k + n_cols + 1]):
                continue
            vals = toks[k:k + n_cols]
            if len(vals) != n_cols:
                continue
            nxt = toks[k + n_cols][0] if k + n_cols < len(toks) else ""
            if NUMBER_RE.match(nxt) and not (nxt.isdigit() and int(nxt) in HORIZONS):
                continue   # mehr Werte als Spalten: Zuordnung unsicher
            head_q = text[h.start:h.end].strip()
            for (topic, _), (val, _, hi) in zip(h.columns, vals):
                if not NUMBER_RE.match(val):
                    continue   # Textwert in Zahlenspalte (z. B. "relu" unter "Heads"): Spalte ist nicht sicher zugeordnet
                quote = f"{head_q} ... {_fragment(text, toks[key][1], hi)}"
                if not quote_in_text(quote, _norm(text)):
                    continue
                out.append(SetupDetailV2(topic=topic, value=val, quote=quote, source="paper",
                                         applies_to={"model": model, "dataset": dataset, "setting": "not stated", "quote": quote},
                                         source_rank="paper", optional=False))
            break   # erster passender Block dieses Datensatzes
    return out
