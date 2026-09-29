"""Spaltenpruefung fuer Zielwerte: Position der Zahl in der Tabellenzeile gegen die Kopfzeile.

Deterministisch, kein LLM. Arbeitet auf normalisiertem Text (claims_source._norm, kleingeschrieben).
Zwei Tabellenformen: (A) Datensaetze als Spalten ("Method Citeseer Cora Pubmed NELL"),
(B) Methoden als Spalten mit Metrik-Unterspalten ("Methods IMP. Linear* ... Repeat *Metric MSE MSE MAE ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NUMBER = r"(?<![\d.])\d+\.\d+(?![\d.]*%)"   # Dezimalzahl, keine Prozentangabe (IMP.-Spalte)


@dataclass
class ColumnCheck:
    status: str   # "ok" | "mismatch" | "nicht pruefbar"
    detail: str


def _clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _compare(expected: float, value: float, percent: bool, where: str) -> ColumnCheck:
    got = value * 100 if percent and value <= 1 else value
    if abs(expected - got) < 1e-9:
        return ColumnCheck("ok", f"Zahl {expected} steht laut Kopfzeile an der richtigen Stelle ({where})")
    return ColumnCheck("mismatch", f"Laut Kopfzeile muesste an Stelle {where} die Zahl {expected} stehen, "
                                   f"das Modell waehlte {got}")


def _check_methods_as_columns(
    text: str, quote: str, value: float, method: str, metric: str, percent: bool
) -> ColumnCheck | None:
    m = re.search(r"\bmethods\s+((?:\S+\s+){1,30}?)\*?metric\s+((?:(?:mse|mae)\b\s*)+)", text)
    if not m:
        return None
    names = [n for n in m.group(1).split() if _clean_name(n) not in ("imp", "")]
    metrics = m.group(2).split()
    kinds = sorted(set(metrics), key=metrics.index)
    k = len(kinds)
    if not names or len(metrics) < len(names) * k:
        return ColumnCheck("nicht pruefbar", "Kopfzeile passt nicht zur Zahl der Methoden")
    per_method = metrics[-len(names) * k:]                    # fuehrende Zusatzspalte (IMP.) abschneiden
    if per_method != kinds * len(names):
        return ColumnCheck("nicht pruefbar", "Metrikreihenfolge der Kopfzeile nicht regelmaessig")
    wanted = _clean_name(method.split(" (")[0])
    idx = [i for i, n in enumerate(names) if _clean_name(n) == wanted]
    kind = next((kd for kd in kinds if re.search(rf"\b{kd}\b", metric.lower())), None)
    if len(idx) != 1 or kind is None:
        return ColumnCheck("nicht pruefbar", f"Methode '{method}' oder Metrik '{metric}' nicht eindeutig in der Kopfzeile")
    row = [float(x) for x in re.findall(NUMBER, quote)]
    if len(row) < len(names) * k:
        return ColumnCheck("nicht pruefbar", f"Zeile hat {len(row)} Zahlen, erwartet {len(names) * k}")
    pos = idx[0] * k + kinds.index(kind)
    return _compare(row[-len(names) * k:][pos], value, percent, f"Methode {names[idx[0]]}, {kind}, Position {pos + 1}")


def _check_datasets_as_columns(text: str, quote: str, value: float, dataset: str, percent: bool) -> ColumnCheck | None:
    want = _clean_name(dataset)
    for m in re.finditer(r"\bmethod\s+((?:[a-z][a-z0-9-]*\s+)+?)(?=[a-z][a-z0-9*-]*\s+(?:\[\d|\d))", text):
        cols = m.group(1).split()
        if want in [_clean_name(c) for c in cols]:
            break
    else:
        return None
    q = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", quote)              # Literaturverweise
    q = re.sub(r"±\s*\d+(?:\.\d+)?", " ", q)                     # Streuung
    row = [float(x) for x in re.findall(NUMBER, q)]
    if len(row) < len(cols):
        return ColumnCheck("nicht pruefbar", f"Zeile hat {len(row)} Zahlen, Kopfzeile {len(cols)} Spalten")
    i = [_clean_name(c) for c in cols].index(want)
    return _compare(row[i], value, percent, f"Spalte {cols[i]}, Position {i + 1} von {len(cols)}")


DATASET_WORDS = {
    "cora", "citeseer", "pubmed", "nell", "full", "amazon", "computer", "computers", "photo", "coauthor", "cs",
    "physics",
}


def _merge_full(words: list[str]) -> list[str]:
    """'cora full' ist eine Spalte."""
    cols: list[str] = []
    for w in words:
        if w == "full" and cols:
            cols[-1] += " full"
        else:
            cols.append(w)
    return cols


def _row_from(text: str, pos: int) -> str:
    """Zeile ab `pos` (Zeilenname, dann Zahlen) bis zum naechsten Wort mit mindestens drei Buchstaben nach der ersten Zahl."""
    out, seen_number = [], False
    for m in re.finditer(r"\S+", text[pos:pos + 400]):
        tok = m.group()
        if re.search(r"\d", tok):
            seen_number = True
        elif seen_number and re.search(r"[a-z]{3,}", tok):
            break
        out.append(tok)
    return " ".join(out)


def _check_dataset_run(text: str, quote: str, value: float, dataset: str, percent: bool) -> ColumnCheck | None:
    """Tabellen ohne 'Method'-Kopf: die letzte Folge von Datensatznamen VOR der Zeile ist die Kopfzeile
    ('CORA CiteSeer PubMed CORA Full')."""
    pos = text.find(quote)
    if pos < 0:
        return None
    want = dataset.lower()
    best: list[str] | None = None
    run: list[str] = []
    for m in re.finditer(r"[a-z0-9-]+", text[:pos]):
        tok = m.group()
        stripped = tok.lstrip("0123456789")   # Fussnotenziffer vor dem Namen ("2cora")
        word = stripped if tok[:1].isdigit() and stripped in DATASET_WORDS else tok
        if word in DATASET_WORDS:
            run.append(word)
            continue
        if len(run) >= 3 and want in _merge_full(run):
            best = _merge_full(run)
        run = []
    if len(run) >= 3 and want in _merge_full(run):
        best = _merge_full(run)
    if best is None:
        return None
    def row_numbers(s: str) -> list[float]:
        s = re.sub(r"±\s*\d+(?:\.\d+)?", " ", re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", s))
        return [float(x) for x in re.findall(NUMBER, s)]

    row = row_numbers(quote)
    if len(row) < len(best):   # Zitat enthaelt nur einen Teil der Zeile: im Text bis zum naechsten Zeilennamen fortsetzen
        row = row_numbers(_row_from(text, pos))
    if len(row) < len(best):
        return ColumnCheck("nicht pruefbar", f"Kopfzeile {best} passt nicht zur Zeile ({len(row)} Zahlen)")
    i = best.index(want)
    return _compare(row[i], value, percent, f"Spalte {best[i]}, Position {i + 1} von {len(best)}")


def check_columns(
    norm_text: str, norm_quote: str, value: float, method: str, dataset: str, metric: str, percent: bool
) -> ColumnCheck:
    """Prueft, ob `value` in der Zeile `norm_quote` an der Stelle steht, die die Kopfzeile fuer
    (method, dataset|metric) vorgibt."""
    for res in (
        _check_methods_as_columns(norm_text, norm_quote, value, method, metric, percent),
        _check_datasets_as_columns(norm_text, norm_quote, value, dataset, percent),
        _check_dataset_run(norm_text, norm_quote, value, dataset, percent),
    ):
        if res is not None:
            return res
    return ColumnCheck("nicht pruefbar", "Kopfzeile der Tabelle nicht erkannt")
