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
    horizon: int | None = None
    source_row: str | None = None


def latex_tabulars(text: str) -> list[str]:
    """Outermost tabular bodies; nested header mini-tables do not truncate results."""
    bodies, stack = [], []
    for token in re.finditer(r"\\begin\{tabular\}(?:\[[^\]]*\])?|\\end\{tabular\}", text):
        if token.group().startswith(r"\begin"):
            stack.append(token.end())
        elif stack:
            start = stack.pop()
            if not stack:
                bodies.append(text[start:token.start()])
    return bodies


def latex_rows(text: str) -> list[str]:
    """Split only top-level row breaks; shortstack and other braced cells may contain \\\\."""
    rows, start, depth, i = [], 0, 0, 0
    while i < len(text):
        if text[i] == "\\":
            if text[i:i + 2] == "\\\\" and depth == 0:
                rows.append(text[start:i])
                i += 2
                end = re.match(r"\[[^\]]*\]", text[i:])
                if end:
                    i += end.end()
                start = i
                continue
            if text[i:i + 2] in (r"\{", r"\}", "\\\\"):
                i += 2
                continue
        elif text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth = max(0, depth - 1)
        i += 1
    rows.append(text[start:])
    return [row for row in rows if row.strip()]


def _clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _compare(expected: float, value: float, percent: bool, where: str) -> ColumnCheck:
    got = value * 100 if percent and value <= 1 else value
    if abs(expected - got) < 1e-9:
        return ColumnCheck("ok", f"Zahl {expected} steht laut Kopfzeile an der richtigen Stelle ({where})")
    return ColumnCheck("mismatch", f"Laut Kopfzeile muesste an Stelle {where} die Zahl {expected} stehen, "
                                   f"das Modell waehlte {got}")


def _result_number(cell: str) -> float | None:
    """One scalar, optionally followed by explicitly marked standard deviation."""
    numbers = re.findall(NUMBER, cell)
    if len(numbers) == 1:
        return float(numbers[0])
    plain = re.sub(r"[${}\s]", "", cell)
    mean_std = re.fullmatch(r"(\d+\.\d+)(?:\\pm|±|\+/-)(\d+\.\d+)", plain)
    return float(mean_std[1]) if mean_std else None


def _latex_cells(row: str) -> list[str]:
    """Expand multicolumn spans so the method and metric headers share column indices."""
    cells = []
    for cell in row.split("&"):
        # A method name may be laid out on two lines inside a header cell.
        cell = re.sub(r"\\begin\{tabular\}(?:\[[^\]]*\])?\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\}",
                      lambda m: m[1].replace("\\\\", " "), cell, flags=re.S)
        while re.search(r"\\(?:textbf|textit|textsc|underline|mathrm)\{([^{}]*)\}", cell):
            cell = re.sub(r"\\(?:textbf|textit|textsc|underline|mathrm)\{([^{}]*)\}", r"\1", cell)
        span = re.search(r"\\multicolumn\{(\d+)\}\{[^{}]*\}\{", cell)
        count = int(span[1]) if span else 1
        text = cell[span.end():].strip() if span else cell
        if span and text.endswith("}"):
            text = text[:-1]
        text = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)\b", "", text).strip()
        cells.extend([text] * min(count, 100))
    return cells


def _check_latex_methods(text: str, quote: str, value: float, method: str, dataset: str,
                         metric: str, percent: bool) -> ColumnCheck | None:
    """Use the enclosing LaTeX table, including full rows when a quote is only a fragment."""
    wanted = _clean_name(method.split(" (")[0])
    kind = next((k for k in ("mse", "mae") if re.search(rf"\b{k}\b", metric.lower())), None)
    if not kind:
        return None
    quoted_rows = latex_rows(quote)
    if not quoted_rows:
        return None
    for table in latex_tabulars(text):
        if quote not in table:
            continue
        rows = latex_rows(table)
        anchor = quoted_rows[-1].strip()
        headers = metrics = None
        matches = []
        source_rows = []
        datasets = []
        active_dataset, remaining, group = None, 0, []
        for row in rows:
            cells = _latex_cells(row)
            if any(re.search(r"\bmodels?\b|\bmethods?\b", c) for c in cells) and wanted in [_clean_name(c.split(" (")[0]) for c in cells]:
                headers = cells
            elif sum(_clean_name(c) in ("mse", "mae") for c in cells) >= 2:
                metrics = cells
            else:
                if len(cells) >= 2 and re.fullmatch(r"\d+|avg", cells[1].strip()):
                    label = re.search(r"\b(etth[12]|ettm[12])\b", cells[0], re.I)
                    span = re.search(r"\\multirow\{(\d+)\}", cells[0])
                    if label:
                        active_dataset = label[1].lower()
                        remaining = min(int(span[1]), 100) if span else 1
                        group = []
                    elif _clean_name(cells[0]):
                        active_dataset, remaining, group = None, 0, []
                    if remaining > 0:
                        group.append(row)
                    if anchor in row:
                        matches.append(cells)
                        datasets.append(active_dataset if remaining > 0 else None)
                        source_rows.append("\\\\".join(group).strip() if remaining > 0 else row.strip())
                    remaining = max(0, remaining - 1)
        if headers is None or metrics is None or len(headers) != len(metrics) or len(matches) != 1:
            return ColumnCheck("nicht pruefbar", "LaTeX-Kopfzeile oder zitierte Zeile nicht eindeutig")
        positions = [i for i, (name, met) in enumerate(zip(headers, metrics))
                     if _clean_name(name.split(" (")[0]) == wanted and _clean_name(met) == kind]
        if len(positions) != 1:
            return ColumnCheck("nicht pruefbar", "LaTeX-Methode und Metrik nicht eindeutig zugeordnet")
        cells = matches[0]
        pos = positions[0]
        if len(cells) != len(headers) or pos >= len(cells):
            return ColumnCheck("nicht pruefbar", "LaTeX-Zeile passt nicht zur Kopfzeile")
        if _clean_name(dataset) != datasets[0]:
            return ColumnCheck("nicht pruefbar", "Datensatz in der zitierten LaTeX-Zeile nicht belegt")
        horizon = re.search(r"prediction length\s+(\d+)", metric.lower())
        if horizon and cells[1].strip() != horizon[1]:
            return ColumnCheck("mismatch", "Zitierte LaTeX-Zeile hat einen anderen Vorhersagehorizont")
        number = _result_number(cells[pos])
        if number is None:
            return ColumnCheck("nicht pruefbar", "Ergebniszelle enthaelt keine eindeutige Zahl")
        result = _compare(number, value, percent, f"Methode {method}, {kind}, LaTeX-Spalte {pos + 1}")
        # Only the dataset-bound, fully aligned row supplies horizon evidence.
        result.horizon = int(cells[1]) if re.fullmatch(r"\d+", cells[1]) else None
        result.source_row = source_rows[0]
        return result
    return None


def latex_table_caption(text: str) -> str | None:
    """Read one caption with balanced braces, including nested formatting/newlines."""
    starts = list(re.finditer(r"\\caption(?:\[[^\]]*\])?\{", text))
    if len(starts) != 1:
        return None
    start, depth = starts[0].end(), 1
    for i in range(start, len(text)):
        if text[i] in "{}" and (i == 0 or text[i - 1] != "\\"):
            depth += 1 if text[i] == "{" else -1
            if depth == 0:
                return text[start:i]
    return None


def _check_latex_datasets_horizons(text: str, quote: str, value: float, method: str,
                                  dataset: str, metric: str, percent: bool) -> ColumnCheck | None:
    """Methods in rows; repeated dataset/horizon headers in columns (e.g. FITS)."""
    horizon = re.search(r"prediction length\s+(\d+)", metric.lower())
    kind = re.search(r"\b(mse|mae)\b", metric.lower())
    if percent or not horizon or not kind:
        return None
    wanted = _clean_name(method.split(" (")[0])
    for table in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, re.S):
        for tab in latex_tabulars(table.group()):
            if quote not in tab:
                continue
            rows = [(row.strip(), _latex_cells(row)) for row in latex_rows(tab)]
            datasets = [cells for _, cells in rows if re.search(r"\bdatasets?\b", cells[0])]
            horizons = [cells for _, cells in rows if _clean_name(cells[0]) in ("horizon", "horizons")]
            if not datasets or not horizons:
                continue
            if len(datasets) != 1 or len(horizons) != 1:
                return ColumnCheck("nicht pruefbar", "Datensatz-/Horizontkopfzeile nicht eindeutig")
            # Metric must belong to this table, not to a nearby results table.
            caption = latex_table_caption(table.group())
            metrics = set(re.findall(r"\b(mse|mae)\b", caption or ""))
            metric_rows = [cells for _, cells in rows if _clean_name(cells[0]) in ("metric", "metrics")]
            metric_cells = metric_rows[0] if len(metric_rows) == 1 else None
            if metrics != {kind[1]} and not metric_cells:
                return ColumnCheck("nicht pruefbar", "Tabellenmetrik nicht eindeutig belegt")
            if metrics and kind[1] not in metrics:
                return ColumnCheck("nicht pruefbar", "Caption und angeforderte Metrik widersprechen sich")
            def method_name(cell):
                span = re.fullmatch(r"\\multirow\{[-+]?\d+\}\{[^{}]*\}\{([^{}]*)\}", cell.strip())
                name = span[1] if span else cell
                return _clean_name(name.split(" (")[0])
            selected = [(row, cells) for row, cells in rows
                        if method_name(cells[0]) == wanted and quote in row]
            if len(selected) != 1:
                return ColumnCheck("nicht pruefbar", "Zitat gehoert nicht eindeutig zur gewaehlten Methode")
            row, cells = selected[0]
            ds, hs = datasets[0], horizons[0]
            if len(ds) != len(hs) or len(cells) != len(ds):
                return ColumnCheck("nicht pruefbar", "Ergebniszeile passt nicht zu Datensatz-/Horizontkopfzeilen")
            if metric_cells is not None and len(metric_cells) != len(ds):
                return ColumnCheck("nicht pruefbar", "Metrikkopfzeile passt nicht zu Datensatz-/Horizontkopfzeilen")
            positions = [i for i in range(1, len(ds))
                         if _clean_name(ds[i]) == _clean_name(dataset) and hs[i].strip() == horizon[1]
                         and (metric_cells is None or _clean_name(metric_cells[i]) == kind[1])]
            if len(positions) != 1:
                return ColumnCheck("nicht pruefbar", "Datensatz und Horizont nicht eindeutig zugeordnet")
            pos = positions[0]
            number = _result_number(cells[pos])
            if number is None:
                return ColumnCheck("nicht pruefbar", "Ergebniszelle enthaelt keine eindeutige Zahl")
            result = _compare(number, value, False,
                              f"{method}, {dataset}, Horizont {horizon[1]}, {kind[1]}, LaTeX-Spalte {pos + 1}")
            result.horizon, result.source_row = int(horizon[1]), row
            return result
    return None


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
    if len(row) < len(cols):   # Kopf nicht sauber getrennt (Zeilenname vor den Zahlen) -> naechste Variante versuchen
        return None
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
        _check_latex_datasets_horizons(norm_text, norm_quote, value, method, dataset, metric, percent),
        _check_latex_methods(norm_text, norm_quote, value, method, dataset, metric, percent),
        _check_methods_as_columns(norm_text, norm_quote, value, method, metric, percent),
        _check_datasets_as_columns(norm_text, norm_quote, value, dataset, percent),
        _check_dataset_run(norm_text, norm_quote, value, dataset, percent),
    ):
        if res is not None:
            return res
    return ColumnCheck("nicht pruefbar", "Kopfzeile der Tabelle nicht erkannt")
