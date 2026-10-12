"""Conservative source-table selection for fresh forecasting discovery claims.

No stored targets are rewritten. Unknown, conflicting, ablation and baseline-only
tables are left to the existing extraction path; this is not a method-fidelity check.
"""
from __future__ import annotations

import re

from .claims_source import _norm
from .table_check import NUMBER, _clean_name, _latex_cells, check_columns, latex_rows


def original_source_quote(text: str, normalized_quote: str) -> str | None:
    """Recover original case/spacing of a verified row or multirow dataset group."""
    for table in re.finditer(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", text, re.S):
        rows = latex_rows(table[1])
        for start in range(len(rows)):
            for end in range(start + 1, min(start + 100, len(rows)) + 1):
                quote = "\\\\".join(rows[start:end]).strip()
                normalized = _norm(quote.replace("|", " "))
                if normalized == normalized_quote or _norm(quote) == normalized_quote:
                    return quote
                if len(normalized) > len(normalized_quote):
                    break
    return None


def _plain(text: str) -> str:
    return re.sub(r"[{}]", " ", re.sub(r"\\[a-zA-Z]+\*?", " ", text)).lower()


def requires_external_weights(abstract: str) -> bool:
    # Explicit dependence, not a discussion of baselines or training from scratch.
    return bool(re.search(
        r"(?:leverag\w*|rely|relies|relying|utiliz\w*|using|uses)\s+(?:on\s+)?"
        r"(?:frozen\s+)?pre[- ]trained\s+(?:vision|language|foundation|large)", _plain(abstract)))


def proposed_method(method: str, abstract: str) -> bool:
    wanted = re.escape(method.strip())
    return any(re.search(rf"(?<![\w-]){wanted}(?![\w-])", clause, re.I)
               for clause in re.findall(r"\bwe\s+(?:propose|introduce|present|develop)\b[^.!?]{0,350}", _plain(abstract)))


def select_source_claim(text: str, dataset: str, abstract: str, metric: str = "mse") -> dict | None:
    """Select a unique proposed-method result for horizon 96 from an aligned main table."""
    if metric not in ("mse", "mae") or requires_external_weights(abstract):
        return None
    found = {}
    for table in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, re.S):
        body = table.group()
        metadata = _plain(body.split(r"\begin{tabular}")[0])
        if re.search(r"ablation|sensitivity|with and without|few[- ]shot|univariate", metadata):
            continue
        if not re.search(r"multivariate|forecasting (?:performance|results)", metadata):
            continue
        length = re.search(r"(?:look[- ]back(?: window)?|input(?: sequence)? length)\s*"
                           r"(?:for all experiments\s*)?(?:of|is|=)?\s*(\d+)", metadata)
        if not length or not 1 <= int(length[1]) <= 720:
            continue
        tabulars = list(re.finditer(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", body, re.S))
        for tab in tabulars:
            rows = latex_rows(tab[1])
            methods = set()
            for row in rows:
                cells = _latex_cells(row)
                if any(re.search(r"\bmodels?\b|\bmethods?\b", _plain(c)) for c in cells):
                    for cell in cells[2:]:
                        name = cell.split(" (")[0].strip()
                        if re.search(r"\b(?:ours|proposed)\b", _plain(cell)) or proposed_method(name, abstract):
                            methods.add(name)
            for row in rows:
                cells = _latex_cells(row)
                if len(cells) < 3 or cells[1].strip() != "96":
                    continue
                for method in methods:
                    for value in set(float(n) for n in re.findall(NUMBER, row)):
                        check = check_columns(_norm(body), _norm(row.strip()), value, method,
                                              dataset, f"{metric.upper()} for prediction length 96", False)
                        if check.status != "ok" or check.horizon != 96 or not check.source_row:
                            continue
                        quote = original_source_quote(body, check.source_row)
                        if not quote:
                            continue
                        key = (_clean_name(method), value, int(length[1]))
                        found[key] = {"method": method, "metric": metric, "target": value,
                                      "horizon": 96, "input_length": int(length[1]), "quote": quote,
                                      "extraction_basis": "verified source table"}
    return next(iter(found.values())) if len(found) == 1 else None
