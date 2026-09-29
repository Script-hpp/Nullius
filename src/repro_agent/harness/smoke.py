"""Formpruefung des Smoke-Laufs (vor dem Training). Laeuft im Orchestrator, kein LLM.

Der Agenten-Code laeuft mit SMOKE=1 auf winziger Datenmenge; die Ausgabe wird gegen die vom Task
festgelegte Form geprueft. Die Meldung geht an die Reparaturschleife (Englisch, Prompt fuer `code`).
"""

from __future__ import annotations

import csv
import io
import math


def _shape_from_ids(ids: list[str], parts: int) -> tuple[int, ...] | None:
    try:
        split = [tuple(int(x) for x in i.split("_")) for i in ids]
    except ValueError:
        return None
    if not split or any(len(s) != parts for s in split):
        return None
    return tuple(max(s[k] for s in split) + 1 for k in range(parts))


def check_smoke_output(
    csv_bytes: bytes, expected_ids: tuple[str, ...], expected_dims: tuple[int, ...] | None = None,
    dim_names: tuple[str, ...] = (),
) -> str | None:
    """None = Form stimmt, sonst praezise Fehlermeldung."""
    try:
        reader = csv.DictReader(io.StringIO(csv_bytes.decode()))
        if reader.fieldnames != ["id", "prediction"]:
            return f"predictions.csv must have header id,prediction, found {reader.fieldnames}"
        rows = [(r["id"], r["prediction"]) for r in reader]
    except (UnicodeDecodeError, csv.Error, KeyError) as exc:
        return f"predictions.csv unreadable: {exc}"
    ids = [i for i, _ in rows]
    if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        msg = f"expected {len(expected_ids)} unique ids, got {len(ids)} rows ({len(set(ids))} unique)."
        if expected_dims:
            got = _shape_from_ids(ids, len(expected_dims))
            names = f" ({', '.join(dim_names)})" if dim_names else ""
            msg += f" Expected output shape{names} = {expected_dims}, got {got if got else 'unparseable ids'}."
        missing = [i for i in expected_ids if i not in set(ids)][:3]
        extra = [i for i in dict.fromkeys(ids) if i not in set(expected_ids)][:3]
        if missing:
            msg += f" Missing e.g. {missing}."
        if extra:
            msg += f" Unexpected e.g. {extra}."
        return msg
    for _, v in rows:
        try:
            if not math.isfinite(float(v)):
                return f"predictions contain a non-finite value ({v})"
        except ValueError:
            return f"predictions contain a non-numeric value ({v!r})"
    return None
