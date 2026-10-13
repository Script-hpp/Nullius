"""Selection targets withheld from fitting; original train/test splits unchanged."""
from __future__ import annotations

import numpy as np

HOLDOUT_FRACTION = 0.2


def selection_ids(ids) -> np.ndarray:
    ids = np.sort(np.asarray(ids))
    if len(ids) < 2:
        raise ValueError('Validation needs both visible and withheld observations')
    n = max(1, int(np.ceil(len(ids) * HOLDOUT_FRACTION)))
    return ids[-n:]


def visible_val_rows(n: int) -> int:
    return int(n * (1 - HOLDOUT_FRACTION))


def graph_public(arrays: dict) -> dict:
    result = {k: v.copy() for k, v in arrays.items()}
    selected = selection_ids(result['val_idx'])
    result['labels'][selected] = -1
    result['val_idx'] = np.setdiff1d(result['val_idx'], selected)
    result['selection_idx'] = selected
    return result


def forecast_public(arrays: dict) -> dict:
    result = {k: v.copy() for k, v in arrays.items()}
    result['val'] = result['val'][:visible_val_rows(len(result['val']))]
    result['test_inputs'] = result['test_inputs'][:0]
    return result
