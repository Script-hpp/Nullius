"""Unprivileged generated predictor. No complete test file or hidden targets here."""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import json
import os
import sys

import numpy as np


def _accepts_calendar(predict):
    # Wrapper **kwargs (e.g. nn.Module.__call__) do not prove that the actual
    # forward function accepts calendar. Opt in through the named parameter.
    try:
        return 'calendar' in inspect.signature(predict).parameters
    except (TypeError, ValueError):
        return False


def main():
    sys.path.insert(0, '/work')
    import nullius_io as nio
    spec = importlib.util.spec_from_file_location('agent_forecaster', '/work/train.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
        factory = getattr(module, 'train_forecaster', None)
        if not callable(factory):
            raise ValueError('Define train_forecaster(data, seed, smoke) returning a callable predictor(history)')
        predict = factory(nio.load(), int(os.environ['SEED']), nio.SMOKE)
        if not callable(predict):
            raise ValueError('train_forecaster must return a callable predictor')
        accepts_calendar = _accepts_calendar(predict)
    for line in sys.stdin:
        payload = json.loads(line)
        history = np.asarray(payload['history'] if isinstance(payload, dict) else payload, dtype=np.float32)
        calendar = np.asarray(payload['calendar'], dtype=np.int64) if isinstance(payload, dict) else None
        if calendar is not None and calendar.shape not in ((720,5),(720,6)):
            raise ValueError('Calendar must contain exactly the past 720 rows and 5 or 6 features')
        if accepts_calendar and calendar is None:
            raise ValueError('This predictor requires calendar metadata; use a calendar-enabled task snapshot')
        with contextlib.redirect_stdout(sys.stderr):
            pred = predict(history, calendar=calendar) if accepts_calendar else predict(history)
            if hasattr(pred, 'detach'):
                pred = pred.detach().cpu().numpy()
            pred = np.asarray(pred)
        if pred.shape != (96, 7) or not np.isfinite(pred).all():
            raise ValueError('predictor(history) must return a finite [96,7] forecast')
        print(json.dumps(pred.tolist(), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
