"""Unprivileged generated predictor. No complete test file or hidden targets here."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys

import numpy as np


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
    for line in sys.stdin:
        history = np.asarray(json.loads(line), dtype=np.float32)
        with contextlib.redirect_stdout(sys.stderr):
            pred = predict(history)
            if hasattr(pred, 'detach'):
                pred = pred.detach().cpu().numpy()
            pred = np.asarray(pred)
        if pred.shape != (96, 7) or not np.isfinite(pred).all():
            raise ValueError('predictor(history) must return a finite [96,7] forecast')
        print(json.dumps(pred.tolist(), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
