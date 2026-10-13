"""Unprivileged generated predictor. No complete test file or hidden targets here."""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import json
import os
import sys
import traceback

import numpy as np

FAILURE_FD_ENV = 'NULLIUS_FAILURE_FD'
SUMMARY_CHARS = 600


def _accepts_calendar(predict):
    # Wrapper **kwargs (e.g. nn.Module.__call__) do not prove that the actual
    # forward function accepts calendar. Opt in through the named parameter.
    try:
        return 'calendar' in inspect.signature(predict).parameters
    except (TypeError, ValueError):
        return False


def _location(frame) -> str:
    code = (frame.line or '').strip()
    return f"{os.path.basename(frame.filename)} line {frame.lineno} in {frame.name}" + (f": {code[:160]}" if code else '')


def _report_failure(phase: str, exc: BaseException, work_dir: str) -> None:
    """Keep the full traceback on stderr and give the broker a short primary cause.

    The broker prints this cause last, so a repair that only sees the end of
    stderr still sees why the generated code failed, not the broker's pipe error.
    """
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    sys.stderr.flush()
    frames = traceback.extract_tb(exc.__traceback__)
    root = os.path.abspath(work_dir) + os.sep
    agent = [f for f in frames if os.path.abspath(f.filename).startswith(root)]
    summary = {'phase': phase, 'error': f'{type(exc).__name__}: {exc}'[:SUMMARY_CHARS],
               'agent_frames': [_location(f) for f in agent[-3:]],
               'last_frame': _location(frames[-1]) if frames else None}
    fd = os.environ.get(FAILURE_FD_ENV, '')
    if fd.isdigit():
        try:
            os.write(int(fd), json.dumps(summary).encode())
        except OSError:
            pass


def main(work_dir: str = '/work'):
    phase = ['loading train.py']
    try:
        _serve(work_dir, phase)
    except BaseException as exc:   # also SystemExit/KeyboardInterrupt raised by generated code
        _report_failure(phase[0], exc, work_dir)
        sys.exit(1)


def _serve(work_dir: str, phase: list[str]):
    sys.path.insert(0, work_dir)
    import nullius_io as nio
    spec = importlib.util.spec_from_file_location('agent_forecaster', os.path.join(work_dir, 'train.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
        factory = getattr(module, 'train_forecaster', None)
        if not callable(factory):
            raise ValueError('Define train_forecaster(data, seed, smoke) returning a callable predictor(history)')
        phase[0] = 'training (train_forecaster)'
        predict = factory(nio.load(), int(os.environ['SEED']), nio.SMOKE)
        if not callable(predict):
            raise ValueError('train_forecaster must return a callable predictor')
        accepts_calendar = _accepts_calendar(predict)
    for n, line in enumerate(sys.stdin):
        phase[0] = f'prediction for history {n} (predictor)'
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
            raise ValueError(f'predictor(history) must return a finite [96,7] forecast, got shape {list(pred.shape)}')
        print(json.dumps(pred.tolist(), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
