"""Trusted causal input broker, inside sandbox only. Agent runs in a separate UID.

Only the broker can read the original NPZ. A forecast is recorded before sending
the next history. Scores remain in another pristine evaluation branch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import secrets

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection_contract import forecast_public


def main(config_path: str):
    cfg = json.loads(Path(config_path).read_text())
    private = Path(cfg['private_data'])
    with np.load(private, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    public = Path(cfg['data'])
    np.savez_compressed(public, **forecast_public(arrays))
    os.chmod(public, 0o444)
    os.makedirs('/output', exist_ok=True)
    # Only the broker owns committed forecasts. The worker must not replace a
    # previous CSV after later histories reveal its future observations.
    os.chmod('/output', 0o700)
    os.makedirs('/tmp/forecast-agent', exist_ok=True)
    os.chown('/tmp/forecast-agent', 65534, 65534)
    # No privileges, supplementary groups or writable interpreter installation.
    cmd = ['setpriv', '--reuid=65534', '--regid=65534', '--clear-groups', '--bounding-set=-all',
           '--inh-caps=-all', '--ambient-caps=-all', '--no-new-privs',
           sys.executable, '-I', str(Path(__file__).with_name('forecast_worker.py'))]
    env = dict(os.environ, HOME='/tmp/forecast-agent', PYTHONNOUSERSITE='1')
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, cwd='/tmp/forecast-agent')
    series = np.concatenate([arrays['train'], arrays['val']])
    train_end = len(arrays['train'])
    val_starts = train_end + cfg['val_stride'] * np.arange(cfg['val_windows'])
    val_inputs = np.stack([series[s-720:s] for s in val_starts])
    val_calendar = test_calendar = None
    if 'train_calendar' in arrays:
        calendar = np.concatenate([arrays['train_calendar'], arrays['val_calendar']])
        val_calendar = np.stack([calendar[s-720:s] for s in val_starts])
        test_calendar = arrays['test_calendar']
    count = cfg['smoke_windows'] if os.environ.get('SMOKE') == '1' else None
    checks = []
    n_seen = 0
    def predict(history, calendar):
        payload = {'history': history.tolist(), 'calendar': calendar.tolist()} if calendar is not None else history.tolist()
        proc.stdin.write(json.dumps(payload, allow_nan=False) + '\n')
        proc.stdin.flush()
        line = proc.stdout.readline(256_000)
        if not line or not line.endswith('\n'):
            raise RuntimeError('Forecast worker stopped or exceeded its output contract')
        pred = np.asarray(json.loads(line), dtype=np.float64)
        if pred.shape != (96, 7) or not np.isfinite(pred).all():
            raise ValueError('Forecast worker output must be finite [96,7]')
        return pred
    try:
        for filename, inputs, calendars in [('val_predictions.csv', val_inputs, val_calendar),
                                            ('predictions.csv', arrays['test_inputs'], test_calendar)]:
            with open('/output/' + filename, 'w') as out:
                out.write('id,prediction\n')
                for i, history in enumerate(inputs[:count]):
                    calendar = calendars[i] if calendars is not None else None
                    pred = predict(history, calendar)
                    # Reservoir sampling keeps eight unknown-to-agent stability probes.
                    n_seen += 1
                    if len(checks) < 8:
                        checks.append((history.copy(), None if calendar is None else calendar.copy(), pred.copy()))
                    else:
                        slot = secrets.randbelow(n_seen)
                        if slot < len(checks):
                            checks[slot] = (history.copy(), None if calendar is None else calendar.copy(), pred.copy())
                    # Commit this prediction before exposing any later input.
                    out.writelines(f'{i}_{h}_{c},{pred[h,c]:.8g}\n' for h in range(96) for c in range(7))
                    out.flush()
        for history, calendar, expected in checks:
            if not np.allclose(predict(history, calendar), expected, rtol=0, atol=1e-4):
                raise ValueError('Prediction changed for a previously seen history; inference must be deterministic and must not update the model')
        proc.stdin.close()
        proc.wait(timeout=10)
        if proc.returncode != 0:
            raise RuntimeError(f'Forecast worker exited with {proc.returncode}')
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


if __name__ == '__main__':
    main(sys.argv[1])
