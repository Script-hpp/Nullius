"""Trusted causal input broker, inside sandbox only. Agent runs in a separate UID.

Only the broker can read the original NPZ. A forecast is recorded before sending
the next history. Scores remain in another pristine evaluation branch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import secrets

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection_contract import forecast_public

FAILURE_FD_ENV = 'NULLIUS_FAILURE_FD'
SUMMARY_BYTES = 8000


class WorkerFailure(RuntimeError):
    """The generated forecaster failed or broke the line contract. Not a broker defect."""


class WorkerChannel:
    """One line-oriented exchange with the unprivileged worker.

    The worker's stderr stays the caller's stderr (full traceback, live, and seen by
    the soft-timeout stack dump). A separate pipe carries only its short failure
    summary, so a broken stdin/stdout pipe is reported with the worker's own cause.
    """

    def __init__(self, cmd: list[str], env: dict, cwd: str):
        self._summary, write_end = os.pipe()
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                         env=dict(env, **{FAILURE_FD_ENV: str(write_end)}), cwd=cwd,
                                         pass_fds=(write_end,))
        finally:
            os.close(write_end)
        os.set_blocking(self._summary, False)

    def request(self, payload, stage: str) -> str:
        try:
            self.proc.stdin.write(json.dumps(payload, allow_nan=False) + '\n')
            self.proc.stdin.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise self.failure(f'sending {stage}') from None
        line = self.proc.stdout.readline(256_000)
        if not line:
            raise self.failure(f'waiting for the forecast of {stage}')
        if not line.endswith('\n'):
            raise WorkerFailure('Forecast worker exceeded its output contract: one JSON forecast line of at most 256000 characters')
        return line

    def finish(self, timeout: float = 10):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait(timeout=timeout)
        if self.proc.returncode != 0:
            raise self.failure('finishing')

    def failure(self, stage: str) -> WorkerFailure:
        """Reap the worker and name its primary error, bounded for repair prompts."""
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        summary = self._read_summary()
        code = self.proc.returncode
        if code is not None and code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = f'signal {-code}'
            ended = f'was killed by {name}' + (' (often the memory limit)' if -code == signal.SIGKILL else '')
        else:
            ended = f'ended with exit code {code}'
        if not summary:
            return WorkerFailure(f'Forecast worker {ended} while the broker was {stage}. '
                                 'The worker printed no Python exception; see its output above.')
        lines = [f"Forecast worker failed during {summary.get('phase', 'an unknown phase')}: {summary.get('error')}"]
        lines += [f'  at {frame}' for frame in summary.get('agent_frames') or []]
        last = summary.get('last_frame')
        if last and last not in (summary.get('agent_frames') or []):
            lines.append(f'  raised in {last}')
        lines.append(f'The trusted broker stopped while {stage} because the worker {ended}. '
                     'Fix the error above in train.py; the full traceback is earlier in stderr.')
        return WorkerFailure('\n'.join(str(line)[:700] for line in lines))

    def _read_summary(self) -> dict | None:
        data = b''
        try:
            while len(data) < SUMMARY_BYTES:
                chunk = os.read(self._summary, SUMMARY_BYTES - len(data))
                if not chunk:
                    break
                data += chunk
        except BlockingIOError:
            pass
        except OSError:
            return None
        try:
            summary = json.loads(data.decode('utf-8', errors='replace')) if data else None
        except ValueError:
            return None
        return summary if isinstance(summary, dict) else None

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        try:   # unsent buffered input must not print a second pipe error at interpreter exit
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            os.close(self._summary)
        except OSError:
            pass


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
    worker = WorkerChannel(cmd, env, '/tmp/forecast-agent')
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
    def predict(history, calendar, stage):
        payload = {'history': history.tolist(), 'calendar': calendar.tolist()} if calendar is not None else history.tolist()
        line = worker.request(payload, stage)
        try:
            pred = np.asarray(json.loads(line), dtype=np.float64)
        except ValueError:
            raise WorkerFailure('Forecast worker wrote a line that is not a JSON forecast; '
                                'print nothing to stdout outside the broker protocol') from None
        if pred.shape != (96, 7) or not np.isfinite(pred).all():
            raise WorkerFailure('Forecast worker output must be finite [96,7]')
        return pred
    try:
        for filename, inputs, calendars in [('val_predictions.csv', val_inputs, val_calendar),
                                            ('predictions.csv', arrays['test_inputs'], test_calendar)]:
            with open('/output/' + filename, 'w') as out:
                out.write('id,prediction\n')
                for i, history in enumerate(inputs[:count]):
                    calendar = calendars[i] if calendars is not None else None
                    pred = predict(history, calendar, f'{filename} history {i}')
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
            if not np.allclose(predict(history, calendar, 'a repeated earlier history'), expected, rtol=0, atol=1e-4):
                raise WorkerFailure('Prediction changed for a previously seen history; inference must be deterministic and must not update the model')
        worker.finish()
    finally:
        worker.close()


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except WorkerFailure as exc:
        # The worker's own traceback is already above; a broker traceback would only
        # push the primary cause out of the repair window.
        print(f'\n{exc}', file=sys.stderr, flush=True)
        sys.exit(1)
