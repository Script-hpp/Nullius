"""A failing generated forecaster must reach repair with its own exception, not the broker's pipe error.

Live incident (FITS, code 83f508ba9484): training compared 186 output steps with 96 targets, the
worker died, the broker then failed with BrokenPipeError, and the 1500-character repair window
showed only the broker traceback. Local and network free; the Docker test is explicit opt-in.
"""
import dataclasses
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import numpy as np
import pytest

HARNESS = Path(__file__).resolve().parents[1] / 'src' / 'repro_agent' / 'harness'
sys.path.insert(0, str(HARNESS))
from forecast_runner import WorkerChannel, WorkerFailure   # noqa: E402
from repro_agent.loop import error_query   # noqa: E402

TRIAGE_CHARS = 1500
IO_STUB = """import numpy as np
SMOKE = False
def load():
    return {'train': np.zeros((800, 7), np.float32), 'val': np.zeros((480, 7), np.float32),
            'test_inputs': np.zeros((0, 720, 7), np.float32)}
"""
# Deep agent call chain, so the worker's own traceback alone exceeds the repair window.
TRAIN_WITH_SHAPE_BUG = """import numpy as np
import sys

def model(x):
    return np.concatenate([x[:, -90:], np.zeros((len(x), 96, 7))], axis=1)   # reconstruction + forecast

def mse(out, y):
    return float(((out - y) ** 2).mean())

def step(x, y, depth):
    if depth:
        return step(x, y, depth - 1)
    return mse(model(x), y)

def train_forecaster(data, seed, smoke):
    for epoch in range(3):
        print(f'epoch {epoch} ' + 'x' * 200)                                   # training log on stdout
    print('progress ' * 400, file=sys.stderr)
    step(np.zeros((32, 720, 7)), np.zeros((32, 96, 7)), 25)
    return lambda history: np.zeros((96, 7))
"""


def _work(tmp_path, train_code):
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'nullius_io.py').write_text(IO_STUB)
    (work / 'train.py').write_text(train_code)
    return work


def _channel(work):
    start = f'import sys; sys.path.insert(0, {str(HARNESS)!r}); import forecast_worker; forecast_worker.main({str(work)!r})'
    return WorkerChannel([sys.executable, '-I', '-c', start], dict(os.environ, SEED='0'), str(work))


def _history():
    # Real contract size: about 100 kB JSON, more than one pipe buffer.
    return np.random.default_rng(0).normal(size=(720, 7)).tolist()


def _repair_window(stderr: str, failure: WorkerFailure) -> str:
    """What forecast_runner prints on WorkerFailure, then what loop.triage keeps."""
    return (stderr + f'\n{failure}').strip()[-TRIAGE_CHARS:]


def test_training_error_survives_the_first_broken_history_write(tmp_path, capfd):
    channel = _channel(_work(tmp_path, TRAIN_WITH_SHAPE_BUG))
    try:
        with pytest.raises(WorkerFailure) as caught:
            channel.request(_history(), 'val_predictions.csv history 0')
    finally:
        channel.close()
    message = str(caught.value)
    assert 'sending val_predictions.csv history 0' in message   # the broker's first write broke
    assert message.startswith('Forecast worker failed during training (train_forecaster): ValueError: '
                              'operands could not be broadcast together with shapes (32,186,7) (32,96,7)')
    assert 'train.py line 8 in mse: return float(((out - y) ** 2).mean())' in message
    assert len(message) < 1200
    err = capfd.readouterr().err
    assert 'Traceback (most recent call last)' in err and 'in step' in err   # full worker traceback kept
    assert len(err) > 3 * TRIAGE_CHARS
    window = _repair_window(err, caught.value)
    assert '(32,186,7) (32,96,7)' in window and 'BrokenPipeError' not in window
    # Role 5 searches the last exception line; it must be the worker's, not the broker's.
    assert error_query(window).startswith('ValueError: operands could not be broadcast together')
    assert channel.proc.returncode == 1


def test_prediction_error_after_successful_training_is_reported_at_readline_eof(tmp_path, capfd):
    code = textwrap.dedent("""
        import numpy as np
        def lookup(table, index):
            return table[index]
        def train_forecaster(data, seed, smoke):
            table = np.zeros((10, 96, 7))
            return lambda history: lookup(table, 12)
    """)
    channel = _channel(_work(tmp_path, code))
    try:
        with pytest.raises(WorkerFailure) as caught:
            channel.request(_history(), 'predictions.csv history 0')
    finally:
        channel.close()
    message = str(caught.value)
    assert 'waiting for the forecast of predictions.csv history 0' in message   # stdout EOF, not a write error
    assert 'prediction for history 0 (predictor): IndexError: index 12 is out of bounds' in message
    assert 'train.py line 4 in lookup' in message
    assert 'IndexError' in _repair_window(capfd.readouterr().err, caught.value)


def test_wrong_forecast_shape_names_the_returned_shape(tmp_path):
    code = 'import numpy as np\ndef train_forecaster(data, seed, smoke):\n    return lambda history: np.zeros((186, 7))\n'
    channel = _channel(_work(tmp_path, code))
    try:
        with pytest.raises(WorkerFailure, match=r'finite \[96,7\] forecast, got shape \[186, 7\]'):
            channel.request(_history(), 'predictions.csv history 0')
    finally:
        channel.close()


def test_exit_from_generated_training_is_a_named_failure(tmp_path):
    code = 'import sys\ndef train_forecaster(data, seed, smoke):\n    sys.exit(0)\n'
    channel = _channel(_work(tmp_path, code))
    try:
        with pytest.raises(WorkerFailure, match=r'during training \(train_forecaster\): SystemExit: 0'):
            channel.request(_history(), 'val_predictions.csv history 0')
    finally:
        channel.close()


def test_worker_killed_without_python_exception_reports_the_signal(tmp_path):
    code = 'import os, signal\ndef train_forecaster(data, seed, smoke):\n    os.kill(os.getpid(), signal.SIGKILL)\n'
    channel = _channel(_work(tmp_path, code))
    try:
        with pytest.raises(WorkerFailure, match='was killed by SIGKILL') as caught:
            channel.request(_history(), 'val_predictions.csv history 0')
    finally:
        channel.close()
    assert 'printed no Python exception' in str(caught.value)
    assert channel.proc.returncode == -signal.SIGKILL


def test_valid_worker_still_serves_and_exits_cleanly(tmp_path):
    code = 'import numpy as np\ndef train_forecaster(data, seed, smoke):\n    return lambda history: history[-96:]\n'
    channel = _channel(_work(tmp_path, code))
    try:
        history = _history()
        line = channel.request(history, 'predictions.csv history 0')
        assert np.allclose(json.loads(line), np.asarray(history, dtype=np.float32)[-96:])
        channel.finish()
        assert channel.proc.returncode == 0
    finally:
        channel.close()


@pytest.mark.skipif(os.getenv('NULLIUS_DOCKER_TESTS') != '1', reason='explicit offline Docker integration check')
def test_real_broker_in_sandbox_shows_the_torch_shape_error_to_repair(tmp_path, monkeypatch):
    """Full product path: privilege drop, causal broker, loop crash detail and triage window."""
    import repro_agent.loop as loop
    from repro_agent.sandbox.local import LocalSandbox
    from repro_agent.tasks import etth1_task
    from test_loop_fakes import make_orch
    image = 'repro-snapshot:gate-etth1'
    subprocess.run(['docker', 'image', 'inspect', image], check=True, capture_output=True)
    series = np.arange(1400 * 7, dtype=np.float32).reshape(1400, 7) / 100
    buf = io.BytesIO()
    np.savez_compressed(buf, train=series[:800], val=series[800:1400],
                        test_inputs=np.stack([series[500 + i:1220 + i] for i in range(3)]))

    class OfflineSandbox(LocalSandbox):
        def __init__(self):
            self.containers = []

        def branch(self, snapshot):
            name = 'nullius-worker-error-' + uuid.uuid4().hex[:8]
            subprocess.run(['docker', 'run', '-d', '--network', 'none', '--name', name, snapshot, 'sleep', 'infinity'],
                           check=True, capture_output=True)
            self.containers.append(name)
            self.put_file(name, '/data/etth1_public.npz', buf.getvalue())
            self.put_file(name, '/data/etth1_public.npz.source.json',
                          json.dumps({'toy': hashlib.sha256(buf.getvalue()).hexdigest()}).encode())
            return name

    cfg = json.loads(etth1_task().io_config)
    cfg.update(test_windows=3, val_windows=22)
    task = dataclasses.replace(etth1_task(), io_config=json.dumps(cfg), smoke_ids=(), smoke_ids_cmd='').with_independent_selection()
    # Same defect as live code 83f508ba9484: FITS-style output (90 reconstruction + 96 forecast) against 96 targets.
    code = textwrap.dedent('''
        import numpy as np
        import torch
        from torch import nn

        class FrequencyInterpolation(nn.Module):
            def __init__(self, seq_len=90, pred_len=96):
                super().__init__()
                self.seq_len = seq_len
                self.up = nn.Linear(seq_len, seq_len + pred_len)
            def forward(self, x):
                return self.up(x[:, -self.seq_len:, :].transpose(1, 2)).transpose(1, 2)

        def train_epoch(model, x, y, optimizer):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        def train_forecaster(data, seed, smoke):
            torch.manual_seed(seed)
            model = FrequencyInterpolation()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            x, y = torch.zeros(32, 720, 7), torch.zeros(32, 96, 7)
            for epoch in range(2):
                print(f"epoch {epoch}")
                train_epoch(model, x, y, optimizer)
            return lambda history: np.zeros((96, 7))
    ''')
    real_harness = loop.HARNESS_DIR
    for name in ('forecast_runner.py', 'forecast_worker.py', 'selection_contract.py', 'nullius_io.py'):
        (tmp_path / name).write_bytes((real_harness / name).read_bytes())
    (tmp_path / 'toy_eval.py').write_text("import json\nprint(json.dumps({'status':'ok','metrics':{'mse':8.0}}))\n")
    task = dataclasses.replace(task, harness_files=['toy_eval.py'], eval_cmd='python {harness_dir}/toy_eval.py')
    monkeypatch.setattr(loop, 'HARNESS_DIR', tmp_path)
    sb = OfflineSandbox()
    o, _ = make_orch(sb)
    try:
        _, result = o._attempt(image, code, 0, task, 'mse')
    finally:
        for name in sb.containers:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, check=True)
    assert result['status'] == 'fehler' and 'measured' not in result
    detail = result['detail']
    window = o.triage(detail)
    assert 'The size of tensor a (186) must match the size of tensor b (96)' in window
    assert 'during training (train_forecaster)' in window and 'in train_epoch' in window
    assert 'BrokenPipeError' not in window
    assert loop.error_query(window).startswith('RuntimeError: The size of tensor a (N) must match the size of tensor b (96)')
    # Full stderr still carries the complete torch traceback ahead of the summary.
    assert 'Traceback (most recent call last)' in detail and 'mse_loss' in detail
    assert len(detail) > TRIAGE_CHARS
