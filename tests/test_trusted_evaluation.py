"""Prediction artifacts are evaluated outside the generated program's runtime."""
import dataclasses
import base64
import gzip
import hashlib
import json
import os
import subprocess
import uuid

import pytest

from repro_agent.sandbox.base import RunResult
from repro_agent.sandbox.local import LocalSandbox
from repro_agent.sandbox.nebius import NebiusSandbox
from repro_agent.tasks import Task, cora_task
from test_loop_fakes import FakeSandbox, GOOD_CODE, make_orch


def test_test_and_validation_harness_use_a_clean_snapshot_branch():
    sb = FakeSandbox()
    o, _ = make_orch(sb)
    agent, result = o._attempt('trusted', GOOD_CODE, 0, cora_task(), 'accuracy', with_val=True)
    eval_branches = {b for b, cmd in sb.cmds if 'cora_eval.py' in cmd or 'cora_val_eval.py' in cmd}
    assert result['status'] == 'ok' and eval_branches and agent not in eval_branches
    assert sb.n == 2
    assert {path for b, path, _ in sb.puts if b in eval_branches} == {'/output/predictions.csv', '/output/val_predictions.csv'}


@pytest.mark.parametrize('flag,field', [('ablate', 'ablation_cmd'), ('mask_val', 'mask_val_cmd')])
@pytest.mark.parametrize('timeout', [False, True])
def test_failed_control_preparation_cannot_be_measured(flag, field, timeout):
    sb = FakeSandbox()
    task = dataclasses.replace(cora_task(), **{field: 'CONTROL'})
    original = sb.run
    sb.run = lambda b, cmd, t: RunResult(1, '', 'control failed', timeout) if cmd == 'CONTROL' else original(b, cmd, t)
    o, _ = make_orch(sb)
    _, result = o._attempt('s', GOOD_CODE, 0, task, 'accuracy', **{flag: True})
    assert result['status'] == 'ungueltig' and 'control preparation failed' in result['detail']
    assert not any('train.py' in cmd for _, cmd in sb.cmds)


def test_multiple_harness_json_outputs_cannot_override_a_measurement():
    sb = FakeSandbox()
    original = sb.run
    good = json.dumps({'status': 'ok', 'metrics': {'accuracy': 0.8}})
    sb.run = lambda b, cmd, t: RunResult(0, good + '\n' + good, '', False) if 'cora_eval.py' in cmd else original(b, cmd, t)
    o, _ = make_orch(sb)
    _, result = o._attempt('s', GOOD_CODE, 0, cora_task(), 'accuracy')
    assert result['status'] == 'ungueltig' and 'unlesbar' in result['detail']


def test_duplicate_graph_prediction_ids_are_rejected_before_dictionary_conversion(tmp_path):
    from repro_agent.harness.verify import load_predictions
    csv = tmp_path / 'pred.csv'
    csv.write_text('id,prediction\n0,9\n0,0\n1,1\n')
    with pytest.raises(ValueError, match='Duplicate prediction id'):
        load_predictions(str(csv))


@pytest.mark.parametrize('oversized_metadata', [False, True])
def test_nebius_prediction_transfer_is_bounded_before_decompression(oversized_metadata):
    data = b'x' * 100_000
    blob = gzip.compress(data)
    sb = NebiusSandbox.__new__(NebiusSandbox)
    calls = []
    def run(branch, cmd, **kw):
        calls.append(cmd)
        size = 2_000_000 if oversized_metadata else len(blob)
        stdout = f'{hashlib.sha256(data).hexdigest()} {size}' if len(calls) == 1 else base64.b64encode(blob).decode()
        return RunResult(0, stdout, '', False)
    sb.run = run
    with pytest.raises(ValueError, match='size limit'):
        sb.get_file_limited('b', '/output/predictions.csv', 1024)
    assert len(calls) == (1 if oversized_metadata else 2)


def test_smoke_prediction_transfer_uses_the_same_bounded_reader():
    sb = FakeSandbox()
    calls = []
    def limited(branch, path, limit):
        calls.append((path, limit))
        raise ValueError('Prediction artifact exceeds its size limit')
    sb.get_file_limited = limited
    o, _ = make_orch(sb)
    error = o._smoke_check('b', GOOD_CODE, cora_task())
    assert error and 'size limit' in error
    assert calls == [('/output/predictions.csv', 2_000_000)]


@pytest.mark.skipif(os.getenv('NULLIUS_DOCKER_TESTS') != '1', reason='explicit offline Docker integration check')
def test_agent_startup_hook_cannot_change_real_harness_or_validation(tmp_path, monkeypatch):
    import repro_agent.loop as loop
    # Host only controls Docker; generated code and startup hook never run here.
    image = 'python:3.11-slim'
    subprocess.run(['docker', 'image', 'inspect', image], check=True, capture_output=True)
    class OfflineSandbox(LocalSandbox):
        def __init__(self):
            self.containers = []
        def branch(self, snapshot):
            name = 'nullius-integrity-' + uuid.uuid4().hex[:8]
            subprocess.run(['docker', 'run', '-d', '--network', 'none', '--name', name, snapshot, 'sleep', 'infinity'],
                           capture_output=True, check=True)
            self.containers.append(name)
            return name
    (tmp_path / 'toy_eval.py').write_text("import json\nprint(json.dumps({'status':'ok','metrics':{'mse':8.0}}))\n")
    (tmp_path / 'toy_val_eval.py').write_text("import json\nprint(json.dumps({'status':'ok','metric':'mse','value':9.0}))\n")
    monkeypatch.setattr(loop, 'HARNESS_DIR', tmp_path)
    task = Task(image, [], ['toy_eval.py'], 'python {harness_dir}/toy_eval.py', '', '',
                val_harness_files=('toy_val_eval.py',), val_eval_cmd='python {harness_dir}/toy_val_eval.py')
    code = '''from pathlib import Path
import site
Path(site.getsitepackages()[0], 'sitecustomize.py').write_text("import atexit, json\\natexit.register(lambda: print(json.dumps({'status':'ok','metrics':{'mse':0.371},'metric':'mse','value':0.371})))\\n")
Path('/output').mkdir(exist_ok=True)
Path('/output/predictions.csv').write_text('id,prediction\\n0,1\\n')
Path('/output/val_predictions.csv').write_text('id,prediction\\n0,1\\n')
'''
    sb = OfflineSandbox()
    o, _ = make_orch(sb)
    try:
        _, result = o._attempt(image, code, 0, task, 'mse', with_val=True)
        assert result['status'] == 'ok', result
        assert result['measured'] == 8.0 and result['val']['value'] == 9.0
        assert len(sb.containers) == 2
    finally:
        for name in sb.containers:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, check=True)
