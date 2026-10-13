import dataclasses
import hashlib
import io
import json
import os
import subprocess
import uuid

import numpy as np
import pytest

from repro_agent.harness.selection_contract import forecast_public, graph_public, selection_ids
from repro_agent.tasks import cora_task, etth1_task
from test_io_helper import _load_helper
from test_loop_fakes import make_orch
from repro_agent.sandbox.local import LocalSandbox


def test_graph_selection_labels_are_hidden_and_checkpoint_labels_remain_visible():
    arrays = {'labels': np.arange(30) % 3, 'train_idx': np.arange(5), 'val_idx': np.arange(5, 25), 'test_idx': np.arange(25, 30)}
    public = graph_public(arrays)
    assert np.array_equal(public['selection_idx'], [21, 22, 23, 24])
    assert (public['labels'][public['selection_idx']] == -1).all()
    assert np.array_equal(public['labels'][public['val_idx']], arrays['labels'][public['val_idx']])
    assert np.array_equal(public['train_idx'], arrays['train_idx'])
    assert np.array_equal(public['test_idx'], arrays['test_idx'])
    assert arrays['labels'][21] != -1


def test_forecast_fitting_has_no_test_histories_or_selection_targets():
    arrays = {'train': np.zeros((800, 7)), 'val': np.ones((600, 7)), 'test_inputs': np.ones((3, 720, 7)) * 9}
    public = forecast_public(arrays)
    assert public['val'].shape == (480, 7) and public['test_inputs'].shape == (0, 720, 7)
    assert np.array_equal(public['train'], arrays['train'])


def test_named_forecast_windows_have_real_correctly_aligned_targets(tmp_path, monkeypatch):
    series = np.arange(1400 * 7, dtype=np.float32).reshape(1400, 7)
    data = {'train': series[:800], 'val': series[800:1280], 'test_inputs': np.empty((0, 720, 7))}
    helper = _load_helper(etth1_task().with_independent_selection(), tmp_path, monkeypatch, data, False)
    windows = helper.forecast_data(336, stride=4)
    assert np.array_equal(windows['train_inputs'][0], series[:336])
    assert np.array_equal(windows['train_targets'][0], series[336:432])
    assert np.array_equal(windows['val_inputs'][0], series[464:800])
    assert np.array_equal(windows['val_targets'][0], series[800:896])
    assert np.array_equal(windows['val_targets'][1], series[824:920])


def test_contract_is_explicit_and_idempotent_and_preserves_legacy_tasks():
    task = etth1_task()
    protected = task.with_independent_selection()
    assert not task.selection_contract and protected.selection_contract
    assert protected.with_independent_selection() == protected
    assert 'train_forecaster' in protected.with_io_helper().data_description
    assert 'withheld' in protected.val_notes
    graph = cora_task().with_independent_selection()
    assert "['selection_idx']" in graph.val_smoke_ids_cmd
    assert 'NULLIUS_SELECTION_HOLDOUT=1' in graph.val_eval_cmd
    assert graph.eval_cmd == cora_task().eval_cmd


@pytest.mark.skipif(os.getenv('NULLIUS_DOCKER_TESTS') != '1', reason='explicit offline Docker integration check')
@pytest.mark.parametrize('adaptive', [False, True])
def test_real_causal_broker_hides_future_data_and_drops_worker_privileges(tmp_path, monkeypatch, adaptive):
    import repro_agent.loop as loop
    image = 'repro-snapshot:gate-etth1'
    subprocess.run(['docker', 'image', 'inspect', image], check=True, capture_output=True)
    series = np.arange(1400 * 7, dtype=np.float32).reshape(1400, 7) / 100
    test = np.stack([series[500+i:1220+i] for i in range(3)])
    buf = io.BytesIO()
    np.savez_compressed(buf, train=series[:800], val=series[800:1400], test_inputs=test)
    class OfflineSandbox(LocalSandbox):
        def __init__(self): self.containers = []
        def branch(self, snapshot):
            name = 'nullius-causal-' + uuid.uuid4().hex[:8]
            subprocess.run(['docker', 'run', '-d', '--network', 'none', '--name', name, snapshot, 'sleep', 'infinity'],
                           check=True, capture_output=True)
            self.containers.append(name)
            self.put_file(name, '/data/etth1_public.npz', buf.getvalue())
            self.put_file(name, '/data/etth1_public.npz.source.json', json.dumps({'toy': hashlib.sha256(buf.getvalue()).hexdigest()}).encode())
            return name
    cfg = json.loads(etth1_task().io_config)
    cfg.update(test_windows=3, val_windows=22)
    task = dataclasses.replace(etth1_task(), io_config=json.dumps(cfg), smoke_ids=(), smoke_ids_cmd='').with_independent_selection()
    code = '''import numpy as np
import os
def train_forecaster(data, seed, smoke):
    assert os.getuid() == 65534
    assert data['test_inputs'].shape == (0,720,7)
    assert len(data['val']) == 480
    assert not os.access('/output', os.W_OK)
    try:
        with open('/output/predictions.csv','w') as f:
            f.write('must not replace committed forecasts')
    except PermissionError:
        pass
    else:
        raise RuntimeError('committed forecasts must not be writable by the worker')
    try:
        with open('/opt/forecast-runtime/source.npz','rb') as f:
            f.read(1)
    except PermissionError:
        pass
    else:
        raise RuntimeError('private future data must not be readable')
    return lambda history: np.repeat(history[-1:,:], 96, axis=0)
'''
    if adaptive:
        code = code.replace('    return lambda history: np.repeat(history[-1:,:], 96, axis=0)',
                            '    calls=[0]\n    def predict(history):\n        calls[0]+=1\n'
                            '        return np.repeat(history[-1:,:],96,axis=0)+calls[0]*.01\n    return predict')
    # Score code remains trusted and network free; this test checks the broker,
    # not a paper number. Actual forecast bytes are checked independently below.
    real_harness = loop.HARNESS_DIR
    for name in ('forecast_runner.py', 'forecast_worker.py', 'selection_contract.py', 'nullius_io.py'):
        (tmp_path / name).write_bytes((real_harness / name).read_bytes())
    (tmp_path / 'toy_eval.py').write_text("import json\nprint(json.dumps({'status':'ok','metrics':{'mse':8.0}}))\n")
    task = dataclasses.replace(task, harness_files=['toy_eval.py'], eval_cmd='python {harness_dir}/toy_eval.py')
    monkeypatch.setattr(loop, 'HARNESS_DIR', tmp_path)
    sb = OfflineSandbox()
    o, _ = make_orch(sb)
    try:
        branch, result = o._attempt(image, code, 0, task, 'mse')
        if adaptive:
            assert result['status'] == 'fehler' and 'Prediction changed' in result['detail']
            assert 'measured' not in result
            return
        assert result['status'] == 'ok', result
        from repro_agent.harness.independence import parse_predictions
        predictions = parse_predictions(sb.get_file(branch, '/output/predictions.csv'), 3)
        assert np.allclose(predictions, test[:, -1:, :])
        assert result['measured'] == 8.0
    finally:
        for name in sb.containers:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, check=True)


def test_report_records_contract_and_selection_sample_count_without_relabeling_old_runs():
    from repro_agent.dossier import Dossier
    from repro_agent.report import render_markdown
    info = etth1_task().with_independent_selection().selection_info()
    assert info['selection_samples'] == 21 and info['total_validation_samples'] == 117
    text = render_markdown(Dossier(paper_id='p', selection_contract=info))
    assert '21 of 117' in text and 'causal_stream_v1' in text
    assert 'withheld_validation_v1' not in render_markdown(Dossier(paper_id='historical'))
