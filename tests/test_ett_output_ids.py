import importlib
import json
import sys
import pytest
from repro_agent.tasks import HARNESS_DIR


@pytest.mark.parametrize('module', ['ett_eval', 'ett_val_eval'])
@pytest.mark.parametrize('bad_id', ['-1_0_0', '0_-1_0', '0_0_-1', '2_0_0', '0_2_0', '0_0_1'])
def test_out_of_range_ids_are_rejected_before_truth_is_loaded(module, bad_id, tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(HARNESS_DIR))
    m = importlib.import_module(module)
    monkeypatch.setattr(m, 'PRED_LEN', 2)
    monkeypatch.setattr(m.ett_data, 'N_CHANNELS', 1)
    monkeypatch.setattr(m.ett_data, 'n_windows', lambda *_: 2)
    monkeypatch.setattr(m.ett_data, 'n_val_windows', lambda *_: 2)
    monkeypatch.setattr(m.ett_data, 'download', lambda *_: pytest.fail('invalid IDs must fail before loading truth'))
    path = tmp_path/'pred.csv'
    path.write_text('id,prediction\n'+bad_id+',0.5\n')
    monkeypatch.setattr(m, 'PREDICTIONS', str(path))
    m.main('etth1')
    out = json.loads(capsys.readouterr().out)
    assert out['status'] == 'ungueltig' and bad_id in out['reason']
