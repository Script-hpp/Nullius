import hashlib
import json

import pytest

from repro_agent.costs import BudgetExceeded
from repro_agent.dossier import Claim, Dossier
from repro_agent.harness import data_identity, ett_data
from repro_agent.tasks import etth1_task
from test_loop_fakes import FakeSandbox, make_orch


def test_changed_raw_dataset_cannot_be_used_for_scoring(tmp_path, monkeypatch):
    raw = tmp_path / 'ETTh1.csv'
    raw.write_text('original dataset bytes')
    manifest = data_identity.hashes([raw])
    monkeypatch.setenv('NULLIUS_DATA_IDENTITY', json.dumps(manifest))
    assert ett_data.download('etth1', tmp_path) == raw  # cache, no network
    raw.write_text('different same-shaped dataset bytes')
    with pytest.raises(ValueError, match='changed since preparation'):
        ett_data.download('etth1', tmp_path)


def test_preparation_manifest_records_actual_source_file(tmp_path):
    raw = tmp_path / 'source.csv'
    raw.write_bytes(b'dataset')
    out = tmp_path / 'public.npz'
    data_identity.write_manifest(out, [raw])
    assert json.loads((tmp_path / 'public.npz.source.json').read_text()) == {'source.csv': hashlib.sha256(b'dataset').hexdigest()}


def test_budget_stop_after_first_candidate_retains_a_validated_candidate(tmp_path):
    code = "print('model')\n"
    o, _ = make_orch(FakeSandbox())
    o.allow_test_feedback = False
    order = []
    def write(*a, **kw):
        order.append('write')
        if len(order) > 1:
            raise BudgetExceeded('next request does not fit the cap')
        return code
    def run_seed(*a, **kw):
        order.append('trial')
        o._val = {(hashlib.sha256(code.encode()).hexdigest()[:12], 0): {'metric': 'mse', 'value': .4}}
        return .4, None, code
    o.write_code, o._run_seed = write, run_seed
    o._seed0_quality_loop = lambda *a, **kw: (.4, None, code, None, [], set(), 0)
    c = Claim('c', 'Toy', 'mse', .4, .01, 'table', 'paper')
    result = o._run_candidates(c, etth1_task(), 's', Dossier(paper_id='p'), tmp_path, 'plan', 0, 3)
    assert order == ['write', 'trial', 'write'] and result[2] == code
    assert any('spending limit' in text for text in o._warnings)


def test_budget_stop_without_an_independently_validated_candidate_still_aborts(tmp_path):
    o, _ = make_orch(FakeSandbox())
    o.allow_test_feedback = False
    o.write_code = lambda *a, **kw: (_ for _ in ()).throw(BudgetExceeded('cap'))
    c = Claim('c', 'Toy', 'mse', .4, .01, 'table', 'paper')
    with pytest.raises(BudgetExceeded):
        o._run_candidates(c, etth1_task(), 's', Dossier(paper_id='p'), tmp_path, 'plan', 0, 3)
