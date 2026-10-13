import hashlib

import pytest

from repro_agent.dossier import Claim, Dossier
from repro_agent.harness.val_use_check import static_val_use
from repro_agent.method_guard import MethodImplementationBlocked
from repro_agent.spec import Spec, SpecItem
from repro_agent.tasks import cora_task, etth1_task
from test_candidates import C0, C1
from test_loop_fakes import FakeSandbox, make_orch
from test_val_label_check import LEAK_CAT, SELECTION_ONLY, setup_mask


@pytest.mark.parametrize('slice_', ["d['val_idx']", "labels >= 0"])
def test_copying_validation_labels_is_not_early_stopping(slice_):
    code = f"labels=d['labels']\npred[{slice_}]=labels[{slice_}]\n"
    assert static_val_use(code)['verdict'] == 'training'
    assert static_val_use(SELECTION_ONLY)['verdict'] == 'selection'


def test_forecast_fit_on_selection_series_is_detected():
    code = "train=d['train']\nval=d['val']\nfull=np.concatenate((train,val))\nmodel.fit(full[:-1],full[1:])\n"
    assert static_val_use(code, forecasting=True)['verdict'] == 'training'
    assert static_val_use("val=d['val']\nloss=criterion(model(x),val)\nif loss<best: save(model)\n", forecasting=True)['verdict'] != 'training'


def test_copied_validation_predictions_cannot_win_selection(tmp_path):
    copied = "import numpy as np\nd=np.load('/data/cora_public.npz')\npred[d['val_idx']]=d['labels'][d['val_idx']]\n"
    honest = "print('honest')\n"
    o, _, events = setup_mask({copied: .95, honest: .8}, {copied: .7}, [copied, honest])
    o.allow_test_feedback, o.n_candidates = False, 2
    c = Claim('c', 'GCN', 'accuracy', .8, .01, 'table', 'paper')
    d = Dossier(paper_id='p', claims=[c])
    o.run_claim({'seeds': 3}, c, cora_task(), 's', d, tmp_path)
    assert next(info for kind, info in events if kind == 'candidates')['chosen'] == 1
    assert d.val_label_check['c']['0']['excluded']


def test_failed_masked_execution_is_excluded_without_claiming_a_successful_control():
    o, _ = make_orch(FakeSandbox())
    o.allow_test_feedback = False
    o._attempt = lambda *a, **k: ('b', {'status': 'fehler', 'detail': 'masked run crashed'})
    result = o.val_label_check('s', SELECTION_ONLY, 0, cora_task(), 'accuracy', {'value': .8})
    assert result['excluded'] and result['verdict'] == 'nicht messbar'


def test_repo_repair_cannot_bypass_validation_admission(tmp_path):
    from test_repo_diff import make, run
    o, _, _, events, claim, dossier, _ = make({C0: .8, LEAK_CAT: .95}, [LEAK_CAT], test_by_code={C0: .8, LEAK_CAT: .82})
    o.allow_test_feedback = False
    run(o, claim, dossier, tmp_path)
    repo = next(info for kind, info in events if kind == 'repo_diff')
    assert repo['result'] == 'kept' and 'independent validation' in repo['reason']
    assert not any(a.chosen and a.code_hash == hashlib.sha256(LEAK_CAT.encode()).hexdigest()[:12] for a in dossier.attempts)


@pytest.mark.parametrize('stage', ['crash', 'quality'])
def test_selection_rescores_the_final_hash_after_every_repair(stage, tmp_path):
    good = "CONFIG={'lr':.001}\nopt=Adam(lr=CONFIG['lr'])\n"
    broken = "CONFIG={'lr':.001}\nopt=Adam(lr=.001)\n"
    honest = good + '# second candidate\n'
    o, _ = make_orch(FakeSandbox())
    o.allow_test_feedback, o.graph_repair = False, True
    o.spec = Spec(model='Toy', family='forecast', items=[SpecItem('lr', '.001', 'q', 'paper', 'paper', .001, enforce=True)])
    o._active_spec = lambda: o.spec
    codes = iter([good, honest])
    o.write_code = lambda *a, **kw: next(codes)
    o._val = {}
    def run_seed(claim, task, snapshot, dossier, run_dir, plan, code, seed, cand):
        final = broken if stage == 'crash' and code == good else code
        h = hashlib.sha256(final.encode()).hexdigest()[:12]
        o._val[(h, seed)] = {'metric': 'mse', 'value': .1 if final == broken else .5}
        return .4, None, final
    def quality(claim, task, snapshot, dossier, run_dir, plan, code, v, note, seed, *args, **kw):
        final = broken if stage == 'quality' and code == good else code
        o._val[(hashlib.sha256(final.encode()).hexdigest()[:12], seed)] = {'metric': 'mse', 'value': .1 if final == broken else .5}
        return v, note, final, None, [], set(), 0
    o._run_seed, o._seed0_quality_loop = run_seed, quality
    c = Claim('c', 'Toy', 'mse', .4, .01, 'table', 'paper')
    result = o._run_candidates(c, etth1_task(), 's', Dossier(paper_id='p'), tmp_path, 'p', 0, 2)
    assert result[2] == honest and result[6] == 1


def test_single_candidate_is_checked_before_any_other_seed(tmp_path):
    o, _ = make_orch(FakeSandbox(), seeds=3, write=LEAK_CAT)
    o.allow_test_feedback, o.graph_repair, o.n_candidates = False, True, 1
    o._selection_check = lambda *a: {'excluded': True, 'verdict': 'Validation training detected'}
    claim = Claim('c', 'Toy', 'accuracy', .815, .01, 'table', 'paper')
    dossier = Dossier(paper_id='p')
    with pytest.raises(MethodImplementationBlocked, match='validation controls'):
        o.run_claim({'seeds': 3}, claim, cora_task(), 's', dossier, tmp_path)
    assert dossier.verdicts == [] and all(a.seed == 0 for a in dossier.attempts)


def test_matching_mse_is_not_overridden_by_a_legitimate_mae_tradeoff():
    o, _ = make_orch(FakeSandbox())
    claim = Claim('c', 'Toy', 'mse', .37, .01, 'table', 'paper')
    v = o._verdict(claim, [.37] * 3, 3, [], {'repeat': {'mse': .5, 'mae': .5}}, {'mse': .37, 'mae': .6})
    assert v.urteil == 'reproduziert'
