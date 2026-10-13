import json
import pytest
from repro_agent.dossier import Claim, Dossier
from repro_agent.report import render_markdown
from repro_agent.tasks import etth1_task
from test_loop_fakes import make_orch, FakeSandbox, GOOD_CODE


def test_secondary_metrics_and_chosen_flags_ignore_old_measurements_even_when_code_hash_is_reused(tmp_path):
    o,_=make_orch(FakeSandbox(),seeds=3,repairs=[GOOD_CODE])
    o.graph_repair=True
    o.n_candidates=1
    calls=[]
    def attempt(snapshot,code,seed,task,metric,**kw):
        first=not calls;calls.append(seed)
        mse,mae=(0.7,5.0) if first else (0.4+0.01*seed,0.5)
        return 'b'+str(len(calls)),{'status':'ok','measured':mse,'metrics':{'mse':mse,'mae':mae},
                                   'baselines':{'zero':{'mse':1.0,'mae':1.0}}}
    o._attempt=attempt
    c=Claim('c','forecast','mse',0.371,0.01,'table','paper');d=Dossier(paper_id='p',claims=[c])
    o.run_claim({'seeds':3},c,etth1_task(),'snapshot',d,tmp_path)
    assert d.verdicts[0].measured_mean==pytest.approx(0.41)
    assert d.verdicts[0].urteil!='Agenten-Fehler vermutet'
    assert [a.chosen for a in d.attempts].count(True)==3 and d.attempts[0].chosen is False


def test_policy_provenance_survives_storage_and_old_records_are_not_relabeled(tmp_path,monkeypatch):
    import repro_agent.dossier as module
    monkeypatch.setattr(module,'REPO_ROOT',tmp_path)
    d=Dossier(paper_id='p',selection_policy='validation_only')
    path=d.save('test')
    assert Dossier.load(path).selection_policy=='validation_only'
    assert 'Testmessungen steuerten keine Code-Aenderung' in render_markdown(d)
    data=json.loads(path.read_text());data.pop('selection_policy');path.write_text(json.dumps(data))
    old=Dossier.load(path)
    assert old.selection_policy is None and 'Testmessungen steuerten' not in render_markdown(old)


def test_objection_report_does_not_promote_rejected_repeated_seed_measurements():
    from repro_agent.dossier import Attempt
    from repro_agent.objection_experiments import main_values
    d=Dossier(paper_id='p')
    for value,chosen in [(0.4,True),(0.8,False)]:
        d.attempts.append(Attempt(claim_id='c',branch_id='b',parent_snapshot='s',code_hash='same',seed=0,
                                  status='stabil',measured=value,error_summary=None,cost_usd=0,chosen=chosen))
    assert main_values(d,'c','same')=={0:0.4}
