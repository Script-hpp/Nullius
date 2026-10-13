"""Changing the hidden test score must not change code repair or acceptance."""
import hashlib
import inspect
import pytest
from repro_agent.loop import Orchestrator, select_candidate
from repro_agent.dossier import Claim, Dossier
from repro_agent.tasks import cora_task
from test_loop_fakes import make_orch, FakeSandbox, GOOD_CODE


def test_default_policy_disables_test_feedback():
    assert inspect.signature(Orchestrator.__init__).parameters['allow_test_feedback'].default is False


@pytest.mark.parametrize('score', [0.01, 0.99])
def test_test_score_and_graph_ablation_never_trigger_default_code_repair(score, tmp_path):
    o, calls = make_orch(FakeSandbox(), seeds=1)
    o.graph_repair, o.allow_test_feedback = True, False
    o._fault_check = lambda *a: pytest.fail('test baseline must not decide repair')
    o.ablation_check = lambda *a: pytest.fail('test ablation must not decide repair')
    claim=Claim('c','GCN','accuracy',0.83,0.01,'table','paper')
    d=Dossier(paper_id='p',claims=[claim])
    h=hashlib.sha256(GOOD_CODE.encode()).hexdigest()[:12]
    o._val={(h,0):{'metric':'accuracy','value':0.8,'baselines':{'majority':0.3}}}
    out=o._seed0_quality_loop(claim,cora_task(),'s',d,tmp_path,'plan',GOOD_CODE,score,None,0)
    assert out[2]==GOOD_CODE and not calls['repair']


def test_forecast_fault_is_detected_using_validation_baselines_only():
    o,_=make_orch(FakeSandbox())
    c=Claim('c','forecast','mse',0.371,0.01,'table','paper')
    o._val={('hash',0):{'metric':'mse','value':1.2,'baselines':{'repeat':0.8,'zero':1.5}}}
    assert o._val_fault_check(c,'hash',0)
    o._val[('hash',0)]['value']=0.7
    assert o._val_fault_check(c,'hash',0) is None


def test_failed_remaining_seed_never_changes_the_selected_code(tmp_path):
    sb=FakeSandbox(train_exit=1)
    o,calls=make_orch(sb, seeds=3)
    o.allow_test_feedback, o.graph_repair=False,True
    c=Claim('c','GCN','accuracy',0.83,0.01,'table','paper');d=Dossier(paper_id='p',claims=[c])
    value,note,code=o._run_seed(c,cora_task(),'s',d,tmp_path,'plan',GOOD_CODE,1)
    assert value is None and code==GOOD_CODE and not calls['repair'] and 'identical' in note


def test_all_bad_test_scores_are_reported_without_retuning_code(tmp_path):
    sb=FakeSandbox();sb.acc=lambda seed,abl:0.01
    o,calls=make_orch(sb,seeds=3)
    o.allow_test_feedback,o.graph_repair=False,True
    c=Claim('c','GCN','accuracy',0.83,0.01,'table','paper');d=Dossier(paper_id='p',claims=[c])
    o.run_claim({'seeds':3},c,cora_task(),'s',d,tmp_path)
    assert not calls['repair'] and d.verdicts[0].measured_mean==pytest.approx(0.01)
    assert len([a for a in d.attempts if a.chosen])==3


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),float('-inf'),True,'0.1'])
def test_invalid_validation_scores_cannot_win_candidate_selection(bad):
    assert select_candidate([{'metric':'mse','value':bad},{'metric':'mse','value':0.7}])==(1,'val')


def test_different_validation_metrics_cannot_be_ranked_as_one_number():
    with pytest.raises(ValueError,match='cannot be compared'):
        select_candidate([{'metric':'mse','value':0.1},{'metric':'accuracy','value':0.9}])


def test_validation_baselines_do_not_depend_on_test_rows(tmp_path):
    from test_ett import make_csv
    from repro_agent.harness import ett_data
    path,_=make_csv(tmp_path,'etth1')
    before=ett_data.val_baselines('etth1',path)
    rows=path.read_text().splitlines()
    for i in range(11521,len(rows)): rows[i]='2016-01-01 00:00:00,'+','.join(['1000000']*7)
    path.write_text('\n'.join(rows)+'\n')
    assert ett_data.val_baselines('etth1',path)==before


def test_candidates_that_all_trained_on_validation_labels_cannot_be_promoted(tmp_path):
    from test_val_label_check import setup_mask, LEAK_CAT
    from test_candidates import run
    from repro_agent.method_guard import MethodImplementationBlocked
    codes=[LEAK_CAT,LEAK_CAT+'\n# v2\n',LEAK_CAT+'\n# v3\n']
    o,_,_=setup_mask(dict(zip(codes,[0.90,0.95,0.92])),dict(zip(codes,[0.3]*3)),codes)
    o.allow_test_feedback=False
    with pytest.raises(MethodImplementationBlocked,match='validation labels'):
        run(o,tmp_path)


def test_missing_validation_never_falls_back_to_test_based_selection(tmp_path):
    from test_candidates import setup, run, C0, C1, C2
    from repro_agent.method_guard import MethodImplementationBlocked
    o,_,_,events=setup({C0:0.83,C1:0.6,C2:0.7},{})
    o.allow_test_feedback=False
    with pytest.raises(MethodImplementationBlocked,match='validation did not complete'):
        run(o,tmp_path)
    assert not any(kind=='candidates' for kind,_ in events)


@pytest.mark.parametrize('exit_code,timed_out,value', [(1,False,0.8),(0,True,0.8),(0,False,float('nan')),
                                                    (0,False,float('inf')),(0,False,True),(0,False,'0.8')])
def test_failed_or_invalid_validation_output_cannot_be_used(exit_code,timed_out,value):
    import json
    from repro_agent.sandbox.base import RunResult
    sb=FakeSandbox()
    o,_=make_orch(sb)
    sb.run=lambda *a: RunResult(exit_code,json.dumps({'status':'ok','metric':'accuracy','value':value}),'',timed_out)
    result=o._val_eval('b','/harness',cora_task(),0)
    assert 'error' in result and 'value' not in result


def test_partial_seed_run_cannot_issue_a_reproduction_verdict(tmp_path):
    from repro_agent.method_guard import MethodImplementationBlocked
    o,_=make_orch(FakeSandbox(),seeds=3)
    o.allow_test_feedback=False
    original=o._run_seed
    def one(claim,task,snapshot,dossier,run_dir,plan,code,seed,*args):
        return original(claim,task,snapshot,dossier,run_dir,plan,code,seed,*args) if seed==0 else (None,'failed seed',code)
    o._run_seed=one
    c=Claim('c','GCN','accuracy',0.81,0.01,'table','paper');d=Dossier(paper_id='p',claims=[c])
    with pytest.raises(MethodImplementationBlocked,match='1 of 3'):
        o.run_claim({'seeds':3},c,cora_task(),'s',d,tmp_path)
    assert d.verdicts==[] and d.attempts
