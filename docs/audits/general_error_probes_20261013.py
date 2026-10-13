"""Offline evidence probes. Assertions confirm CURRENT defects, not fixes.

Run: uv run python docs/audits/general_error_probes_20261013.py
No provider, Tavily, sandbox or network requests. Only synthetic data and fakes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'src/repro_agent/harness'))

import numpy as np
from repro_agent import spec_check
from repro_agent.dossier import Claim, Dossier
from repro_agent.harness import ett_data, verify
from repro_agent.harness.code_check import find_violations
from repro_agent.harness.training_check import checkpoint_issues
from repro_agent.method_guard import placeholder_issues
from repro_agent.source_context import author_excerpt
from repro_agent.spec import Spec, SpecItem, build_spec, drop_run_script_defaults, raw_items, resolve
from repro_agent.setup_details import Doc
from repro_agent.costs import BudgetExceeded
from repro_agent.repo_source import clone_plan
from repro_agent.sandbox.base import RunResult
from repro_agent.tasks import cora_task, etth1_task
from test_loop_fakes import FakeSandbox, make_orch
from test_val_label_check import LEAK_CAT


def snapshot():
    files = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files
            if p.startswith(('src/', 'web/src/', 'tests/', 'benchmark/', 'deploy/')) or p == 'Dockerfile'}


def numeric_spec(key='lr', value=0.001):
    return Spec(model='Toy', family='forecast', items=[SpecItem(key, str(value), 'q', 'paper', 'paper', value, enforce=True)])


def small_csv(directory, changed=False):
    data = np.random.default_rng(11).normal(size=(200, 2))
    if changed:
        data[160:] = -data[160:] + 3
    path = directory / ('changed.csv' if changed else 'original.csv')
    path.write_text('date,a,b\n' + '\n'.join('2020-01-01,' + ','.join(map(str, row)) for row in data) + '\n')
    return path


def future_window_leak():
    with tempfile.TemporaryDirectory() as temp, patch.dict(ett_data.SPECS, {'toy': ('toy.csv', 10, 1)}), patch.object(ett_data, 'MAX_HISTORY', 8):
        path = small_csv(Path(temp))
        out = Path(temp) / 'public.npz'
        ett_data.prepare_public('toy', path, out, pred_len=4)
        public = dict(np.load(out)); truth = ett_data.targets('toy', path, pred_len=4)
        w, h, c = truth.shape
        recovered = np.full_like(truth, np.nan)
        for i in range(w):
            for step in range(h):
                if i + step + 1 < w:
                    recovered[i, step] = public['test_inputs'][i + step + 1, -1]
        mask = np.isfinite(recovered)
        assert not any('target' in key or 'label' in key for key in public)
        assert np.allclose(recovered[mask], truth[mask], atol=1e-6)
        assert mask.mean() > 0.9 and etth1_task().mask_val_cmd == ''
        return {'toy_recovered_fraction': float(mask.mean()), 'max_abs_error': float(np.max(np.abs(recovered[mask]-truth[mask]))),
                'etth1_recoverable_fraction': 1 - 97 / (2 * 2785), 'independence_probe_in_standard_run': False}


def data_version_drift():
    with tempfile.TemporaryDirectory() as temp, patch.dict(ett_data.SPECS, {'toy': ('toy.csv', 10, 1)}), patch.object(ett_data, 'MAX_HISTORY', 8):
        original, changed = small_csv(Path(temp)), small_csv(Path(temp), True)
        out = Path(temp) / 'public.npz'; ett_data.prepare_public('toy', original, out, pred_len=4)
        first, second = ett_data.targets('toy', original, pred_len=4), ett_data.targets('toy', changed, pred_len=4)
        assert first.shape == second.shape and not np.allclose(first, second)
        keys = list(np.load(out).files)
        assert not any('hash' in k or 'version' in k for k in keys)
        return {'same_shape': list(first.shape), 'target_mse_between_versions': float(((first-second)**2).mean()), 'public_keys': keys}


def training_vs_inference_batch():
    quote = 'We compare inference speed with batch size 32.'
    arg = "parser.add_argument('--batch_size', type=int, default=8)"
    docs = [Doc('paper', quote), Doc('repo:models/Toy.py@abc', 'class Model:\n    pass\n'), Doc('repo:run.py@abc', arg)]
    reply = [{'key': 'batch_size', 'value': '32', 'quote': quote, 'source': 'paper'},
             {'key': 'batch_size', 'value': '8', 'quote': arg, 'source': 'repo:run.py@abc'}]
    spec = build_spec(lambda *_: json.dumps(reply), [], docs, 'Toy erreicht MSE 0.4', 'ETTh1', numeric_only=True)
    chosen = spec.get('batch_size')
    assert chosen.number == 32 and chosen.enforce and chosen.rank == 'paper'
    return {'enforced_batch': chosen.number, 'training_default': 8, 'source_quote': chosen.quote}


def unrelated_default_stripping():
    excerpt = "=== repo:run.py@abc ===\nparser.add_argument('--time_aware_weight', default=0.5)\n=== repo:scripts/both.sh@abc ===\npython run.py --data Wanted --pred_len 96\npython run.py --data Other --pred_len 192 --time_aware_weight 0.3\n"
    changed, dropped = drop_run_script_defaults(excerpt)
    assert 'default=0.5' not in changed and 'time_aware_weight' in dropped
    return {'wrong_default_removed': True, 'dropped': dropped, 'wanted_command_has_override': False}


def multi_command_values():
    d = Doc('repo:scripts/both.sh@abc', 'python run.py --data Wanted --pred_len 96 --learning_rate 0.01\npython run.py --data Other --pred_len 192 --learning_rate 0.02\n')
    items = raw_items([d], [d.name], 'Toy', 'forecast')
    assert not any(item.key == 'lr' for item in items)
    return {'wanted_lr': 0.01, 'lr_in_raw_items': False}


def unrelated_keyword_conflicts():
    spec = Spec(model='Toy', family='forecast', items=numeric_spec('batch_size', 32).items + numeric_spec('dropout', 0.1).items)
    code = "CONFIG={'batch_size':32,'dropout':0.1}\ntrain=DataLoader(train_set,batch_size=CONFIG['batch_size'])\nretrieval=DataLoader(memory,batch_size=1024)\ny=F.normalize(x,p=2)\nlayer=nn.Dropout(CONFIG['dropout'])\n"
    _, report = spec_check.config_check(code, spec)
    assert {(c['key'], c['literal']) for c in report['conflicts']} == {('batch_size', 1024), ('dropout', 2)}
    return {'false_conflicts': report['conflicts']}


def silent_config_overwrites():
    endings = ["cfg=CONFIG\ncfg['lr']=1.0", "CONFIG={'lr':1.0}", "if True:\n    CONFIG['lr']=1.0"]
    reports = []
    for ending in endings:
        code = "CONFIG={'lr':0.001}\n" + ending + "\noptimizer=Adam(lr=CONFIG['lr'])\n"
        changed, report = spec_check.config_check(code, numeric_spec())
        assert not report['conflicts'] and not report['overrides'] and not report['unused']
        env = {'Adam': lambda **kw: kw}
        exec(compile(changed, '<synthetic-config>', 'exec'), env)
        assert env['optimizer']['lr'] == 1.0
        reports.append({'case': ending, 'actual_lr': 1.0, 'reported_score': spec_check.score(numeric_spec(), report, [])})
    return {'cases': reports}


def paper_number_collision():
    a = SpecItem('batch_size', '16', 'batch size 16', 'repo:a.conf', 'config', 16, specific=True)
    b = SpecItem('batch_size', '32', 'batch size 32', 'repo:b.conf', 'config', 32, specific=True)
    chosen, _ = resolve([a,b], 'forecast', True, corroborate={32})
    assert chosen[0].number == 32 and chosen[0].enforce and not chosen[0].tie
    return {'chosen_batch': 32, 'corroborating_number_may_be_unrelated_horizon': 32, 'tie_removed': True}


class ControlSandbox(FakeSandbox):
    def __init__(self, failure):
        super().__init__(); self.failure, self.masked = failure, set()

    def run(self, branch, cmd, timeout_s):
        task = cora_task()
        if cmd == task.mask_val_cmd:
            if self.failure == 'mask_command': return RunResult(1,'','mask did not run',False)
            self.masked.add(branch); return RunResult(0,'','',False)
        if cmd == task.ablation_cmd and self.failure == 'ablation_command':
            return RunResult(1,'','ablation did not run',False)
        if 'train.py' in cmd and branch in self.masked and self.failure == 'masked_training':
            return RunResult(1,'','masked candidate crashed',False)
        if 'cora_val_eval.py' in cmd:
            return RunResult(0,json.dumps({'status':'ok','metric':'accuracy','value':0.95,'baselines':{'majority':0.3}}),'',False)
        return super().run(branch,cmd,timeout_s)


def validation_control_failures():
    cases = []
    for failure in ['mask_command','masked_training']:
        o,_=make_orch(ControlSandbox(failure));o.allow_test_feedback=False
        result=o.val_label_check('s',LEAK_CAT,0,cora_task(),'accuracy',{'value':0.95})
        assert result['static']=='training' and result['excluded'] is False
        cases.append({'failure':failure,**result})
    return {'cases':cases}


def failed_ablation():
    o,_=make_orch(ControlSandbox('ablation_command'));o.allow_test_feedback=False
    result=o.ablation_check('s',"print('model')\n",0,cora_task(),'accuracy',0.815)
    assert result['verdict']=='Graph nicht genutzt' and result['detail'] is None
    return result


def duplicate_classification_ids():
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'predictions.csv';path.write_text('id,prediction\n0,9\n0,0\n1,1\n')
        predictions=verify.load_predictions(str(path))
        assert predictions=={'0':'0','1':'1'} and verify.check_plausibility(predictions,{'0','1'}) is None
        return {'rows':3,'unique_ids':2,'accepted':predictions,'discarded_prediction':'0,9'}


def placeholder_blind_spots():
    ref={'layers/retrieval.py':'class Retrieval:\n    def retrieve(self,x):\n        return x.mean(0)\n'}
    cases=["class Retrieval:\n    def retrieve(self,x):\n        result=torch.zeros_like(x)\n        return result\n",
           "class Retrieval:\n    def fetch_context(self,x):\n        return torch.zeros_like(x)\n"]
    assert all(placeholder_issues(c,ref)==[] for c in cases)
    return {'constant_output_variants_not_detected':len(cases)}


def checkpoint_blind_spots():
    cases=["for epoch in range(2):\n    optimizer.step()\n    best=model.state_dict().copy()\nmodel.load_state_dict(state_dict=best)\n",
           "for epoch in range(2):\n    trainer.step()\n    best=model.state_dict().copy()\nmodel.load_state_dict(best)\n",
           "for epoch in range(2):\n    optimizer.step()\n    self.best=model.state_dict().copy()\nmodel.load_state_dict(self.best)\n"]
    assert all(checkpoint_issues(c)==[] for c in cases)
    return {'shallow_checkpoint_variants_not_detected':len(cases),'variants':['keyword restore','optimizer name trainer','attribute checkpoint']}


def protocol_dependency_omission():
    docs=[Doc('repo:models/Toy.py@abc','class Model:\n    pass\n'),Doc('repo:exp/train.py@abc','from utils.schedule import update\nloss.backward()\n'),
          Doc('repo:utils/schedule.py@abc','from utils.decay import rate\ndef update(x):\n    return rate(x)\n'),Doc('repo:utils/decay.py@abc','def rate(x):\n    return x*0.5\n')]
    excerpt,names=author_excerpt(docs,'Toy','ETTh1')
    assert 'repo:utils/schedule.py@abc' in names and 'repo:utils/decay.py@abc' not in names
    missing=[docs[0],docs[1]]
    _,missing_names=author_excerpt(missing,'Toy','ETTh1',listing=['models/Toy.py','exp/train.py','utils/schedule.py'])
    assert 'repo:exp/train.py@abc' in missing_names
    return {'transitive_training_helper_missing':True,'known_missing_first_helper_does_not_block':True,'included':names}


def requested_repo_ref_lost():
    tag=clone_plan('https://github.com/example/project/tree/v1/model')
    head=clone_plan('https://github.com/example/project/tree/main/model')
    assert tag==head
    return {'requested_v1':tag,'requested_main':head,'ref_distinction_lost':True}


def comments_rejected_as_operations():
    samples=['# See https://example.com/paper\nprint(1)\n',"def fetch_neighbors_from_memory(x):\n    return x\n"]
    reports=[find_violations(c) for c in samples]
    assert all(reports)
    return {'harmless_code_rejections':reports}


def candidate_orch(codes, vals, spec=None):
    o,_=make_orch(FakeSandbox());o.graph_repair=True;o.allow_test_feedback=False
    o.spec_mode,o.spec=bool(spec),spec
    sequence=iter(codes);o.write_code=lambda *a,**k:next(sequence)
    o._emit=lambda *a,**k:None
    def run_seed(claim,task,snapshot,dossier,run_dir,plan,code,seed,candidate=None):
        h=hashlib.sha256(code.encode()).hexdigest()[:12]
        o.__dict__.setdefault('_val',{})[(h,seed)]={'metric':'mse','value':vals[code]}
        return 0.4,None,code
    o._run_seed=run_seed
    o._seed0_quality_loop=lambda claim,task,snapshot,dossier,run_dir,plan,code,v,note,seed,*a,**k:(v,note,code,None,[],set(),0)
    return o


def unchecked_forecast_validation():
    codes=["print('copies validation labels')\n","print('independent model')\n"]
    o=candidate_orch(codes,dict(zip(codes,[0.,0.5])))
    o.val_label_check=lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not be called for forecast'))
    c=Claim('c','Toy','mse',0.4,0.01,'q','paper');d=Dossier(paper_id='p',claims=[c])
    with tempfile.TemporaryDirectory() as temp:
        result=o._run_candidates(c,etth1_task(),'s',d,Path(temp),'p',0,2)
    assert result[6]==0 and not d.val_label_check
    return {'winning_candidate':0,'reported_val_mse':0.,'forecast_label_use_check':False}


def stale_spec_score():
    good="CONFIG={'lr':0.001}\nrate=CONFIG['lr']\n# candidate A\n"
    other="CONFIG={'lr':0.001}\nrate=CONFIG['lr']\n# candidate B\n"
    bad="CONFIG={'lr':0.001}\nprint('unused configuration')\n"
    s=numeric_spec();o=candidate_orch([good,other],{good:0.5,other:0.3},s)
    original=o._seed0_quality_loop
    def repaired(claim,task,snapshot,dossier,run_dir,plan,code,v,note,seed,*a,**k):
        if code!=good: return original(claim,task,snapshot,dossier,run_dir,plan,code,v,note,seed,*a,**k)
        o._val[(hashlib.sha256(bad.encode()).hexdigest()[:12],seed)]={'metric':'mse','value':0.1}
        return v,note,bad,None,[],set(),1
    o._seed0_quality_loop=repaired
    c=Claim('c','Toy','mse',0.4,0.01,'q','paper');d=Dossier(paper_id='p',claims=[c])
    with tempfile.TemporaryDirectory() as temp:
        result=o._run_candidates(c,etth1_task(),'s',d,Path(temp),'p',0,2)
    _,report=spec_check.config_check(result[2],s)
    assert result[2]==bad and spec_check.score(s,report,[])==(0,1)
    return {'selected_candidate':result[6],'actual_final_spec_score':[0,1],'selection_notes':result[-1]}


def budget_aborts_before_valid_candidate_trial():
    o=candidate_orch(["print('valid')\n"],{"print('valid')\n":0.3})
    calls=[]
    def write(*a,**k):
        calls.append('write')
        if len(calls)==2: raise BudgetExceeded('synthetic exhausted budget')
        return "print('valid')\n"
    o.write_code=write;o._run_seed=lambda *a,**k:calls.append('trial')
    c=Claim('c','Toy','mse',0.4,0.01,'q','paper');d=Dossier(paper_id='p',claims=[c])
    with tempfile.TemporaryDirectory() as temp:
        try: o._run_candidates(c,etth1_task(),'s',d,Path(temp),'p',0,2)
        except BudgetExceeded: pass
        else: raise AssertionError('Expected budget stop')
    assert calls==['write','write']
    return {'call_order':calls,'first_candidate_already_generated':True,'first_candidate_tried':False}


def copied_graph_validation_labels_are_selected():
    from test_val_label_check import setup_mask
    copied="import numpy as np\nd=np.load('/data/cora_public.npz')\npred[d['val_idx']]=d['labels'][d['val_idx']]\n"
    honest="print('honest model')\n"
    o,_,events=setup_mask({copied:0.95,honest:0.8},{copied:0.7},[copied,honest])
    o.allow_test_feedback=False
    o.n_candidates=2
    c=Claim('c','Toy','accuracy',0.8,0.01,'q','paper');d=Dossier(paper_id='p',claims=[c])
    with tempfile.TemporaryDirectory() as temp:
        o.run_claim({'seeds':3},c,cora_task(),'s',d,Path(temp))
    selection=next(data for kind,data in events if kind=='candidates')
    control=d.val_label_check['c']['0']
    assert selection['chosen']==0 and control['verdict']=='Validierungslabels genutzt'
    assert control['static']=='selection' and control['excluded'] is False
    return {'selected_copied_labels':True,'control':control,'chosen_candidate':0}


def single_candidate_skips_label_check():
    from repro_agent.harness.val_use_check import static_val_use
    from repro_agent.loop import CANDIDATES
    o,_=make_orch(ControlSandbox('masked_training'));o.graph_repair=True;o.allow_test_feedback=False;o.n_candidates=1
    o.write_code=lambda *a,**k:LEAK_CAT
    o.val_label_check=lambda *a,**k: (_ for _ in ()).throw(AssertionError('single path called label check'))
    c=Claim('c','Toy','accuracy',0.815,0.01,'q','paper');d=Dossier(paper_id='p',claims=[c])
    with tempfile.TemporaryDirectory() as temp:
        o.run_claim({'seeds':3},c,cora_task(),'s',d,Path(temp))
    training_hit=static_val_use(LEAK_CAT)['verdict']=='training'
    assert not d.val_label_check and d.verdicts and training_hit
    return {'label_check_called':False,'static_training_hit':training_hit,'verdict_produced':True,'api_default_candidates':CANDIDATES}


def aborted_live_job_link_is_volatile():
    import asyncio, time
    from repro_agent import api
    from fastapi import HTTPException
    job=api.ReproJob('synthetic-audit-job','toy',True,created=time.time()-7200,done=True)
    with patch.dict(api.repro_jobs,{job.id:job},clear=True):
        api._prune_jobs()
        assert job.id not in api.repro_jobs
        try: asyncio.run(api.stream_reproduction(job.id,None))
        except HTTPException as error: assert error.status_code==404
        else: raise AssertionError('Expired job should be unavailable')
    return {'job_link_after_prune':404,'registry_persistent':False,'local_event_files_may_still_exist':True}


def inactive_model_setting_enforced():
    arg="parser.add_argument('--dropout', type=float, default=0.1)"
    docs=[Doc('repo:models/Toy.py@abc','class Model:\n    def forward(self,x):\n        return x\n'),Doc('repo:run.py@abc',arg)]
    reply=[{'key':'dropout','value':'0.1','quote':arg,'source':'repo:run.py@abc'}]
    s=build_spec(lambda *_:json.dumps(reply),[],docs,'Toy erreicht MSE 0.4','ETTh1',numeric_only=True)
    assert s.get('dropout').enforce
    code="CONFIG={'dropout':0.1}\nclass Model:\n    def forward(self,x):\n        return x\n"
    _,report=spec_check.config_check(code,s)
    assert report['unused']==['dropout']
    assert spec_check.failures(report,[])
    return {'reference_model_uses_dropout':False,'dropout_enforced':True,'faithful_model_gets_failure':report['unused']}


def extraction_calls_missing_from_event_count():
    from types import SimpleNamespace
    from repro_agent.setup_details import collect_raw
    calls=[];events=[]
    def complete(*a,**k):
        calls.append(a[0]);return SimpleNamespace(content='[]',finish_reason='stop')
    o,_=make_orch(FakeSandbox());o.llm=SimpleNamespace(complete=complete)
    o.on_event=lambda e:events.append(e)
    collect_raw(o.llm,[Doc('paper','some text'),Doc('repo:README.md','other text')],'Toy','ETTh1')
    assert len(calls)==2 and events==[]
    return {'direct_setup_calls':len(calls),'live_llm_events':len(events)}


def unclaimed_metric_can_override_matched_claim():
    from repro_agent.harness.verify import agent_fault_suspected
    truth=np.zeros(2);pred=np.array([0.5,0.7]);base=np.array([0.,1.])
    mse=float(((pred-truth)**2).mean());mae=float(np.abs(pred-truth).mean())
    baseline={'repeat':{'mse':float(((base-truth)**2).mean()),'mae':float(np.abs(base-truth).mean())}}
    reason=agent_fault_suspected('mse',mse,mse,baseline,{'mae':mae})
    o,_=make_orch(FakeSandbox());o.allow_test_feedback=False
    c=Claim('c','Toy','mse',mse,0.01,'q','paper')
    verdict=o._verdict(c,[mse]*3,3,[],baseline,{'mse':mse,'mae':mae})
    assert reason and verdict.urteil=='Agenten-Fehler vermutet'
    return {'target_mse':mse,'measured_mse':mse,'measured_mae':mae,'baseline':baseline,'verdict':verdict.urteil,'diagnostic_reason':reason}


PROBES=[future_window_leak,data_version_drift,training_vs_inference_batch,unrelated_default_stripping,multi_command_values,
        unrelated_keyword_conflicts,silent_config_overwrites,paper_number_collision,validation_control_failures,failed_ablation,
        duplicate_classification_ids,placeholder_blind_spots,checkpoint_blind_spots,protocol_dependency_omission,
        requested_repo_ref_lost,comments_rejected_as_operations,unchecked_forecast_validation,stale_spec_score,
        budget_aborts_before_valid_candidate_trial,copied_graph_validation_labels_are_selected,
        single_candidate_skips_label_check,aborted_live_job_link_is_volatile,inactive_model_setting_enforced,
        extraction_calls_missing_from_event_count,unclaimed_metric_can_override_matched_claim]


if __name__=='__main__':
    before=snapshot();results=[]
    with patch('socket.socket.connect',side_effect=RuntimeError('Network disabled during audit')):
        for fn in PROBES:
            result=fn();results.append({'probe':fn.__name__,'confirmed':True,'observed':result})
            print(fn.__name__+': confirmed',flush=True)
    after=snapshot()
    assert before==after,'Product/test/benchmark files changed during audit'
    output={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'network_disabled':True,'product_files_unchanged':True,'confirmed_probes':len(results),'results':results}
    path=Path(__file__).with_suffix('.json');path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print('Evidence:',path)
