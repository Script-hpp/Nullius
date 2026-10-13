"""Cross-paper local check with fixed FITS author preset, no provider calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.review.saraf_contract_full import OfflineSandbox
from repro_agent.harness import ett_data
from repro_agent.harness.code_check import find_violations
from repro_agent.harness.training_check import checkpoint_issues
from repro_agent.tasks import etth1_task
import repro_agent.loop as loop

ADAPTER='''import copy,json,sys
from types import SimpleNamespace
import numpy as np
import torch
import nullius_io as nio
sys.path.insert(0,'/opt/reference')
from models.FITS import Model

def train_forecaster(data,seed,smoke):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    length,horizon,batch=360,96,128
    args=SimpleNamespace(seq_len=length,pred_len=horizon,individual=False,enc_in=7,cut_freq=106)
    model=Model(args)
    windows=nio.forecast_data(length)
    xtr,ytr=windows['train_inputs'],windows['train_targets']
    # Dense checkpoint windows, all targets inside the visible validation prefix.
    series=np.concatenate([data['train'][-length:],data['val']])
    starts=np.arange(length,len(series)-horizon+1)
    xva=np.stack([series[s-length:s] for s in starts])
    yva=np.stack([series[s:s+horizon] for s in starts])
    if smoke:
        xtr,ytr,xva,yva=xtr[:8],ytr[:8],xva[:8],yva[:8]
        batch=8
    optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
    best=float('inf')
    best_state=None
    bad=0
    records=[]
    metadata={'model':'FITS','train_windows':len(xtr),'checkpoint_windows':len(xva),'epochs':records,
              'fixed_parameters':{'length':360,'horizon':96,'cut_freq':106,'H_order':6,'base_T':24,
              'batch_cli':64,'effective_batch':128,'learning_rate':.0005,'epochs_max':50,
              'patience':20,'lradj':'type3','train_mode':1,'individual':False}}
    for epoch in range(1 if smoke else 50):
        model.train()
        order=torch.randperm(len(xtr)).numpy()
        losses=[]
        for start in range(0,len(order)-batch+1,batch):
            indices=order[start:start+batch]
            x,y=torch.tensor(xtr[indices]),torch.tensor(ytr[indices])
            optimizer.zero_grad()
            output,_=model(x)
            loss=((output[:,-horizon:]-y)**2).mean()
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        val_losses=[]
        with torch.no_grad():
            order=torch.randperm(len(xva)).numpy()
            for start in range(0,len(order)-batch+1,batch):
                indices=order[start:start+batch]
                output,_=model(torch.tensor(xva[indices]))
                val_losses.append(float(((output[:,-horizon:]-torch.tensor(yva[indices]))**2).mean()))
        value=float(np.mean(val_losses))
        if not np.isfinite(value): raise ValueError('Non-finite checkpoint validation')
        if value<=best:
            best,best_state,bad=value,copy.deepcopy(model.state_dict()),0
            metadata['best_epoch']=epoch+1
        else: bad+=1
        records.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),
                        'visible_validation_loss':value,'lr':optimizer.param_groups[0]['lr']})
        with open('/tmp/forecast-agent/training-summary.json','w') as stream: json.dump(metadata,stream)
        if bad>=20: break
        for group in optimizer.param_groups: group['lr']=.0005*(.95**epoch)
    model.load_state_dict(best_state)
    model.eval()
    def predict(history):
        with torch.no_grad():
            output,_=model(torch.tensor(history[None,-length:],dtype=torch.float32))
            return output[0,-horizon:].numpy()
    return predict
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=Path('/tmp/nullius-fits-independent-20261013'))
    parser.add_argument('--csv',type=Path,default=Path('/tmp/nullius-saraf-local-audit/ETTh1.csv'))
    parser.add_argument('--out',type=Path,default=Path('docs/audits/fits_independent_20261013.json'))
    args=parser.parse_args()
    errors=find_violations(ADAPTER)+checkpoint_issues(ADAPTER)
    if errors: raise ValueError(errors)
    compile(ADAPTER,'fits_reference_adapter.py','exec')
    files={name:args.repo/name for name in ('models/FITS.py','models/NLinear.py','scripts/FITS/etth1_best.sh','run_longExp_F.py','exp/exp_main_F.py','utils/tools.py','data_provider/data_factory.py')}
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo,text=True).strip()
    task=etth1_task().with_independent_selection().with_io_helper()
    report={'paper':'FITS','paper_url':'https://arxiv.org/html/2307.03756v3','target':.372,'tolerance':.01,
            'seeds':[0,1,2],'settings_source':'scripts/FITS/etth1_best.sh + run_longExp_F.py defaults/derived batch and cutoff',
            'author_commit':source_commit,'source_hashes':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()},
            'adapter_sha256':hashlib.sha256(ADAPTER.encode()).hexdigest(),'contract':task.selection_info(),
            'product_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'dataset_sha256':hashlib.sha256(args.csv.read_bytes()).hexdigest(),'runs':[],
            'scope':'Fixed human adapter around author model; independent cross-paper pipeline test, not an autonomous agent test',
            'protocol_differences':'Visible validation prefix/dense checkpoint windows, hidden last20%; seeds0/1/2; CPU; no author FLOP profiling RNG consumption',
            'no_test_based_tuning':True,'network_during_training':'none'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.with_suffix('.adapter.py').write_text(ADAPTER)
    # Publish immutable configuration BEFORE the first training/test result.
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    original_harness=loop.HARNESS_DIR
    with tempfile.TemporaryDirectory() as directory:
        work=Path(directory)
        public=work/'public.npz'
        os.environ['NULLIUS_FORECAST_CALENDAR']='1'
        ett_data.prepare_public('etth1',args.csv,public)
        data,manifest=public.read_bytes(),Path(str(public)+'.source.json').read_bytes()
        for path in original_harness.glob('*.py'): (work/path.name).write_bytes(path.read_bytes())
        with (work/'ett_data.py').open('a') as stream:
            stream.write("\ndef download(name,target):\n    assert name=='etth1'\n    path=Path('/opt/evaluation-source/ETTh1.csv')\n    data_identity.verify([path])\n    return path\n")
        loop.HARNESS_DIR=work
        try:
            for seed in report['seeds']:
                print(f'FITS seed {seed}: fixed published best preset, no grid/test tuning',flush=True)
                sb=OfflineSandbox(data,manifest,args.csv.read_bytes(),{f'/opt/reference/{name}':files[name].read_bytes() for name in ('models/FITS.py','models/NLinear.py')})
                orch=loop.Orchestrator.__new__(loop.Orchestrator)
                orch.sandbox,orch._smoke_ok,orch.graph_repair,orch.io_helper=sb,set(),True,True
                orch.train_timeout_s=600
                start=time.monotonic()
                try:
                    branch,result=orch._attempt('repro-snapshot:gate-etth1',ADAPTER,seed,task,'mse',with_val=True)
                    try: training=json.loads(sb.get_file_limited(branch,'/tmp/forecast-agent/training-summary.json',30_000))
                    except Exception: training=None
                    report['runs'].append({'seed':seed,'seconds':round(time.monotonic()-start,2),'result':result,'training':training})
                    args.out.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps({'seed':seed,'status':result['status'],'mse':result.get('measured'),'seconds':report['runs'][-1]['seconds']}),flush=True)
                finally: sb.close()
        finally: loop.HARNESS_DIR=original_harness
    values=[r['result']['measured'] for r in report['runs'] if r['result'].get('status')=='ok']
    report.update(complete=len(values)==3,mean_mse=statistics.mean(values) if values else None,
                  std_mse=statistics.stdev(values) if len(values)>1 else None)
    report['within_numeric_tolerance']=report['complete'] and abs(report['mean_mse']-.372)<=.01
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('complete','mean_mse','std_mse','within_numeric_tolerance')}),flush=True)


if __name__=='__main__': main()
