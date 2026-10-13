"""Full local SARAF-core reference, fixed settings, no provider calls.

Human adapter around cached author code. Not an autonomous agent reproduction.
Training/forecasting run only inside network-free Docker; scoring uses a fresh
branch and the production harness with its dataset download replaced by a cache.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.review.saraf_contract_smoke import ADAPTER
from repro_agent.harness import ett_data
from repro_agent.harness.code_check import find_violations
from repro_agent.harness.training_check import checkpoint_issues
from repro_agent.sandbox.local import LocalSandbox
from repro_agent.tasks import etth1_task
import repro_agent.loop as loop

FULL_TRAINING = '''
    def precompute(split):
        x,marks=windows[split+'_inputs'],windows[split+'_calendar']
        chunks=[]
        with torch.no_grad():
            for start in range(0,len(x),1024):
                index=torch.arange(start,min(start+1024,len(x)))
                chunks.append(model.rt.retrieve(torch.tensor(x[start:start+1024]),index,
                    train=split=='train',x_mark=torch.tensor(marks[start:start+1024],dtype=torch.float32)))
        return torch.cat(chunks,dim=1)
    model.retrieval_temporal_dict={'train':precompute('train'),'valid':precompute('val')}
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    best_loss=float('inf')
    best_state=None
    records=[]
    metadata={'train_windows':len(windows['train_inputs']),'checkpoint_windows':len(windows['val_inputs']),
              'epochs':records,'fixed_parameters':{'batch_size':8,'max_epochs':10,'lr':.001,'lradj':'type1',
              'seq_len':720,'pred_len':96,'n_period':1,'topm':20,'time_aware_weight':.5}}
    for epoch in range(10):
        model.train()
        order=torch.randperm(len(windows['train_inputs']))
        losses=[]
        for start in range(0,len(order),8):
            index=order[start:start+8]
            x=torch.tensor(take_batch(windows['train_inputs'],index))
            marks=torch.tensor(take_batch(windows['train_calendar'],index),dtype=torch.float32)
            y=torch.tensor(take_batch(windows['train_targets'],index))
            optimizer.zero_grad()
            loss=((model(x,marks,index,mode='train')-y)**2).mean()
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        val_losses=[]
        with torch.no_grad():
            # The original validation averages batch losses. No test data here.
            order=torch.randperm(len(windows['val_inputs']))
            for start in range(0,len(order),8):
                index=order[start:start+8]
                x=torch.tensor(take_batch(windows['val_inputs'],index))
                marks=torch.tensor(take_batch(windows['val_calendar'],index),dtype=torch.float32)
                y=torch.tensor(take_batch(windows['val_targets'],index))
                loss=((model(x,marks,index,mode='valid')-y)**2).mean()
                val_losses.append(float(loss))
        value=float(np.mean(val_losses))
        if not np.isfinite(value): raise ValueError('Non-finite checkpoint validation')
        if value<=best_loss:
            best_loss=value
            best_state=copy.deepcopy(model.state_dict())
            metadata['best_epoch']=epoch+1
        records.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),
                        'visible_validation_loss':value,'lr':optimizer.param_groups[0]['lr']})
        with open('/tmp/forecast-agent/training-summary.json','w') as stream:
            json.dump(metadata,stream)
        # Author type1 schedule is updated at each epoch end.
        for group in optimizer.param_groups: group['lr']=.001*(.5**epoch)
    model.load_state_dict(best_state)
    model.eval()
'''


def take_batch(values, indices):
    """One-element/scalar indices must retain the batch axis, like DataLoader."""
    return values[np.asarray(indices,dtype=np.int64).reshape(-1)]


def adapter():
    code=ADAPTER.replace("    if not smoke:\n        raise ValueError('This diagnostic adapter permits only SMOKE=1, not a paper reproduction')\n",'')
    # No dynamic imports: the normal product code guard applies to this adapter.
    # Vendor globals retain their unused import stubs; optional-package discovery
    # by the optimizer must see the actual (absent) packages, not those stubs.
    for line in ('from importlib.machinery import ModuleSpec\n',
                 "sys.modules['pandas'].__spec__ = ModuleSpec('pandas',loader=None)\n",
                 "progress.__spec__ = ModuleSpec('tqdm',loader=None)\n"):
        assert line in code
        code=code.replace(line,'',1)
    code=code.replace('from models.SARAF import Model\n',
                      "from models.SARAF import Model\nsys.modules.pop('pandas',None)\nsys.modules.pop('tqdm',None)\n",1)
    begin=code.index('    index = torch.arange(8)')
    end=code.index('    retrieval_cache = {}')
    smoke=code[begin:end]
    replacement='    if smoke:\n'+textwrap.indent(smoke,'    ')+'    else:\n'+textwrap.indent(FULL_TRAINING.lstrip('\n'),'    ')
    return 'import copy,json\nimport numpy as np\n'+inspect.getsource(take_batch)+'\n'+code[:begin]+replacement+code[end:]


class OfflineSandbox(LocalSandbox):
    def __init__(self,public,manifest,csv,authors):
        self.public,self.manifest,self.csv,self.authors=public,manifest,csv,authors
        self._branches=set()
    def branch(self,snapshot):
        name='nullius-saraf-full-'+uuid.uuid4().hex[:8]
        subprocess.run(['docker','run','-d','--network','none','--cpus','4','--memory','6g','--name',name,snapshot,'sleep','infinity'],check=True,capture_output=True)
        self._branches.add(name)
        self.put_file(name,'/data/etth1_public.npz',self.public)
        self.put_file(name,'/data/etth1_public.npz.source.json',self.manifest)
        if len(self._branches)==1:
            self.put_files(name,self.authors)
            self.run(name,'chmod 444 /opt/reference/*/*.py',30)
        else:
            # Only the fresh evaluation branch receives the raw scoring file.
            self.put_file(name,'/opt/evaluation-source/ETTh1.csv',self.csv)
        return name


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--author-root',type=Path,default=Path('/tmp/nullius-saraf-local-audit/author_repo'))
    parser.add_argument('--csv',type=Path,default=Path('/tmp/nullius-saraf-local-audit/ETTh1.csv'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--out',type=Path,default=Path('docs/audits/saraf_contract_full_20261013.json'))
    args=parser.parse_args()
    code=adapter()
    compile(code,'saraf_reference_adapter.py','exec')  # Syntax only, never execute author/model code on host.
    violations=find_violations(code)+checkpoint_issues(code)
    if violations: raise ValueError(violations)
    original_harness=loop.HARNESS_DIR
    files={key:args.author_root/key for key in ('models/SARAF.py','layers/Retrieval.py')}
    task=etth1_task().with_independent_selection().with_io_helper()
    report={'product_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'scope':'Human adapter around original SARAF core; full fixed training, not autonomous generated code',
            'contract':task.selection_info(),'target':.371,'tolerance':.01,'requested_seeds':args.seeds,
            'dataset_sha256':hashlib.sha256(args.csv.read_bytes()).hexdigest(),
            'source_hashes':{key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in files.items()},
            'adapter_sha256':hashlib.sha256(code.encode()).hexdigest(),'runs':[],'network':'none',
            'protocol_difference':'Visible checkpoint validation sampled daily (93 windows); hidden selection last20%. Original author validation is dense/full.',
            'sampling_difference':'Retrieval sampled online and cached for each history+calendar; random-number order differs from whole-test precomputation.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.with_suffix('.adapter.py').write_text(code)
    with tempfile.TemporaryDirectory() as directory:
        work=Path(directory)
        public=work/'public.npz'
        os.environ['NULLIUS_FORECAST_CALENDAR']='1'
        ett_data.prepare_public('etth1',args.csv,public)
        data,manifest=public.read_bytes(),Path(str(public)+'.source.json').read_bytes()
        for path in original_harness.glob('*.py'):
            (work/path.name).write_bytes(path.read_bytes())
        with (work/'ett_data.py').open('a') as stream:
            stream.write("\n# Trusted offline scoring source; no agent branch receives this file.\ndef download(name,target):\n    assert name=='etth1'\n    path=Path('/opt/evaluation-source/ETTh1.csv')\n    data_identity.verify([path])\n    return path\n")
        loop.HARNESS_DIR=work
        try:
            for seed in args.seeds:
                print(f'Starting seed {seed}, fixed training and 600s runtime limit',flush=True)
                sb=OfflineSandbox(data,manifest,args.csv.read_bytes(),{f'/opt/reference/{key}':path.read_bytes() for key,path in files.items()})
                orch=loop.Orchestrator.__new__(loop.Orchestrator)
                orch.sandbox,orch._smoke_ok,orch.graph_repair,orch.io_helper=sb,set(),True,True
                orch.train_timeout_s=600
                start=time.monotonic()
                try:
                    branch,result=orch._attempt('repro-snapshot:gate-etth1',code,seed,task,'mse',with_val=True)
                    try: training=json.loads(sb.get_file_limited(branch,'/tmp/forecast-agent/training-summary.json',20_000))
                    except Exception: training=None
                    report['runs'].append({'seed':seed,'seconds':round(time.monotonic()-start,2),'result':result,'training':training})
                    args.out.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps(report['runs'][-1]),flush=True)
                finally:
                    sb.close()
        finally:
            loop.HARNESS_DIR=original_harness
    values=[r['result']['measured'] for r in report['runs'] if r['result'].get('status')=='ok']
    report['complete']=len(values)==len(args.seeds)
    report['mean_mse']=statistics.mean(values) if values else None
    report['std_mse']=statistics.stdev(values) if len(values)>1 else None
    report['within_numeric_tolerance']=report['complete'] and abs(report['mean_mse']-.371)<=.01
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('complete','mean_mse','std_mse','within_numeric_tolerance')}),flush=True)


if __name__=='__main__':
    main()
