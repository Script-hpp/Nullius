"""Offline smoke of original SARAF core under the causal input contract.

No provider calls, no paper score, no full reproduction. Author code and one
training batch execute only in an isolated disposable Docker container.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from repro_agent.harness import ett_data
from repro_agent.loop import Orchestrator
from repro_agent.sandbox.local import LocalSandbox
from repro_agent.tasks import etth1_task

ADAPTER = '''import sys, types, hashlib
from importlib.machinery import ModuleSpec
import numpy as np
import torch
from types import SimpleNamespace
import nullius_io as nio
# Neither optional package is used by prepare_dataset/retrieve/encoder.
sys.modules['pandas'] = types.ModuleType('pandas')
sys.modules['pandas'].__spec__ = ModuleSpec('pandas',loader=None)
progress = types.ModuleType('tqdm')
progress.__spec__ = ModuleSpec('tqdm',loader=None)
progress.tqdm = lambda items, **kw: items
sys.modules['tqdm'] = progress
sys.path.insert(0, '/opt/reference')
from models.SARAF import Model

def train_forecaster(data, seed, smoke):
    if not smoke:
        raise ValueError('This diagnostic adapter permits only SMOKE=1, not a paper reproduction')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    windows = nio.forecast_data(720)
    class TrainDataset:
        pred_len, label_len = 96, 48
        def __len__(self): return len(windows['train_inputs'])
        def __getitem__(self, index):
            return (index, windows['train_inputs'][index], windows['train_targets'][index],
                    windows['train_calendar'][index], np.zeros((96,5),dtype=np.float32))
    args = SimpleNamespace(use_gpu=False,gpu=0,task_name='long_term_forecast',seq_len=720,
                           pred_len=96,features='M',enc_in=7,n_period=1,topm=20,freq='h',
                           time_aware_weight=.5,fusion_method='weighted')
    model = Model(args)
    model.rt.prepare_dataset(TrainDataset())
    index = torch.arange(8)
    x = torch.tensor(windows['train_inputs'][:8])
    marks = torch.tensor(windows['train_calendar'][:8],dtype=torch.float32)
    y = torch.tensor(windows['train_targets'][:8])
    with torch.no_grad():
        retrieved = model.rt.retrieve(x,index,train=True,x_mark=marks)
    model.retrieval_temporal_dict = {'train':retrieved}
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    model.train()
    optimizer.zero_grad()
    loss = ((model(x,marks,index,mode='train')-y)**2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    model.eval()
    retrieval_cache = {}
    def predict(history, *, calendar):
        assert calendar.shape==(720,5)
        x = torch.tensor(history[None],dtype=torch.float32)
        marks = torch.tensor(calendar[None],dtype=torch.float32)
        index = torch.tensor([0])
        with torch.no_grad():
            # Original SARAF samples retrieval once per window and stores it.
            # Lazy memoization keeps that property without exposing future windows.
            key = hashlib.sha256(history.tobytes()+calendar.tobytes()).digest()
            if key not in retrieval_cache:
                retrieval_cache[key] = model.rt.retrieve(x,index,train=False,x_mark=marks)
            model.retrieval_temporal_dict['test'] = retrieval_cache[key]
            return model(x,marks,index,mode='test').detach().numpy()[0]
    return predict
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--author-root',type=Path,default=Path('/tmp/nullius-saraf-local-audit/author_repo'))
    parser.add_argument('--csv',type=Path,default=Path('/tmp/nullius-saraf-local-audit/ETTh1.csv'))
    parser.add_argument('--out',type=Path,default=Path('docs/audits/saraf_contract_smoke_20261013.json'))
    args=parser.parse_args()
    sb=LocalSandbox()
    branch='nullius-saraf-smoke-'+uuid.uuid4().hex[:8]
    image='repro-snapshot:gate-etth1'
    subprocess.run(['docker','run','-d','--network','none','--cpus','4','--memory','6g','--name',branch,image,'sleep','infinity'],check=True,capture_output=True)
    sb._branches={branch}
    started=time.monotonic()
    files={'models/SARAF.py':args.author_root/'models/SARAF.py',
           'layers/Retrieval.py':args.author_root/'layers/Retrieval.py'}
    try:
        with tempfile.TemporaryDirectory() as directory:
            public=Path(directory)/'public.npz'
            os.environ['NULLIUS_FORECAST_CALENDAR']='1'
            ett_data.prepare_public('etth1',args.csv,public)
            sb.put_file(branch,'/data/etth1_public.npz',public.read_bytes())
        sb.put_files(branch,{f'/opt/reference/{key}':path.read_bytes() for key,path in files.items()})
        sb.run(branch,'chmod 444 /opt/reference/*/*.py',30)
        task=etth1_task().with_independent_selection().with_io_helper()
        orch=Orchestrator.__new__(Orchestrator)
        orch.sandbox,orch._smoke_ok,orch.graph_repair=sb,set(),True
        sb.put_file(branch,'/work/train.py',ADAPTER.encode())
        error=orch._prepare_selection_contract(branch,task)
        if error is None:
            error=orch._smoke_check(branch,ADAPTER,task,seed=0)
        result={'status':'passed' if error is None else 'stopped','error':error,'seconds':round(time.monotonic()-started,2),
                'contract':task.selection_info(),'seed':0,'training':'one batch of eight; smoke only',
                'test_windows':8,'validation_windows':8,'paper_score':None,'network':'none',
                'unused_import_stubs':['pandas','tqdm'],
                'retrieval_sampling':'sample once and memoize per history+calendar, as in the original stored retrieval',
                'source_hashes':{key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in files.items()},
                'adapter_sha256':hashlib.sha256(ADAPTER.encode()).hexdigest(),
                'dataset_sha256':hashlib.sha256(args.csv.read_bytes()).hexdigest()}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        sb.close()


if __name__=='__main__':
    main()
