"""Offline SARAF calendar-input diagnosis; no training or paper score.

Author code executes only in a disposable, network-free Docker container.
Usage: uv run python docs/audits/saraf_calendar_contract_probe_20261013.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[2]

PROBE = r'''
import csv, datetime, json, sys, types
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
# These optional imports are unused by the exercised numerical functions.
sys.modules['pandas'] = types.ModuleType('pandas')
progress = types.ModuleType('tqdm')
progress.tqdm = lambda items, **kwargs: items
sys.modules['tqdm'] = progress
sys.path.insert(0, '/opt/probe')
from Retrieval import RetrievalTool
from selection_contract import forecast_public
with np.load('/data/etth1_public.npz', allow_pickle=False) as source:
    original = {key: source[key] for key in source.files}
public = forecast_public(original)
with open('/opt/probe/ETTh1.csv', newline='') as stream:
    dates = [datetime.datetime.fromisoformat(row['date']) for row in csv.DictReader(stream)][:8640]
marks = np.asarray([[d.year,d.month,d.day,d.weekday(),d.hour] for d in dates], dtype=np.float32)
starts = [0,24,1800]
train = torch.tensor(np.stack([public['train'][s:s+720] for s in starts]))
train_marks = torch.tensor(np.stack([marks[s:s+720] for s in starts]))
query = torch.tensor(public['train'][3500:4220][None])
query_marks = torch.tensor(marks[3500:4220][None])
rt = RetrievalTool(720,96,7,n_period=1,topm=20,freq='h',time_aware_weight=.5)
rt.train_data_all_mg, _ = rt.decompose_mg(train)
rt.train_data_all_mark = train_marks
captured = []
def capture(sim, batch_size, index):
    captured.append(sim.detach().clone())
    return torch.zeros((1,batch_size,96,7))
# Capture the intermediate similarity, without executing MMR or training.
rt.generate_retrieval_predictions_2 = capture
rt.retrieve(query,torch.tensor([0]),train=False,x_mark=query_marks)
rt.retrieve(query,torch.tensor([0]),train=False,x_mark=None)
delta = float((captured[0]-captured[1]).abs().max())
assert delta > 0.01, delta
assert 'train_dates' not in public and 'val_dates' not in public
worker = Path('/opt/probe/forecast_worker.py').read_text()
assert 'history = np.asarray(json.loads(line)' in worker and 'pred = predict(history)' in worker
result = {'status':'confirmed_contract_gap','dataset':'ETTh1','probe':'author intermediate retrieval similarity',
          'public_fields':sorted(public),'author_calendar_features':['year','month','day','weekday','hour'],
          'author_time_aware_weight':.5,'max_similarity_difference_without_calendar':delta,
          'trained':False,'paper_score':None,'network':'none',
          'unused_import_stubs':['pandas','tqdm'],
          'scope':'Original correlation/time-bonus functions; final retrieval and training not executed'}
Path('/opt/probe/result.json').write_text(json.dumps(result,indent=2))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--author-root', type=Path, default=Path('/tmp/nullius-saraf-local-audit/author_repo'))
    parser.add_argument('--csv', type=Path, default=Path('/tmp/nullius-saraf-local-audit/ETTh1.csv'))
    parser.add_argument('--out', type=Path, default=Path(__file__).with_suffix('.json'))
    parser.add_argument('--base-commit', default='07f53b614084bd6a760545756aa1f3e4d11e952b')
    args = parser.parse_args()
    from repro_agent.sandbox.local import LocalSandbox
    name = 'nullius-saraf-calendar-' + uuid.uuid4().hex[:8]
    image = 'repro-snapshot:gate-etth1'
    subprocess.run(['docker','run','-d','--network','none','--name',name,image,'sleep','infinity'], check=True,capture_output=True)
    files = {'Retrieval.py':args.author_root/'layers/Retrieval.py',
             'selection_contract.py':ROOT/'src/repro_agent/harness/selection_contract.py',
             'forecast_worker.py':ROOT/'src/repro_agent/harness/forecast_worker.py',
             'ETTh1.csv':args.csv}
    contents = {key:path.read_bytes() for key,path in files.items()}
    for key in ('selection_contract.py','forecast_worker.py'):
        contents[key] = subprocess.check_output(['git','show',f'{args.base_commit}:src/repro_agent/harness/{key}'],cwd=ROOT)
    try:
        sb = LocalSandbox()
        sb.put_files(name,{f'/opt/probe/{key}':data for key,data in contents.items()})
        sb.put_file(name,'/opt/probe/probe.py',PROBE.encode())
        execution = sb.run(name,'python /opt/probe/probe.py',60)
        if execution.exit_code != 0 or execution.timed_out:
            raise RuntimeError(execution.stderr[-2000:] or execution.stdout[-2000:])
        result = json.loads(sb.get_file(name,'/opt/probe/result.json'))
        result['source_hashes'] = {key:hashlib.sha256(data).hexdigest() for key,data in contents.items()}
        result['product_commit'] = args.base_commit
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        subprocess.run(['docker','rm','-f',name], check=True,capture_output=True)


if __name__ == '__main__':
    main()
