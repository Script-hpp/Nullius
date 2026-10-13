"""Cross-paper gate: real cached author repos through ceo build_spec (deterministic part only). No network/model."""
import sys, json, gzip
code = sys.argv[1]
sys.path.insert(0, code + '/src')
from pathlib import Path
from repro_agent.spec import build_spec
from repro_agent.setup_details import Doc
from repro_agent.source_context import SourceContextBlocked
R = Path(code)
def repo(path):
    d = json.loads((R / path).read_text())
    commit = (d.get('commit') or 'x')[:7]
    return [Doc(f'repo:{p}@{commit}', t) for p, t in d['files'].items()], list(d['files'])
cases = [
 ('FITS ETTh1 (tests fixture)', 'tests/fixtures/fits_runtime_settings.json', 'FITS erreicht 0.372 MSE auf ETTh1 (Horizont 96)', 'ETTh1'),
 ('DLinear ETTh1 (LTSF-Linear fixture)', 'benchmark/fixtures/repos/cure-lab_LTSF-Linear.json', 'DLinear erreicht MSE 0.386 auf ETTh1', 'ETTh1'),
 ('NLinear ETTh1 (same repo, script launches DLinear)', 'benchmark/fixtures/repos/cure-lab_LTSF-Linear.json', 'NLinear erreicht MSE 0.374 auf ETTh1', 'ETTh1'),
 ('MoHETS ETTh1 (weather scripts only)', 'benchmark/fixtures/repos/evortigosa_mohe_forecast.json', 'MoHETS erreicht MSE 0.350 auf ETTh1', 'ETTh1'),
 ('GCN Cora (graph family)', 'benchmark/fixtures/repos/tkipf_gcn.json', 'GCN erreicht 81.5 % Accuracy auf Cora', 'Cora'),
]
out = {}
for name, path, claim, dataset in cases:
    docs, listing = repo(path)
    try:
        spec = build_spec(lambda *_: '[]', [], [Doc('paper', 'paper text')] + docs, claim, dataset, numeric_only=True, repo_listing=listing)
        out[name] = {'enforced': {i.key: i.value for i in spec.enforced()},
                     'runtime': sorted({i.key for i in spec.items if getattr(i, 'origin', '') == 'runtime'}),
                     'not_enforced': {i.key: i.value for i in spec.items if not i.enforce}}
    except SourceContextBlocked as x:
        out[name] = f'BLOCKED: {x}'
    except Exception as x:
        out[name] = f'ERROR {type(x).__name__}: {x}'
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps(out, indent=1))
