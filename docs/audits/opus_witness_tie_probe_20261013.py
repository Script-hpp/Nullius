"""Runtime witnesses must be real fragments of the named source and carry the enforced value; ties stop. No network."""
import sys, json, re
code = sys.argv[1]
sys.path.insert(0, code + '/src')
from pathlib import Path
from repro_agent.spec import build_spec, runtime_items, _norm, quote_in_text
from repro_agent.setup_details import Doc
from repro_agent.source_context import SourceContextBlocked
out = {}
def check(name, path, claim):
    d = json.loads((Path(code) / path).read_text()); commit = (d.get('commit') or 'x')[:7]
    docs = [Doc('paper', 'paper text')] + [Doc(f'repo:{p}@{commit}', t) for p, t in d['files'].items()]
    by = {x.name: x.text for x in docs}
    spec = build_spec(lambda *_: '[]', [], docs, claim, 'ETTh1', numeric_only=True, repo_listing=list(d['files']))
    rows = []
    for i in spec.items:
        if getattr(i, 'origin', '') != 'runtime': continue
        src = by.get(i.source, '')
        rows.append({'key': i.key, 'value': i.value, 'source': i.source, 'quote': i.quote[:90],
                     'quote_in_source': quote_in_text(i.quote, _norm(src)),
                     'value_in_quote': bool(re.search(rf'(?<![\d.]){re.escape(i.value)}(?![\d])', i.quote)) or '*' in i.quote or '//' in i.quote})
    out[name] = rows
check('FITS', 'tests/fixtures/fits_runtime_settings.json', 'FITS erreicht 0.372 MSE auf ETTh1 (Horizont 96)')
check('DLinear', 'benchmark/fixtures/repos/cure-lab_LTSF-Linear.json', 'DLinear erreicht MSE 0.386 auf ETTh1')
# Literal value only in a later call of the same script; first call is another horizon with another value.
ENTRY = ("import argparse\nparser = argparse.ArgumentParser()\n" + "".join(
    f"parser.add_argument('--{n}', type={t}, default={v!r})\n" for n, t, v in [
        ('model', 'str', 'X'), ('data', 'str', 'ETTh1'), ('seq_len', 'int', 96), ('pred_len', 'int', 96), ('batch_size', 'int', 32)])
    + "args = parser.parse_args()\n")
MODEL = "class Model:\n    def __init__(self, configs):\n        self.seq_len = configs.seq_len\n"
sh = ('bs=16\npython run.py --model FITS --data ETTh1 --seq_len 336 --pred_len 192 --batch_size $bs\n'
      'python run.py --model FITS --data ETTh1 --seq_len 336 --pred_len 96 --batch_size 64\n')
docs = [Doc('repo:models/FITS.py@x', MODEL), Doc('repo:run.py@x', ENTRY), Doc('repo:scripts/FITS/etth1.sh@x', sh)]
out['later_literal_witness'] = [(i.key, i.value, i.source, i.quote) for i in runtime_items(docs, [d.name for d in docs], 'FITS', 'forecast', 'ETTh1')]
# Two equally ranked launch scripts disagree on batch size -> visible stop
tie = [Doc('repo:models/FITS.py@x', MODEL), Doc('repo:run.py@x', ENTRY),
       Doc('repo:scripts/FITS/etth1_a.sh@x', 'python run.py --model FITS --data ETTh1 --seq_len 336 --pred_len 96 --batch_size 64\n'),
       Doc('repo:scripts/FITS/etth1_b.sh@x', 'python run.py --model FITS --data ETTh1 --seq_len 336 --pred_len 96 --batch_size 128\n')]
try:
    spec = build_spec(lambda *_: '[]', [], [Doc('paper', 'p')] + tie, 'FITS erreicht 0.372 MSE auf ETTh1', 'ETTh1', numeric_only=True,
                      repo_listing=['models/FITS.py', 'run.py', 'scripts/FITS/etth1_a.sh', 'scripts/FITS/etth1_b.sh'])
    out['tie'] = {i.key: (i.value, i.enforce) for i in spec.items}
except SourceContextBlocked as x:
    out['tie'] = f'BLOCKED: {x}'
json.dump(out, open(sys.argv[2], 'w'), indent=1); print(json.dumps(out, indent=1))
