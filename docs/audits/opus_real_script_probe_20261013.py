"""Real SparseTSF/SegRNN launch scripts (from earlier local dossiers) through ceo runtime_items. No network."""
import sys, json
code, research = sys.argv[1], sys.argv[2]
sys.path.insert(0, code + '/src')
from repro_agent.spec import runtime_items
from repro_agent.setup_details import Doc
from repro_agent.source_context import SourceContextBlocked
def scripts(path, marker):
    d = json.load(open(path)); found = {}
    def walk(x):
        if isinstance(x, dict):
            q = (x.get('applies_to') or {}).get('quote', '')
            if marker in q and 'for pred_len in' in q:
                found[x.get('source')] = q
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(d); return found
ENTRY = ("import argparse\nparser = argparse.ArgumentParser()\n" + "".join(
    f"parser.add_argument('--{n}', type={t}, default={d!r})\n" for n, t, d in [
        ('model', 'str', 'X'), ('data', 'str', 'ETTh1'), ('seq_len', 'int', 96), ('pred_len', 'int', 96),
        ('batch_size', 'int', 32), ('learning_rate', 'float', 0.0001), ('train_epochs', 'int', 10), ('patience', 'int', 3),
        ('enc_in', 'int', 7), ('period_len', 'int', 24), ('seg_len', 'int', 48), ('d_model', 'int', 512), ('dropout', 'float', 0.05)])
    + "args = parser.parse_args()\n")
MODELS = {'SparseTSF': "class Model:\n    def __init__(self, configs):\n        self.seq_len = configs.seq_len\n        self.pred_len = configs.pred_len\n        self.period_len = configs.period_len\n",
          'SegRNN': "class Model:\n    def __init__(self, configs):\n        self.seq_len = configs.seq_len\n        self.pred_len = configs.pred_len\n        self.seg_len = configs.seg_len\n        self.d_model = configs.d_model\n"}
out = {}
for model, path in [('SparseTSF', research + '/assistant/runs/disc_2405_00946/20261009-220822/dossier.json'),
                    ('SegRNN', research + '/assistant/runs/disc_2308_11200/20261009-224634/dossier.json')]:
    for src, text in scripts(path, 'model_name=' + model).items():
        rel = src.split(':', 1)[1].split('@')[0]
        docs = [Doc(f'repo:models/{model}.py@x', MODELS[model]), Doc('repo:run_longExp.py@x', ENTRY), Doc(src, text)]
        try:
            items = runtime_items(docs, [d.name for d in docs], model, 'forecast', 'ETTh1')
            out[f'{model} {rel}'] = sorted((i.key, i.value) for i in items)
        except SourceContextBlocked as x:
            out[f'{model} {rel}'] = f'BLOCKED: {x}'
json.dump(out, open(sys.argv[3], 'w'), indent=1)
print(json.dumps(out, indent=1))
