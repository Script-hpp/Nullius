"""A loop over other horizons in a second ETTh1 script must not stop a run whose 96 call is concrete. No network."""
import sys, json
sys.path.insert(0, sys.argv[1] + '/src')
from repro_agent.spec import runtime_items
from repro_agent.setup_details import Doc
from repro_agent.source_context import SourceContextBlocked
ENTRY = ("import argparse\nparser = argparse.ArgumentParser()\n" + "".join(
    f"parser.add_argument('--{n}', type={t}, default={d!r})\n" for n, t, d in [
        ('model', 'str', 'X'), ('data', 'str', 'ETTh1'), ('seq_len', 'int', 96), ('pred_len', 'int', 96),
        ('batch_size', 'int', 32), ('learning_rate', 'float', 0.0001)]) + "args = parser.parse_args()\n")
MODEL = "class Model:\n    def __init__(self, configs):\n        self.seq_len = configs.seq_len\n        self.pred_len = configs.pred_len\n"
base = [Doc('repo:models/FITS.py@x', MODEL), Doc('repo:run.py@x', ENTRY),
        Doc('repo:scripts/FITS/etth1.sh@x', 'python run.py --model FITS --data ETTh1 --seq_len 336 --pred_len 96 --batch_size 64\n')]
out = {}
for name, extra in [('only_concrete_96', []),
                    ('plus_long_horizon_loop', [Doc('repo:scripts/FITS/etth1_long.sh@x',
                        'for pred_len in 336 720\ndo\npython run.py --model FITS --data ETTh1 --seq_len 336 --pred_len $pred_len --batch_size 64\ndone\n')])]:
    docs = base + extra
    try:
        out[name] = sorted((i.key, i.value) for i in runtime_items(docs, [d.name for d in docs], 'FITS', 'forecast', 'ETTh1'))
    except SourceContextBlocked as x:
        out[name] = f'BLOCKED: {x}'
json.dump(out, open(sys.argv[2], 'w'), indent=1); print(json.dumps(out, indent=1))
