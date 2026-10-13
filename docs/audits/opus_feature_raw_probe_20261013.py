"""Does the raw CLI path still use a launch call whose effective features are S? No network."""
import sys, json
sys.path.insert(0, sys.argv[1] + '/src')
from repro_agent.spec import build_spec
from repro_agent.setup_details import Doc
from repro_agent.source_context import SourceContextBlocked
HEAD = ("import argparse\nparser = argparse.ArgumentParser()\n" + "".join(
    f"parser.add_argument('--{n}', type={t}, default={v!r})\n" for n, t, v in [
        ('model', 'str', 'X'), ('data', 'str', 'ETTh1'), ('features', 'str', 'M'), ('enc_in', 'int', 7),
        ('seq_len', 'int', 96), ('pred_len', 'int', 96), ('batch_size', 'int', 32), ('learning_rate', 'float', 0.001)]) + "args = parser.parse_args()\n")
MODEL = "class Model:\n    def __init__(self, configs):\n        self.seq_len = configs.seq_len\n        self.channels = configs.enc_in\n"
CALL = 'python run.py --model FITS --data ETTh1 --seq_len 336 --pred_len 96 --batch_size 64 --learning_rate 0.01\n'
out = {}
for name, entry, call in [('module_level_features_S', HEAD + "args.features = 'S'\n", CALL),
                          ('cli_feature_S', HEAD, CALL.replace('\n', ' --feature S\n')),
                          ('default_S_no_cli_flag', HEAD.replace("default='M'", "default='S'"), CALL),
                          ('module_level_enc_in_1', HEAD + "args.enc_in = 1\n", CALL),
                          ('default_enc_in_1_no_cli_flag', HEAD.replace("'--enc_in', type=int, default=7", "'--enc_in', type=int, default=1"), CALL),
                          ('known_M', HEAD, CALL)]:
    docs = [Doc('paper', 'p'), Doc('repo:models/FITS.py@x', MODEL), Doc('repo:run.py@x', entry), Doc('repo:scripts/FITS/etth1.sh@x', call)]
    try:
        spec = build_spec(lambda *_: '[]', [], docs, 'FITS erreicht 0.372 MSE auf ETTh1', 'ETTh1', numeric_only=True,
                          repo_listing=['models/FITS.py', 'run.py', 'scripts/FITS/etth1.sh'])
        out[name] = {i.key: (i.value, i.enforce, getattr(i, 'origin', '')) for i in spec.items}
    except SourceContextBlocked as x:
        out[name] = f'BLOCKED: {x}'
json.dump(out, open(sys.argv[2], 'w'), indent=1); print(json.dumps(out, indent=1))
