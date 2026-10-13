"""Feature/channel identity guard: unresolved -> stop, known M/7 -> unchanged, S/1 -> no match. No network."""
import sys, json
sys.path.insert(0, sys.argv[1] + '/src')
from repro_agent.spec import runtime_items
from repro_agent.setup_details import Doc
from repro_agent.source_context import SourceContextBlocked
HEAD = ("import argparse\nparser = argparse.ArgumentParser()\n" + "".join(
    f"parser.add_argument('--{n}', type={t}, default={v!r})\n" for n, t, v in [
        ('model', 'str', 'X'), ('data', 'str', 'ETTh1'), ('features', 'str', 'M'), ('enc_in', 'int', 7),
        ('seq_len', 'int', 96), ('pred_len', 'int', 96), ('batch_size', 'int', 32)]) + "args = parser.parse_args()\n")
MODEL = "class Model:\n    def __init__(self, configs):\n        self.seq_len = configs.seq_len\n        self.channels = configs.enc_in\n"
CALL = 'python run.py --model FITS --data ETTh1 --seq_len 336 --pred_len 96 --batch_size 64'
cases = {
 'known_M_7': (HEAD, CALL),
 'main_sets_features_S': (HEAD + "def main():\n    args.features = 'S'\n", CALL),
 'main_sets_enc_in_1': (HEAD + "def main():\n    args.enc_in = 1\n", CALL),
 'module_level_features_S': (HEAD + "args.features = 'S'\n", CALL),
 'module_level_enc_in_from_data': (HEAD + "args.enc_in = load_data().shape[1]\n", CALL),
 'cli_feature_abbrev_S': (HEAD, CALL + ' --feature S --enc_in 1'),
 'cli_features_MS': (HEAD, CALL + ' --features MS'),
 'no_features_flag_at_all': (HEAD.replace("parser.add_argument('--features', type=str, default='M')\n", ''), CALL),
}
out = {}
for name, (entry, call) in cases.items():
    docs = [Doc('repo:models/FITS.py@x', MODEL), Doc('repo:run.py@x', entry), Doc('repo:scripts/FITS/etth1.sh@x', call + '\n')]
    try:
        out[name] = sorted((i.key, i.value) for i in runtime_items(docs, [d.name for d in docs], 'FITS', 'forecast', 'ETTh1'))
    except SourceContextBlocked as x:
        out[name] = f'BLOCKED: {x}'
json.dump(out, open(sys.argv[2], 'w'), indent=1); print(json.dumps(out, indent=1))
