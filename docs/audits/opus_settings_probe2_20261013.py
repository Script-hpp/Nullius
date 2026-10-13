"""Second-round probes for ceo experiment settings. No author code is executed; no network."""
import sys, json
sys.path.insert(0, sys.argv[1] + '/src')
from repro_agent.experiment_settings import python_settings
from repro_agent.spec import runtime_items
from repro_agent.setup_details import Doc
from repro_agent.source_context import SourceContextBlocked
out = {}
P = ("import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--model', type=str, default='X')\n"
     "parser.add_argument('--data', type=str, default='ETTh1')\nparser.add_argument('--seq_len', type=int, default=96)\n"
     "parser.add_argument('--pred_len', type=int, default=96)\nparser.add_argument('--batch_size', type=int, default=32)\n"
     "parser.add_argument('--learning_rate', type=float, default=0.001)\nparser.add_argument('--label_len', type=int, default=48)\n"
     "parser.add_argument('--e', type=int, default=0)\n")
CALL = 'python run.py --batch_size 64 --seq_len 360'
muts = {
 'function_body_mutation': P + "def main():\n    args = parser.parse_args()\n    args.batch_size = args.batch_size * 2\nmain()\n",
 'set_defaults': P + "parser.set_defaults(learning_rate=0.0005)\nargs = parser.parse_args()\n",
 'setattr_from_config_loop': P + "args = parser.parse_args()\nfor k, v in cfg.items():\n    setattr(args, k, v)\n",
 'dict_update': P + "args = parser.parse_args()\nargs.__dict__.update({'batch_size': 128})\n",
 'vars_update': P + "args = parser.parse_args()\nvars(args).update(batch_size=128)\n",
 'other_name_opt': P + "opt = parser.parse_args()\nopt.batch_size = opt.batch_size * 2\n",
 'known_args_load_config': P + "args, _ = parser.parse_known_args()\nargs = load_config(args)\n",
 'class_self_args_store': P + "args = parser.parse_args()\nclass Exp:\n    def __init__(self, args):\n        self.args = args\n        self.args.batch_size = 256\nExp(args)\n",
 'namespace_copy_alias': P + "args = parser.parse_args()\nhp = args\nhp.batch_size = 256\n",
}
for name, text in muts.items():
    values, _ = python_settings(text, CALL)
    out[name] = {k: values.get(k) for k in ('batch_size', 'learning_rate', 'seq_len')}
model = 'class Model:\n    def __init__(self, configs):\n        self.seq_len = configs.seq_len\n        self.pred_len = configs.pred_len\n'
def run(name, script, entry=P + "args = parser.parse_args()\n", extra=()):
    docs = [Doc('repo:models/FITS.py@x', model), Doc('repo:run.py@x', entry), Doc('repo:scripts/FITS/etth1.sh@x', script)] + list(extra)
    try:
        items = runtime_items(docs, [d.name for d in docs], 'FITS', 'forecast', 'ETTh1')
        out[name] = sorted((i.key, i.value) for i in items)
    except SourceContextBlocked as x:
        out[name] = f'BLOCKED: {x}'
# Common author layout: fixed seq_len, loop over all horizons (e.g. SparseTSF/SegRNN style)
run('horizon_loop_fixed_seq', 'model_name=FITS\nseq_len=720\nfor pred_len in 96 192 336 720\ndo\npython -u run.py --model $model_name --data ETTh1 --seq_len $seq_len --pred_len $pred_len --batch_size 256 --learning_rate 0.02\ndone\n')
run('dataset_loop', 'for data in ETTh1 ETTh2\ndo\npython -u run.py --model FITS --data $data --seq_len 336 --pred_len 96\ndone\n')
run('explicit_unread_flags', 'python -u run.py --model FITS --data ETTh1 --seq_len 336 --pred_len 96 --e 1 --label_len 48\n')
run('mutation_in_imported_module', 'python -u run.py --model FITS --data ETTh1 --seq_len 336 --pred_len 96 --batch_size 64\n',
    entry=P + "args = parser.parse_args()\nfrom exp.exp_main import Exp_Main\nExp_Main(args).train()\n",
    extra=[Doc('repo:exp/exp_main.py@x', "class Exp_Main:\n    def __init__(self, args):\n        self.args = args\n        self.args.batch_size = self.args.batch_size * 4\n")])
json.dump(out, open(sys.argv[2], 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
