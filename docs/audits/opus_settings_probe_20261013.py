"""Edge probes for ceo experiment_settings/runtime_items. No author code is executed; no network."""
import sys, json
sys.path.insert(0, sys.argv[1] + '/src')
from repro_agent.experiment_settings import python_settings, shell_calls
from repro_agent.spec import runtime_items, resolve, call_matches
from repro_agent.setup_details import Doc
out = {}
P = "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--batch_size', type=int, default=32)\nparser.add_argument('--learning_rate', type=float, default=0.001)\nparser.add_argument('--seq_len', type=int, default=96)\n"
CALL = 'python run.py --batch_size 64 --seq_len 360'
cases = {
 'function_body_mutation': P + "def main():\n    args = parser.parse_args()\n    args.batch_size = args.batch_size * 2\nmain()\n",
 'set_defaults': P + "parser.set_defaults(learning_rate=0.0005)\nargs = parser.parse_args()\n",
 'setattr_from_config_loop': P + "args = parser.parse_args()\nimport yaml\ncfg = yaml.safe_load(open('c.yaml'))\nfor k, v in cfg.items():\n    setattr(args, k, v)\n",
 'setattr_module_level': P + "args = parser.parse_args()\nsetattr(args, 'batch_size', 128)\n",
 'dict_update': P + "args = parser.parse_args()\nargs.__dict__.update({'batch_size': 128})\n",
 'vars_update': P + "args = parser.parse_args()\nvars(args).update(batch_size=128)\n",
 'other_name_opt': P + "opt = parser.parse_args()\nopt.batch_size = opt.batch_size * 2\n",
 'main_guard_mutation': P + "if __name__ == '__main__':\n    args = parser.parse_args()\n    args.batch_size = args.batch_size * 2\n",
 'module_level_known_branch': P + "args = parser.parse_args()\nif args.seq_len == 360:\n    args.batch_size = args.batch_size * 2\n",
 'parse_known_args_config_file': P + "parser.add_argument('--config', default='cfg.yaml')\nargs, _ = parser.parse_known_args()\nargs = load_config(args.config, args)\n",
}
for name, text in cases.items():
    values, _ = python_settings(text, CALL)
    out[name] = {k: values.get(k) for k in ('batch_size', 'learning_rate', 'seq_len')}
# Shell: conditional single assignment, loop over values, horizon loop
sh = {
 'conditional_assignment': 'if [ "$1" = "fast" ]; then\n  lr=0.01\nfi\npython run.py --data ETTh1 --model FITS --pred_len 96 --learning_rate $lr\n',
 'loop_multi_seq': 'for seq_len in 90 180 360 720\ndo\npython run.py --data ETTh1 --model FITS --pred_len 96 --seq_len $seq_len\ndone\n',
 'loop_horizons_fixed_seq': 'for pred_len in 96 192 336 720\ndo\npython run.py --data ETTh1 --model FITS --pred_len $pred_len --seq_len 336\ndone\n',
}
for name, text in sh.items():
    calls = shell_calls(text)
    out['sh_' + name] = [{'expanded': e, 'matches_ETTh1_96_FITS': call_matches(e, 'ETTh1', 96, 'FITS')} for _, e, _ in calls]
# Two matching launch scripts with different explicit values
docs = [Doc('repo:models/FITS.py@x', 'class Model:\n    def __init__(self, configs):\n        self.seq_len = configs.seq_len\n'),
        Doc('repo:run.py@x', P + "args = parser.parse_args()\n"),
        Doc('repo:scripts/FITS/etth1_a.sh@x', 'python run.py --data ETTh1 --model FITS --pred_len 96 --seq_len 360 --batch_size 64\n'),
        Doc('repo:scripts/FITS/etth1_b.sh@x', 'python run.py --data ETTh1 --model FITS --pred_len 96 --seq_len 720 --batch_size 64\n')]
items = runtime_items(docs, [d.name for d in docs], 'FITS', 'forecast', 'ETTh1')
out['two_scripts_runtime_items'] = [(i.key, i.value, i.source) for i in items]
resolved, missing = resolve(items, 'forecast', True)
out['two_scripts_resolved'] = [(i.key, i.value, i.enforce) for i in resolved]
# Graph family must not use runtime launch settings
out['graph_runtime_items'] = len(runtime_items(docs, [d.name for d in docs], 'GCN', 'graph', 'Cora'))
json.dump(out, open(sys.argv[2], 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
