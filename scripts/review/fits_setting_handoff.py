"""Complete FITS source -> settings -> local reference, no model/provider calls.

The training adapter is human-written, as in fits_independent_reference. Only
its numeric settings are supplied by the production source/spec compiler.
This proves the handoff, not autonomous code generation or method fidelity.
"""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.review import fits_independent_reference as reference
from repro_agent import catalog, repo_source, spec_check
from repro_agent.setup_details import Doc
from repro_agent.spec import build_spec


def source_settings(repository):
    listing = subprocess.check_output(['git', 'ls-files'], cwd=repository, text=True).splitlines()

    class CachedRepository:
        # Read the pinned local checkout; never clone or execute author scripts.
        def base_snapshot(self, *args, **kwargs): return 'cached-repository'
        def branch(self, *args): return 'cached-branch'
        def run(self, *args):
            data = ('\0'.join(listing) + '\0').encode()
            return SimpleNamespace(stdout=f'{len(data)}\n{data.count(0)}\n{hashlib.sha256(data).hexdigest()}\n', exit_code=0, timed_out=False)
        def get_file(self, branch, path):
            if path == '/tmp/.nullius_repo_listing':
                return ('\0'.join(listing) + '\0').encode()
            if path.endswith('.head_commit'):
                return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository)
            return (repository / path.removeprefix('/repo/')).read_bytes()

    # Reproduce the actual wrong model file selection from the live incident.
    with patch.object(repo_source, 'select_files', return_value=['README.md', 'AD/FITS.py', 'AD/main.py', 'AD/solver.py']):
        dump = repo_source.read_repo(CachedRepository(), None, 'https://github.com/VEWOXIC/FITS',
            'FITS erreicht MSE 0.372', 'ETTh1', fixture_dir=repository / '.unused-fixtures',
            unit_key='FITS', model='FITS', mode='live')
    primary = json.loads(gzip.decompress((ROOT / 'benchmark/source_reserves/2307.03756v3.json.gz').read_bytes()))['text']
    config, _ = catalog.build_paper(catalog.find_entry(catalog.load_catalog(), 'disc_2307_03756'))
    docs = [Doc('paper', primary)] + [Doc('repo:' + name + '@' + dump.commit[:7], text) for name, text in dump.files.items()]
    spec = build_spec(lambda *_: '[]', catalog.discovery_details(config, primary), docs,
        'FITS erreicht MSE 0.372', 'ETTh1', numeric_only=True, repo_listing=dump.listing)
    return spec, dump


def configured_adapter(spec):
    values = {item.key: item.number for item in spec.enforced()}
    replacements = {
        'length,horizon,batch=360,96,128': "length,horizon,batch=CONFIG['seq_len'],CONFIG['pred_len'],CONFIG['batch_size']\n    assert CONFIG['train_mode']==1",
        'enc_in=7,cut_freq=106': "enc_in=CONFIG['enc_in'],cut_freq=CONFIG['cut_freq']",
        'lr=.0005': "lr=CONFIG['lr']",
        'range(1 if smoke else 50)': "range(1 if smoke else CONFIG['epochs'])",
        'if bad>=20:': "if bad>=CONFIG['patience']:",
        "group['lr']=.0005*(.95**epoch)": "group['lr']=CONFIG['lr']*(.95**epoch)",
    }
    code = reference.ADAPTER
    for old, new in replacements.items():
        assert code.count(old) == 1, old
        code = code.replace(old, new, 1)
    code = 'CONFIG=' + repr(values) + '\n' + code
    code, report = spec_check.config_check(code, spec)
    assert not report['unused'], report
    # This diagnostic uses the frozen reference implementation, not a parameter
    # search. Refuse changed settings rather than relabeling its old metadata.
    assert values == dict(lr=.0005, epochs=50, batch_size=128, patience=20, seq_len=360,
                          cut_freq=106, enc_in=7, pred_len=96, train_mode=1), values
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('/tmp/nullius-fits-independent-20261013'))
    parser.add_argument('--csv', type=Path, default=Path('/tmp/nullius-saraf-local-audit/ETTh1.csv'))
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/audits/fits_setting_handoff_20261013.json')
    parser.add_argument('--settings-only', action='store_true')
    args = parser.parse_args()
    spec, dump = source_settings(args.repo)
    adapter = configured_adapter(spec)
    proof = {'author_commit': dump.commit, 'files': list(dump.files), 'spec': asdict(spec),
             'settings': {item.key: item.number for item in spec.enforced()},
             'provider_calls': 0, 'scope': __doc__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.settings.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof['settings']), flush=True)
    if args.settings_only: return
    reference.ADAPTER = adapter
    sys.argv = [sys.argv[0], '--repo', str(args.repo), '--csv', str(args.csv), '--out', str(args.out)]
    reference.main()
    report = json.loads(args.out.read_text())
    report['settings_handoff'] = proof
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
