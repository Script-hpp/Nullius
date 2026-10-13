"""Netzlose Emulation: read_repo gegen lokalen FITS-Klon d040bb0, Nebius-Kopfkappung 65535 vs. voll."""
import json, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
CLONE = Path('/tmp/nullius-fits-independent-20261013')
CAP = 65535

class Sb:
    def __init__(self, cap): self.cap = cap; self.cmds = []
    def base_snapshot(self, image, cmds, label=None): return 'snap'
    def branch(self, snap): return 'b'
    def run(self, branch, cmd, timeout_s, **kw):
        self.cmds.append(cmd)
        out = subprocess.run(cmd.replace('/repo', str(CLONE)), shell=True, capture_output=True).stdout
        trunc = self.cap is not None and len(out) > self.cap
        if trunc: out = out[:self.cap]
        return SimpleNamespace(stdout=out.decode(errors='replace'), stderr='', exit_code=0, timed_out=False, truncated=trunc)
    def get_file(self, branch, path):
        if path == '/repo/.head_commit':
            return subprocess.run(['git', '-C', str(CLONE), 'rev-parse', 'HEAD'], capture_output=True).stdout
        return (CLONE / path[len('/repo/'):]).read_bytes()

LIVE_CHOICE = ["AD/FITS.py", "AD/main.py", "README.md", "data_provider/data_factory.py", "data_provider/data_loader.py",
               "exp/exp_basic.py", "exp/exp_main.py", "exp/exp_main_F.py", "layers/utils.py"]
class Llm:
    def __init__(self): self.prompts = []
    def complete(self, role, messages, **kw):
        self.prompts.append(messages[-1]['content'])
        return SimpleNamespace(content=json.dumps(LIVE_CHOICE))

from repro_agent.repo_source import read_repo
res = {}
for name, cap in (('nebius_cap', CAP), ('full', None)):
    sb, llm = Sb(cap), Llm()
    dump = read_repo(sb, llm, 'https://github.com/VEWOXIC/FITS', 'FITS ETTh1 MSE 0.372 horizon 96', 'ETTh1',
                     focus_model=True, unit_key='fits_etth1', model='FITS', mode='live')
    prompt = llm.prompts[0]
    res[name] = {'n_listing': len(dump.listing), 'last_listing_entry': dump.listing[-1],
                 'files': list(dump.files), 'chars': sum(len(t) for t in dump.files.values()),
                 'truncation': dump.truncation,
                 'llm_prompt_has_models_FITS': 'models/FITS.py' in prompt,
                 'llm_prompt_has_etth1_best': 'scripts/FITS/etth1_best.sh' in prompt}
print(json.dumps(res, indent=1))

from repro_agent.setup_details import Doc, target_docs, _chunks
for name in ('nebius_cap', 'full'):
    docs = [Doc('paper', 'x')] + [Doc(f'repo:{p}@d040bb0', (CLONE / p).read_text(errors='replace')) for p in res[name]['files']] + [Doc('repo-readme', 'r')]
    picked = target_docs(docs)
    print(name, 'target_docs:', [d.name for d in picked],
          'repo chunks:', sum(len(_chunks(d.text)) for d in picked if d.name.startswith('repo:')))
