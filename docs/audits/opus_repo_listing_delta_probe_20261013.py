"""Netzlose Delta-Probe verified_listing: echter FITS-Klon d040bb0, Nebius-Kopfkappung 65535 auf JEDEM run-stdout."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
CAP = 65535
SCRATCH = Path(sys.argv[1])

class Sb:
    """Runs the real shell command against a local clone; stdout capped like the SDK default."""
    def __init__(self, repo): self.repo = repo; self.gets = []; self.listing_file = SCRATCH / 'listing.bin'
    def base_snapshot(self, image, cmds, label=None): return 'snap'
    def branch(self, snap): return 'b'
    def _map(self, cmd): return cmd.replace('/tmp/.nullius_repo_listing', str(self.listing_file)).replace('/repo', str(self.repo))
    def run(self, branch, cmd, timeout_s, **kw):
        p = subprocess.run(self._map(cmd), shell=True, capture_output=True)
        return SimpleNamespace(stdout=p.stdout[:CAP].decode(errors='replace'), stderr=p.stderr.decode(errors='replace'),
                               exit_code=p.returncode, timed_out=False)
    def get_file(self, branch, path):
        self.gets.append(path)
        if path == '/repo/.head_commit':
            return subprocess.run(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], capture_output=True).stdout
        return Path(self._map(path)).read_bytes()
    def get_file_limited(self, branch, path, max_bytes):
        data = self.get_file(branch, path)
        if len(data) > max_bytes: raise ValueError('limit')
        return data

class Llm:
    def complete(self, role, messages, **kw):
        return SimpleNamespace(content=json.dumps(["AD/FITS.py", "AD/main.py", "README.md", "exp/exp_main_F.py"]))

from repro_agent.repo_source import read_repo
from repro_agent.source_context import SourceContextBlocked
out = {}
FITS = Path('/tmp/nullius-fits-independent-20261013')
sb = Sb(FITS)
d = read_repo(sb, Llm(), 'https://github.com/VEWOXIC/FITS', 'FITS ETTh1 MSE 0.372', 'ETTh1', focus_model=True,
              unit_key='fits_etth1', model='FITS', mode='live')
out['fits_root'] = {'n_listing': len(d.listing), 'partial_lo_entry': 'lo' in d.listing, 'files': list(d.files)[:4],
                    'AD_read': any(p.startswith('AD/') for p in d.files), 'listing_bytes': sb.listing_file.stat().st_size}
sb = Sb(FITS)
d = read_repo(sb, Llm(), 'https://github.com/VEWOXIC/FITS/tree/main/logs', 'FITS', 'ETTh1', mode='live')
out['fits_subdir_logs'] = {'n_listing': len(d.listing), 'expected': 1340, 'subdir': d.subdir, 'listing_bytes': sb.listing_file.stat().st_size}

def toy(names, size_each=1):
    repo = Path(tempfile.mkdtemp(dir=SCRATCH))
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    for n in names:
        (repo / n).parent.mkdir(parents=True, exist_ok=True); (repo / n).write_bytes(b'x' * size_each)
    subprocess.run(['git', '-C', str(repo), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'i'], check=True)
    return repo
odd = ['a b.py', 'models/Ünïcode.py', 'quote"d.py', 'new\nline.py', 'tab\tname.py']
sb = Sb(toy(odd))
d = read_repo(sb, Llm(), 'https://github.com/x/odd', 'X', 'ETTh1', mode='live')
out['odd_names'] = {'listing': d.listing, 'all_exact': sorted(d.listing) == sorted(odd)}
old = subprocess.run('git ls-files', shell=True, cwd=sb.repo, capture_output=True, text=True).stdout.splitlines()
out['odd_names_old_ls_files'] = old
big = [f'deep/{"d" * 180}/f{i:05d}.txt' for i in range(5200)]
sb = Sb(toy(big, 0))
try:
    read_repo(sb, Llm(), 'https://github.com/x/big', 'X', 'ETTh1', mode='live'); out['big_1MB'] = 'NOT BLOCKED'
except SourceContextBlocked as e:
    out['big_1MB'] = f'BLOCKED: {e}; transferred={"/tmp/.nullius_repo_listing" in sb.gets}'
print(json.dumps(out, indent=1, ensure_ascii=False))
