"""Nebius defaults to 64 KiB stdout; canonical models can follow large data trees."""
import hashlib
from types import SimpleNamespace

import pytest

from repro_agent import repo_source


class TruncatedSandbox:
    def __init__(self, files, listing):
        self.files, self.listing = files, listing.replace('\n','\0').encode()
        self.metadata_size = len(self.listing)
        self.metadata_count = self.listing.count(0)
        self.metadata_hash = hashlib.sha256(self.listing).hexdigest()
        self.transferred = self.listing
        self.gets = []
    def base_snapshot(self, *args, **kwargs): return 'snapshot'
    def branch(self, *args): return 'branch'
    def run(self, branch, cmd, timeout_s):
        if '.nullius_repo_listing' in cmd:
            return SimpleNamespace(stdout=f'{self.metadata_size}\n{self.metadata_count}\n{self.metadata_hash}\n', exit_code=0, timed_out=False, stderr='')
        return SimpleNamespace(stdout=self.listing[:65536].decode(), exit_code=0, timed_out=False, stderr='')
    def get_file(self, branch, path):
        self.gets.append(path)
        if path == '/tmp/.nullius_repo_listing': return self.transferred
        if path.endswith('.head_commit'): return b'abc123'
        return self.files[path.removeprefix('/repo/')].encode()


def test_canonical_forecast_files_after_stdout_limit_survive_transport(tmp_path, monkeypatch):
    files={'AD/Toy.py':'class Model: pass\n', 'README.md':'Use `_best` for final reported results.',
           'models/Toy.py':'class Model: pass\n',
           'scripts/Toy/etth1_best.sh':'python run.py --model Toy --data ETTh1 --seq_len 360 --pred_len 96\n',
           'run.py':'p.add_argument("--seq_len",default=360)\nargs=p.parse_args()\n'}
    listing='AD/Toy.py\nREADME.md\n'+''.join(f'dataset/unused_data_directory/long_series_file_{i:04d}.csv\n' for i in range(1400))+'models/Toy.py\nrun.py\nscripts/Toy/etth1_best.sh\n'
    assert listing.index('models/Toy.py')>65536
    sb=TruncatedSandbox(files,listing)
    monkeypatch.setattr(repo_source,'select_files',lambda *_:['AD/Toy.py','README.md'])
    dump=repo_source.read_repo(sb,None,'https://github.com/example/toy','Toy erreicht MSE 0.4','ETTh1',
        fixture_dir=tmp_path,unit_key='Toy',model='Toy',mode='live')
    assert {'models/Toy.py','scripts/Toy/etth1_best.sh','run.py'}<=set(dump.files)
    assert 'AD/Toy.py' not in dump.files and len(dump.listing)==1405


@pytest.mark.parametrize('corruption',['truncated','hash','size','too_large','metadata','count','failure'])
def test_unverified_listing_never_reaches_model_selection(tmp_path, monkeypatch, corruption):
    from repro_agent.source_context import SourceContextBlocked
    sb=TruncatedSandbox({},'models/Toy.py\n')
    if corruption=='truncated':sb.transferred=sb.transferred[:-1]
    if corruption=='count':sb.metadata_count+=1
    if corruption=='hash':sb.metadata_hash='0'*64
    if corruption=='size':sb.metadata_size+=1
    if corruption=='too_large':sb.metadata_size=2_000_000
    if corruption=='metadata':sb.metadata_hash='not-a-hash'
    if corruption=='failure':sb.run=lambda *args:SimpleNamespace(stdout='',exit_code=1,timed_out=False,stderr='git failed')
    monkeypatch.setattr(repo_source,'select_files',lambda *_:pytest.fail('unverified listing reached the model'))
    with pytest.raises(SourceContextBlocked,match='listing'):
        repo_source.read_repo(sb,None,'https://github.com/example/toy','Toy','ETTh1',fixture_dir=tmp_path,mode='live')
    if corruption in ('too_large','metadata','failure'):assert '/tmp/.nullius_repo_listing' not in sb.gets
