import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from repro_agent import repo_source as rs


class Sandbox:
    def __init__(self, text): self.text, self.clones = text, 0
    def base_snapshot(self, *a, **k): self.clones += 1; return 'snapshot'
    def branch(self, *a): return 'branch'
    def run(self, *a):
        data=b'models/NewForecast.py\0'
        return SimpleNamespace(stdout=f'{len(data)}\n{data.count(0)}\n{hashlib.sha256(data).hexdigest()}\n', exit_code=0)
    def get_file(self, b, path):
        if path == '/tmp/.nullius_repo_listing': return b'models/NewForecast.py\0'
        return b'new-commit' if path.endswith('.head_commit') else self.text.encode()


def test_live_repository_ignores_old_fixture_without_overwriting_it(tmp_path, monkeypatch):
    url='https://github.com/example/new'
    path=tmp_path/rs.unit_fixture_name(url,'unit')
    old={'url':url,'commit':'old','files':{'models/NewForecast.py':'class Model: pass'}}
    path.write_text(json.dumps(old));before=path.read_bytes()
    sb=Sandbox('# preamble\n'*3000+'class Model:\n    def forward(self,x):\n        return x\n')
    monkeypatch.setattr(rs,'select_files',lambda *_:['models/NewForecast.py'])
    dump=rs.read_repo(sb,None,url,'NewForecast','ETTh1',fixture_dir=tmp_path,unit_key='unit',model='NewForecast',mode='live')
    assert dump.commit=='new-commit' and 'def forward' in dump.files['models/NewForecast.py']
    assert len(dump.files['models/NewForecast.py'])>rs.MAX_FILE_CHARS
    assert sb.clones==1 and path.read_bytes()==before


def test_repository_reader_records_extreme_truncation(tmp_path, monkeypatch):
    sb=Sandbox('# x\n'*(rs.IMPORT_CHAR_BUDGET//4+10))
    monkeypatch.setattr(rs,'select_files',lambda *_:['models/NewForecast.py'])
    dump=rs.read_repo(sb,None,'https://github.com/example/new','NewForecast','ETTh1',fixture_dir=tmp_path,
                      unit_key='unit',model='NewForecast',mode='live')
    assert dump.truncation and dump.truncation[0]['of_chars']>dump.truncation[0]['kept_chars']


def test_large_binary_data_file_is_skipped_without_counting_as_truncated(tmp_path, monkeypatch):
    # Local catalog check 2026-10-14: Kipf's gcn/data/ind.cora.allx (pickle) blocked author-code mode as a "truncated source"
    class DataSandbox(Sandbox):
        def get_file(self, b, path):
            if path.endswith('ind.cora.allx'):
                return b'\x80\x02cnumpy\x00' * (rs.IMPORT_CHAR_BUDGET // 8 + 10)
            return super().get_file(b, path)
    sb = DataSandbox('class Model:\n    pass\n')
    monkeypatch.setattr(rs, 'select_files', lambda *_: ['models/NewForecast.py', 'gcn/data/ind.cora.allx'])
    dump = rs.read_repo(sb, None, 'https://github.com/example/new', 'NewForecast', 'ETTh1', fixture_dir=tmp_path,
                        unit_key='unit', model='NewForecast', mode='live')
    assert 'gcn/data/ind.cora.allx' not in dump.files and dump.truncation == []
