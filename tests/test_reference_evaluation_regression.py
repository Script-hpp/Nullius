"""Offline reference run: moving evaluation must preserve predictions and score.

The oracle is synthetic (node ID modulo 7), NOT a new Cora benchmark result.
The human GCN reference trains on the unchanged cached public Cora data.
"""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import uuid

import pytest

from repro_agent.sandbox.local import LocalSandbox
from repro_agent.tasks import cora_task
from test_loop_fakes import make_orch


@pytest.mark.skipif(os.getenv('NULLIUS_DOCKER_TESTS') != '1', reason='explicit offline Docker reference comparison')
def test_honest_reference_is_identical_with_old_and_separate_evaluation(tmp_path, monkeypatch):
    import repro_agent.loop as current
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(['git', 'show', 'e945e01:src/repro_agent/loop.py'], cwd=root, text=True)
    old = types.ModuleType('repro_agent.old_audit_loop')
    old.__package__ = 'repro_agent'
    sys.modules[old.__name__] = old
    exec(compile(source, 'old_audit_loop.py', 'exec'), old.__dict__)
    oracle = """import csv,json
with open('/output/predictions.csv') as f:
    rows=list(csv.DictReader(f))
score=sum(int(r['prediction']) == int(r['id']) % 7 for r in rows)/len(rows)
print(json.dumps({'status':'ok','metrics':{'accuracy':score}}))
"""
    (tmp_path / 'toy_eval.py').write_text(oracle)
    monkeypatch.setattr(current, 'HARNESS_DIR', tmp_path)
    monkeypatch.setattr(old, 'HARNESS_DIR', tmp_path)
    class OfflineSandbox(LocalSandbox):
        def __init__(self): self.containers = []
        def branch(self, snapshot):
            name = 'nullius-reference-' + uuid.uuid4().hex[:8]
            subprocess.run(['docker', 'run', '-d', '--network', 'none', '--name', name, snapshot, 'sleep', 'infinity'],
                           capture_output=True, check=True)
            self.containers.append(name)
            return name
        def run(self, branch, cmd, timeout_s):
            # Both paths use Docker's actual network isolation. Historical
            # unshare failed without SYS_ADMIN; never grant that capability.
            return super().run(branch, cmd.replace('unshare -n ', ''), timeout_s)
    sb = OfflineSandbox()
    o, _ = make_orch(sb, seeds=1)
    o.train_timeout_s = 300
    code = (root / 'tests/reference/gcn_cora_reference.py').read_text().replace('import torch\n', 'import torch\ntorch.set_num_threads(1)\n', 1)
    task = dataclasses.replace(cora_task(), smoke_ids=(), smoke_ids_cmd='', harness_files=['toy_eval.py'],
                               eval_cmd='python {harness_dir}/toy_eval.py', val_eval_cmd='', io_config='')
    prior = old.Orchestrator.__new__(old.Orchestrator)
    prior.__dict__.update(o.__dict__)
    try:
        a, before = prior._attempt('repro-snapshot:gate-cora', code, 0, task, 'accuracy')
        b, after = o._attempt('repro-snapshot:gate-cora', code, 0, task, 'accuracy')
        assert before['status'] == after['status'] == 'ok', (before, after)
        assert before['measured'] == after['measured']
        assert sb.get_file(a, '/output/predictions.csv') == sb.get_file(b, '/output/predictions.csv')
    finally:
        sys.modules.pop(old.__name__, None)
        for name in sb.containers:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, check=True)
