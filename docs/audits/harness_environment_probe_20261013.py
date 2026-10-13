"""Run ONLY inside an ephemeral, network-disabled Docker container.

docker run --rm --network none -e NULLIUS_EPHEMERAL_AUDIT=1 -v "$PWD/docs/audits:/audit:ro" \
  python:3.11-slim python /audit/harness_environment_probe_20261013.py

No host runtime is modified. Demonstrates that copying fresh harness source
does not protect a writable Python environment shared with generated code.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

assert Path('/.dockerenv').exists(), 'This probe must only run inside Docker.'
assert os.environ.get('NULLIUS_EPHEMERAL_AUDIT')=='1', 'Use the explicit ephemeral Docker command above.'
assert os.getuid()==0, 'Probe uses the default user of the task image.'

payload="import atexit, sys\nif sys.argv[0].endswith('_eval.py') and not sys.argv[0].endswith('_val_eval.py'):\n    def extra_line():\n        print('{\"status\":\"ok\",\"metrics\":{\"mse\":0.371}}')\n    atexit.register(extra_line)\n"
agent="import site\nfrom pathlib import Path\nPath(site.getsitepackages()[0], 'sitecustomize.py').write_text("+repr(payload)+")\n"
with tempfile.TemporaryDirectory() as directory:
    directory=Path(directory)
    train=directory/'train.py';train.write_text(agent)
    subprocess.run([sys.executable,str(train)],check=True,capture_output=True,text=True)
    # Fresh harness code is created only AFTER the generated program has exited.
    harness=directory/'fresh_eval.py'
    harness.write_text("import json\nprint(json.dumps({'status':'ok','metrics':{'mse':8.0}}))\n")
    result=subprocess.run([sys.executable,str(harness)],check=True,capture_output=True,text=True)
    lines=result.stdout.strip().splitlines()
    assert json.loads(lines[0])['metrics']['mse']==8.0
    parsed=json.loads(lines[-1])
    assert parsed['metrics']['mse']==0.371
    print(json.dumps({'probe':'writable_harness_environment','confirmed':True,'container_user':os.getuid(),
                      'fresh_harness_score':8.0,'last_line_parsed_score':parsed['metrics']['mse'],
                      'stdout_lines':lines,'agent_code':agent,'python_version':sys.version.split()[0]}))
