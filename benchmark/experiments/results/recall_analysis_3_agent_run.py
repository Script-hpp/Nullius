"""Run a stored agent script locally under the harness data/metric (no LLM, no sandbox). Usage: agent_run.py <code.py> <cora|lcc> <n> [patchfile.py]"""
import os, statistics, subprocess, sys, tempfile, time, json
from pathlib import Path
import numpy as np
ROOT = Path("/home/onuralp/Desktop/hackathon/research")
sys.path.insert(0, str(ROOT / "src" / "repro_agent" / "harness"))
import cora_data, cora_lcc_data
CACHE = Path(__file__).parent / "cache"
code0 = Path(sys.argv[1]).read_text(); task = sys.argv[2]; n = int(sys.argv[3])
patches = []
if len(sys.argv) > 4:
    ns = {}; exec(Path(sys.argv[4]).read_text(), ns); patches = ns["PATCHES"]
for old, new in patches:
    assert code0.count(old) >= 1, old
    code0 = code0.replace(old, new)
vals = []; t0 = time.time()
for s in range(n):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp); raw = CACHE / "raw"
        if task == "cora":
            cora_data.prepare_public(raw, d / "p.npz"); code = code0.replace("/data/cora_public.npz", str(d / "p.npz"))
        else:
            cora_lcc_data.prepare_public(raw, s, d / "p.npz"); code = code0.replace("/data/cora_lcc_public.npz", str(d / "p.npz"))
        code = code.replace("/output", str(d / "out"))
        (d / "train.py").write_text(code)
        r = subprocess.run([sys.executable, str(d / "train.py")], capture_output=True, text=True, cwd=tmp,
                           env={"SEED": str(s), "SPLIT": str(s), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "2"})
        if r.returncode: print("ERR", r.stderr[-800:]); sys.exit(1)
        rows = (d / "out" / "predictions.csv").read_text().splitlines()[1:]
        truth = cora_data.test_labels(raw) if task == "cora" else cora_lcc_data.test_labels(raw, s)
        assert len(rows) == len(truth), (len(rows), len(truth))
        vals.append(float(np.mean([truth[i] == p for i, p in (l.split(",") for l in rows)])))
print(f"{Path(sys.argv[1]).name} {sys.argv[4] if len(sys.argv)>4 else ''}: n={n} mean {statistics.mean(vals):.4f} std {statistics.stdev(vals) if n>1 else 0:.4f} {[round(v,4) for v in vals]} {time.time()-t0:.0f}s", flush=True)
