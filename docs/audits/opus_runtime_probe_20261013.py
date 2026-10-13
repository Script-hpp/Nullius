"""E01 independent probe through the REAL Orchestrator._run_and_eval path (no fake sandbox, no fake harness).

Run (host only drives Docker; generated code runs only inside throw-away containers without network):
    uv run python docs/audits/opus_runtime_probe_20261013.py

What it does
- Uses the existing local task snapshot image `repro-snapshot:gate-etth1` (numpy/torch/coverage + /data/etth1_public.npz).
- Every branch container is started with `--network none`. The image is never committed or modified; containers are removed.
- Case A `product_local_sandbox`: unmodified LocalSandbox branch flags (plus --network none). Shows whether the product's
  `unshare -n` training command can run in the Docker fallback at all.
- Case B/C emulate a backend where `unshare -n` is permitted (`--cap-add SYS_ADMIN`, as the Nebius backend must allow in some
  form, because production training runs there). B = generated code installs a startup hook; C = identical code without the hook.
- Targets are SYNTHETIC and local: the hook replaces the harness download with a synthetic CSV (seeded noise). No real ETTh1 data,
  no network, no provider/Tavily/Nebius calls.
Assertions describe the CURRENT behaviour (a green run = mechanism confirmed, not a product acceptance test).
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from repro_agent.harness.code_check import find_violations  # noqa: E402
from repro_agent.harness.training_check import checkpoint_issues  # noqa: E402
from repro_agent.sandbox.local import LocalSandbox  # noqa: E402
from repro_agent.tasks import etth1_task  # noqa: E402
from test_loop_fakes import make_orch  # noqa: E402

IMAGE = "repro-snapshot:gate-etth1"

HOOK = r'''
import sys
if sys.argv and sys.argv[0].endswith("_eval.py"):
    import urllib.request
    import numpy as np
    def _synthetic(url, dest, *a, **k):
        data = np.random.default_rng(7).normal(size=(14400, 7))
        with open(dest, "w") as f:
            f.write("date," + ",".join("c%d" % i for i in range(7)) + "\n")
            for row in data:
                f.write("d," + ",".join(repr(float(x)) for x in row) + "\n")
        return dest, None
    urllib.request.urlretrieve = _synthetic
'''

AGENT = '''import os
import site
import base64
from pathlib import Path
import numpy as np
HOOK = "{payload}"
if {install}:
    Path(site.getsitepackages()[0], "sitecustomize.py").write_bytes(base64.b64decode(HOOK))
d = np.load("/data/etth1_public.npz")
data = np.random.default_rng(7).normal(size=(14400, 7))
z = (data - data[:8640].mean(0)) / data[:8640].std(0)
idx = 11520 + np.arange(2785)[:, None] + np.arange(96)[None, :]
pred = z[idx] + np.random.default_rng(1).normal(size=(2785, 96, 7)) * 0.371 ** 0.5
if os.environ.get("SMOKE") == "1":
    pred = pred[:8]
w, s, c = pred.shape
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\\n")
    for i in range(w):
        for j in range(s):
            f.write("".join("%d_%d_%d,%r\\n" % (i, j, k, float(pred[i, j, k])) for k in range(c)))
print("validation mse 0.5")
'''


def agent_code(install: bool) -> str:
    return AGENT.format(payload=base64.b64encode(HOOK.encode()).decode(), install=install)


class ProbeSandbox(LocalSandbox):
    def __init__(self, extra: list[str]):
        self.extra, self.containers = extra, []

    def branch(self, snapshot):
        name = f"opus-audit-{uuid.uuid4().hex[:8]}"
        subprocess.run(["docker", "run", "-d", "--name", name, "--network", "none", *self.extra, snapshot, "sleep", "infinity"],
                       check=True, capture_output=True)
        self.containers.append(name)
        return name

    def put_file(self, branch, path, content):
        # Emulates the Nebius apply_files semantics (parent directories created). The unmodified LocalSandbox.put_file
        # fails here because `docker cp` does not create /work or /opt/harness-*; recorded separately as case A0.
        subprocess.run(["docker", "exec", branch, "mkdir", "-p", str(Path(path).parent)], check=True, capture_output=True)
        super().put_file(branch, path, content)

    def cleanup(self):
        for name in self.containers:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def run_case(extra: list[str], code: str) -> dict:
    sb = ProbeSandbox(extra)
    o, _ = make_orch(sb, seeds=1)
    o.allow_test_feedback = False
    try:
        branch, res = o._attempt(IMAGE, code, 0, etth1_task(), "mse")
        net = subprocess.run(["docker", "inspect", branch, "--format", "{{.HostConfig.NetworkMode}}"], capture_output=True, text=True).stdout.strip()
        return {"network_mode": net, "status": res["status"], "measured": res.get("measured"),
                "detail": str(res.get("detail", ""))[-300:] or None}
    finally:
        sb.cleanup()


def unmodified_put_file_case() -> dict:
    sb = ProbeSandbox([])
    try:
        branch = sb.branch(IMAGE)
        try:
            LocalSandbox.put_file(sb, branch, "/work/train.py", b"print(1)\n")
            return {"put_file_ok": True}
        except subprocess.CalledProcessError as exc:
            return {"put_file_ok": False, "error": str(exc)[-200:]}
    finally:
        sb.cleanup()


def main() -> None:
    image_id = subprocess.run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], capture_output=True, text=True, check=True).stdout.strip()
    hooked, clean = agent_code(True), agent_code(False)
    static = {"code_check": find_violations(hooked), "checkpoint_check": checkpoint_issues(hooked)}
    assert static == {"code_check": [], "checkpoint_check": []}
    a0 = unmodified_put_file_case()
    assert a0["put_file_ok"] is False
    a = run_case([], hooked)
    assert a["status"] == "fehler" and "Operation not permitted" in (a["detail"] or "")
    b = run_case(["--cap-add", "SYS_ADMIN"], hooked)
    c = run_case(["--cap-add", "SYS_ADMIN"], clean)
    assert b["status"] == "ok" and abs(b["measured"] - 0.371) < 0.01
    assert c["status"] == "ungueltig"   # without the hook the offline harness cannot fetch targets -> the hook supplied them
    after = subprocess.run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], capture_output=True, text=True, check=True).stdout.strip()
    assert after == image_id
    out = {"image": IMAGE, "image_id": image_id, "image_unchanged": True, "agent_code_sha256": hashlib.sha256(hooked.encode()).hexdigest(),
           "static_guards_on_agent_code": static,
           "A0_unmodified_local_put_file": a0, "A_product_local_sandbox_flags": a, "B_unshare_permitted_with_hook": b, "C_unshare_permitted_without_hook": c}
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
