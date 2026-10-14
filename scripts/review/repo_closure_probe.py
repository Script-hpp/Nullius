"""Offline probe: does read_repo collect every local module the author-source guard later requires?

Runs the real listing/read shell commands against local clones (no network, no model calls; the file
choice of the model is replaced by a fixed list). Usage:
  uv run python scripts/review/repo_closure_probe.py <scratch_dir> <name>=<clone>:<Model>:<url>:<file,file,...> ...
"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from repro_agent.repo_source import read_repo  # noqa: E402
from repro_agent.setup_details import Doc  # noqa: E402
from repro_agent.source_context import SourceContextBlocked, author_excerpt  # noqa: E402

CAP = 65535
SCRATCH = Path(sys.argv[1])


class Sb:
    def __init__(self, repo):
        self.repo, self.listing_file = repo, SCRATCH / "listing.bin"

    def base_snapshot(self, image, cmds, label=None):
        return "snap"

    def branch(self, snap):
        return "b"

    def _map(self, cmd):
        return cmd.replace("/tmp/.nullius_repo_listing", str(self.listing_file)).replace("/repo", str(self.repo))

    def run(self, branch, cmd, timeout_s, **kw):
        p = subprocess.run(self._map(cmd), shell=True, capture_output=True)
        return SimpleNamespace(stdout=p.stdout[:CAP].decode(errors="replace"), stderr=p.stderr.decode(errors="replace"),
                               exit_code=p.returncode, timed_out=False)

    def get_file(self, branch, path):
        if path == "/repo/.head_commit":
            return subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], capture_output=True).stdout
        return Path(self._map(path)).read_bytes()

    def get_file_limited(self, branch, path, max_bytes):
        return self.get_file(branch, path)


out = {}
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    clone, model, rest2 = rest.split(":", 2)
    url, chosen = rest2.rsplit(":", 1)
    llm = SimpleNamespace(complete=lambda role, messages, _c=chosen, **kw: SimpleNamespace(content=json.dumps(_c.split(","))))
    dump = read_repo(Sb(Path(clone)), llm, url, f"{model} ETTh1 MSE", "ETTh1", focus_model=True,
                     unit_key=name, model=model, mode="live")
    docs = [Doc(f"repo:{p}@{dump.commit[:7]}", t) for p, t in dump.files.items()]
    res = {"read": sorted(dump.files), "read_chars": sum(len(t) for t in dump.files.values()), "n_listing": len(dump.listing)}
    try:
        excerpt, names = author_excerpt(docs, model, "ETTh1", listing=dump.listing)
        res.update(guard="ok", excerpt_chars=len(excerpt), excerpt_docs=len(names))
        from repro_agent.spec import runtime_items
        try:
            res["runtime"] = {i.key: i.value for i in runtime_items(docs, names, model, "forecast", "ETTh1")}
        except SourceContextBlocked as exc:
            res["runtime"] = f"BLOCKED: {exc}"
    except SourceContextBlocked as exc:
        res.update(guard=f"BLOCKED: {exc}")
    out[name] = res
print(json.dumps(out, indent=1))
