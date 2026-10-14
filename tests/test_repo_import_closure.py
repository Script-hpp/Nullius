"""The reader must collect every local module that the author-source guard later requires (SARAF, 2026-10-14)."""
import hashlib
from types import SimpleNamespace

from repro_agent import repo_source as rs
from repro_agent.setup_details import Doc
from repro_agent.source_context import author_excerpt

PAD = "# " + "x" * 98 + "\n"


def _repo():
    files = {
        "models/Net.py": "import torch\nfrom layers.Block import Block\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x\n",
        "layers/Block.py": "class Block: pass\n" + PAD * 300,
        "run.py": "import argparse\nfrom exp.exp_main import Exp\nparser = argparse.ArgumentParser()\nparser.add_argument('--lr', default=0.1)\n",
        "exp/exp_main.py": ("from utils.tools import adjust\nfrom utils.augmentation import augment\nfrom data_provider.loader import Loader\n"
                            "def train(model, loader):\n    for x in loader:\n        loss = model(x)\n        loss.backward()\n"),
        "utils/tools.py": "def adjust(): pass\n" + PAD * 300,
        "utils/augmentation.py": "def augment():\n    import utils.dtw as dtw\n    return dtw\n" + PAD * 300,
        "utils/dtw.py": "def dtw(): pass\n" + PAD * 60,   # vorher: nach ~120k gelesenen Zeichen nie gelesen
        "data_provider/loader.py": "class Loader: pass\n" + PAD * 300,
    }
    return files


class Sandbox:
    def __init__(self, files):
        self.files = files
        self.listing = "".join(p + "\0" for p in files).encode()

    def base_snapshot(self, *a, **k):
        return "snap"

    def branch(self, *a):
        return "b"

    def run(self, *a, **k):
        data = self.listing
        return SimpleNamespace(stdout=f"{len(data)}\n{data.count(0)}\n{hashlib.sha256(data).hexdigest()}\n", exit_code=0)

    def get_file(self, b, path):
        if path == "/tmp/.nullius_repo_listing":
            return self.listing
        if path.endswith(".head_commit"):
            return b"abc1234"
        return self.files[path.removeprefix("/repo/")].encode()


def test_reader_collects_lazy_training_imports_the_guard_requires(monkeypatch):
    files = _repo()
    monkeypatch.setattr(rs, "select_files", lambda *_a, **_k: ["models/Net.py", "run.py", "exp/exp_main.py"])
    dump = rs.read_repo(Sandbox(files), None, "https://github.com/example/net", "Net ETTh1", "ETTh1",
                        unit_key="unit", model="Net", mode="live")
    assert "utils/dtw.py" in dump.files
    docs = [Doc(f"repo:{p}@abc1234", t) for p, t in dump.files.items()]
    excerpt, names = author_excerpt(docs, "Net", "ETTh1", listing=dump.listing)
    assert "repo:utils/dtw.py@abc1234" in names


def test_incomplete_author_source_stops_before_paid_setup_extraction(monkeypatch):
    import pytest
    from repro_agent import loop
    from repro_agent.source_context import SourceContextBlocked

    files = {"models/Net.py": "from layers.Block import Block\nclass Model:\n    def forward(self, x):\n        return x\n",
             "layers/Block.py": "class Block: pass\n"}
    dump = rs.RepoDump(url="https://github.com/example/net", commit="abc1234", files=files, n_tracked=3,
                       listing=list(files) + ["layers/Missing.py"])
    files["models/Net.py"] = "from layers.Missing import M\n" + files["models/Net.py"]
    monkeypatch.setattr(loop, "read_repo", lambda *a, **k: dump)
    o = loop.Orchestrator.__new__(loop.Orchestrator)
    o.paper_only, o.repo_verify, o.code_grounded, o.spec_mode, o._paper_id = False, True, True, True, "p"
    o.paper_text = SimpleNamespace(text="paper text")
    o.sandbox, o.llm, o.tavily_mode, o.setup_v2 = None, None, "live", True
    o._tavily_getter = lambda: None
    o._resolve_repo = lambda claim, get: ("https://github.com/example/net", "", {})
    o._warn = o._emit = lambda *a, **k: None
    with pytest.raises(SourceContextBlocked, match="layers/Missing.py"):
        o._gather_docs("Net ETTh1 MSE", "ETTh1")
