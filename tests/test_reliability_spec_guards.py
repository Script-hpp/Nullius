"""Invariants for the reliability fixes E07-E12, E15-E19 (docs/opus_general_audit_review_20261013.md). No network, no model calls.

Each test is the inverted form of a counter-example from the audits: it fails on e945e01 and passes after the fix.
"""
import json
from types import SimpleNamespace

import pytest

from repro_agent import spec_check
from repro_agent.harness.code_check import find_violations
from repro_agent.harness.training_check import checkpoint_issues
from repro_agent.method_guard import placeholder_issues
from repro_agent.repo_source import RepoLink, clone_plan, find_repo_links, read_repo, requested_ref
from repro_agent.setup_details import Doc, SetupDetail
from repro_agent.source_context import SourceContextBlocked, author_excerpt
from repro_agent.spec import Spec, SpecItem, build_spec, drop_run_script_defaults, paper_numbers, raw_items, resolve


def numeric_spec(*pairs):
    return Spec(model="Toy", family="forecast",
                items=[SpecItem(k, str(v), "q", "paper", "paper", v, enforce=True) for k, v in pairs])


# ---- E07 / N04: purpose of a number ------------------------------------------------------------------------------------
def test_inference_batch_quote_does_not_override_training_default():
    quote = "We compare inference speed with batch size 32."
    arg = "parser.add_argument('--batch_size', type=int, default=8)"
    docs = [Doc("paper", quote), Doc("repo:models/Toy.py@abc", "class Model:\n    pass\n"), Doc("repo:run.py@abc", arg)]
    reply = [{"key": "batch_size", "value": "32", "quote": quote, "source": "paper"},
             {"key": "batch_size", "value": "8", "quote": arg, "source": "repo:run.py@abc"}]
    spec = build_spec(lambda *_: json.dumps(reply), [], docs, "Toy erreicht MSE 0.4", "ETTh1", numeric_only=True)
    assert spec.get("batch_size").number == 8
    assert any("Inferenz" in d["reason"] for d in spec.dropped)


def test_training_sentence_that_also_mentions_inference_is_kept():
    quote = "All models are trained for 10 epochs under the same protocol, and we report average inference speed."
    docs = [Doc("paper", quote), Doc("repo:models/Toy.py@abc", "class Model:\n    pass\n")]
    reply = [{"key": "epochs", "value": "10", "quote": quote, "source": "paper"}]
    spec = build_spec(lambda *_: json.dumps(reply), [], docs, "Toy erreicht MSE 0.4", "ETTh1", numeric_only=True)
    assert spec.get("epochs").number == 10


def test_dropout_is_not_enforced_for_a_model_without_dropout():
    arg = "parser.add_argument('--dropout', type=float, default=0.1)"
    reply = [{"key": "dropout", "value": "0.1", "quote": arg, "source": "repo:run.py@abc"}]
    plain = [Doc("repo:models/Toy.py@abc", "class Toy:\n    def forward(self, x):\n        return x\n"), Doc("repo:run.py@abc", arg)]
    spec = build_spec(lambda *_: json.dumps(reply), [], plain, "Toy erreicht MSE 0.4", "ETTh1", numeric_only=True)
    assert spec.get("dropout") is None
    assert any(d["item"]["key"] == "dropout" and "Mechanismus" in d["reason"] for d in spec.dropped)
    with_dropout = [Doc("repo:models/Toy.py@abc", "from layers.Block import Block\nclass Toy:\n    pass\n"),
                    Doc("repo:layers/Block.py@abc", "import torch.nn as nn\nclass Block:\n    def __init__(self, c):\n        self.d = nn.Dropout(c.dropout)\n"),
                    Doc("repo:run.py@abc", arg)]
    spec = build_spec(lambda *_: json.dumps(reply), [], with_dropout, "Toy erreicht MSE 0.4", "ETTh1", numeric_only=True)
    assert spec.get("dropout").enforce   # mechanism in an imported layer of the model


# ---- E08 / E09: one run-script invocation per experiment --------------------------------------------------------------
REAL_SH = """for seed in 2021 0 42
do
    python -u run.py \\
      --model_id ETTh1_720_96 \\
      --data ETTh1 \\
      --pred_len 96 \\
      --learning_rate 0.001 \\
      --topm 20

    python -u run.py \\
      --model_id ETTh1_720_336 \\
      --data ETTh1 \\
      --pred_len 336 \\
      --learning_rate 0.0001 \\
      --topm 20

    python -u run.py \\
      --model_id ETTm1_720_96 \\
      --data ETTm1 \\
      --pred_len 96 \\
      --time_aware_weight 0.1 \\
      --topm 20
done
"""
REAL_ARG = "    parser.add_argument('--time_aware_weight', type=float, default=0.5, help='w')\n    parser.add_argument('--topm', type=int, default=20)\n"


def test_default_kept_when_only_other_experiments_set_the_flag():
    excerpt = f"=== repo:run.py@a ===\n{REAL_ARG}\n=== repo:scripts/ETTh_720.sh@a ===\n{REAL_SH}"
    out, dropped = drop_run_script_defaults(excerpt, dataset="ETTh1", horizon=96, model="SARAF")
    assert "default=0.5" in out and "time_aware_weight" not in dropped
    assert dropped == ["topm"]   # the ETTh1/96 call sets it, so its default is replaced
    _, legacy = drop_run_script_defaults(excerpt)   # without experiment context: unchanged behaviour
    assert "time_aware_weight" in legacy


def test_value_of_the_matching_call_survives_other_calls():
    sh = Doc("repo:scripts/ETTh_720.sh@a", REAL_SH)
    items = {i.key: i.number for i in raw_items([sh], [sh.name], "SARAF", "forecast", dataset="ETTh1", horizon=96)}
    assert items["lr"] == 0.001
    other = Doc("repo:scripts/both.sh@a", "python run.py --data Wanted --pred_len 96 --learning_rate 0.01\n"
                                          "python run.py --data Other --pred_len 192 --learning_rate 0.02\n")
    assert {i.key: i.number for i in raw_items([other], [other.name], "Toy", "forecast", dataset="Wanted", horizon=96)}["lr"] == 0.01


def test_shell_variables_and_lcc_prefix_still_match():
    sh = Doc("repo:run.sh@a", "for h in 96 192\ndo\npython main.py --dataset cora --pred_len $h --lr 0.005\ndone\n")
    assert {i.key: i.number for i in raw_items([sh], [sh.name], "GCN", "graph", dataset="Cora-LCC")}["lr"] == 0.005


# ---- E10: keyword role --------------------------------------------------------------------------------------------------
def test_unrelated_keywords_are_not_conflicts_but_training_literals_are():
    spec = numeric_spec(("batch_size", 32), ("dropout", 0.1))
    code = ("CONFIG={'batch_size':32,'dropout':0.1}\ntrain=DataLoader(train_set,batch_size=CONFIG['batch_size'])\n"
            "rt_loader=DataLoader(data,batch_size=1024,shuffle=False)\ny=F.normalize(x,p=2)\nlayer=nn.Dropout(CONFIG['dropout'])\n")
    assert spec_check.config_check(code, spec)[1]["conflicts"] == []
    bad = code + "train_loader=DataLoader(train_set,batch_size=64,shuffle=True)\nd=nn.Dropout(p=0.5)\nenc=nn.TransformerEncoderLayer(8,2,dropout=0.3)\n"
    got = {(c["key"], c["literal"]) for c in spec_check.config_check(bad, spec)[1]["conflicts"]}
    assert got == {("batch_size", 64), ("dropout", 0.5), ("dropout", 0.3)}


# ---- E11: every module-level write to CONFIG ------------------------------------------------------------------------------
@pytest.mark.parametrize("ending", [
    "cfg=CONFIG\ncfg['lr']=1.0",
    "CONFIG={'lr':1.0}",
    "if True:\n    CONFIG['lr']=1.0",
    "CONFIG['lr'] *= 1000",
    "CONFIG={**CONFIG,'lr':1.0}",
    "CONFIG.__setitem__('lr', 1.0)",
    "CONFIG.pop('lr')",
    "CONFIG.update(vars(args))",
    "CONFIG={'other':1}",
    "try:\n    CONFIG['lr']=args.lr\nexcept Exception:\n    pass",
])
def test_config_overwrites_are_conflicts(ending):
    code = "CONFIG={'lr':0.001}\n" + ending + "\noptimizer=Adam(lr=CONFIG['lr'])\n"
    _, report = spec_check.config_check(code, numeric_spec(("lr", 0.001)))
    assert [c["key"] for c in report["conflicts"]] == ["lr"]
    assert spec_check.score(numeric_spec(("lr", 0.001)), report, []) == (0, 1)
    assert "changes CONFIG" in spec_check.repair_message(spec_check.failures(report, []), numeric_spec(("lr", 0.001)))


@pytest.mark.parametrize("ending", ["CONFIG={**CONFIG,'other':2}", "cfg=CONFIG\ncfg['other']=3", "CONFIG['lr']=0.001",
                                    "for epoch in range(3):\n    CONFIG['lr'] *= 0.5", "CONFIG.setdefault('lr', 5.0)"])
def test_consistent_writes_and_runtime_schedules_still_pass(ending):
    code = "CONFIG={'lr':0.001}\n" + ending + "\noptimizer=Adam(lr=CONFIG['lr'])\n"
    assert spec_check.config_check(code, numeric_spec(("lr", 0.001)))[1]["conflicts"] == []


# ---- E12: corroboration belongs to the same parameter -----------------------------------------------------------------
def test_horizon_number_does_not_decide_a_batch_conflict():
    a = SpecItem("batch_size", "16", "batch_size: 16", "repo:a.conf", "config", 16, specific=True)
    b = SpecItem("batch_size", "32", "batch_size: 32", "repo:b.conf", "config", 32, specific=True)
    paper = [SetupDetail("prediction horizon", "32", "We use a prediction length of 32 steps.", "paper")]
    chosen, _ = resolve([a, b], "forecast", True, corroborate=paper_numbers(paper, []))
    assert not chosen[0].enforce and chosen[0].tie
    batch = [SetupDetail("batch", "32", "The batch size is 32.", "paper")]
    chosen, _ = resolve([a, b], "forecast", True, corroborate=paper_numbers(batch, []))
    assert chosen[0].number == 32 and chosen[0].enforce


# ---- E15: training dependencies are as complete as the model core ------------------------------------------------------
def test_transitive_training_helpers_are_included_or_block():
    docs = [Doc("repo:models/Toy.py@abc", "class Model:\n    pass\n"), Doc("repo:exp/train.py@abc", "from utils.schedule import update\nloss.backward()\n"),
            Doc("repo:utils/schedule.py@abc", "from utils.decay import rate\ndef update(x):\n    return rate(x)\n"),
            Doc("repo:utils/decay.py@abc", "def rate(x):\n    return x*0.5\n")]
    _, names = author_excerpt(docs, "Toy", "ETTh1")
    assert "repo:utils/decay.py@abc" in names
    with pytest.raises(SourceContextBlocked, match="training dependency utils/schedule.py"):
        author_excerpt(docs[:2], "Toy", "ETTh1", listing=["models/Toy.py", "exp/train.py", "utils/schedule.py"])


def test_registry_of_other_models_is_not_required():
    docs = [Doc("repo:models/Toy.py@abc", "class Model:\n    pass\n"),
            Doc("repo:exp/exp_basic.py@abc", "from models import Toy, Other\nloss.backward()\n")]
    _, names = author_excerpt(docs, "Toy", "ETTh1", listing=["models/Toy.py", "models/Other.py", "exp/exp_basic.py"])
    assert "repo:exp/exp_basic.py@abc" in names


# ---- E16: explicit git refs ---------------------------------------------------------------------------------------------
class CloneSandbox:
    def __init__(self):
        self.setup = []

    def base_snapshot(self, image, cmds, label=None):
        self.setup = cmds
        return "snap"

    def branch(self, snap):
        return "b"

    def run(self, branch, cmd, timeout_s):
        import hashlib
        self.listing=b'100644 blob a 10\tmodel/train.py\0'
        return SimpleNamespace(stdout=f'{len(self.listing)}\n{self.listing.count(0)}\n{hashlib.sha256(self.listing).hexdigest()}\n', stderr="", exit_code=0)

    def get_file(self, branch, path):
        if path == '/tmp/.nullius_repo_listing': return self.listing
        return b"abcdef1234567\n" if path.endswith(".head_commit") else b"x = 1\n"


class OneLLM:
    def complete(self, role, messages, **kw):
        return SimpleNamespace(content='["model/train.py"]')


def test_requested_ref_is_checked_out_and_recorded(tmp_path):
    assert requested_ref("https://github.com/a/b/tree/v1/model") == "v1"
    assert requested_ref("https://github.com/a/b/tree/main/model") == "main"
    assert requested_ref("https://github.com/a/b/tree/HEAD/model") == requested_ref("https://github.com/a/b") == ""
    sb = CloneSandbox()
    dump = read_repo(sb, OneLLM(), "https://github.com/a/b/tree/v1/model", "Toy", "ETTh1", fixture_dir=tmp_path, mode="live")
    clone = " ".join(sb.setup)
    assert "git checkout -q v1" in clone and "sparse-checkout set model" in clone and "--depth 1" not in clone
    assert dump.ref == "v1" and clone_plan("https://github.com/a/b/tree/v1/model") == ("https://github.com/a/b", "model")
    read_repo(sb, OneLLM(), "https://github.com/a/b/tree/HEAD/model", "Toy", "ETTh1", fixture_dir=tmp_path, mode="live")
    assert "checkout -q" not in " ".join(sb.setup)
    with pytest.raises(ValueError):
        requested_ref("https://github.com/a/b/tree/-x/model")


class DefaultMasterSandbox(CloneSandbox):
    """Repository whose default branch is master and that has no branch main: checking out main fails like git would."""
    def __init__(self):
        super().__init__()
        self.attempts = []

    def base_snapshot(self, image, cmds, label=None):
        from repro_agent.sandbox.base import SandboxSetupError
        self.attempts.append(list(cmds))
        if any("git checkout -q main" in c for c in cmds):
            raise SandboxSetupError("error: pathspec 'main' did not match any file(s) known to git")
        return super().base_snapshot(image, cmds, label)


def test_requested_main_is_not_replaced_by_a_different_default_branch(tmp_path):
    from repro_agent.sandbox.base import SandboxSetupError
    sb = DefaultMasterSandbox()
    with pytest.raises(SandboxSetupError, match="pathspec 'main'"):
        read_repo(sb, OneLLM(), "https://github.com/a/b/tree/main", "Toy", "ETTh1", fixture_dir=tmp_path, mode="live")
    assert len(sb.attempts) == 1   # no silent second clone of the default branch
    dump = read_repo(sb, OneLLM(), "https://github.com/a/b/tree/master", "Toy", "ETTh1", fixture_dir=tmp_path, mode="live")
    assert dump.ref == "master" and "git checkout -q master" in " ".join(sb.setup)


def test_paper_link_keeps_every_explicit_ref():
    links = find_repo_links("Code is available at https://github.com/a/b/tree/v2.1/src and more.")
    assert links[0].target == "https://github.com/a/b/tree/v2.1/src"
    assert RepoLink("a", "b", "", "strong", 0, "v3").target == "https://github.com/a/b/tree/v3"
    links = find_repo_links("Code is available at https://github.com/a/b/tree/master/src and more.")
    assert links[0].target == "https://github.com/a/b/tree/master/src" and links[0].ref == "master"
    links = find_repo_links("Code is available at https://github.com/a/b/tree/HEAD/src and more.")
    assert links[0].target == "https://github.com/a/b/tree/HEAD/src" and links[0].ref == ""


# ---- E17-E19: guard variants and false positives ---------------------------------------------------------------------
REF = {"layers/retrieval.py": "class Retrieval:\n    def retrieve(self, x):\n        return x.mean(0)\n"}


@pytest.mark.parametrize("code", [
    "class Retrieval:\n    def retrieve(self, x):\n        result = torch.zeros_like(x)\n        return result\n",
    "class Retrieval:\n    def fetch_context(self, x):\n        return torch.zeros_like(x)\n",
    "class Retrieval:\n    def retrieve(self, x):\n        return 0 * x\n",
    "class Retrieval:\n    def retrieve(self, x):\n        return torch.full_like(x, 0)\n",
])
def test_placeholder_variants_are_detected(code):
    assert placeholder_issues(code, REF)


@pytest.mark.parametrize("code", [
    "class Retrieval:\n    def retrieve(self, x):\n        out = torch.zeros_like(x)\n        out[0] = x[0]\n        return out\n",
    "class Retrieval:\n    def retrieve(self, x):\n        out = torch.zeros_like(x)\n        out.add_(x)\n        return out\n",
    "class Retrieval:\n    def init_cache(self, x):\n        return torch.zeros_like(x)\n",
])
def test_working_retrieval_and_initialisation_are_allowed(code):
    assert placeholder_issues(code, REF) == []
    assert placeholder_issues("class Retrieval:\n    def fetch_context(self, x):\n        return torch.zeros_like(x)\n", {}) == []


@pytest.mark.parametrize("code", [
    "for epoch in range(2):\n    optimizer.step()\n    best = model.state_dict().copy()\nmodel.load_state_dict(state_dict=best)\n",
    "for epoch in range(2):\n    trainer.step()\n    best = model.state_dict().copy()\nmodel.load_state_dict(best)\n",
    "for epoch in range(2):\n    optimizer.step()\n    self.best = model.state_dict().copy()\nmodel.load_state_dict(self.best)\n",
    "for epoch in range(2):\n    loss.backward()\n    best = {k: v for k, v in model.state_dict().items()}\nmodel.load_state_dict(best)\n",
    "for epoch in range(2):\n    opt.step()\n    best = {k: v.detach() for k, v in model.state_dict().items()}\nmodel.load_state_dict(best)\n",
    "for epoch in range(2):\n    opt.step()\n    best = OrderedDict(model.state_dict())\nmodel.load_state_dict(best)\n",
])
def test_shallow_checkpoint_variants_are_rejected(code):
    assert checkpoint_issues(code)


def test_independent_checkpoint_copies_are_allowed():
    code = "for epoch in range(2):\n    trainer.step()\n    best = {k: v.detach().clone() for k, v in model.state_dict().items()}\nmodel.load_state_dict(best)\n"
    assert checkpoint_issues(code) == []


@pytest.mark.parametrize("code", [
    "# See https://example.com/paper\nprint(1)\n",
    "def fetch_neighbors_from_memory(x):\n    return x\n",
    '"""Unlike socket-based loaders we read a local npz (Planetoid split)."""\nimport numpy as np\n',
    "def f():\n    '''download nothing; uses /data only'''\n    return 1\n",
])
def test_descriptive_text_and_local_names_are_not_operations(code):
    assert find_violations(code) == []


@pytest.mark.parametrize("code", [
    "from sklearn.datasets import fetch_openml",
    "x = sklearn.datasets.fetch_california_housing()",
    "url = 'https://x.org/a.csv'  # comment",
    "import socket  # harmless comment",
    "s = '''import urllib'''\n",
    "def g():\n    '''doc'''\n    return download_file()\n",
])
def test_real_operations_are_still_rejected(code):
    assert find_violations(code)


def test_build_spec_passes_the_experiment_to_script_parsing():
    """Product path: build_spec strips only defaults set by the ETTh1/96 call and reads its learning rate."""
    docs = [Doc("paper", "SARAF on ETTh1."),
            Doc("repo:models/SARAF.py@a", "class SARAF:\n    def forward(self, x):\n        return x\n"),
            Doc("repo:run.py@a", REAL_ARG + "    parser.add_argument('--learning_rate', type=float, default=0.01)\nloss.backward()\n"),
            Doc("repo:scripts/ETTh_720.sh@a", REAL_SH)]
    spec = build_spec(lambda *_: "[]", [], docs, "SARAF erreicht MSE 0.371", "ETTh1", numeric_only=True)
    assert "default=0.5" in spec.excerpt and "time_aware_weight" not in spec.dropped_defaults
    assert "learning_rate" in spec.dropped_defaults and spec.get("lr").number == 0.001
