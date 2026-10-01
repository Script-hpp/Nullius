"""A3/M7: Repo-Defaults (argparse) kennzeichnen, Konflikte Paper vs Repo im Plan-Prompt. Attrappen, kein Netz."""

from types import SimpleNamespace

from repro_agent.loop import Orchestrator
from repro_agent.setup_details import (
    REPO_DEFAULT_MARK, SetupDetail, find_conflicts, format_conflicts, format_for_prompt, is_repo_default,
)
from repro_agent.tasks import cora_task

WD_PAPER = SetupDetail("weight decay", "5e-5", "weight decay is tuned with hyperopt, best 5e-5", "paper")
WD_REPO = SetupDetail("Weight decay default", "5e-6", "parser.add_argument('--weight_decay', type=float, default=5e-6)", "repo:args.py@abc1234")
LR_REPO = SetupDetail("learning rate", "0.2", "lr = 0.2  # set in the training loop", "repo:train.py@abc1234")
EPOCHS = SetupDetail("epochs", "100", "we train for 100 epochs", "paper")


def test_is_repo_default_only_for_argparse_default_in_repo_files():
    assert is_repo_default(WD_REPO)
    assert not is_repo_default(LR_REPO)                                  # Repo, aber kein Default
    assert not is_repo_default(SetupDetail("x", "1", "default=1", "paper"))   # Paper nie


def test_format_marks_defaults_only_when_asked():
    plain = format_for_prompt([WD_PAPER, WD_REPO, LR_REPO])
    assert REPO_DEFAULT_MARK not in plain                                # alter Pfad: unveraendert
    marked = format_for_prompt([WD_PAPER, WD_REPO, LR_REPO], mark_defaults=True)
    lines = marked.splitlines()
    assert REPO_DEFAULT_MARK in lines[2] and REPO_DEFAULT_MARK not in lines[1] and REPO_DEFAULT_MARK not in lines[3]
    assert "the paper may tune this per dataset" in lines[2]


def test_conflicts_pair_paper_and_repo_with_different_values():
    assert find_conflicts([WD_PAPER, WD_REPO, EPOCHS, LR_REPO]) == [(WD_PAPER, WD_REPO)]
    same = SetupDetail("Weight decay", "5e-5", "default=5e-5", "repo:args.py@abc1234")
    assert find_conflicts([WD_PAPER, same]) == []                        # gleicher Wert: kein Konflikt
    assert find_conflicts([WD_REPO, LR_REPO]) == []                      # kein Paper-Wert
    text = format_conflicts([WD_PAPER, WD_REPO])
    assert "paper says 5e-5" in text and "repository says 5e-6" in text and "repo:args.py@abc1234" in text and "paper:" not in text.split("\n")[0]
    assert format_conflicts([EPOCHS]) == ""


def _orch(flag, details):
    o = Orchestrator.__new__(Orchestrator)
    o.details, o.graph_repair = details, flag
    seen = []

    def ask(role, system, user, **kw):
        seen.append(user)
        return SimpleNamespace(content="1. plan")

    o._ask = ask
    return o, seen


def test_plan_prompt_has_conflict_block_and_marks_only_in_product_path():
    paper = {"title": "T"}
    claim = SimpleNamespace(text="GCN 81.5", target=0.815, metric="accuracy")
    details = [WD_PAPER, WD_REPO]
    old, seen_old = _orch(False, details)
    old.plan(paper, claim, cora_task())
    new, seen_new = _orch(True, details)
    new.plan(paper, claim, cora_task())
    assert "different values for the same setting" not in seen_old[0] and REPO_DEFAULT_MARK not in seen_old[0]
    assert "different values for the same setting" in seen_new[0] and REPO_DEFAULT_MARK in seen_new[0]
    assert "paper says 5e-5" in seen_new[0] and "repository says 5e-6" in seen_new[0]
    # Pfad ohne Flag: Prompt wie vorher (Setup-Details unmarkiert, kein Konfliktblock)
    t = cora_task()
    assert seen_old[0] == (
        f"Paper: T\nClaim: GCN 81.5 (target 0.815, accuracy)\n\n{t.data_description}\n\n{format_for_prompt(details)}\n\n"
        "Write a short reproduction plan: model architecture, all hyperparameters and training "
        "protocol as described in the paper. Mark every value you are NOT sure about with "
        "'(uncertain)'. Max 15 lines.")
