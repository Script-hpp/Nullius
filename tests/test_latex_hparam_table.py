"""Hyperparameter tables in LaTeX sources bind to the spec; no provider calls."""
import json
from pathlib import Path

from repro_agent.hparam_table import table_details
from repro_agent.setup_details import Doc
from repro_agent.spec import build_spec

ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ("parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')\n"
          "parser.add_argument('--hidden_dim', type=int, default=64, help='hidden feature dimension')\n")
DEFAULTS = json.dumps([
    {"key": "dropout", "value": "0.05", "source": "repo:run.py@1",
     "quote": "parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')"},
    {"key": "hidden", "value": "64", "source": "repo:run.py@1",
     "quote": "parser.add_argument('--hidden_dim', type=int, default=64, help='hidden feature dimension')"}])


def tsmixer():
    return (ROOT / "tests/fixtures/tsmixer_hparams.tex").read_text()


def test_latex_hyperparameter_row_for_dataset_and_horizon():
    # Local catalog check 2026-10-14: the pipe-only reader found nothing in LaTeX; argparse defaults won
    got = {d.topic: d.value for d in table_details([Doc("paper", tsmixer())], "TSMixer", "ETTh1", 96)}
    assert got == {"learning rate": "0.0001", "number of blocks / layers": "6", "dropout": "0.9", "hidden size": "512"}


def test_other_dataset_section_and_horizon_are_never_taken():
    got = {d.topic: d.value for d in table_details([Doc("paper", tsmixer())], "TSMixer", "ETTh1", 192)}
    assert got.get("number of blocks / layers") == "4"
    assert table_details([Doc("paper", tsmixer())], "TFT", "Cora", 96) == []


def test_table_row_outranks_argparse_defaults_in_the_spec():
    details = table_details([Doc("paper", tsmixer())], "TSMixer", "ETTh1", 96)
    spec = build_spec(lambda *_: DEFAULTS, details, [Doc("paper", tsmixer()), Doc("repo:run.py@1", RUN_PY)],
                      "TSMixer erreicht MSE 0.361", "ETTh1", numeric_only=True)
    assert {"dropout": 0.9, "hidden": 512, "layers": 6, "lr": 0.0001}.items() <= {i.key: i.number for i in spec.items if i.enforce}.items()
