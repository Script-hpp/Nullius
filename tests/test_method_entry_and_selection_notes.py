"""Method repos without a model class, and the Cora selection-file contract in paper mode; no provider calls."""
from repro_agent.setup_details import Doc
from repro_agent.spec import model_core_docs, model_files
from repro_agent.tasks import TASKS


def test_selection_contract_keeps_the_full_val_file_instructions():
    # Live 2026-10-14 (BANGS, paper mode): the short note lost format and smoke subset; every candidate guessed the ids
    notes = TASKS["Cora"]().with_independent_selection().val_notes
    assert "first 20 ids of sorted selection_idx" in notes
    assert "header `id,prediction`" in notes and "never val_idx" in notes


RUN_SH = ("#!/usr/bin/env bash\npython3 -W ignore baseline/drgst.py --dataset $data\n"
          "#python3 -W ignore main_node.py --dataset Cora --top 100\n#python3 -W ignore main_node.py --dataset Citeseer\n")
MAIN = "import argparse\nfrom ST_src.banzhaf import select\nparser = argparse.ArgumentParser()\n"
DOCS = [Doc("repo:run.sh@1", RUN_SH), Doc("repo:main_node.py@1", MAIN), Doc("repo:ST_src/banzhaf.py@1", "def select():\n    pass\n"),
        Doc("repo:baseline/drgst.py@1", "x = 1\n")]


def test_training_method_without_a_model_class_uses_the_authors_entry_script():
    # BANGS 2410.09348 is a self-training procedure; no file or class is called BANGS
    assert [d.name for d in model_files(DOCS, "BANGS")] == ["repo:main_node.py@1"]
    assert {d.name for d in model_core_docs(DOCS, "BANGS")} == {"repo:main_node.py@1", "repo:ST_src/banzhaf.py@1"}
    assert model_files(DOCS, "BANGS", entry_fallback=False) == []


def test_a_real_model_file_still_wins_and_ties_are_never_guessed():
    docs = DOCS + [Doc("repo:models/BANGS.py@1", "class BANGS:\n    pass\n")]
    assert [d.name for d in model_files(docs, "BANGS")] == ["repo:models/BANGS.py@1"]
    tie = [Doc("repo:run.sh@1", "python a.py\npython b.py\n"), Doc("repo:a.py@1", ""), Doc("repo:b.py@1", "")]
    assert model_files(tie, "X") == []
