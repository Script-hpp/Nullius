"""Live-Fund 2026-10-09 (Kipf GCN, Katalogeintrag disc_1609_02907): Die Dateiwahl lieferte keinen gueltigen Pfad, der Repo-Ausschnitt blieb leer
und der Lauf fiel still auf "Nachbau aus Paper" zurueck. Offline mit dem gespeicherten Repo-Ausschnitt von tkipf/gcn (recall cg v1), kein Netz, kein LLM."""

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from repro_agent import repo_source
from repro_agent.repo_source import heuristic_files, read_repo, select_files
from repro_agent.setup_details import Doc
from repro_agent.spec import focused_excerpt, model_files, model_name_of

FIXTURE = Path(__file__).resolve().parents[1] / "benchmark" / "experiments" / "recall_cg_v1_fixtures" / "repos" / "tkipf_gcn__kipf_gcn_2017.json"
URL = "https://github.com/tkipf/gcn"
CLAIM = "GCN (this paper) erreicht 81,5 % Accuracy auf Cora (Standard-Split)"   # so baut catalog.build_paper den Claim-Text
# git ls-files von tkipf/gcn (36 Dateien): die 8 gespeicherten Quelldateien plus Daten, die der Ausschnitt nie liest
LISTING = ["README.md", "requirements.txt", "setup.py", "gcn/__init__.py", "gcn/inits.py", "gcn/layers.py", "gcn/metrics.py", "gcn/models.py",
           "gcn/train.py", "gcn/utils.py", "gcn/data/ind.cora.x", "gcn/data/ind.cora.graph", "gcn/data/ind.cora.test.index"]


class FakeSandbox:
    def __init__(self, files):
        self.files = files

    def base_snapshot(self, *a, **k):
        return "snap"

    def branch(self, snap):
        return "b"

    def run(self, branch, cmd, timeout_s):
        data=('\0'.join(LISTING)+'\0').encode()
        return SimpleNamespace(stdout=f'{len(data)}\n{data.count(0)}\n{hashlib.sha256(data).hexdigest()}\n', stderr="", exit_code=0)

    def get_file(self, branch, path):
        if path == '/tmp/.nullius_repo_listing': return ('\0'.join(LISTING)+'\0').encode()
        rel = path.removeprefix("/repo/")
        if rel == ".head_commit":
            return b"39a4089fe72ad9f055ed6fdb9746abdcfebc4d81\n"
        if rel not in self.files:
            raise FileNotFoundError(rel)
        return self.files[rel].encode()


class ScriptedLLM:
    def __init__(self, *answers):
        self.answers, self.n = list(answers), 0

    def complete(self, role, messages, **kw):
        self.n += 1
        return SimpleNamespace(content=self.answers.pop(0) if self.answers else "")


@pytest.fixture
def sandbox():
    stored = json.loads(FIXTURE.read_text())["files"]
    return FakeSandbox({**stored, "setup.py": "from setuptools import setup\nsetup(name='gcn')\n", "gcn/__init__.py": ""})


def run(sandbox, llm, notes):
    return read_repo(sandbox, llm, URL, CLAIM, "Cora", fixture_dir=Path("/nonexistent-fixture-dir"), focus_model=True, unit_key="disc_1609_02907",
                     model=model_name_of(CLAIM), mode="live", on_note=notes.append)


def test_model_name_drops_table_suffix():
    assert model_name_of(CLAIM) == "GCN"
    assert model_name_of("PatchTST/42 erreicht MSE 0,375 auf ETTh1") == "PatchTST" and model_name_of("SL-GCN erreicht 82,92 % Accuracy auf Cora") == "SL-GCN"


@pytest.mark.parametrize("answer", ["", "I would pick the training loop.", "[]", '["nonexistent/file.py"]'])
def test_unusable_answers_still_give_nonempty_excerpt_with_model_file(sandbox, answer):
    notes = []
    dump = run(sandbox, ScriptedLLM(answer, answer), notes)
    assert "gcn/models.py" in dump.files and "gcn/train.py" in dump.files and "README.md" in dump.files
    assert model_files([Doc(f"repo:{p}", t) for p, t in dump.files.items()], model_name_of(CLAIM))   # Klasse GCN steht im Ausschnitt
    assert not [p for p in dump.files if "/data/" in p]
    ex, names = focused_excerpt([Doc(f"repo:{p}@39a4089", t) for p, t in dump.files.items()], model_name_of(CLAIM), "Cora", [])
    assert ex and any("gcn/models.py" in n for n in names)   # Steckbrief bekommt einen Ausschnitt: Modus bleibt "Nachbau aus Paper und Autorencode"
    assert len(notes) == 1 and "Namensheuristik" in notes[0]   # sichtbar, nie still


def test_second_ask_recovers_before_heuristic(sandbox):
    llm = ScriptedLLM("no list here", 'Pick:\n```json\n["./gcn/train.py", "`gcn/models.py`", "GCN/Layers.py"]\n```')
    notes = []
    dump = run(sandbox, llm, notes)
    assert llm.n == 2 and not notes
    assert list(dump.files)[:3] == ["gcn/models.py", "gcn/train.py", "gcn/layers.py"] or {"gcn/train.py", "gcn/models.py", "gcn/layers.py"} <= set(dump.files)


def test_paths_mentioned_without_json_list_are_used():
    llm = ScriptedLLM("The relevant files are gcn/train.py and gcn/models.py.")
    assert select_files(llm, LISTING, CLAIM, "Cora") == ["gcn/train.py", "gcn/models.py"] and llm.n == 1


def test_good_answer_unchanged_single_call(sandbox):
    llm = ScriptedLLM('["README.md", "gcn/train.py", "gcn/models.py"]')
    notes = []
    dump = run(sandbox, llm, notes)
    assert llm.n == 1 and not notes and {"README.md", "gcn/train.py", "gcn/models.py"} <= set(dump.files)


def test_model_file_added_when_choice_misses_it(sandbox):
    dump = run(sandbox, ScriptedLLM('["README.md", "gcn/utils.py"]'), [])
    assert "gcn/models.py" in dump.files   # nur Hilfsdateien gewaehlt: Modelldatei wird nachgeladen


def test_heuristic_ranks_readme_model_and_train_first_and_skips_data():
    got = heuristic_files(LISTING, "GCN", "Cora")
    assert got[0] == "README.md" and "gcn/models.py" in got[:2] and "gcn/train.py" in got and not [p for p in got if "/data/" in p]
    assert heuristic_files([], "GCN", "Cora") == []


def test_heuristic_finds_model_named_like_slmodel_and_ignores_other_models():
    listing = ["README.md", "src/run.py", "src/slmodel.py", "src/layer.py", "gnnbench/models/monet.py", "gnnbench/models/base_model.py"]
    got = heuristic_files(listing, "SL-GCN", "Cora")
    assert "src/slmodel.py" in got and "gnnbench/models/base_model.py" in got and "gnnbench/models/monet.py" not in got
