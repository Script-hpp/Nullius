"""Repo-Wahl mit Pruefung, Unterordner in Sammel-Repos, Hyperparameter-Tabellen im Anhang, Modellwahl-Regel (Live-Fund TSMixer 2303.06053,
2026-10-09: M5-Wettbewerbs-Repo als "Autorencode", batch_size=32 daraus erzwungen, Anhang-Tabellen nicht gelesen, "each model with" als Modellwahl).
Attrappen und gespeicherte Auszuege, kein Netz, kein LLM."""

from types import SimpleNamespace

import pytest

from repro_agent import loop as loop_mod
from repro_agent import repo_source
from repro_agent.hparam_table import table_details
from repro_agent.repo_source import (
    find_repo_links, find_repo_url, parse_github_url, read_repo, repo_evidence, repo_status, subdir_listing,
)
from repro_agent.sandbox.base import SandboxSetupError
from repro_agent.setup_details import MAX_TEXT_CHARS, Doc, extraction_text
from repro_agent.spec import other_model_selected
from test_tavily_mode import _make

TITLE = "TSMixer: An All-MLP Architecture for Time Series Forecasting"
ARXIV = "2303.06053"
# Auszug aus dem gespeicherten Paper-Text (runs/disc_2303_06053): Abstract-Link mit PDF-Leerzeichen, Fussnoten am Ende des Anhangs
PAPER_HEAD = ("Published in Transactions on Machine Learning Research (09/2023) TSMixer: An All-MLP Architecture for Time Series Fore-casting "
              "Si-An Chen Chun-Liang Li Google Cloud AI Research Abstract Real-world time-series datasets are often multivariate. "
              "The implementation is available at: https://github.com/google-research/google-research/tree/master/ tsmixer.\n1 Introduction ")
PAPER_FOOT = ("We refer to the example script in GluonTS (Alexandrov et al., 2020)2 and the repository of the third place solution3 in the "
              "competition to implement our basic feature engineering. "
              "TSMixer 96 0.0001 6 0.9 512 relu 192 0.001 4 0.9 256 relu 1https://www.kaggle.com/competitions/m5-forecasting-accuracy/data "
              "2https://github.com/awslabs/gluonts/blob/dev/examples/m5_gluonts_template.ipynb 3https://github.com/devmofl/M5_Accuracy_3rd 21")
M5_URL = "https://github.com/devmofl/M5_Accuracy_3rd"
M5_README = "# M5 Accuracy 3rd place solution\nPyTorch-ts based LightGBM and DeepAR ensembles for the Kaggle M5 forecasting competition. " * 3
OFFICIAL = "https://github.com/google-research/google-research/tree/HEAD/tsmixer"
OFFICIAL_LINK = "https://github.com/google-research/google-research/tree/master/tsmixer"   # E16: Ref des Paperlinks bleibt erhalten
OFFICIAL_README = "# TSMixer\nTSMixer: An All-MLP Architecture for Time Series Forecasting (arXiv:2303.06053). Run train.py --model tsmixer."
UNOFFICIAL = "https://github.com/giuliomattolin/tsmixer-pytorch"
UNOFFICIAL_README = "# tsmixer-pytorch\nUnofficial PyTorch re-implementation of TSMixer."


# ---- A: Links im Paper-Text ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("", None),
    ("This paper contains no implementation link.", None),
    ("matrix where aij \n\n1https://github.com/Tiiiger/SGC \n\nvi and", "https://github.com/Tiiiger/SGC"),
    ("Code is available at https://github.com/lss-1138/SegRNN", "https://github.com/lss-1138/SegRNN"),
])
def test_repo_links_come_only_from_the_supplied_paper(text, expected):
    assert find_repo_url(text) == expected
    assert all(link.target in text for link in find_repo_links(text))


def test_paper_link_with_subfolder_beats_footnote_repo():
    text = PAPER_HEAD + "x " * 3000 + PAPER_FOOT
    links = find_repo_links(text)
    assert [(l.strength, l.target) for l in links][0] == ("strong", OFFICIAL_LINK)
    assert links[0].subdir == "tsmixer" and links[0].root == "https://github.com/google-research/google-research"
    assert find_repo_url(text) == OFFICIAL_LINK
    assert all("gluonts" not in l.target for l in links)   # Blob-Link auf ein Notebook ist kein Repo


def test_footnote_repo_without_strong_link_is_only_a_weak_candidate_and_infra_repos_are_ignored():
    text = "see the implementation of the third place solution 3 https://github.com/devmofl/M5_Accuracy_3rd . Report errors at https://github.com/arXiv/html_feedback/issues"
    assert [l.strength for l in find_repo_links(text)] == ["weak"]
    assert parse_github_url("https://github.com/brucemiller/LaTeXML/issues") is None
    assert parse_github_url("https://github.com/awslabs/gluonts/blob/dev/examples/x.ipynb") is None
    assert parse_github_url("https://github.com/a/b/tree/main/sub/dir") == ("a", "b", "sub/dir")
    assert parse_github_url("https://github.com/a/b.git") == ("a", "b", "")


def test_evidence_and_status():
    assert repo_evidence(M5_README, M5_URL, TITLE, ARXIV, "TSMixer") == ("", "")   # M5-README nennt weder Modell, Titel noch arXiv-ID
    assert repo_evidence(OFFICIAL_README, OFFICIAL, TITLE, ARXIV, "TSMixer")[0] == "paper"
    assert repo_evidence("Implements the model.", OFFICIAL, "Other title", "9999.99999", "TSMixer")[0] == "model"   # Pfad "tsmixer"
    assert repo_status(UNOFFICIAL, UNOFFICIAL_README, False, "model")[0] == "unofficial"
    assert repo_status(OFFICIAL, OFFICIAL_README, True, "paper")[0] == "official"
    # kein Link im Paper, README belegt das Paper, Besitzer passt zu einem Autor: offiziell; sonst nicht
    r = f"Code for {TITLE} (arXiv:{ARXIV})"
    assert repo_status("https://github.com/chenxx/tsmixer", r, False, "paper", PAPER_HEAD, ["Si-An Chen"])[0] == "official"
    assert repo_status("https://github.com/someone/tsmixer", r, False, "paper", PAPER_HEAD, ["Si-An Chen"])[0] == "unofficial"


# ---- A: Orchestrator-Wahl --------------------------------------------------------------------------------------------------
def orch(monkeypatch, text, readmes, search=None, title=TITLE):
    o = _make(monkeypatch, graph_repair=True, repo_verify=True, tavily_mode="fixture")
    o.paper_text = SimpleNamespace(text=text, arxiv_id=ARXIV)
    o._paper_title = title
    o._sources, o.events, o.on_event = [], None, None
    seen = {"fetched": [], "searched": 0, "emitted": []}
    o._emit = lambda type, **kw: seen["emitted"].append((type, kw))

    def fetch(url, get_tavily, mode="fixture"):
        seen["fetched"].append(url)
        return readmes.get(url, "")

    def search_fn(*a, **k):
        seen["searched"] += 1
        return list(search or [])

    monkeypatch.setattr(loop_mod, "fetch_readme", fetch)
    monkeypatch.setattr(loop_mod, "search_repo_candidates", search_fn)
    return o, seen


CLAIM = "TSMixer erreicht 0,361 MSE auf ETTh1 (Horizont 96)"


def test_wrong_repo_m5_is_rejected_and_unofficial_is_not_author_code(monkeypatch):
    """M5-Repo war die Fussnote des Papers: README nennt weder Modell noch Titel noch arXiv-ID -> abgelehnt. Die Suche liefert nur ein
    inoffizielles Repo -> nicht als Autorencode, Rueckfall-Grund benennt es."""
    text = "x " * 500 + " the repository of the third place solution3 in the competition 3https://github.com/devmofl/M5_Accuracy_3rd 21"
    o, seen = orch(monkeypatch, text, {M5_URL: M5_README, UNOFFICIAL: UNOFFICIAL_README}, search=[UNOFFICIAL])
    url, readme, status = o._resolve_repo(CLAIM, lambda: None)
    assert url is None and readme == "" and status == {}
    verdicts = {kw["url"]: (kw["verdict"], kw["status"]) for t, kw in seen["emitted"] if t == "repo_candidate"}
    assert verdicts[M5_URL][0] == "rejected" and verdicts[UNOFFICIAL] == ("accepted", "unofficial")
    assert "unofficial" in o._repo_fallback_reason and UNOFFICIAL in o._repo_fallback_reason
    assert any(t == "warning" and "Inoffizielles Repo" in kw["text"] for t, kw in seen["emitted"])
    o._base_modes = {"code_grounded": True, "io_helper": True, "spec_mode": False}
    o.code_grounded = o.io_helper = o.spec_mode = True
    mode, reason = o._resolve_mode(None)   # kein Autorencode-Modus ohne offizielles Repo
    assert mode == loop_mod.MODE_PAPER and "unofficial" in reason


def test_paper_link_is_preferred_over_search_and_needs_no_search_call(monkeypatch):
    o, seen = orch(monkeypatch, PAPER_HEAD + "x " * 3000 + PAPER_FOOT, {OFFICIAL_LINK: OFFICIAL_README, M5_URL: M5_README}, search=[UNOFFICIAL])
    url, readme, status = o._resolve_repo(CLAIM, lambda: None)
    assert url == OFFICIAL_LINK and readme == OFFICIAL_README
    assert status["status"] == "official" and status["source"] == "paper"
    assert seen["searched"] == 0 and seen["fetched"] == [OFFICIAL_LINK]   # kein weiterer Lesezugriff, keine Suche
    assert [kw["verdict"] for t, kw in seen["emitted"] if t == "repo_candidate"] == ["accepted"]


def test_official_search_hit_beats_unofficial_one_and_search_uses_several_hits(monkeypatch):
    official_hit = "https://github.com/chenxx/tsmixer"
    o, seen = orch(monkeypatch, "no code link in this text", {
        UNOFFICIAL: UNOFFICIAL_README, official_hit: f"Official code for {TITLE} arXiv:{ARXIV}", M5_URL: M5_README},
        search=[UNOFFICIAL, official_hit])
    o.paper_text.text = "Si-An Chen et al. no code link. " * 5
    url, readme, status = o._resolve_repo(CLAIM, lambda: None)
    assert url == official_hit and status["status"] == "official" and status["source"] == "search"
    assert seen["fetched"] == [UNOFFICIAL, official_hit]


def test_search_hit_with_unreadable_readme_is_rejected(monkeypatch):
    o, seen = orch(monkeypatch, "no link", {}, search=["https://github.com/x/y"])
    assert o._resolve_repo(CLAIM, lambda: None)[0] is None
    assert [kw["verdict"] for t, kw in seen["emitted"] if t == "repo_candidate"] == ["rejected"]


def test_search_candidates_collects_all_matching_hits_and_keeps_subfolders(tmp_path):
    class T:
        def find_material(self, title, domains=None):
            return {"results": [
                {"url": "https://github.com/x/other", "content": "unrelated"},
                {"url": "https://github.com/giuliomattolin/tsmixer-pytorch", "content": "Unofficial TSMixer", "title": "tsmixer-pytorch"},
                {"url": "https://github.com/google-research/google-research/tree/master/tsmixer", "content": f"{TITLE}", "title": "tsmixer"}]}
    got = repo_source.search_repo_candidates(TITLE, ARXIV, lambda: T(), fixture_dir=tmp_path, model="TSMixer")
    assert got == [OFFICIAL, UNOFFICIAL]   # Titel-Treffer zuerst, Modellname-Treffer danach, Unterordner erhalten
    assert repo_source.search_repo_candidates(TITLE, ARXIV, lambda: None, fixture_dir=tmp_path) == got   # aus der Fixture


# ---- A: Unterordner-Klon ---------------------------------------------------------------------------------------------------
class SubSandbox:
    def __init__(self):
        self.setup, self.cmds = [], []

    def base_snapshot(self, image, cmds, label=None):
        self.setup = cmds
        return "snap"

    def branch(self, snap):
        return "b"

    def run(self, branch, cmd, timeout_s):
        self.cmds.append(cmd)
        return SimpleNamespace(stdout="100644 blob abc 120\ttsmixer/train.py\n100644 blob def 9000000\ttsmixer/data/big.npy\n100644 blob ghi 50\ttsmixer/README.md",
                               stderr="", exit_code=0)

    def get_file(self, branch, path):
        if path.endswith(".head_commit"):
            return b"abcdef1234567\n"
        return b"x = 1\n"


class OneLLM:
    def complete(self, role, messages, **kw):
        return SimpleNamespace(content='["tsmixer/train.py", "tsmixer/README.md"]')


def test_subfolder_clone_is_sparse_and_listing_is_limited(tmp_path):
    sb = SubSandbox()
    dump = read_repo(sb, OneLLM(), OFFICIAL, CLAIM, "ETTh1", fixture_dir=tmp_path, mode="live")
    clone = " ".join(sb.setup)
    assert "--sparse" in clone and "--filter=blob:none" in clone and "sparse-checkout set tsmixer" in clone
    assert "https://github.com/google-research/google-research " in clone + " " and "/tree/" not in clone   # Wurzel-URL, nicht die Baum-URL
    assert "git ls-tree -r -l HEAD -- tsmixer" in sb.cmds[0]
    assert sorted(dump.files) == ["tsmixer/README.md", "tsmixer/train.py"] and dump.subdir == "tsmixer" and dump.n_tracked == 2   # big.npy (9 MB) fehlt


def test_subfolder_listing_size_limits():
    out = "100644 blob a 10\tsub/a.py\n100644 blob b 5000000\tsub/w.bin"
    assert subdir_listing(out) == ["sub/a.py"]
    with pytest.raises(SandboxSetupError):
        subdir_listing("100644 blob a 150000000\tsub/a.py\n100644 blob b 150000000\tsub/b.py")
    assert repo_source.clone_plan("https://github.com/a/b/tree/main/x;rm -rf y") == ("https://github.com/a/b", "x")   # Shell-Zeichen enden den Pfad


# ---- B: Hyperparameter-Tabellen --------------------------------------------------------------------------------------------
TABLE_FLAT = ("E Hyperparameters Table 8: Hyerparamter tuning spaces and best configurations for TSMixer and TFT on long-term forecasting benchmarks. "
              "ETTh1 Search space Learing rate Blocks Dropout Hidden size Heads Model T 0.001, 0.0001 1, 2, 4, 6, 8 0.1, 0.3, 0.5, 0.7, 0.9 8, 16, 32, 64 4, 8 "
              "TSMixer 96 0.0001 6 0.9 512 relu 192 0.001 4 0.9 256 relu 336 0.001 4 0.9 256 relu 720 0.001 2 0.9 64 relu "
              "TFT 96 0.001 0.3 192 0.001 0.5 "
              "ETTh2 Search space Learing rate Blocks Dropout Hidden size Heads Model T 0.001, 0.0001 1, 2 TSMixer 96 0.001 2 0.7 64 relu")


def test_flattened_appendix_table_is_read_with_quotes_and_right_row():
    got = {d.topic: d for d in table_details([Doc("paper", TABLE_FLAT)], "TSMixer", "ETTh1")}
    assert {t: d.value for t, d in got.items()} == {"learning rate": "0.0001", "number of blocks / layers": "6", "dropout": "0.9", "hidden size": "512"}
    q = got["learning rate"].quote
    assert q == "Learing rate Blocks Dropout Hidden size Heads ... TSMixer 96 0.0001"   # Spaltenkoepfe + Zeile bis zum Wert, woertlich
    assert got["dropout"].quote.endswith("TSMixer 96 0.0001 6 0.9") and got["learning rate"].applies_to["dataset"] == "ETTh1"
    eth2 = {d.topic: d.value for d in table_details([Doc("paper", TABLE_FLAT)], "TSMixer", "ETTh2")}
    assert eth2["learning rate"] == "0.001" and eth2["dropout"] == "0.7"   # Datensatz-Zeile korrekt zugeordnet, nicht die erste Tabelle
    assert table_details([Doc("paper", TABLE_FLAT)], "TSMixer", "Cora") == []   # Datensatz nicht belegt: nichts raten
    assert table_details([Doc("paper", TABLE_FLAT)], "PatchTST", "ETTh1") == []


def test_markdown_table_and_column_mismatch():
    md = ("Table 5: Hyperparameters on ETTh1\n| Model | T | Learning rate | Batch size | Dropout |\n|---|---|---|---|---|\n"
          "| DLinear | 96 | 0.005 | 32 | 0.1 |\n| | 192 | 0.005 | 32 | 0.1 |\n")
    # Kopfzeile steht hinter "Model | T": die Spaltennamen beginnen bei "Learning rate"
    got = {d.topic: d.value for d in table_details([Doc("paper", md)], "DLinear", "ETTh1")}
    assert got == {"learning rate": "0.005", "batch size": "32", "dropout": "0.1"}
    bad = md.replace("| DLinear | 96 | 0.005 | 32 | 0.1 |", "| DLinear | 96 | 0.005 | 32 | 0.1 | 7 |")
    assert table_details([Doc("paper", bad)], "DLinear", "ETTh1") == []   # mehr Werte als Spalten: keine Zuordnung


def test_appendix_tail_beyond_text_limit_is_kept():
    text = "intro " * (MAX_TEXT_CHARS // 6 + 500) + " E Hyperparameters Table 8: Hyerparamter tuning spaces " + TABLE_FLAT
    assert len(text) > MAX_TEXT_CHARS
    assert "TSMixer 96 0.0001 6 0.9 512" in extraction_text(text) and "TSMixer 96 0.0001 6 0.9 512" not in text[:MAX_TEXT_CHARS]
    assert extraction_text("short text") == "short text"


# ---- C: "each model with" ist keine Modellwahl -------------------------------------------------------------------------------
@pytest.mark.parametrize("quote", [
    "We train each model with a maximum 100 epochs and do early stopping if the validation loss is not improved after 5 epochs.",
    "each model w/ dropout 0.5 is trained for 30 epochs", "For every model and dataset we use batch size 32", "the model is trained with lr 0.01",
])
def test_generic_words_after_model_are_not_another_model(quote):
    assert other_model_selected(quote, "TSMixer") == ""


@pytest.mark.parametrize("quote,expected", [
    ("model_name = 'DLinear'", "DLinear"), ("python run.py --model gat --lr 0.01", "gat"), ("args.model == DLinear", "DLinear"),
    ("model: PatchTST", "PatchTST"), ("we select model DLinear for ETTh1", "DLinear"),
])
def test_real_model_selections_still_detected(quote, expected):
    assert other_model_selected(quote, "TSMixer") == expected
    assert other_model_selected(quote, expected) == ""


def test_gather_docs_returns_status_and_root_for_subfolder_repo(monkeypatch):
    o, seen = orch(monkeypatch, PAPER_HEAD + "x " * 100, {OFFICIAL_LINK: OFFICIAL_README})
    o.paper_only = False
    o._tavily_calls = 0
    monkeypatch.setattr(loop_mod, "read_repo", lambda *a, **k: repo_source.RepoDump(
        url=OFFICIAL_LINK, commit="abcdef1234567", files={"tsmixer/train.py": "x = 1"}, n_tracked=1, subdir="tsmixer", ref="master"))
    docs, info = o._gather_docs(CLAIM, "ETTh1")
    assert [d.name for d in docs][:2] == ["paper", "repo-readme"] and docs[-1].name == "repo:tsmixer/train.py@abcdef1"
    assert info["url"] == OFFICIAL_LINK and info["root"] == "https://github.com/google-research/google-research" and info["subdir"] == "tsmixer"
    assert info["status"] == "official" and info["source"] == "paper"
    assert [s["kind"] for s in o._sources] == ["readme"]   # README als Quelle festgehalten


def test_gather_docs_without_official_repo_reads_no_repo_files(monkeypatch):
    o, seen = orch(monkeypatch, "no code link " * 50, {UNOFFICIAL: UNOFFICIAL_README}, search=[UNOFFICIAL])
    o.paper_only = False
    o._tavily_calls = 0
    monkeypatch.setattr(loop_mod, "read_repo", lambda *a, **k: pytest.fail("unofficial repo must not be cloned or read"))
    docs, info = o._gather_docs(CLAIM, "ETTh1")
    assert info is None and [d.name for d in docs] == ["paper"]
