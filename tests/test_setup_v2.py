"""v5 Schritt A (Schalter setup_v2): Pflichtliste, Quellenrang, Geltungsbereich. Attrappen, kein LLM, kein Netz."""

import dataclasses
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import yaml

from repro_agent import setup_details as sd
from repro_agent.dossier import Dossier
from repro_agent.loop import Orchestrator
from repro_agent.report import render_markdown
from repro_agent.setup_details import Doc, SetupDetail, SetupDetailV2

ROOT = Path(__file__).resolve().parents[1]
GCN_CLAIM = "GAT erreicht 83,0 % Accuracy auf Cora (Standard-Split)"
TS_CLAIM = "DLinear erreicht MSE 0,375 auf ETTh1 bei Vorhersagehorizont 96 (Eingabefenster 336)"


class FakeLLM:
    """Antwortet je Prompt aus einer Liste (Funktion prompt -> JSON-Liste); merkt sich alle Prompts."""

    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def complete(self, role, messages, **kw):
        user = messages[-1]["content"]
        self.prompts.append(user)
        return SimpleNamespace(content="thinking...\n" + json.dumps(self.answer(user)))


def item(topic, value, quote, model="all", dataset="all", setting="not stated", scope=None):
    return {"topic": topic, "value": value, "quote": quote,
            "applies_to": {"model": model, "dataset": dataset, "setting": setting}, "scope_quote": scope or quote}


def v2(topic, value="x", source="paper", rank="paper", applies=None, excluded="", optional=False, quote="q"):
    return SetupDetailV2(topic, value, quote, source, applies, rank, optional, excluded)


def test_source_rank():
    assert sd.source_rank("paper", "x") == "paper" and sd.source_rank("repo-readme", "x") == "readme text"
    assert sd.source_rank("repo:scripts/etth1.sh@abc1234", "--learning_rate 1") == "start script"
    assert sd.source_rank("repo:execute_cora.py@abc1234", "lr = 1") == "start script"
    assert sd.source_rank("repo:run_longExp.py@abc1234", "parser.add_argument('--lr', default=1)") == "argparse default"
    assert sd.source_rank("repo:train.py@abc1234", "flags.DEFINE_float('lr', 1, 'x')") == "argparse default"
    assert sd.source_rank("repo:config/gcn.conf.yaml@abc1234", "lr: 1") == "config"
    assert sd.source_rank("repo:layers.py@abc1234", "x = 1") == "code"
    assert [sd.RANKS.index(r) for r in ("start script", "config", "argparse default", "readme text", "paper")] == sorted(
        sd.RANKS.index(r) for r in ("start script", "config", "argparse default", "readme text", "paper"))


def test_verify_v2_scope_needs_own_quote_and_marks_optional():
    docs = [Doc("repo-readme", "You can normalize features to unit norm. We use learning rate 5 for Cora and 7 for Reddit.")]
    raw = [{"doc": "repo-readme", "item": item("normalisation", "unit norm", "You can normalize features to unit norm.")},
           {"doc": "repo-readme", "item": item("learning rate", "5", "learning rate 5 for Cora", model="GAT", dataset="Cora",
                                               scope="learning rate 5 for Cora and 7 for Reddit")},
           {"doc": "repo-readme", "item": item("learning rate", "7", "7 for Reddit", dataset="Reddit", scope="invented scope sentence")}]
    kept, dropped = sd.verify_raw_v2(raw, docs)
    assert dropped == [] and [d.source_rank for d in kept] == ["readme text"] * 3
    assert kept[0].optional is True and kept[1].optional is False                            # "can" im README-Satz
    assert kept[1].applies_to["dataset"] == "Cora" and kept[1].applies_to["quote"].startswith("learning rate 5")
    assert kept[2].applies_to is None                                                       # Zitat nicht im Dokument: Geltungsbereich unbelegt


def test_scope_exclusion_rules():
    ap = lambda m="all", d="all", s="not stated": {"model": m, "dataset": d, "setting": s, "quote": "q"}
    ex = lambda a, claim=GCN_CLAIM, ds="Cora": sd.scope_exclusion(v2("t", applies=a), claim, ds)
    assert ex(ap("GAT", "Cora")) == "" and ex(ap("GAT, GCN", "Cora, Citeseer, Pubmed")) == "" and ex(ap()) == ""
    assert ex(None) == ""                                                                    # Bereich unbekannt: bleibt im Plan
    assert ex(ap("GCN")).startswith("other model")                                           # Angabe fuer anderes Modell
    assert ex(ap("all", "Reddit, PPI", "inductive")).startswith("other dataset")
    assert ex(ap("all", "all", "inductive")) == "inductive setting"
    assert ex(ap("all", "all", "transductive and inductive")) == ""
    assert ex(ap("Informer, Transformer"), TS_CLAIM, "ETTh1").startswith("other model")
    assert ex(ap("DLinear", "ETTh1"), TS_CLAIM, "ETTh1") == "" and ex(ap("Linear models", "ETTh1"), TS_CLAIM, "ETTh1") == ""
    assert ex(ap("all", "Electricity"), TS_CLAIM, "ETTh1").startswith("other dataset")
    assert ex(ap("all", "Cora"), GCN_CLAIM, "Cora-LCC") == ""                                # Cora-LCC zaehlt als Cora


def test_missing_topics_by_family():
    graph = [v2("Learning rate"), v2("dropout rate"), v2("feature normalisation"), v2("adjacency normalization")]
    miss = sd.missing_topics(graph, "graph")
    assert "learning rate" not in miss and "feature normalisation" not in miss and "adjacency normalisation" not in miss
    assert miss == ["weight decay", "hidden size / heads", "epochs", "early stopping rule"]
    assert sd.missing_topics([v2("adjacency normalisation")], "graph")[0] == "feature normalisation"   # Adjazenz erfuellt Merkmals-Thema nicht
    assert sd.missing_topics([v2("dropout", excluded="other model (GCN)")], "graph").count("dropout") == 1   # ausgeschlossen zaehlt nicht
    ts = [v2("input length"), v2("learning rate"), v2("learning rate schedule"), v2("batch size"), v2("epochs"), v2("early stopping patience")]
    assert sd.missing_topics(ts, "forecast") == [] and sd.missing_topics([], "forecast")[0] == "input length"
    assert sd.family_for("Cora") == "graph" and sd.family_for("Cora-LCC") == "graph" and sd.family_for("ETTh1") == "forecast"


def test_target_docs_data_loader_and_start_scripts():
    docs = [Doc("paper", "p"), Doc("repo-readme", "r"), Doc("repo:execute_cora.py@c1", "a"), Doc("repo:models/gat.py@c1", "b"),
            Doc("repo:utils/process.py@c1", "c"), Doc("repo:scripts/etth1.sh@c1", "d"), Doc("repo:config/x.yaml@c1", "e")]
    names = [d.name.split("@")[0] for d in sd.target_docs(docs)]
    assert "repo:models/gat.py" not in names and "paper" not in names
    assert names[:3] == ["repo:execute_cora.py", "repo:scripts/etth1.sh", "repo:config/x.yaml"] and "repo:utils/process.py" in names and "repo-readme" in names


GRAPH_DOCS = [
    Doc("paper", "We use dropout 0.6 and early stopping with patience 100. Features are row-normalized in the original code for GAT."),
    Doc("repo:execute_cora.py@abc1234", "lr = 0.005  # learning rate\nl2_coef = 0.0005  # weight decay\nhid_units = 8  # hidden units\nnb_epochs = 100000"),
    Doc("repo:utils/process.py@abc1234", "def preprocess_features(features):\n    \"\"\"Row-normalize feature matrix\"\"\"\n"),
]


def first_pass(user):
    """Erste Extraktion: nur Paper-Absatz; Datenlade- und Skriptdateien liefern in der ersten Runde nichts."""
    if "Earlier passes found no statement" in user:
        out = []
        if "preprocess_features" in user:
            out.append(item("feature normalisation", "row-normalized", "Row-normalize feature matrix", model="GAT", dataset="Cora"))
        if "lr = " in user:
            out += [item("learning rate", "0.005", "lr = 0.005  # learning rate", model="GAT", dataset="Cora"),
                    item("weight decay", "0.0005", "l2_coef = 0.0005  # weight decay", model="GAT", dataset="Cora"),
                    item("hidden units", "8", "hid_units = 8  # hidden units", model="GAT", dataset="Cora"),
                    item("epochs", "100000", "nb_epochs = 100000", model="GAT", dataset="Cora")]
        return out
    if "We use dropout" in user:
        return [item("dropout", "0.6", "We use dropout 0.6", model="GAT", dataset="Cora"),
                item("early stopping patience", "100", "early stopping with patience 100", model="GAT", dataset="Cora"),
                item("dropout", "0.6", "dropout 0.6", model="GCN", dataset="Cora")]   # fuer ein anderes Modell
    return []


def test_two_pass_extraction_fills_missing_topics_from_loader_and_start_script():
    llm = FakeLLM(first_pass)
    details, dropped, missing, info = sd.extract_setup_details_v2(llm, GRAPH_DOCS, GCN_CLAIM, "Cora")
    topics = {d.topic: d for d in details if not d.excluded}
    assert {"dropout", "early stopping patience", "feature normalisation", "learning rate", "weight decay", "hidden units", "epochs"} <= set(topics)
    assert topics["learning rate"].source_rank == "start script" and topics["feature normalisation"].source == "repo:utils/process.py@abc1234"
    assert info["missing_first"][0] == "feature normalisation" and info["added"] == 5 and info["targeted_docs"]
    assert missing == ["adjacency normalisation"]                                           # nicht gefunden: gueltiges Ergebnis
    assert any(d.excluded.startswith("other model") for d in details)                       # GCN-Angabe: im Dossier, nicht im Plan
    targeted = [p for p in llm.prompts if "Earlier passes found no statement" in p]
    assert targeted and all("adjacency normalisation" in p and "feature normalisation" in p for p in targeted)
    assert not any("We use dropout" in p for p in targeted)                                 # gezielt nur Datenlade-/Startdateien, nicht das Paper


def test_no_second_pass_when_nothing_missing():
    full = [item(t, "v", f"quote {i}") for i, (t, _, _) in enumerate(sd.MANDATORY["graph"])]
    docs = [Doc("paper", "\n\n".join(f"quote {i}" for i in range(len(full))))]
    llm = FakeLLM(lambda user: full)
    details, _, missing, info = sd.extract_setup_details_v2(llm, docs, GCN_CLAIM, "Cora")
    assert missing == [] and info["added"] == 0 and not any("Earlier passes" in p for p in llm.prompts)


def test_quote_check_still_drops_invented_values():
    llm = FakeLLM(lambda user: [item("learning rate", "0.5", "learning rate of 0.5")])
    details, dropped, missing, _ = sd.extract_setup_details_v2(llm, [Doc("paper", "learning rate of 0.01")], GCN_CLAIM, "Cora")
    assert details == [] and dropped >= 1 and "learning rate" in missing


def test_prompt_format_rank_marks_and_conflicts():
    d_script = v2("learning rate", "0.005", "repo:etth1.sh@c", "start script", quote="--learning_rate 0.005")
    d_default = v2("learning rate", "0.0001", "repo:run.py@c", "argparse default", quote="default=0.0001")
    d_readme = v2("normalisation", "unit norm", "repo-readme", "readme text", optional=True)
    d_other = v2("dropout", "0.8", "paper", "paper", excluded="other model (GCN)")
    details = [d_default, d_other, d_script, d_readme]
    text = sd.format_for_prompt_v2(details)
    assert "[start script]" in text and "[argparse default]" in text and "optional advice" in text and "repository default" in text
    assert "dropout" not in text                                                            # ausgeschlossen: nicht im Plan
    conf = sd.format_conflicts_v2(details)
    assert "0.005" in conf and "0.0001" in conf and conf.index("0.005") < conf.index("0.0001") and "highest-ranked source (start script)" in conf
    assert sd.format_conflicts_v2([d_script, dataclasses.replace(d_default, value="0.005")]) == ""   # gleicher Wert: kein Konflikt
    assert sd.format_for_prompt_v2([d_other]) == "" and sd.format_for_prompt_v2([SetupDetail("a", "b", "c")]) == ""


def test_switch_default_off_and_v1_prompt_unchanged():
    assert inspect.signature(Orchestrator.__init__).parameters["setup_v2"].default is False
    llm = FakeLLM(lambda user: [])
    sd._extract_chunk(llm, "text", "claim", "Cora")
    assert 'Answer with a JSON list of {"topic": short label, "value": the setting, "quote": a VERBATIM excerpt' in llm.prompts[0]
    assert "applies_to" not in llm.prompts[0] and "scope_quote" not in llm.prompts[0]
    sd._extract_chunk(llm, "text", "claim", "Cora", v2=True)
    assert "applies_to" in llm.prompts[1] and "scope_quote" in llm.prompts[1]
    o = Orchestrator.__new__(Orchestrator)
    o.details = [SetupDetail("learning rate", "0.01", "lr 0.01", "paper")]
    assert o._details_prompt() == sd.format_for_prompt(o.details)                          # ohne Schalter wie bisher
    o.setup_v2 = True
    o.details = [v2("learning rate", "0.01", "paper", "paper", quote="lr 0.01"), v2("dropout", "0.8", excluded="other model (GCN)")]
    assert "[paper]" in o._details_prompt() and "dropout" not in o._details_prompt()


def test_prompts_contain_no_number_from_the_soll_list():
    """Kein Wert aus der Soll-Liste in Prompt-Texten oder Pflichtliste; sie nennt nur Themen."""
    soll = yaml.safe_load((ROOT / "benchmark" / "experiments" / "v5_step_a_soll.yaml").read_text())["soll"]
    numbers = {n for pts in soll.values() for v in pts.values() for n in re.findall(r"\d+(?:\.\d+)?(?:e-?\d+)?", str(v))}
    llm = FakeLLM(lambda user: [])
    sd._extract_chunk(llm, "EXCERPT", "claim", "dataset", v2=True)
    sd._targeted_chunk(llm, "EXCERPT", "claim", "dataset", [n for fam in sd.MANDATORY.values() for n, _, _ in fam])
    texts = llm.prompts + [sd.format_for_prompt_v2([v2("t", "v")]).split("\n")[0], sd.format_conflicts_v2([])]
    from repro_agent import repo_source

    class Cap:
        def complete(self, role, messages, **kw):
            texts.append(messages[-1]["content"])
            return SimpleNamespace(content="[]")
    repo_source.select_files(Cap(), ["a.py"], "claim", "Cora", focus_model=True)
    prompt_text = " ".join(texts) + " " + " ".join(n for fam in sd.MANDATORY.values() for n, _, _ in fam)
    prompt_text = re.sub(r"\bmax 300 chars\b|\bup to \d+ files\b", "", prompt_text)            # feste Laengengrenzen, keine Soll-Werte
    for n in numbers:
        assert not re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", prompt_text), n


def test_select_files_focus_model_only_in_v2():
    seen = []

    class Cap:
        def complete(self, role, messages, **kw):
            seen.append(messages[-1]["content"])
            return SimpleNamespace(content="[]")
    from repro_agent.repo_source import select_files
    select_files(Cap(), ["a.py"], "GAT claim", "Cora")
    select_files(Cap(), ["a.py"], "GAT claim", "Cora", focus_model=True)
    assert "skip files that only concern other models" not in seen[0] and "skip files that only concern other models" in seen[1]


def test_dossier_and_report_show_exclusions_and_missing():
    d = Dossier(paper_id="p", claims=[], setup_details=[dataclasses.asdict(v2("dropout", "0.8", excluded="other model (GCN)"))],
                setup_missing=["adjacency normalisation"])
    md = render_markdown(d)
    assert "not in the plan: other model (GCN)" in md and "Not found in the sources: adjacency normalisation" in md
    assert json.loads(json.dumps(dataclasses.asdict(d)))["setup_missing"] == ["adjacency normalisation"]
    assert Dossier(paper_id="p", claims=[]).setup_missing == []
