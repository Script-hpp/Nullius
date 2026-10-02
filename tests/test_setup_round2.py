"""Auftrag 32: Tests zu Runde 2 an Rolle 4 (Schalter setup_v2). Attrappen, kein LLM, kein Netz, kein Tavily.

Tests mit xfail(strict=True) zeigen Regressionen oder Abweichungen von der Vorab-Festlegung (v5_step_a.md); sie werden rot (XPASS),
sobald der Fehler in src/ behoben ist. Beispielwerte sind erfunden."""

import json
import re
from types import SimpleNamespace

import pytest

from repro_agent import setup_details as sd
from repro_agent.setup_details import Doc, SetupDetail, SetupDetailV2

CLAIM = "GAT erreicht 83,0 % Accuracy auf Cora (Standard-Split)"
NEUTRAL_CLAIM = "GAT erreicht Accuracy auf Cora"
SGC_SENTENCE = ("Practical advice: it is often very helpful to normalize the features to have zero mean "
                "with standard deviation one")


class FakeLLM:
    def __init__(self, answer=lambda user: []):
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


def applies(model, dataset="all", setting="not stated"):
    return {"model": model, "dataset": dataset, "setting": setting, "quote": "q"}


# ---- Rangfolge: Paper vor argparse-Default und README-Text -------------------------------------------------------------------

def test_paper_ranks_above_argparse_default_and_readme_text():
    assert sd.RANKS.index("paper") < sd.RANKS.index("readme text")      # ohne Runde 2 stand "paper" an letzter Stelle
    assert sd.RANKS.index("paper") < sd.RANKS.index("argparse default")


def test_conflict_paper_value_wins_over_readme_text():
    paper = v2("dropout", "0.6", "paper", "paper", quote="dropout 0.6")
    readme = v2("dropout", "0.5", "repo-readme", "readme text", quote="dropout 0.5")
    text = sd.format_conflicts_v2([readme, paper])
    assert "highest-ranked source (paper)" in text and text.index("0.6") < text.index("0.5")


def test_conflict_paper_value_wins_over_argparse_default():
    paper = v2("learning rate", "0.01", "paper", "paper", quote="learning rate 0.01")
    default = v2("learning rate", "0.001", "repo:train.py@c", "argparse default", quote="default=0.001")
    assert "highest-ranked source (paper)" in sd.format_conflicts_v2([default, paper])


def test_start_script_and_config_still_above_paper():
    assert sd.RANKS.index("start script") < sd.RANKS.index("config") < sd.RANKS.index("paper")


@pytest.mark.xfail(strict=True, reason="Abweichung von V2 (v5_step_a.md): argparse default muss vor README-Fliesstext stehen; "
                                       "RANKS setzt argparse default ans Ende, hinter readme text")
def test_rank_order_follows_preregistration_v2():
    order = ("start script", "config", "argparse default", "readme text")
    assert [sd.RANKS.index(r) for r in order] == sorted(sd.RANKS.index(r) for r in order)


# ---- Akronym-Abgleich Modellname <-> Claim-Methode --------------------------------------------------------------------------

def test_acronym_of_full_model_name_matches_claim_method():
    d = v2("learning rate", "0.2", applies=applies("Simplifying Graph Convolution"))
    assert sd.scope_exclusion(d, "SGC erreicht 81,0 % auf Cora", "Cora") == ""      # vorher: "other model"


def test_other_model_still_excluded_after_acronym_matching():
    d = v2("learning rate", "0.005", applies=applies("GAT"))
    assert sd.scope_exclusion(d, "GCN erreicht 81,5 % auf Cora", "Cora").startswith("other model")
    d = v2("learning rate", "0.2", applies=applies("Simplifying Graph Convolution"))
    assert sd.scope_exclusion(d, "GAT erreicht 83,0 % auf Cora", "Cora").startswith("other model")


def test_acronym_does_not_apply_without_erreicht_claim_format():
    d = v2("learning rate", "0.2", applies=applies("Simplifying Graph Convolution"))
    assert sd.scope_exclusion(d, "SGC reaches 81.0 on Cora", "Cora").startswith("other model")   # claim_method leer: alter Weg


def test_path_model_exclusion_uses_acronym_of_file_name():
    paper = ("We compare a, Simplifying baseline. We use Simplifying nets and a, Simplifying graph. "
             "Also a, Convolution and the Convolution, a, Convolution step. We use a, Graph and use Graph and a, Graph.")
    d = v2("lr", "0.2", "repo:config/simplifying_graph_convolution.yaml@abc1234", "config")
    assert sd.path_model_exclusion(d, "SGC erreicht 81,0 % auf Cora", paper) == ""
    assert sd.path_model_exclusion(d, "GAT erreicht 83,0 % auf Cora", paper).startswith("other model")


@pytest.mark.xfail(strict=True, reason="Akronym wird aus den gefilterten Woertern gebildet ('network' ist generisch): "
                                       "'Graph Convolutional Network' ergibt 'gc', nicht 'gcn', die GCN-Angabe wird ausgeschlossen")
def test_acronym_with_generic_word_in_full_name():
    d = v2("learning rate", "0.01", applies=applies("Graph Convolutional Network"))
    assert sd.scope_exclusion(d, "GCN erreicht 81,5 % auf Cora", "Cora") == ""


@pytest.mark.xfail(strict=True, reason="Teilstring-Abgleich 'claim_method in model': GCNII-Angabe gilt als GCN-Angabe")
def test_substring_of_other_model_name_is_not_a_match():
    d = v2("layers", "64", applies=applies("GCNII"))
    assert sd.scope_exclusion(d, "GCN erreicht 81,5 % auf Cora", "Cora").startswith("other model")


# ---- Paper im gezielten zweiten Durchgang -----------------------------------------------------------------------------------

PAPER_TEXT = "Node features are row-normalized before training. We use dropout 0.6."


def test_target_docs_put_paper_first_and_keep_readme():
    docs = [Doc("repo-readme", "r"), Doc("repo:execute_cora.py@c1", "a"), Doc("paper", "p")]
    names = [d.name.split("@")[0] for d in sd.target_docs(docs)]
    assert names[0] == "paper" and "repo-readme" in names


def test_second_pass_finds_topic_stated_only_in_paper():
    def answer(user):
        if "Earlier passes found no statement" in user and "row-normalized" in user:
            return [item("feature normalisation", "row-normalized", "Node features are row-normalized before training",
                         model="GAT", dataset="Cora")]
        return []                                                       # erste Extraktion findet nichts
    llm = FakeLLM(answer)
    details, _, missing, info = sd.extract_setup_details_v2(llm, [Doc("paper", PAPER_TEXT)], CLAIM, "Cora")
    assert info["missing_first"][0] == "feature normalisation" and info["targeted_docs"] == ["paper"]
    assert any(d.topic == "feature normalisation" and d.source == "paper" for d in details)
    assert "feature normalisation" not in missing


def test_second_pass_prompts_name_missing_topics_and_carry_paper_text():
    llm = FakeLLM()
    sd.extract_setup_details_v2(llm, [Doc("paper", PAPER_TEXT)], CLAIM, "Cora")
    targeted = [p for p in llm.prompts if "Earlier passes found no statement" in p]
    assert targeted and all("EXCERPT from the paper" in p and "row-normalized" in p for p in targeted)


# ---- gezielte Fragen je Pflichtthema (erste Extraktion) ---------------------------------------------------------------------

def topic_of(prompt):
    return re.search(r"Topic to extract: (.+)", prompt).group(1).strip()


def test_first_pass_asks_one_question_per_mandatory_topic_graph():
    llm = FakeLLM()
    sd.collect_middle_ground(llm, [Doc("paper", PAPER_TEXT)], CLAIM, "Cora", "Some Paper")
    asked = [topic_of(p) for p in llm.prompts]
    expected = [name for name, _, _ in sd.MANDATORY["graph"]] + ["other settings"]
    assert sorted(asked) == sorted(expected)                            # genau eine Frage je Thema und Dokument


def test_first_pass_asks_forecast_topics_for_time_series():
    llm = FakeLLM()
    sd.collect_middle_ground(llm, [Doc("paper", "We train with batch size 32.")], "DLinear erreicht MSE auf ETTh1", "ETTh1")
    asked = {topic_of(p) for p in llm.prompts}
    assert {name for name, _, _ in sd.MANDATORY["forecast"]} <= asked and "feature normalisation" not in asked


def test_first_pass_jobs_scale_with_documents():
    llm = FakeLLM()
    docs = [Doc("paper", PAPER_TEXT), Doc("repo-readme", "README"), Doc("repo:execute_cora.py@c1", "lr = 0.005")]
    sd.collect_middle_ground(llm, docs, CLAIM, "Cora")
    assert len(llm.prompts) == 3 * (len(sd.MANDATORY["graph"]) + 1)


def test_first_pass_prompt_has_title_and_target_method():
    llm = FakeLLM()
    sd.extract_setup_details_v2(llm, [Doc("paper", PAPER_TEXT)], CLAIM, "Cora", "Graph Attention Networks")
    first = [p for p in llm.prompts if "Topic to extract" in p]
    assert first and all("Paper Title: Graph Attention Networks" in p and "Target Model/Method: gat" in p for p in first)


def test_big_document_without_topic_match_is_skipped_and_hits_are_windowed():
    big = "filler text. " * 3000                                        # > 15000 Zeichen, kein Themen-Treffer
    llm = FakeLLM()
    sd.collect_middle_ground(llm, [Doc("repo:big.py@c1", big)], CLAIM, "Cora")
    assert llm.prompts == []
    text = big + " the learning rate is 0.01 " + big
    win = sd._regex_windows(text, r"learning.?rate")
    assert "learning rate is 0.01" in win and len(win) < len(text)


def test_first_pass_prompts_contain_no_numbers_outside_claim_and_excerpt():
    """Leak-Schutz: Prompt-Vorlage nennt Themen, nie Werte (V1). Anspruch ohne Zahlen, Ausschnitt ohne Zahlen, Antwortformat abgezogen."""
    llm = FakeLLM()
    sd.collect_middle_ground(llm, [Doc("paper", "plain text without values")], NEUTRAL_CLAIM, "Cora", "Some Paper")
    assert llm.prompts
    for p in llm.prompts:
        template = p.replace(sd.ANSWER_FORMAT_V2, "").split("EXCERPT from")[0].replace("Cora", "")
        assert not re.search(r"\d", template), template


def test_quote_check_drops_item_with_quote_missing_in_document():
    llm = FakeLLM(lambda user: [item("learning rate", "0.5", "learning rate of 0.5")])
    details, dropped, missing, _ = sd.extract_setup_details_v2(llm, [Doc("paper", "learning rate of 0.01")], CLAIM, "Cora")
    assert details == [] and dropped >= 1 and "learning rate" in missing


def test_quote_from_windowed_excerpt_is_verified_against_whole_document():
    doc = Doc("repo:big.py@c1", "x = 1\n" * 3000 + "lr = 0.005  # learning rate\n" + "y = 2\n" * 3000)
    llm = FakeLLM(lambda user: [item("learning rate", "0.005", "lr = 0.005  # learning rate", model="GAT", dataset="Cora")]
                  if "lr = 0.005" in user else [])
    details, _, _, _ = sd.extract_setup_details_v2(llm, [doc], CLAIM, "Cora")
    assert any(d.topic == "learning rate" and d.value == "0.005" for d in details)


def test_old_path_unchanged_without_switch():
    from repro_agent.setup_details import extract_setup_details
    llm = FakeLLM()
    extract_setup_details(llm, [Doc("paper", PAPER_TEXT)], CLAIM, "Cora")
    assert llm.prompts and not any("Topic to extract" in p or "Paper Title" in p for p in llm.prompts)


# ---- Hinweis zur induktiven Einstellung -------------------------------------------------------------------------------------

def test_first_pass_prompt_has_inductive_hint():
    llm = FakeLLM()
    sd.collect_middle_ground(llm, [Doc("paper", PAPER_TEXT)], CLAIM, "Cora")
    assert all("inductive" in p for p in llm.prompts)


def test_inductive_setting_detail_is_excluded_for_graph_claim():
    d = v2("batch size", "2", applies=applies("GAT", "PPI", "inductive"))
    assert sd.scope_exclusion(d, CLAIM, "Cora").startswith("other dataset") or sd.scope_exclusion(d, CLAIM, "Cora") == "inductive setting"
    d = v2("batch size", "2", applies=applies("GAT", "Cora", "inductive"))
    assert sd.scope_exclusion(d, CLAIM, "Cora") == "inductive setting"


@pytest.mark.xfail(strict=True, reason="Hinweis 'transductive setting (e.g., Cora)' ist fest im Prompt und wird auch fuer Zeitreihen "
                                       "(ETTh1) ausgegeben; er stammt aus dem Fall DGI der Soll-Liste, nicht aus einer allgemeinen Regel")
def test_inductive_hint_only_for_graph_family():
    llm = FakeLLM()
    sd.collect_middle_ground(llm, [Doc("paper", "We train with batch size 32.")], "DLinear erreicht MSE auf ETTh1", "ETTh1")
    assert llm.prompts and not any("Cora" in p or "inductive" in p for p in llm.prompts)


# ---- OPTIONAL_RE ------------------------------------------------------------------------------------------------------------

def test_optional_re_matches_sgc_readme_sentence_verbatim():
    assert sd.OPTIONAL_RE.search(SGC_SENTENCE)


def test_sgc_readme_sentence_is_marked_optional_after_verification():
    readme = Doc("repo-readme", f"Setup.\n\n{SGC_SENTENCE}.\n")
    raw = [{"doc": "repo-readme", "item": item("feature normalisation", "zero mean, unit std", SGC_SENTENCE)}]
    details, dropped = sd.verify_raw_v2(raw, [readme])
    assert not dropped and len(details) == 1 and details[0].optional and details[0].source_rank == "readme text"


@pytest.mark.parametrize("sentence", [
    "Weight decay is 5e-4 on the first layer.",
    "The learning rate is 0.01 and the hidden size is 16.",
    "Training stops after 200 epochs or when validation loss rises for 10 epochs.",
    "It is recommended to use dropout 0.5.",
    "We use row-normalised features and the adjacency D^-1/2 (A+I) D^-1/2.",
])
def test_optional_re_leaves_binding_statements_alone(sentence):
    assert not sd.OPTIONAL_RE.search(sentence)


@pytest.mark.parametrize("sentence", [
    "For example, set --lr 0.01.",
    "Optionally, use --epochs 300.",
    "You might want to tune the hidden size.",
])
def test_optional_re_still_matches_earlier_patterns(sentence):
    assert sd.OPTIONAL_RE.search(sentence)


def test_optional_mark_only_for_readme_text():
    docs = [Doc("paper", f"{SGC_SENTENCE}.")]
    raw = [{"doc": "paper", "item": item("feature normalisation", "zero mean", SGC_SENTENCE)}]
    details, _ = sd.verify_raw_v2(raw, docs)
    assert details and not details[0].optional                          # nur README-Fliesstext wird markiert


@pytest.mark.xfail(strict=True, reason="OPTIONAL_RE prueft das ganze Zitat, nicht den Satz: 'can' in einem Nebensatz markiert die "
                                       "verbindliche Angabe (--lr 0.01) als optional, und optionale Angaben fallen nun aus dem Plan")
def test_optional_re_binding_value_with_unrelated_modal_verb():
    assert not sd.OPTIONAL_RE.search("Run python train.py --lr 0.01 --epochs 200; the script can be run on CPU.")


# ---- optionale Angaben: Plan und Abdeckung ---------------------------------------------------------------------------------

def test_optional_detail_is_not_in_plan_and_not_in_conflicts():
    opt = v2("dropout", "0.9", "repo-readme", "readme text", optional=True)
    bind = v2("dropout", "0.5", "paper", "paper")
    assert sd.plan_details_v2([opt, bind]) == [bind]
    assert sd.format_conflicts_v2([opt, bind]) == ""


@pytest.mark.xfail(strict=True, reason="Regression (Pruefung in test_setup_v2 entfernt): optionale Angaben kommen nicht mehr in den Plan, "
                                       "die Markierung 'optional advice' erreicht den Plan-Text nie (toter Zweig in format_for_prompt_v2)")
def test_optional_advice_mark_reaches_plan_text():
    opt = v2("feature normalisation", "zero mean, unit std", "repo-readme", "readme text", optional=True, quote=SGC_SENTENCE)
    assert "optional advice" in sd.format_for_prompt_v2([opt])


@pytest.mark.xfail(strict=True, reason="Regression: optionale README-Angabe zaehlt als Abdeckung des Pflichtthemas (kein zweiter Durchgang, "
                                       "kein 'fehlt' im Dossier), steht aber nicht im Plan; die Angabe geht still verloren")
def test_optional_only_feature_normalisation_is_not_silently_covered():
    opt = v2("feature normalisation", "zero mean, unit std", "repo-readme", "readme text", optional=True, quote=SGC_SENTENCE)
    covered = "feature normalisation" not in sd.missing_topics([opt], "graph")
    in_plan = "feature normalisation" in sd.format_for_prompt_v2([opt])
    assert covered == in_plan                                           # entweder fehlt sie (zweiter Durchgang) oder sie steht im Plan


# ---- Abdeckung Merkmals-Normalisierung (abgeschwaechte Pruefungen) ----------------------------------------------------------

def test_feature_normalisation_coverage_accepts_preprocess_call_and_row_value():
    assert "feature normalisation" not in sd.missing_topics([v2("feature normalisation", "process.preprocess_features(features)")], "graph")
    assert "feature normalisation" not in sd.missing_topics([v2("data preprocessing", "features are row-normalised")], "graph")


def test_feature_normalisation_still_missing_for_unrelated_topics():
    unrelated = [v2("learning rate", "0.01"), v2("dropout", "0.5"), v2("hidden units", "16"), v2("epochs", "200")]
    assert "feature normalisation" in sd.missing_topics(unrelated, "graph")


def test_feature_normalisation_not_covered_by_graph_standardisation():
    assert "feature normalisation" in sd.missing_topics([v2("standardization", "dataset_graph.standardize()")], "graph")


@pytest.mark.xfail(strict=True, reason="Regression (Pruefung in test_setup_v2 umgekehrt): 'preprocess' in FEATURE_NORM_RE laesst "
                                       "'removing self-loops' als Merkmals-Normalisierung gelten")
def test_self_loop_removal_does_not_cover_feature_normalisation():
    assert "feature normalisation" in sd.missing_topics([v2("data preprocessing", "removing self-loops")], "graph")


@pytest.mark.xfail(strict=True, reason="Regression: Muster 'l2' in FEATURE_NORM_RE trifft jede L2-Weight-Decay-Angabe; "
                                       "fast jedes Dossier gilt damit als normalisierungsabgedeckt (Hauptursache laut failure_analysis_v4)")
def test_l2_weight_decay_does_not_cover_feature_normalisation():
    assert "feature normalisation" in sd.missing_topics([v2("weight decay", "L2 5e-4")], "graph")
