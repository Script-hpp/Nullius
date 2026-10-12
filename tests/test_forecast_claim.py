"""Actual public source tables, network/model/sandbox calls forbidden."""
from pathlib import Path

import pytest

from repro_agent.claims_source import _norm
from repro_agent.forecast_claim import proposed_method, requires_external_weights, select_source_claim
from repro_agent.table_check import check_columns

FIX = Path(__file__).parent / "fixtures"
SARAF = (FIX / "saraf_main_results.tex").read_text()
GLINEAR = (FIX / "glinear_main_results.tex").read_text()
EMBEDDINGS = (FIX / "data_embeddings_main_results.tex").read_text()
G_ABSTRACT = "In this paper, we propose a novel data-efficient architecture, Gaussian-activated Linear model (GLinear), for multivariate TSF."


@pytest.mark.parametrize("source,abstract,method,target,length", [
    (SARAF, "", "SARAF", .371, 720),
    (GLINEAR, G_ABSTRACT, "GLinear", .382, 336),
])
def test_selects_the_own_main_result_with_verbatim_source_evidence(source, abstract, method, target, length):
    claim = select_source_claim(source, "ETTh1", abstract)
    assert claim and claim["target"] == target and claim["method"] == method
    assert claim["input_length"] == length and claim["horizon"] == 96
    assert claim["quote"] in source
    check = check_columns(_norm(source), _norm(claim["quote"]), target, method,
                          "ETTh1", "MSE for prediction length 96", False)
    assert check.status == "ok" and check.horizon == 96


@pytest.mark.parametrize("source", [
    SARAF.replace("SARAF (ours)", "SARAF"),
    SARAF.replace("Multivariate", "Ablation multivariate"),
    SARAF.replace("look-back window of 720", "look-back window of 1440"),
    SARAF.replace("& 96 &", "& 192 &"),
    SARAF.replace("& 96 &", "& Avg &"),
    SARAF.replace(r"\underline{0.371}", "not reported"),
    SARAF + SARAF.replace("0.371", "0.372"),
    GLINEAR,  # no '(ours)' or abstract proposal evidence
    EMBEDDINGS,  # a comparison/study of existing methods, no proposed method
])
def test_unknown_ambiguous_ablation_baseline_or_unsupported_tables_do_not_select_a_claim(source):
    assert select_source_claim(source, "ETTh1", "We study forecasting benchmarks.") is None


def test_baselines_are_not_proposed_methods_and_external_dependencies_are_explicit():
    assert proposed_method("GLinear", G_ABSTRACT)
    assert not proposed_method("DLinear", G_ABSTRACT + " We compare it against DLinear.")
    assert not proposed_method("Times2D", "We study the data embeddings of Times2D.")
    assert requires_external_weights("Time-VLM leverages pre-trained Vision-Language Models.")
    assert not requires_external_weights("We train from scratch and compare against pre-trained language models.")


def test_multirow_dataset_context_expires_and_never_leaks_to_another_dataset():
    claim = select_source_claim(GLINEAR, "ETTh1", G_ABSTRACT)
    assert claim
    assert check_columns(_norm(GLINEAR), _norm(claim["quote"]), .382, "GLinear", "ETTh2", "mse", False).status != "ok"
    broken = GLINEAR.replace(r"\multirow{7}{*}{ETTh1}", r"\multirow{2}{*}{ETTh1}")
    assert select_source_claim(broken, "ETTh1", G_ABSTRACT) is None
