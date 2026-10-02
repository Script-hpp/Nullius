"""paper_head: gemeinsame Funktionen von scripts/make_summaries.py und der Pipeline. Attrappen, kein Netz."""

from repro_agent.paper_head import SYSTEM, build_head, claim_for_prompt, head_of_text, prompt, source_label, verified_institutions
from repro_agent.dossier import Claim

DOC = {"text": "# Title\n\nAda L.\nUniversity of Amsterdam, NL\n\nAbstract. Text.", "source_kind": "pdf", "arxiv_version": "v4"}


def test_prompt_turns_fractions_into_percent_and_uses_the_method():
    c = claim_for_prompt({"title": "T", "dataset": "Cora", "fidelity": {"method": "GCN"}}, Claim("c", "t", "accuracy", 0.815, 0.01, "s", "paper"))
    p = prompt(c, "An abstract.")
    assert "reaches 81.5% accuracy" in p and "the method GCN" in p and "An abstract." in p
    assert claim_for_prompt({"title": "T"}, Claim("c", "t", "mse", 0.37, 0.01, "s", "paper"))["claim"]["method"] == "T"   # no fidelity entry: the title


def test_head_of_text_starts_at_the_title_for_html_and_before_the_first_affiliation():
    nav = "navigation " * 100
    h = head_of_text(nav + "\n# The Title\nbody")
    assert "# The Title" in h and len(h) < 200 and h.count("navigation") <= 11   # navigation far in front is cut, a few characters before the title stay
    assert "Affiliation: Edinburgh" in head_of_text(nav + "Ada Affiliation: Edinburgh")


def test_source_label_names_the_real_format():
    assert source_label({"source_kind": "pdf", "arxiv_version": "v4"}) == "arXiv PDF v4, read with Tavily Extract"
    assert source_label({"source_kind": "arxiv_html", "arxiv_version": None}) == "arXiv HTML, read with Tavily Extract"
    assert source_label({"source_kind": "ar5iv"}) == "arXiv HTML (ar5iv), read with Tavily Extract"
    assert source_label(None) is None


def test_verified_institutions_need_the_quote_in_the_head_and_the_name_in_the_quote():
    head = "Ada L.\nUniversity of Amsterdam, NL\nDepartment of Physics\n"
    items = [{"name": "University of Amsterdam", "quote": "University of Amsterdam, NL"}, {"name": "Oxford", "quote": "University of Oxford"},
             {"name": "Department of Physics", "quote": "Department of Physics"}, {"name": "University of Amsterdam", "quote": "University of Amsterdam"}]
    assert [i["name"] for i in verified_institutions(items, head)] == ["University of Amsterdam"]   # Oxford not in the text, department dropped, duplicate dropped


def test_build_head_collects_summary_authors_affiliations_and_venue():
    answers = {SYSTEM: '{"summary": "' + "A sentence about the paper in plain words. " * 3 + '"}'}

    def ask(system, user, t):
        return answers.get(system, '{"institutions": [{"name": "University of Amsterdam", "quote": "University of Amsterdam, NL"}]}')

    doc = {**DOC, "text": "Published as a conference paper at ICLR 2017\n" + DOC["text"]}
    h = build_head(ask, {"title": "T", "dataset": "Cora", "claim": {"method": "GCN", "metric": "accuracy", "target": 0.815}}, "Abstract.", {"authors": ["Ada L."], "year": 2016}, doc, "1609.02907")
    assert h["text"].startswith("A sentence") and h["authors"] == ["Ada L."] and h["year"] == 2016
    assert [i["name"] for i in h["institutions"]] == ["University of Amsterdam"] and h["venue"]["name"] == "ICLR 2017"
    assert build_head(lambda s, u, t: "", {"title": "T", "dataset": "", "claim": {"method": "m", "metric": "mse", "target": 0.3}}, "A.", None, None, "x") == {}
