"""Loop-level B1/B2 probe on ceo snapshot: drifted LaTeX quote and unavailable LaTeX source. No network/model calls."""
import sys, hashlib, json, pathlib, tempfile
code = sys.argv[1]
sys.path[:0] = [code + '/src', code + '/tests']
import pytest
from repro_agent import catalog, loop, claims_source, discovery
from repro_agent.claims_source import PaperText, TargetVerificationBlocked, ExtractionInvalid, fetch_paper_text
from test_discovery_target_handoff import target_gate
T = open(code + '/tests/fixtures/fits_results_tables.tex').read()
def paper(text):
    return PaperText("2307.03756", "u", "latex", "t", hashlib.sha256(text.encode()).hexdigest(), "v3", text)
out = {}
mp = pytest.MonkeyPatch()
for name, text in [("unchanged", T),
                   ("main_row_changed_appendix_kept", T.replace(r"FITS & \textbf{0.372}", r"FITS & \textbf{0.370}")),
                   ("main_table_removed_appendix_kept", T[:T.index(r"\begin{table*}", 10)])]:
    config, claims = catalog.build_paper(catalog.find_entry(catalog.load_catalog(), "disc_2307_03756"))
    o, events, _ = target_gate(mp, pathlib.Path(tempfile.mkdtemp()), paper(text))   # LLM raises AssertionError
    try:
        o.resolve_targets(config, claims)
        out[name] = f"resolved {claims[0].target} {claims[0].target_evidence['table']}"
    except TargetVerificationBlocked as x:
        out[name] = f"blocked: {x}"
    except AssertionError as x:
        out[name] = f"LLM CALLED: {x}"
# B2: LaTeX unavailable, strict LaTeX claim must not touch Tavily
mp.setattr(discovery, "_fetch_arxiv_source", lambda aid: "")
mp.setattr(claims_source, "latest_arxiv_version", lambda aid: "v3")
def no_tavily():
    raise AssertionError("Tavily requested")
try:
    fetch_paper_text("2307.03756", no_tavily, fixture_dir=pathlib.Path(tempfile.mkdtemp()), mode="live", require_latex=True)
    out["latex_unavailable"] = "returned paper"
except ExtractionInvalid as x:
    out["latex_unavailable"] = f"stopped: {x}"
except AssertionError as x:
    out["latex_unavailable"] = f"TAVILY CALLED: {x}"
seen = []
mp.setattr(discovery, "_fetch_arxiv_source", lambda aid: seen.append(aid) or T * 3)
fetch_paper_text("2307.03756", no_tavily, fixture_dir=pathlib.Path(tempfile.mkdtemp()), mode="live", require_latex=True)
out["versioned_eprint_id"] = seen
mp.undo()
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
