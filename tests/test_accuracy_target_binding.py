"""Accuracy claims bind to their own LaTeX table; no provider calls."""
import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from repro_agent.claims_source import ExtractionInvalid, PaperText, verify_selected_target
from repro_agent.discovery import _decode_arxiv_source

ROOT = Path(__file__).resolve().parents[1]


def bangs():
    text = (ROOT / "tests/fixtures/bangs_tables.tex").read_text()
    return PaperText("2410.09348", "u", "latex", "now", hashlib.sha256(text.encode()).hexdigest(), "v1", text)


def test_main_table_value_is_verified_against_its_own_column():
    # Live 2026-10-14: a second model read chose the appendix table's 83.47 and stopped the correct 84.23 claim
    ev = verify_selected_target(bangs(), "BANGS", "Cora", "accuracy", 0.8423, r"\textbf{84.23±0.62*}", True)
    assert ev.value == pytest.approx(0.8423) and ev.table == "tab:GCN" and ev.column_check == "ok"


def test_a_value_from_another_column_is_rejected():
    with pytest.raises(ExtractionInvalid, match="Kopfzeile"):
        verify_selected_target(bangs(), "BANGS", "Cora", "accuracy", 0.8372, r"{\ul 83.72±0.52}", True)


def test_unparsed_accuracy_layout_keeps_the_independent_extraction_path():
    text = r"\begin{table}\begin{tabular}{l} GCN reaches 81.5 on Cora \\ \end{tabular}\end{table}"
    paper = PaperText("x", "u", "latex", "now", hashlib.sha256(text.encode()).hexdigest(), None, text)
    assert verify_selected_target(paper, "GCN", "Cora", "accuracy", 0.815, "81.5", True) is None


def test_latex_inputs_are_inlined_in_document_order():
    files = {"main.tex": "\\documentclass{article}\nMAIN\n\\input{sec/results}\n\\appendix\n\\input{appendix}\n",
             "appendix.tex": "APPENDIX TABLE", "sec/results.tex": "\\section{Results}\nMAIN TABLE"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("appendix.tex", "sec/results.tex", "main.tex"):   # appendix first in the archive
            data = files[name].encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    text = _decode_arxiv_source(buf.getvalue())
    assert text.index("MAIN TABLE") < text.index("\\appendix") < text.index("APPENDIX TABLE")
    assert text.count("APPENDIX TABLE") == 1
