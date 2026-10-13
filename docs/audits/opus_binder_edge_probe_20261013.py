"""Edge cases for ceo verify_selected_target on the real FITS tables. No network/model calls."""
import sys, json, hashlib, re
sys.path.insert(0, sys.argv[1] + '/src')
from repro_agent.claims_source import PaperText, verify_selected_target, ExtractionInvalid
T = open(sys.argv[1] + '/tests/fixtures/fits_results_tables.tex').read()
main = T[T.index(r'\begin{table*}', 10):]           # tab:FITSett
row = r"FITS & \textbf{0.372} & \textbf{0.404}"
def run(name, text, value=.372, quote=r"FITS & \textbf{0.372}", metric="MSE for prediction length 96"):
    p = PaperText("2307.03756", "u", "latex", "t", hashlib.sha256(text.encode()).hexdigest(), None, text)
    try:
        ev = verify_selected_target(p, "FITS", "ETTh1", metric, value, quote, False)
        res = "None (LLM fallback)" if ev is None else f"verified {ev.value} {ev.table}"
    except ExtractionInvalid as x:
        res = f"blocked: {x}"
    print(f"{name}: {res}"); return res
out = {}
# second copy of the main row in a wraptable with a different ETTh1/96 value
wrap = "\n\\begin{wraptable}{r}{0.5\\textwidth}\\caption{Look-back 720 in MSE}\\begin{tabular}{c|cccc}\nDataset & \\multicolumn{4}{c}{ETTh1}\\\\\nHorizon & 96 & 192 & 336 & 720\\\\\nFITS & \\textbf{0.372} & 0.5 & 0.5 & 0.5\\\\\n\\end{tabular}\\end{wraptable}\n"
out['copy_in_wraptable_is_ignored'] = run('copy_in_wraptable_is_ignored', T + wrap)
# same copy inside a table* whose tabular has a nested tabular header (SegRNN style)
nested = wrap.replace('wraptable}{r}{0.5\\textwidth}', 'table*}').replace('wraptable}', 'table*}').replace('Dataset &', 'Dataset & \\begin{tabular}[c]{@{}c@{}}x\\\\ y\\end{tabular} &', 1)
out['copy_in_nested_tabular_table'] = run('copy_in_nested_tabular_table', T + nested)
# std inside result cells
out['std_in_cell'] = run('std_in_cell', T.replace(r"FITS & \textbf{0.372}", r"FITS & \textbf{0.372}$\pm$0.000"))
# caption after the tabular (common style)
cap = re.search(r"\\caption\{Long-term forecasting results on ETT dataset in MSE\..*?\}\s*\n", main, re.S)
moved = T.replace(cap.group(), "").replace(r"\label{tab:FITSett}", cap.group() + r"\label{tab:FITSett}")
out['caption_after_tabular'] = run('caption_after_tabular', moved)
# two-metric caption
out['caption_mse_and_mae'] = run('caption_mse_and_mae', T.replace("on ETT dataset in MSE.", "on ETT dataset in MSE and MAE."))
# quote that only contains the number
out['number_only_quote'] = run('number_only_quote', T, quote=r"\textbf{0.372}")
# drifted source: main row renumbered, appendix unchanged
out['drift_main_row_changed'] = run('drift_main_row_changed', T.replace(r"FITS & \textbf{0.372}", r"FITS & \textbf{0.370}"))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
