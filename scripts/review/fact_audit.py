"""Decisive-fact audit for the v4 arm-B runs (read only, stdlib, no API calls).

Usage: python3 scripts/review/fact_audit.py <main checkout> [store.jsonl]
       python3 scripts/review/fact_audit.py <main checkout> --s4 <clone dir>   (V0 of docs/recall_steckbrief_plan.md, no LLM)

For every arm-B real-claim run and every pre-listed decisive fact of its unit it reports:
  run    - is the fact in a file the run had (dossier.repo.files, text from the repo fixture)?
           "F1" = the file holding it was NOT in the run (needs a new file selection)
  prompt - did the fact reach the plan/code prompt as a verified setup-detail quote/value?
  code   - does the final measured agent code implement it? (regex heuristic, "-" = not checkable)
The facts come from benchmark/experiments/failure_analysis_v4.md and docs/recall_root_cause.md
(dev units, post hoc). The code regexes are heuristics; read the printed code lines before relying on one.
"""
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
S4_MODE = "--s4" in sys.argv
STORE = sys.argv[2] if len(sys.argv) > 2 and not S4_MODE else "runs/benchmark_v4.jsonl"
FIX = Path(__file__).resolve().parents[2] / "benchmark" / "fixtures" / "repos"
REF = {"kipf_gcn_2017": "reproduziert", "wu_sgc_2019": "reproduziert", "zeng_dlinear_2023": "reproduziert",
       "shchur_pitfalls_2018": "reproduziert", "velickovic_gat_2018": "reproduziert",
       "velickovic_dgi_2019": "teilweise", "shchur_gat_2018": "reproduziert", "zeng_nlinear_2023": "reproduziert"}
# Normalisation and lr-schedule use in the code is NOT regex-checkable reliably (adjacency sums, other schedules match too);
# for those the code column is "-" and failure_analysis_v4.md (read code lines) is the source.
LTSF = {   # fact id: (repo file, regex in that file, regex in prompt quote/value, regex in code or None)
    "seq336": ("scripts/EXP-LongForecasting/Linear/etth1.sh", r"seq_len=336", r"\b336\b", r"\b336\b"),
    "lr0.005": ("scripts/EXP-LongForecasting/Linear/etth1.sh", r"--learning_rate 0\.005", r"0\.005", r"0\.005|5e-3"),
    "halve_type1": ("utils/tools.py", r"0\.5 \*\* \(\(epoch - 1\) // 1\)", r"0\.5\s*\*\*|halv", None),
}
FACTS = {
    "zeng_dlinear_2023": LTSF,
    "zeng_nlinear_2023": {**LTSF,
        "subtract_last": ("models/NLinear.py", r"seq_last", r"seq_last|subtract", r"\[:,\s*-1:|seq_last|last_val|x\[:,\s*-1"),
        "MISLEADING model_name=DLinear": ("scripts/EXP-LongForecasting/Linear/etth1.sh", r"model_name=DLinear",
                                          r"model_name=DLinear|--model \$model_name|DLinear", r"class DLinear|moving_avg|series_decomp|trend"),
    },
    "wu_sgc_2019": {
        "row_normalize": ("utils.py", r"features = row_normalize\(features\)", r"row_normalize|Row-normalize|row-normaliz", None),
        "MISLEADING README standardise": ("README.md", r"zero mean", r"zero mean", r"\.std\(|StandardScaler"),
    },
    "velickovic_gat_2018": {
        "feat_rownorm": ("execute_cora.py", r"preprocess_features", r"preprocess_features|row-normaliz|Row-normalize", None),
    },
    "velickovic_dgi_2019": {
        "one_gcn_encoder": ("models/dgi.py", r"self\.gcn = GCN\(n_in, n_h, activation\)", r"self\.gcn = GCN|single|one[- ]layer", None),
        "hid512": ("execute.py", r"hid_units = 512", r"\b512\b", r"\b512\b"),
        "lr0.001": ("execute.py", r"lr = 0\.001", r"0\.001", r"0\.001|1e-3"),
        "prelu": ("execute.py", r"nonlinearity = 'prelu'", r"prelu|PReLU", r"PReLU"),
        "feat_rownorm": ("execute.py", r"preprocess_features", r"preprocess_features|row-normaliz|Row-normalize", None),
        "gcn_layer_body": ("layers/gcn.py", r"", r"nn\.Linear\(in_ft, out_ft, bias=False\)", None),
    },
    "shchur_gat_2018": {
        "gat_model_file": ("gnnbench/models/gat.py", r"", r"gat\.py", None),
        "gat_config": ("config/optimized/gat.conf.yaml", r"", r"gat\.conf", None),
        "patience50": ("config/train.conf.yaml", r"early_stopping_tolerance: 50", r"tolerance: 50|\b50\b", r"patience\s*=\s*50|tolerance\s*=\s*50|PATIENCE\s*=\s*50"),
    },
    "shchur_pitfalls_2018": {
        "glorot_init": ("gnnbench/models/gcn.py", r"", r"glorot|xavier|Glorot", r"xavier|glorot|Glorot"),
        "hid64": ("config/optimized/gcn.conf.yaml", r"hidden_size: 64", r"\b64\b", r"\b64\b"),
    },
    "kipf_gcn_2017": {
        "feat_rownorm": ("gcn/utils.py", r"def preprocess_features", r"preprocess_features|row-normaliz|Row-normalize", None),
        "glorot_init": ("gcn/layers.py", r"glorot\(", r"glorot|Glorot|xavier", r"xavier|glorot|Glorot"),
    },
}
FIXTURE = {"zeng_dlinear_2023": "cure-lab_LTSF-Linear", "zeng_nlinear_2023": "cure-lab_LTSF-Linear",
           "wu_sgc_2019": "Tiiiger_SGC", "velickovic_gat_2018": "PetarV-_GAT", "velickovic_dgi_2019": "PetarV-_DGI",
           "shchur_gat_2018": "shchur_gnn-benchmark", "shchur_pitfalls_2018": "shchur_gnn-benchmark", "kipf_gcn_2017": "tkipf_gcn"}


def final_code(run_dir: Path, claim: str, dossier: dict) -> tuple[str | None, str | None]:
    hashes = collections.Counter(a["code_hash"] for a in dossier["attempts"]
                                 if a["claim_id"] == claim and a.get("seed", 0) > 0 and a.get("measured") is not None)
    if not hashes:
        return None, None
    h = hashes.most_common(1)[0][0]
    p = run_dir / f"{claim}_{h}.py"
    return h, (p.read_text(errors="replace") if p.exists() else None)


def code_body(src: str) -> str:
    """Code without comment lines and docstring-ish header comments, so a plan comment does not count as implemented."""
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


S4_CHOSEN_NOTE = (
    "Auswahl der LLM-Dateiwahl (select_files) = die Dateien der gespeicherten Fixture (Proxy, offline); dazu die deterministischen "
    "S4-Schritte aus repo_source.read_repo: Pflicht-Modelldatei, relative Imports zwei Ebenen, Quelltext-Filter, Zeichenbudget.")


class _CloneSandbox:
    """Sandbox-Attrappe ueber einen lokalen Klon (nur Lesen); read_repo laeuft unveraendert durch."""

    def __init__(self, clone: Path):
        self.clone = clone

    def base_snapshot(self, *a, **k):
        return "snap"

    def branch(self, snap):
        return "b"

    def get_file(self, branch, path):
        rel = path.removeprefix("/repo/")
        if rel == ".head_commit":
            import subprocess
            return subprocess.run(["git", "-C", str(self.clone), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.encode()
        return (self.clone / rel).read_bytes()

    def run(self, branch, cmd, timeout_s):
        import subprocess
        from types import SimpleNamespace
        out = subprocess.run(["git", "-C", str(self.clone), "ls-files"], capture_output=True, text=True).stdout
        return SimpleNamespace(stdout=out, stderr="", returncode=0)


def s4_audit(clones: Path) -> int:
    """V0: steht jeder vorab gelistete entscheidende Fakt im S4-Ausschnitt (Steckbrief-Aufruf) der Einheit? Rueckgabe: Zahl fehlender Fakten."""
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repro_agent.loop import load_claims
    from repro_agent.repo_source import read_repo
    from repro_agent.setup_details import Doc, SetupDetail
    from repro_agent.spec import focused_excerpt, model_name_of

    recs = [json.loads(l) for l in open(ROOT / "runs/benchmark_v4.jsonl")]
    print(S4_CHOSEN_NOTE, "\n")
    missing_total = 0
    for paper_id, facts in FACTS.items():
        paper, claims = load_claims(paper_id)
        claim, dataset = claims[0], paper["dataset"]
        model = model_name_of(claim.text)
        rec = next(r for r in recs if r["paper"] == paper_id and r["arm"].startswith("B") and r["origin"] == "paper")
        d = json.loads((ROOT / rec["run_dir"] / "dossier.json").read_text())
        details = [SetupDetail(**{k: v for k, v in x.items() if k in SetupDetail.__dataclass_fields__}) for x in d["setup_details"]]
        fxname = FIXTURE[paper_id]
        fx = json.loads((FIX / f"{fxname}.json").read_text())
        chosen = [p for p in fx["files"] if (clones / fxname / p).exists()]
        llm = type("L", (), {"complete": lambda self, *a, **k: type("R", (), {"content": json.dumps(chosen)})()})()
        with tempfile.TemporaryDirectory() as tmp:
            dump = read_repo(_CloneSandbox(clones / fxname), llm, fx["url"], claim.text, dataset, focus_model=True,
                             unit_key=paper_id, model=model, fixture_dir=Path(tmp))
        docs = [Doc(f"repo:{p}@{dump.commit[:7]}", t) for p, t in dump.files.items()]
        excerpt, names = focused_excerpt(docs, model, dataset, details)
        pieces = {}
        for block in excerpt.split("\n\n=== "):
            head, _, body = block.removeprefix("=== ").partition(" ===\n")
            pieces[head.split(":", 1)[-1].split("@")[0].replace(" (function body)", "")] = pieces.get(head.split(":", 1)[-1].split("@")[0].replace(" (function body)", ""), "") + body
        print(f"{paper_id}  model={model}  read={len(dump.files)} files, excerpt={len(excerpt)} chars, pieces={len(names)}")
        for fid, (path, repo_re, prompt_re, code_re) in facts.items():
            if fid.startswith("MISLEADING"):
                continue
            body = pieces.get(path)
            in_file = path in dump.files and (not repo_re or re.search(repo_re, dump.files[path]))
            ok = body is not None and (not repo_re or re.search(repo_re, body))
            why = "" if ok else (" (Datei nicht gelesen)" if path not in dump.files else
                                 " (Fakt nicht in der Datei)" if not in_file else
                                 " (Datei gelesen, aber nicht im Ausschnitt)" if body is None else " (Fakt abgeschnitten)")
            missing_total += 0 if ok else 1
            print(f"   {fid:32} {path:50} {'IN EXCERPT' if ok else 'MISSING'}{why}")
        print("   Ausschnitt-Stuecke:", ", ".join(names), "\n")
    print(f"Fehlende Fakten im Ausschnitt: {missing_total}")
    return missing_total


if S4_MODE:
    sys.exit(1 if s4_audit(Path(sys.argv[sys.argv.index("--s4") + 1])) else 0)

recs = [json.loads(l) for l in open(ROOT / STORE)]
summary = collections.defaultdict(lambda: collections.defaultdict(list))
for r in recs:
    if r["origin"] != "paper" or not r["arm"].startswith("B"):
        continue
    run_dir = ROOT / r["run_dir"]
    d = json.loads((run_dir / "dossier.json").read_text())
    fx = json.loads((FIX / f"{FIXTURE[r['paper']]}.json").read_text())["files"]
    run_files = set(d["repo"]["files"])
    prompt_items = [f"{x['topic']} | {x['value']} | {x['quote']}" for x in d["setup_details"]]
    h, code = final_code(run_dir, r["claim"], d)
    body = code_body(code) if code else ""
    miss = r["urteil"] != REF[r["paper"]]
    print(f"{r['paper'][:20]:20} g{r['generation']} {r['urteil'][:12]:12} {'MISS' if miss else 'ok  '} mean={r['mean']:.4f} code={h}")
    for fid, (path, repo_re, prompt_re, code_re) in FACTS[r["paper"]].items():
        if path in run_files:
            in_run = "yes" if (not repo_re or re.search(repo_re, fx.get(path, ""))) else "file-without-fact"
        else:
            in_run = "F1"
        src_ok = [s for s in prompt_items if re.search(prompt_re, s, re.I)]   # loose: any delivered item matching
        in_prompt = "yes" if src_ok else "no"
        in_code = "-" if (code_re is None or not code) else ("yes" if re.search(code_re, body, re.I) else "no")
        if miss and not fid.startswith("MISLEADING"):
            summary[(r["paper"], r["generation"])][{"F1": "not in run", "yes": "in run", "file-without-fact": "in run"}[in_run] + (", in prompt" if in_prompt == "yes" else ", not in prompt")].append(fid)
        print(f"     {fid:32} run={in_run:4} prompt={in_prompt:3} code={in_code:3}"
              + (f"   e.g. {src_ok[0][:110]!r}" if src_ok else ""))
print("\nPer missed run: where each listed decisive fact stood (misleading items excluded)")
for (paper, gen), groups in summary.items():
    print(f"  {paper[:20]:20} g{gen}: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(groups.items())))
