"""Referenzloesungen LOKAL auf der CPU laufen lassen (kein Sandbox-Aufruf, kein LLM). Fuer neue Benchmark-Einheiten (Block B1).

    PYTHONPATH=src python scripts/run_reference_local.py sparsetsf 5 [--cache DIR] [--out DATEI]

Schreibt die Daten wie der Harness (cora_data / etth1_data), fuehrt tests/reference/<name>_reference.py mit SEED=0..n-1 aus, rechnet die Metrik mit den
Harness-Funktionen gegen die Original-Labels und das Urteil mit verify.judge. Zusatz fuer ETTh1: MSE auch ohne die letzten Testfenster, die ein Test-Loader mit
drop_last=True bei Batchgroesse 128 verwerfen wuerde (Nachbildung des bekannten Fehlers, nur Information, kein Urteil).
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "src" / "repro_agent" / "harness"
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(ROOT / "src"))

import cora_data  # noqa: E402
import ett_data  # noqa: E402
import etth1_data  # noqa: E402
import metrics  # noqa: E402
import planetoid_data  # noqa: E402
from repro_agent.harness.verify import judge  # noqa: E402

UNITS = {   # name: (Task, Metrik, Zielwert aus dem Paper (b1_candidates.md), Toleranz)
    "sparsetsf": ("ETTh1", "mse", 0.362, 0.01),
    "rlinear": ("ETTh1", "mse", 0.366, 0.01),
    "gcnii": ("Cora", "accuracy", 0.855, 0.01),
}
# Neue Datensaetze (datasets_v2_validation.md): <modell>_<datensatz>; Zielwerte woertlich aus den gespeicherten Paper-Texten (benchmark/fixtures/), Toleranz vorab festgelegt
for _m, _tgt in {"gcn": (0.703, 0.790), "sgc": (0.719, 0.789), "gat": (0.725, 0.790)}.items():
    UNITS[f"{_m}_citeseer"], UNITS[f"{_m}_pubmed"] = ("citeseer", "accuracy", _tgt[0], 0.01), ("pubmed", "accuracy", _tgt[1], 0.01)
for _ds, (_dl, _nl) in {"etth2": (0.289, 0.277), "ettm1": (0.299, 0.306), "ettm2": (0.167, 0.167)}.items():
    UNITS[f"dlinear_{_ds}"], UNITS[f"nlinear_{_ds}"] = (_ds, "mse", _dl, 0.01), (_ds, "mse", _nl, 0.01)
FILES = {"gcnii": "gcnii_cora_reference.py"}   # sonst <name>_reference.py, neue Einheiten: Modelldatei (_PLANETOID_REF / <modell>_reference.py)
_PLANETOID_REF = {"gcn": "gcn_cora_reference.py", "sgc": "sgc_cora_reference.py", "gat": "gat_cora_reference.py"}


def patch_planetoid(code: str, model: str, ds: str, data_path: str) -> str:
    """Referenz fuer einen anderen Planetoid-Datensatz: Pfad und Klassenzahl; GAT auf PubMed mit der Paper-Anpassung (8 Ausgabekoepfe, L2 0,001)."""
    def sub(old: str, new: str) -> None:
        nonlocal code
        assert old in code, f"Patch-Stelle fehlt: {old}"
        code = code.replace(old, new)

    sub("/data/cora_public.npz", data_path)
    sub("N, n_cls = X.shape[0], 7", "N, n_cls = X.shape[0], int(d['labels'].max()) + 1")
    if model == "gat" and ds == "pubmed":
        sub("P, HEADS, HID, LR, L2, PATIENCE = 0.6, 8, 8, 0.005, 5e-4, 100", "P, HEADS, HID, LR, L2, PATIENCE = 0.6, 8, 8, 0.005, 1e-3, 100\nOUT_HEADS = 8")
        sub("GATLayer(HID * HEADS, n_cls, 1)", "GATLayer(HID * HEADS, n_cls, OUT_HEADS)")
        sub("return l2(F.elu(l1(X, train)), train)", "return l2(F.elu(l1(X, train)), train).view(N, OUT_HEADS, n_cls).mean(1)")
    return code
DROP_LAST_BATCH = 128


def run_one(name: str, task: str, seed: int, cache: Path) -> tuple[float, float, dict]:
    model = name.split("_")[0]
    if name in FILES:
        fname = FILES[name]
    elif task in planetoid_data.SPECS:
        fname = _PLANETOID_REF[model]
    else:
        fname = f"{model}_reference.py"
    code = (ROOT / "tests" / "reference" / fname).read_text()
    new_ett, new_planetoid = task in ett_data.SPECS and task != "etth1", task in planetoid_data.SPECS and task != "cora"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        if new_ett:
            csv = ett_data.download(task, cache / task)
            ett_data.prepare_public(task, csv, d / f"{task}_public.npz")
            code = code.replace("/data/etth1_public.npz", str(d / f"{task}_public.npz"))
        elif new_planetoid:
            raw = cache / task
            planetoid_data.download_raw(task, raw)
            planetoid_data.prepare_public(task, raw, d / f"{task}_public.npz")
            code = patch_planetoid(code, model, task, str(d / f"{task}_public.npz"))
        elif task == "ETTh1":
            csv = etth1_data.download(cache / "ett")
            etth1_data.prepare_public(csv, d / "etth1_public.npz")
            code = code.replace("/data/etth1_public.npz", str(d / "etth1_public.npz"))
        else:
            raw = cache / "raw"
            if not (raw / "ind.cora.x").exists():
                cora_data.download_raw(raw)
            cora_data.prepare_public(raw, d / "cora_public.npz")
            code = code.replace("/data/cora_public.npz", str(d / "cora_public.npz"))
        code = code.replace('"/output"', repr(str(d / "out"))).replace('"/output/predictions.csv"', repr(str(d / "out" / "predictions.csv")))
        (d / "train.py").write_text(code)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(d / "train.py")], capture_output=True, text=True, env={"SEED": str(seed), "PATH": "/usr/bin:/bin"})
        secs = time.time() - t0
        if r.returncode:
            raise RuntimeError(r.stderr[-800:])
        rows = (d / "out" / "predictions.csv").read_text().splitlines()[1:]
        extra: dict = {"stdout_tail": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""}
        if new_ett:
            tgt = ett_data.targets(task, csv)
            pred = np.array([float(l.split(",")[1]) for l in rows]).reshape(tgt.shape)
            value = metrics.mse(pred.reshape(-1).tolist(), tgt.reshape(-1).tolist())
        elif new_planetoid:
            truth = planetoid_data.test_labels(task, raw)
            ids = [l.split(",") for l in rows]
            value = float(np.mean([truth[i] == p for i, p in ids]))
        elif task == "ETTh1":
            tgt = etth1_data.targets(csv)
            pred = np.array([float(l.split(",")[1]) for l in rows]).reshape(tgt.shape)
            value = metrics.mse(pred.reshape(-1).tolist(), tgt.reshape(-1).tolist())
            keep = (len(pred) // DROP_LAST_BATCH) * DROP_LAST_BATCH
            extra["mse_drop_last_128"] = float(((pred[:keep] - tgt[:keep]) ** 2).mean())
            extra["windows_dropped"] = len(pred) - keep
        else:
            truth = cora_data.test_labels(cache / "raw")
            ids = [l.split(",") for l in rows]
            value = float(np.mean([truth[i] == p for i, p in ids]))
        return value, secs, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(UNITS))
    ap.add_argument("seeds", type=int)
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "ref_cache"))
    ap.add_argument("--out")
    a = ap.parse_args()
    task, metric, target, tol = UNITS[a.name]
    vals, secs, extras = [], [], []
    for s in range(a.seeds):
        v, t, e = run_one(a.name, task, s, Path(a.cache))
        vals.append(v); secs.append(t); extras.append(e)
        print(f"seed {s}: {metric} {v:.4f} ({t:.0f}s) {e['stdout_tail']}", flush=True)
    mean, std = statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)
    lines = [f"{a.name}: {len(vals)}/{a.seeds} gueltig; {metric} Mittel {mean:.4f}, Std {std:.4f}; Ziel {target} +- {tol}, "
             f"Abstand {abs(mean - target):.4f} -> Referenzurteil {judge(mean, target, tol)}",
             f"Laufzeit je Seed (s): Mittel {statistics.mean(secs):.0f}, max {max(secs):.0f}; Einzelwerte {[round(v, 4) for v in vals]}"]
    if task == "ETTh1":
        dl = [e["mse_drop_last_128"] for e in extras]
        lines.append(f"Nachbildung drop_last (Batch {DROP_LAST_BATCH}, {extras[0]['windows_dropped']} von 2785 Fenstern verworfen): "
                     f"MSE Mittel {statistics.mean(dl):.4f} (statt {mean:.4f}), Urteil dazu {judge(statistics.mean(dl), target, tol)}")
    print("\n".join(lines))
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
