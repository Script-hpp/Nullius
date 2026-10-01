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
import etth1_data  # noqa: E402
import metrics  # noqa: E402
from repro_agent.harness.verify import judge  # noqa: E402

UNITS = {   # name: (Task, Metrik, Zielwert aus dem Paper (b1_candidates.md), Toleranz)
    "sparsetsf": ("ETTh1", "mse", 0.362, 0.01),
    "rlinear": ("ETTh1", "mse", 0.366, 0.01),
    "gcnii": ("Cora", "accuracy", 0.855, 0.01),
}
FILES = {"gcnii": "gcnii_cora_reference.py"}   # sonst <name>_reference.py
DROP_LAST_BATCH = 128


def run_one(name: str, task: str, seed: int, cache: Path) -> tuple[float, float, dict]:
    code = (ROOT / "tests" / "reference" / FILES.get(name, f"{name}_reference.py")).read_text()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        if task == "ETTh1":
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
        if task == "ETTh1":
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
