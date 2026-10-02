"""Fasst out/<einheit>_seed<N>/harness.json und die Repo-Ausgaben zusammen. Aufruf: summarize.py <work-dir>"""
import json
import re
import statistics as st
import sys
from pathlib import Path

W = Path(sys.argv[1])
UNITS = {"rtsf": ("RLinear", 0.366, "1024", "128"), "ltsf": ("DLinear", 0.375, "2021", "32")}
for key, (name, target, repo_seed, bs) in UNITS.items():
    print(f"== {name} (Ziel {target}, Toleranz 0.01, droplast-Batch {bs})")
    all_mse, rows = [], []
    for seed in [repo_seed, "0", "1", "2", "3"]:
        h = json.loads((W / "out" / f"{key}_seed{seed}" / "harness.json").read_text())
        log = (W / "out" / f"{key}_seed{seed}.txt").read_text()
        own = float(re.search(r"^mse:([0-9.]+)", log, re.M).group(1))
        real = re.search(r"^real\s+(\S+)", log, re.M) or re.search(r"([\d,.]+) total", log)
        t = real.group(1) if real else "?"
        m, dl = h["metrics"]["mse"], h["metrics_droplast"][bs]["mse"]
        all_mse.append(m)
        print(f"seed {seed:>5}: repo {own:.4f} | harness alle {m:.4f} | harness droplast{bs} {dl:.4f} | |droplast-repo| {abs(dl-own):.5f} | |alle-repo| {abs(m-own):.5f} | Zeit {t}")
    mean, sd = st.mean(all_mse), st.stdev(all_mse)
    print(f"Mittel {mean:.4f} std(ddof=1) {sd:.4f} | Abstand zum Ziel {abs(mean-target):.4f}")
