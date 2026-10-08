"""Prototype for docs/gat_failure_analysis.md, F1 (analysis only, reads spec.json files, executes nothing).

For every enforced item with rank `paper` it looks for a literal assignment `key = number` of the same quantity in the raw repository
excerpt stored in spec.json and reports numbers that differ. Usage: python3 -I spec_conflict_audit_prototype.py <recall_cg_gate_samples dir>
"""
import glob, json, re, sys

ALIAS = {"epochs": ["epochs", "nb_epochs", "num_epochs", "n_epochs", "max_epochs"], "batch_size": ["batch_size"], "patience": ["patience"],
         "lr": ["lr", "learning_rate"], "weight_decay": ["weight_decay", "l2_coef"], "dropout": ["dropout"],
         "hidden": ["hidden", "hidden_size", "nhid"], "heads": ["heads", "nheads", "num_heads"]}
NUM = r"([0-9]*\.?[0-9]+(?:[eE]-?[0-9]+)?)"

for f in sorted(glob.glob(f"{sys.argv[1]}/*/spec.json")):
    s = json.load(open(f))
    ex = s.get("excerpt", "")
    conflicts = []
    for it in s["items"]:
        if not it.get("enforce") or it["rank"] != "paper" or it.get("number") is None:
            continue
        for a in ALIAS.get(it["key"], [it["key"]]):
            for m in re.finditer(rf"(?m)^\s*{a}\s*[=:]\s*{NUM}", ex):
                if abs(float(m.group(1)) - float(it["number"])) > 1e-12:
                    conflicts.append((it["key"], it["number"], a, m.group(1)))
    print(f.split("/")[-2].ljust(22), "paper-ranked enforced:", [i["key"] for i in s["items"] if i.get("enforce") and i["rank"] == "paper"], "conflicts:", sorted(set(conflicts)))
