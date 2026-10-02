"""Adapter RTSF (plumprc/RTSF, Commit 0fec001): trainiertes Modell laden, alle 2785 Testfenster vorhersagen.

Aufruf (cwd egal):
  python adapter_rtsf.py --repo <rtsf-klon> --ckpt <checkpoint.pth> --csv <ETTh1.csv> --out <dir> [--batch-size 128]
Schreibt <dir>/predictions.csv und <dir>/mapping.json.
Repo-spezifisch (nur dieser Teil): Argument-Namespace, Modellklasse `models.RLinear.Model`, Dataset `Dataset_ETT_hour`.
"""
import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch-size", type=int, default=128)
a = ap.parse_args()

repo = Path(a.repo).resolve()
sys.path.insert(0, str(repo))
os.chdir(repo)
from data_provider.data_loader import Dataset_ETT_hour  # noqa: E402
from models import RLinear  # noqa: E402

# Entspricht den Argumenten des Startbefehls (Defaults von run.py, uebrige aus script.md:3)
cfg = Namespace(seq_len=336, pred_len=96, channel=7, individual=False, drop=0.1, rev=True)
model = RLinear.Model(cfg).float()
model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
model.eval()

ds = Dataset_ETT_hour(root_path=str(Path(a.csv).parent), data_path=Path(a.csv).name, flag="test",
                      size=[336, 96], features="M", target="OT", timeenc=1, freq="h")
loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, drop_last=False, num_workers=0)
preds, trues = [], []
with torch.no_grad():
    for bx, by, _, _ in loader:
        out, _ = model(bx.float(), by.float())
        preds.append(out.numpy())
        trues.append(by.numpy())
pred, true = np.concatenate(preds), np.concatenate(trues)

mapping = common.check_mapping(true, a.csv)
out_dir = Path(a.out)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "mapping.json").write_text(json.dumps(mapping))
print("mapping", mapping)
if not mapping["ok"]:
    sys.exit("Zuordnungspruefung fehlgeschlagen, keine Vorhersagen geschrieben")
print("rows", common.write_predictions(pred, out_dir / "predictions.csv"))
