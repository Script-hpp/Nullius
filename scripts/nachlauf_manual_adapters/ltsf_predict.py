"""Hand-Adapter LTSF-Linear/DLinear (Adapter-Vertrag des Modus Nachlauf): Modellklasse und Gewichte des Repos, Eingaben aus /data/etth1_public.npz.

Kein Datensatzcode des Repos, keine Rohdaten. Checkpoint ueber NACHLAUF_CKPT. SMOKE=1: nur die ersten 8 Fenster.
"""
import os
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, "/work/repo")
from models import DLinear  # noqa: E402

SEQ_LEN, PRED_LEN, CHANNELS = 336, 96, 7
cfg = Namespace(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=CHANNELS, individual=False)
model = DLinear.Model(cfg).float()
model.load_state_dict(torch.load(os.environ["NACHLAUF_CKPT"], map_location="cpu"))
model.eval()

x = np.load("/data/etth1_public.npz")["test_inputs"][:, -SEQ_LEN:, :]
if os.environ.get("SMOKE") == "1":
    x = x[:8]
parts = []
with torch.no_grad():
    for i in range(0, len(x), 256):
        xb = torch.from_numpy(x[i:i + 256]).float()
        parts.append(model(xb).numpy())
pred = np.concatenate(parts)

n = pred.shape[0]
w, s, c = np.meshgrid(np.arange(n), np.arange(PRED_LEN), np.arange(CHANNELS), indexing="ij")
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    f.write("\n".join(f"{a}_{b}_{d},{v:.8g}" for a, b, d, v in zip(w.ravel(), s.ravel(), c.ravel(), pred.astype(np.float64).ravel())))
    f.write("\n")
