"""NEGATIVKONTROLLE (c), muss abgelehnt werden: ehrliche Vorhersagen, Fensterreihenfolge zufaellig permutiert (feste Permutation).
Sonst wie der Hand-Adapter RTSF/RLinear (Adapter-Vertrag des Modus Nachlauf): Modellklasse und Gewichte des Repos, Eingaben aus /data/etth1_public.npz.

Kein Datensatzcode des Repos, keine Rohdaten. Checkpoint ueber NACHLAUF_CKPT. SMOKE=1: nur die ersten 8 Fenster.
RTSF: `model(x, y)` rechnet intern einen Verlust mit y; wir geben Nullen als y hinein, die Vorhersage haengt nicht davon ab.
"""
import os
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, "/work/repo")
from models import RLinear  # noqa: E402

SEQ_LEN, PRED_LEN, CHANNELS = 336, 96, 7
cfg = Namespace(seq_len=SEQ_LEN, pred_len=PRED_LEN, channel=CHANNELS, individual=False, drop=0.1, rev=True)
model = RLinear.Model(cfg).float()
model.load_state_dict(torch.load(os.environ["NACHLAUF_CKPT"], map_location="cpu"))
model.eval()

x = np.load("/data/etth1_public.npz")["test_inputs"][:, -SEQ_LEN:, :]
if os.environ.get("SMOKE") == "1":
    x = x[:8]
parts = []
with torch.no_grad():
    for i in range(0, len(x), 256):
        xb = torch.from_numpy(x[i:i + 256]).float()
        parts.append(model(xb, torch.zeros(xb.shape[0], PRED_LEN, CHANNELS))[0].numpy())
pred = np.concatenate(parts)
# --- negative control c: begin
pred = pred[np.random.default_rng(0).permutation(len(pred))]
# --- negative control c: end

n = pred.shape[0]
w, s, c = np.meshgrid(np.arange(n), np.arange(PRED_LEN), np.arange(CHANNELS), indexing="ij")
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    f.write("\n".join(f"{a}_{b}_{d},{v:.8g}" for a, b, d, v in zip(w.ravel(), s.ravel(), c.ravel(), pred.astype(np.float64).ravel())))
    f.write("\n")
