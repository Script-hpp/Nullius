"""Gegenproben Auftrag 24: Referenzloesungen lokal mit Hyperparametern/Protokoll des Agenten. Kein LLM, kein Tavily, keine Sandbox.
Aufruf: PYTHONPATH=src .venv/bin/python gp.py <variante> [n_seeds]"""
import os, statistics, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "repro_agent" / "harness")); sys.path.insert(0, str(ROOT / "src"))
import cora_data, cora_lcc_data, etth1_data, metrics
CACHE = Path(__file__).parent / "cache"   # needs raw/ind.cora.* (Planetoid); set before running
REF = ROOT / "tests" / "reference"

GCN_ES = ("""hist = []
for epoch in range(200):
    opt.zero_grad()
    F.cross_entropy(forward(True)[tr], y[tr]).backward()
    opt.step()
    with torch.no_grad():
        v = F.cross_entropy(forward(False)[va], y[va]).item()
    if epoch > 10 and v > np.mean(hist[-10:]):
        break
    hist.append(v)
""", """best_acc, best_state, wait = 0.0, None, 0
for epoch in range(200):
    opt.zero_grad()
    F.cross_entropy(forward(True)[tr], y[tr]).backward()
    opt.step()
    with torch.no_grad():
        a = (forward(False)[va].argmax(1) == y[va]).float().mean().item()
    if a > best_acc:
        best_acc, best_state, wait = a, [W1.detach().clone(), W2.detach().clone()], 0
    else:
        wait += 1
        if wait >= 10:
            break
with torch.no_grad():
    W1.copy_(best_state[0]); W2.copy_(best_state[1])
""")
NORM = ("X = X / X.sum(1, keepdim=True).clamp(min=1)\n", "")
SGC_GRID = ("(1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3)", "(5e-6,)")
NL_MODEL = ("""        self.linear = nn.Linear(L, H)

    def forward(self, x):  # x [B, L, C]
        last = x[:, -1:, :].detach()
        out = self.linear((x - last).transpose(1, 2)).transpose(1, 2)   # eine Schicht, von allen Kanaelen geteilt
        return out + last  # [B, H, C]
""", """        self.linear = nn.Linear(L * 7, H * 7)

    def forward(self, x):  # x [B, L, C]
        return self.linear(x.reshape(x.shape[0], -1)).view(x.shape[0], H, 7)
""")
def lin_train(lr, epochs, pat):   # ETTh1-Referenzen: feste Lernrate, andere Epochenzahl/Geduld
    return [("lr=0.005)", f"lr={lr})"), ("    if LR_HALVE:", "    if False:"), ("else 10):", f"else {epochs}):"), ("if patience >= 3:", f"if patience >= {pat}:")]

V = {  # name: (Referenzdatei, Task, Ersetzungen)
 "gcn_ref": ("gcn_cora", "cora", []),
 "gcn_es_valacc10": ("gcn_cora", "cora", [GCN_ES]),
 "gcn_nonorm": ("gcn_cora", "cora", [NORM]),
 "sgc_ref": ("sgc_cora", "cora", []),
 "sgc_wd5e-6": ("sgc_cora", "cora", [SGC_GRID]),
 "sgc_nonorm_wd5e-6": ("sgc_cora", "cora", [NORM, SGC_GRID]),
 "sgc_std_wd5e-6": ("sgc_cora", "cora", [(NORM[0], "X = (X - X.mean(0)) / X.std(0).clamp(min=1e-6)\n"), SGC_GRID]),
 "sgc_l2norm_wd5e-6": ("sgc_cora", "cora", [(NORM[0], "X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)\n"), SGC_GRID]),
 "sgc_nonorm_grid": ("sgc_cora", "cora", [NORM]),
 "sgc_nonorm_trainval": ("sgc_cora", "cora", [NORM, ("F.cross_entropy(lin(Z[tr]), y[tr]).backward()", "F.cross_entropy(lin(Z[torch.cat([tr, va])]), y[torch.cat([tr, va])]).backward()")]),
 "sgc_wd_repo_tuned": ("sgc_cora", "cora", [("(1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3)", "(1.3026973714043257e-05,)")]),
 "sgc_grid_repo_space": ("sgc_cora", "cora", [("(1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3)", "(1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4)")]),
 "dgi_ref": ("dgi_cora", "cora", []),
 "dgi_200ep": ("dgi_cora", "cora", [("range(1, 2001)", "range(1, 201)")]),
 "dgi_nonorm": ("dgi_cora", "cora", [NORM]),
 "dgi_nonorm_200ep": ("dgi_cora", "cora", [NORM, ("range(1, 2001)", "range(1, 201)")]),
 "dgi_h64_200ep": ("dgi_cora", "cora", [("7, 512", "7, 64"), ("range(1, 2001)", "range(1, 201)")]),
 "gat_ref": ("gat_cora", "cora", []),
 "gat_nonorm": ("gat_cora", "cora", [NORM]),
 "gat_4x4_noloops_nonorm": ("gat_cora", "cora", [NORM, ("0.6, 8, 8, 0.005", "0.6, 4, 4, 0.005"), ("src = torch.cat([edges[:, 0], loops])", "src = edges[:, 0]"), ("dst = torch.cat([edges[:, 1], loops])", "dst = edges[:, 1]")]),

 "gat_bothimprove": ("gat_cora", "cora", [("""    if v_acc >= vacc_mx or v_loss <= vlss_mn:
        if v_acc >= vacc_mx and v_loss <= vlss_mn:
            best_state = state()
        vacc_mx, vlss_mn, curr = max(v_acc, vacc_mx), min(v_loss, vlss_mn), 0
""", """    if v_acc > vacc_mx and v_loss < vlss_mn:
        best_state = state()
        vacc_mx, vlss_mn, curr = v_acc, v_loss, 0
""")]),

 "dgi_official_clf": ("dgi_cora", "cora", [("""mu, sd = emb[tr].mean(0), emb[tr].std(0).clamp(min=1e-6)
emb = (emb - mu) / sd
clf = torch.nn.Linear(H, n_cls)
copt = torch.optim.Adam(clf.parameters(), lr=0.01)
for _ in range(300):""", """clf = torch.nn.Linear(H, n_cls)
torch.nn.init.xavier_uniform_(clf.weight); torch.nn.init.zeros_(clf.bias)
copt = torch.optim.Adam(clf.parameters(), lr=0.01)
for _ in range(100):""")]),
 "gat_esacc": ("gat_cora", "cora", [("""    if v_acc >= vacc_mx or v_loss <= vlss_mn:
        if v_acc >= vacc_mx and v_loss <= vlss_mn:
            best_state = state()
        vacc_mx, vlss_mn, curr = max(v_acc, vacc_mx), min(v_loss, vlss_mn), 0
""", """    if v_acc > vacc_mx:
        best_state = state()
        vacc_mx, curr = v_acc, 0
""")]),
 "gcnlcc_ref": ("gcn_lcc", "lcc", []),
 "gcnlcc_kipf_hp": ("gcn_lcc", "lcc", [("H, P = 64, 0.8", "H, P = 16, 0.5"), ('"weight_decay": 1e-3}', '"weight_decay": 5e-4}'), ("0.5 * 1e-3 *", "0.5 * 5e-4 *")]),
 "gatlcc_ref": ("gat_lcc", "lcc", []),
 "gatlcc_l2x2_allparams": ("gat_lcc", "lcc", [("reg = 0.5 * L2 * sum((p ** 2).sum() for m in (l1, l2) for n, p in m.named_parameters() if n != \"bias\")", "reg = L2 * sum((p ** 2).sum() for m in (l1, l2) for n, p in m.named_parameters())")]),
 "gatlcc_velickovic_hp": ("gat_lcc", "lcc", [("0.6, 0.3, 8, 8, 0.01, 1e-2, 50", "0.6, 0.6, 8, 8, 0.005, 5e-4, 100")]),
 "gatlcc_hid64perhead": ("gat_lcc", "lcc", [("0.6, 0.3, 8, 8, 0.01", "0.6, 0.3, 8, 64, 0.01")]),
 "gatlcc_noextra": ("gat_lcc", "lcc", [("        h = F.dropout(h, P, train)  ", "        pass  ")]),
 "gatlcc_onedir": ("gat_lcc", "lcc", [("src = torch.cat([edges[:, 0], loops])", "keep = edges[:, 0] < edges[:, 1]\nsrc = torch.cat([edges[keep, 0], loops])"), ("dst = torch.cat([edges[:, 1], loops])", "dst = torch.cat([edges[keep, 1], loops])")]),
 "gcnlcc_randn_init": ("gcn_lcc", "lcc", [("torch.nn.init.xavier_uniform_(torch.empty(X.shape[1], H))", "torch.randn(X.shape[1], H)"), ("torch.nn.init.xavier_uniform_(torch.empty(H, n_cls))", "torch.randn(H, n_cls)")]),
 "dlinear_ref": ("dlinear", "etth1", []),
 "dlinear_L96_lr1e-4": ("dlinear", "etth1", [("L, H, KERNEL = 336", "L, H, KERNEL = 96")] + lin_train(0.0001, 10, 7)),
 "dlinear_lr1e-3_20ep": ("dlinear", "etth1", lin_train(0.001, 20, 5)),
 "dlinear_lr5e-3_nohalve": ("dlinear", "etth1", lin_train(0.005, 10, 5)),
 "nlinear_ref": ("nlinear", "etth1", []),
 "nlinear_lr5e-3_nohalve": ("nlinear", "etth1", lin_train(0.005, 10, 5)),
 "nlinear_flat_nolast_lr1e-3": ("nlinear", "etth1", [NL_MODEL] + lin_train(0.001, 50, 5)),
 "nlinear_lr5e-4": ("nlinear", "etth1", lin_train(0.0005, 20, 7)),
}

def run(name, seed):
    ref, task, patches = V[name]
    code = (REF / f"{ref}_reference.py").read_text()
    for old, new in patches:
        assert code.count(old) >= 1, (name, old)
        code = code.replace(old, new)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp); raw = CACHE / "raw"
        if task == "etth1":
            csv = etth1_data.download(CACHE / "ett"); etth1_data.prepare_public(csv, d / "p.npz"); code = code.replace("/data/etth1_public.npz", str(d / "p.npz"))
        elif task == "cora":
            cora_data.prepare_public(raw, d / "p.npz"); code = code.replace("/data/cora_public.npz", str(d / "p.npz"))
        else:
            cora_lcc_data.prepare_public(raw, seed, d / "p.npz"); code = code.replace("/data/cora_lcc_public.npz", str(d / "p.npz"))
        code = code.replace('"/output/predictions.csv"', repr(str(d / "pred.csv"))).replace('"/output"', repr(str(d)))
        (d / "train.py").write_text(code)
        r = subprocess.run([sys.executable, str(d / "train.py")], capture_output=True, text=True, env={"SEED": str(seed), "SPLIT": str(seed), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "2", "LR_HALVE": "1"})
        if r.returncode: raise RuntimeError(r.stderr[-600:])
        rows = (d / "pred.csv").read_text().splitlines()[1:]
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if task == "etth1":
            tgt = etth1_data.targets(csv); pred = np.array([float(l.split(",")[1]) for l in rows])
            return metrics.mse(pred.tolist(), tgt.reshape(-1).tolist()), tail
        truth = cora_data.test_labels(raw) if task == "cora" else cora_lcc_data.test_labels(raw, seed)
        assert len(rows) == len(truth)
        return float(np.mean([truth[i] == p for i, p in (l.split(",") for l in rows)])), tail

if __name__ == "__main__":
    name = sys.argv[1]; task = V[name][1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else {"cora": 10, "lcc": 30, "etth1": 5}[task]
    t0 = time.time(); vals = []; tails = []
    for s in range(n):
        v, t = run(name, s); vals.append(v); tails.append(t)
    print(f"{name}: n={n} Mittel {statistics.mean(vals):.4f} Std {statistics.stdev(vals) if n > 1 else 0:.4f} min {min(vals):.4f} max {max(vals):.4f} | {time.time()-t0:.0f}s | Werte {[round(v, 4) for v in vals]} | letzte Ausgabe Seed 0: {tails[0][:70]}", flush=True)
