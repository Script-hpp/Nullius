"""ETT-Datensaetze (ETTh1, ETTh2, ETTm1, ETTm2; Horizont 96): eine parametrisierte Logik, ETTh1-Verhalten unveraendert, Leak-Schutz je Datensatz.

Ohne Netz: synthetische CSV im Format der ETDataset-Dateien. Echte Daten (Hashes) nur mit ETT_NETWORK_TESTS=1.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from repro_agent.harness import ett_data
from repro_agent.tasks import ETT, HARNESS_DIR, etth1_task, etth2_task, ett_data_description, ett_task, ettm1_task, ettm2_task

NAMES = ("etth1", "etth2", "ettm1", "ettm2")
EXPECT = {"etth1": (8640, 11520, 14400, 2785, 117), "etth2": (8640, 11520, 14400, 2785, 117),
          "ettm1": (34560, 46080, 57600, 11425, 120), "ettm2": (34560, 46080, 57600, 11425, 120)}


def make_csv(tmp_path, name, extra_rows=500):
    """CSV wie ETTh1.csv: date + HUFL..OT, mehr Zeilen als der Split nutzt (wie die echten Dateien)."""
    n = ett_data.borders(name)[2] + extra_rows
    rng = np.random.default_rng(3)
    data = np.cumsum(rng.normal(size=(n, 7)), axis=0) * 0.1 + np.arange(n)[:, None] * 1e-4
    path = tmp_path / ett_data.SPECS[name][0]
    with open(path, "w") as f:
        f.write("date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n")
        f.writelines(f"2016-01-01 00:00:00,{','.join(repr(float(v)) for v in row)}\n" for row in data)
    return path, data


@pytest.mark.parametrize("name", NAMES)
def test_borders_and_window_counts(name):
    tr, va, te, nw, nv = EXPECT[name]
    assert ett_data.borders(name) == (tr, va, te) and ett_data.n_windows(name) == nw and ett_data.n_val_windows(name) == nv
    assert ETT[name][3:7] == (tr, va - tr, nw, nv) and ETT[name][7] == ett_data.val_stride(name)


def test_etth1_binding_matches_frozen_old_formulas(tmp_path):
    from repro_agent.harness import etth1_data
    path, data = make_csv(tmp_path, "etth1")
    tr, va, te = 8640, 11520, 14400
    d = data[:te]
    z_old = (d - d[:tr].mean(0)) / d[:tr].std(0)
    assert np.array_equal(etth1_data.load_standardized(path), z_old)
    idx = va + np.arange(2785)[:, None] + np.arange(96)[None, :]
    assert np.array_equal(etth1_data.targets(path), z_old[idx])
    vidx = tr + 24 * np.arange(117)[:, None] + np.arange(96)[None, :]
    assert np.array_equal(etth1_data.val_targets(path), z_old[vidx])
    out = tmp_path / "p.npz"
    etth1_data.prepare_public(path, out)
    pub = np.load(out)
    starts = va + np.arange(2785)
    assert np.array_equal(pub["test_inputs"], z_old[starts[:, None] - 720 + np.arange(720)[None, :]].astype(np.float32))
    assert (etth1_data.TRAIN_END, etth1_data.VAL_END, etth1_data.TEST_END, etth1_data.VAL_STRIDE, etth1_data.PRED_LEN) == (8640, 11520, 14400, 24, 96)
    assert etth1_data.n_windows() == 2785 and etth1_data.n_val_windows() == 117


@pytest.mark.parametrize("name", NAMES)
def test_public_file_has_no_targets_and_val_never_touches_test(tmp_path, name):
    path, _ = make_csv(tmp_path, name)
    tr, va, te, nw, nv = EXPECT[name]
    z = ett_data.load_standardized(name, path)
    assert z.shape == (te, 7) and np.allclose(z[:tr].mean(0), 0) and np.allclose(z[:tr].std(0), 1)   # Rest der Datei ungenutzt
    out = tmp_path / "pub.npz"
    ett_data.prepare_public(name, path, out)
    d = np.load(out)
    assert sorted(d.files) == ["test_inputs", "train", "val"]
    assert d["train"].shape == (tr, 7) and d["val"].shape == (va - tr, 7) and d["test_inputs"].shape == (nw, 720, 7)
    assert np.array_equal(d["test_inputs"][0, -1], z[va - 1].astype(np.float32))      # Fenster 0: Eingabe endet direkt vor der Testprognose
    truth = ett_data.targets(name, path)
    assert truth.shape == (nw, 96, 7) and np.array_equal(truth[0, 0], z[va]) and np.array_equal(truth[-1, -1], z[te - 1])
    # keine Zielwerte im Agenten-Datensatz: Fenster i sieht nur Zeilen < va + i
    for i in (0, nw // 2, nw - 1):
        assert np.array_equal(d["test_inputs"][i], z[va + i - 720:va + i].astype(np.float32))
    vt = ett_data.val_targets(name, path)
    assert vt.shape == (nv, 96, 7) and np.array_equal(vt[-1, -1], z[tr + ett_data.val_stride(name) * (nv - 1) + 95])
    assert tr + ett_data.val_stride(name) * (nv - 1) + 95 < va                     # letztes Val-Ziel endet in Val


@pytest.mark.parametrize("name", NAMES)
def test_harness_eval_metric_and_validity(tmp_path, monkeypatch, name, capsys):
    path, _ = make_csv(tmp_path, name)
    monkeypatch.syspath_prepend(str(HARNESS_DIR))
    import ett_data as flat
    import ett_eval
    monkeypatch.setattr(flat, "download", lambda n, target: path)
    truth = flat.targets(name, path)
    pred_file = tmp_path / "predictions.csv"
    monkeypatch.setattr(ett_eval, "PREDICTIONS", str(pred_file))
    nw = truth.shape[0]

    def write(arr):
        w, h, c = np.meshgrid(np.arange(nw), np.arange(96), np.arange(7), indexing="ij")
        ids = np.char.add(np.char.add(np.char.add(w.ravel().astype(str), "_"), np.char.add(h.ravel().astype(str), "_")), c.ravel().astype(str))
        pred_file.write_text("id,prediction\n" + "\n".join(f"{i},{v!r}" for i, v in zip(ids, arr.ravel().tolist())) + "\n")

    write(truth + 0.5)
    ett_eval.main(name)
    res = json.loads(capsys.readouterr().out)
    assert res["status"] == "ok" and res["metrics"]["mse"] == pytest.approx(0.25) and res["metrics"]["mae"] == pytest.approx(0.5)
    assert set(res["baselines"]) == {"repeat", "zero"} and ("metrics_droplast" in res) == (name == "etth1")
    pred_file.write_text("id,prediction\n0_0_0,1.0\n")
    ett_eval.main(name)
    assert json.loads(capsys.readouterr().out)["status"] == "ungueltig"


@pytest.mark.parametrize("name,task", [("etth1", etth1_task), ("etth2", etth2_task), ("ettm1", ettm1_task), ("ettm2", ettm2_task)])
def test_task_definition_per_dataset(name, task):
    t = task()
    label, gran, unit, n_train, n_val, n_win, n_val_win, stride = ETT[name]
    d = t.data_description
    assert f"/data/{name}_public.npz" in d and f"[{n_train}, 7]" in d and f"[{n_val}, 7]" in d and f"[{n_win}, 720, 7]" in d
    assert f"0..{n_win - 1}" in d and f"{n_win}*96*7 = {n_win * 96 * 7} rows" in d and label in d
    assert f"window 0..{n_val_win - 1}" in t.val_notes and f"{n_train} + {stride}*w" in t.val_notes
    assert f"all {n_win} windows" in t.code_notes and f"/data/{name}_public.npz" in t.code_notes
    assert t.smoke_dims == (8, 96, 7) and len(t.smoke_ids) == 8 * 96 * 7 and "SMOKE=1" in t.code_notes
    assert t.eval_cmd == f"python {{harness_dir}}/{name}_eval.py" and t.setup_cmds[-1].startswith(f"test -s /data/{name}_public.npz")
    for fname in t.harness_files + list(t.val_harness_files):
        assert (HARNESS_DIR / fname).exists(), fname
    # Zielwerte nicht im Agenten-Datensatz: das Setup schreibt nur die public-Datei, die Rohdatei wird geloescht
    setup = (HARNESS_DIR / "ett_data.py").read_text()
    assert "shutil.rmtree(tmp)" in setup and "test_inputs" in setup and "targets" not in setup.split("def prepare_public")[1].split("def bind")[0]


def test_etth1_description_unchanged():
    d = ett_data_description("etth1")
    assert "test_inputs: float32 [2785, 720, 7]" in d and "(window i forecasts hours i .. i+95 of the 4-month test period)" in d
    assert "15-minute" in ett_data_description("ettm1") and "15-minute steps i .. i+95" in ett_data_description("ettm2")


@pytest.mark.skipif(not os.environ.get("ETT_NETWORK_TESTS"), reason="braucht Netz (ETDataset), ETT_NETWORK_TESTS=1")
def test_real_data_etth1_hashes_and_sizes(tmp_path):
    golden = {"test_inputs": "aa57959ee3720c08a905f6ea4063e3453ea0206e1765ac96410cd4cd4864b2a2",
              "train": "ae5ee3c28cf6884185c4c90a8273c63e007923db312382848d3a2a957dcc5811",
              "val": "00452ce8be2862546ad3e7871f06193a9d98bb1273f4cc5cb3117d3d1d904932"}
    from repro_agent.harness import etth1_data
    csv = etth1_data.download(tmp_path / "e")
    etth1_data.prepare_public(csv, tmp_path / "p.npz")
    d = np.load(tmp_path / "p.npz")
    assert {k: hashlib.sha256(np.ascontiguousarray(d[k]).tobytes()).hexdigest() for k in golden} == golden
    assert hashlib.sha256(np.ascontiguousarray(etth1_data.targets(csv)).tobytes()).hexdigest() == "6d2e61b4c3e9f74356f739a016635993c7bdd0e848b3a560e2d5c65d06ca076e"
    assert etth1_data.baselines(csv)["repeat"]["mse"] == pytest.approx(1.2943705947845083, abs=1e-12)
    for name in ("etth2", "ettm1", "ettm2"):
        c = ett_data.download(name, tmp_path / name)
        assert ett_data.load_standardized(name, c).shape == (ett_data.borders(name)[2], 7)
